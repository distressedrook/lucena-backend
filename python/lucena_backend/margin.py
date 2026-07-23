"""The MARGIN's content builder — deterministic, engine-free, per-cursor-fast.

Serves the v1 right column (mac-client/V1_LAYOUT.md). Wired states, per the
owner's staging (2026-07-24):

  1. move 1 (plies 0–1)  → the EPIGRAPH: a fact-checked quote, seeded by the
     session so a game keeps its quote like a book keeps its epigraph.
  2. from move 2, in book → the THEORY card: the opening names itself
     (masthead), the authored annotation leads, and the labeled DOORS show
     where each next move goes (child-position name lookups).
  3. out of book → NOT YET WIRED (the |eval| ≤ 1.5 plans zone is a pending
     owner conversation); returns a bare position status so the margin never
     404s mid-game.

The INSTANT layer (this module's synchronous path) stays engine-free — table
lookups and board geometry, safe per navigator scrub. The DEEP layer (owner
rulings, 2026-07-24 stage 2) runs in ONE background worker for the LIVE
position only: an eval probe routes the out-of-book margin between the plans
zone (|eval| ≤ 1.5, middlegame → the lucena-plans sheet as per-side PLAN
cards + position-card assessment/structure/weaknesses) and the loud zone /
endgame (eval-in-words only; census facts are already instant). Results are
cached per position; the app polls while `plansPending`.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from lucena_core import openings
from lucena_core.board import Board
from lucena_core import content as authored
from lucena_core.reads import material

_log = logging.getLogger(__name__)

DOOR_CAP = 4          # labeled continuations shown on the theory card
IDEA_SENTENCES = 2    # authored annotations are essays; the card takes the lead
DEEP_NODES = 300_000  # the routing eval probe (background, live position only)
WEAKNESS_ROWS = 3     # per side, on the position card

# -- the deep layer's plumbing (configured once by httpserver) ---------------
_pool = None          # EnginePool — leased per job
_maia = None          # MaiaEngine | None (plans verify degrades without it)
_deep_cache: dict[str, dict] = {}      # norm fen -> deep result
_inflight: set[str] = set()
_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)   # ONE: plans rolls are heavy


def configure(pool, maia) -> None:
    """Called once at server build; without it the deep layer stays off and
    the margin serves the instant layer only (tests, offline)."""
    global _pool, _maia
    _pool, _maia = pool, maia


def _plies_played(fen: str) -> int:
    """Half-moves since the start, derived from the FEN's move counters."""
    parts = fen.split()
    fullmove = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
    white_to_move = len(parts) > 1 and parts[1] == "w"
    return (fullmove - 1) * 2 + (0 if white_to_move else 1)


def _lead_sentences(text: str, n: int) -> str:
    """The first n sentences of an authored annotation — the card's budget."""
    out, count, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1] in " \n"):
            out.append(text[start:i + 1].strip())
            count += 1
            start = i + 1
            if count >= n:
                break
    return " ".join(out) if out else text.strip()


def _doors(board: Board, current: str | None, cap: int = DOOR_CAP) -> list[dict]:
    """Named continuations: each legal move whose resulting position the
    openings table can name. Ordered THEORY-FIRST, not legal-move-order
    (caught in tests: a b3 sideline stole 'Scandinavian Defense' from exd5):
    moves that stay in the current opening's family come first, deeper
    variation names before generic ones. Typicality (Maia %) is a later wire."""
    family = (current or "").split(":")[0].strip()

    def best_grandchild_name(child: Board) -> str | None:
        """The openings table skips forced intermediate positions (after
        2.exd5 the name lives on 2...Qxd5) — look one reply deeper and take
        the strongest name by the same family-first, most-specific order."""
        names = []
        for reply in child.legal_moves():
            n = openings.name_for(child.apply(reply).fen)
            if n:
                names.append(n)
        if not names:
            return None
        names.sort(key=lambda n: (0 if family and n.startswith(family) else 1,
                                  -len(n), n))
        return names[0]

    found = []
    for uci in board.legal_moves():
        child = board.apply(uci)
        name = openings.name_for(child.fen) or best_grandchild_name(child)
        if name:
            found.append((uci, name))
    # family continuations first; within a bucket, more specific (longer)
    # names first — the mainline door names the deepest known theory.
    found.sort(key=lambda p: (0 if family and p[1].startswith(family) else 1,
                              -len(p[1]), p[1]))
    doors, seen = [], set()
    for uci, name in found:
        if name in seen:
            continue
        seen.add(name)
        doors.append({"san": board.san(uci), "variation": name, "typicalPct": None})
        if len(doors) >= cap:
            break
    return doors


def _eval_words(cp_white: int) -> str:
    """cp (White POV) → the margin's words. Never a number (house rule)."""
    side = "White" if cp_white >= 0 else "Black"
    mag = abs(cp_white)
    if mag < 40: return "Roughly equal"
    if mag < 120: return f"{side} is slightly better"
    if mag < 300: return f"{side} is clearly better"
    return f"{side} is winning"


def _sheet_sections(sheet: str) -> dict[str, list[str]]:
    """Parse the lucena-plans fact sheet (HEADER: + two-space-indented lines)
    into {header: [lines]}. Inline headers (ASSESSMENT:, STRUCTURE:) yield
    their own single line."""
    out: dict[str, list[str]] = {}
    current = None
    for raw in sheet.splitlines():
        is_header = (raw and not raw.startswith(" ") and ":" in raw
                     and (raw.partition(":")[0].isupper() or raw.startswith("===")))
        if is_header:
            head, _, rest = raw.partition(":")
            current = head.strip("= ").strip()
            if rest.strip():
                out.setdefault(current, []).append(rest.strip())
        elif current and raw.strip():
            # section content: two-space indents (weaknesses, reads) AND
            # dash bullets (plan lines — caught live: plans start '- ',
            # not '  ', and were silently dropped)
            out.setdefault(current, []).append(raw.strip().lstrip("- ").strip())
    return out


def _deep_job(fen: str) -> None:
    """The background pass for one live position: eval probe → route →
    (plans sheet | loud/endgame words). Every failure caches an empty deep
    result so the app's polling terminates."""
    result: dict = {"evalLine": None, "cards": [], "positionRows": []}
    try:
        from .plans import service as _plans
        with _pool.lease() as engine:
            engine.new_game()
            a = engine.analyse(fen, nodes=DEEP_NODES, multipv=1)
            cp = a.best.score.to_ceiled_cp()
        white_to_move = " w " in f" {fen} "
        cp_white = cp if white_to_move else -cp
        result["evalLine"] = _eval_words(cp_white)
        if abs(cp_white) <= _plans.PLANS_CP_BAND and not _plans.is_endgame(fen):
            sheet, _pid = _plans.sheet_for(fen, _pool, _maia)
            sec = _sheet_sections(sheet)
            if sec.get("ASSESSMENT"):
                result["evalLine"] = sec["ASSESSMENT"][0].rstrip(".")
            rows = []
            if sec.get("STRUCTURE"):
                rows.append({"text": sec["STRUCTURE"][0], "moves": [], "squares": []})
            for side in ("WHITE", "BLACK"):
                for ln in sec.get(f"WEAKNESSES FOR {side}", [])[:WEAKNESS_ROWS]:
                    rows.append({"text": f"{side.title()}: {ln}", "moves": [], "squares": []})
            result["positionRows"] = rows
            for side in ("WHITE", "BLACK"):
                lines = [ln for ln in sec.get(f"PLAN FOR {side}", [])
                         if not ln.startswith("no plan is confirmed")]
                if lines:
                    result["cards"].append({
                        "id": f"plan-{side.lower()}", "title": f"Plan for {side.title()}",
                        "count": None,
                        "sections": [{"heading": None, "rows": [
                            {"text": ln, "moves": [], "squares": []} for ln in lines
                        ]}],
                    })
    except Exception:
        _log.warning("margin deep layer failed for %s", fen, exc_info=True)
    key = " ".join(fen.split()[:4])
    with _lock:
        _deep_cache[key] = result
        _inflight.discard(key)
        if len(_deep_cache) > 64:                 # bounded: a session's worth
            _deep_cache.pop(next(iter(_deep_cache)))


def build(fen: str, *, seed: str = "", live: bool = False) -> dict:
    """MarginContent for `fen`, shaped exactly as the mac model decodes it."""
    board = Board(fen)                    # validates; raises ValueError on garbage
    plies = _plies_played(fen)
    name = openings.name_for(fen)
    move_no = (plies // 2) + 1

    out: dict = {
        "masthead": name,
        "statusLine": None,
        "epigraph": None,
        "theory": None,
        "rookLine": None,
        "cards": [],
        "urgent": None,
        "commandHints": [],
    }

    # 1. move 1 — the book opens. The seed fallback is the DAY, never the fen
    # (a fen seed changed the quote between ply 0 and ply 1 — caught live;
    # a book keeps its epigraph).
    if plies <= 1:
        import datetime
        q = authored.epigraph(seed or datetime.date.today().isoformat())
        out["epigraph"] = {"quote": q["quote"], "author": q["author"],
                           "source": q.get("source")}
        return out

    # 2. in book — theory until the table stops naming positions
    if name:
        out["statusLine"] = f"OPENING · MOVE {move_no}"
        idea = authored.annotation_for(name)
        out["theory"] = {
            "idea": _lead_sentences(idea, IDEA_SENTENCES) if idea else None,
            "doors": _doors(board, name),
        }
        return out

    # 3. out of book — the INSTANT layer: ONE position card (owner: a separate
    # facts card was the same thing twice). Rows: material standing, then the
    # census facts; the deep layer folds its eval/structure/weaknesses into
    # the same card when it lands.
    out["statusLine"] = f"MOVE {move_no}"
    pos_rows = [{"text": material(board)["standing"], "moves": [], "squares": []}]
    try:
        from .grounding_tools.facts import build_fact_sheet
        for f in build_fact_sheet(board, None):
            if f.kind != "opening":
                pos_rows.append({"text": f.text, "moves": [], "squares": f.squares[:3]})
    except Exception:
        _log.warning("census facts unavailable", exc_info=True)
    out["cards"] = [{"id": "position", "title": "Position", "count": None,
                     "sections": [{"heading": None, "rows": pos_rows}]}]

    # the DEEP layer — live position only (owner: scrubs never trigger rolls)
    key = " ".join(fen.split()[:4])
    deep = _deep_cache.get(key)
    if deep is not None:
        if deep["evalLine"]:
            pos_rows.insert(0, {"text": deep["evalLine"], "moves": [], "squares": []})
        pos_rows.extend(deep["positionRows"])
        out["cards"] = [out["cards"][0]] + deep["cards"]
        out["plansPending"] = False
    elif live and _pool is not None:
        with _lock:
            if key not in _inflight:
                _inflight.add(key)
                _worker.submit(_deep_job, fen)
        out["plansPending"] = True
    return out
