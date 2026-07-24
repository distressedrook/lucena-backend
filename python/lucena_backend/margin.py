"""The MARGIN's content builder — authored content + the plans layer.

Staged by ply (owner 2026-07-24, "look inside /content, wire that up"):

  1. move 1 (plies 0-1) → the EPIGRAPH, from the fact-checked quote table.
  2. from move 2, in book → the THEORY card: masthead + the AUTHORED opening
     annotation + labeled doors.
  3. out of book → the plans layer's artifacts. ONE background worker rolls
     the position and emits the PRE-VERIFY JSON the moment it exists (no
     verify_plan calls yet) — shown while `plansPending` stays true; the
     verify gate then replaces it with POST-VERIFY and `plansPending` drops,
     ending the app's polling.

Stages 1-2 (the card builder) were removed in the 2026-07-23 JSON-inspection
pass and are restored here verbatim from backend fb4b2d7, now that
/content is wired through `lucena_core.content`. They are the INSTANT layer:
authored lookups + board geometry, engine-free, safe on every navigator
scrub, and carrying NO model output — the annotations are human-authored and
source-checked, not generated per turn.

Results are cached per position; the deep pass runs for the LIVE position
only (scrubs never trigger rolls — standing rule).
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from lucena_core import content as authored
from lucena_core import openings
from lucena_core.board import Board

_log = logging.getLogger(__name__)

DOOR_CAP = 4          # labeled continuations shown on the theory card
IDEA_SENTENCES = 2    # authored annotations are essays; the card takes the lead

# -- plumbing (configured once by httpserver) ---------------------------------
_pool = None          # EnginePool — leased per job
_maia = None          # MaiaEngine | None (plans verify degrades without it)
_deep_cache: dict[str, dict] = {}      # norm fen -> {"raw", "statusLine", "pending"}
_inflight: set[str] = set()
_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)   # ONE: plans rolls are heavy


def configure(pool, maia) -> None:
    """Called once at server build; without it the deep layer stays off and
    the margin serves nothing (tests, offline)."""
    global _pool, _maia
    _pool, _maia = pool, maia


def _cache(key: str, sheet: dict, status: str, pending: bool) -> None:
    with _lock:
        _deep_cache[key] = {"sheet": sheet, "raw": json.dumps(sheet, indent=2),
                            "statusLine": status, "pending": pending}
        if not pending:
            _inflight.discard(key)
        if len(_deep_cache) > 64:                 # bounded: a session's worth
            _deep_cache.pop(next(iter(_deep_cache)))


def _deep_job(fen: str) -> None:
    """Roll once; publish PRE the moment it exists, POST when verify lands.
    Every failure caches a terminal result so the app's polling terminates."""
    key = " ".join(fen.split()[:4])
    try:
        from .plans import service as _plans
        _pre, post = _plans.sheet_json_staged(
            fen, _pool, _maia,
            on_pre=lambda pre: _cache(key, pre, "PRE-VERIFY · VERIFYING…", True))
        _cache(key, post, "POST-VERIFY", False)
    except Exception:
        _log.warning("margin deep layer failed for %s", fen, exc_info=True)
        _cache(key, {"error": "sheet failed — see backend log"}, "ERROR", False)


# -- the INSTANT layer's card helpers (restored verbatim from backend
# fb4b2d7 — audited code, unchanged; only build()'s staging was rewired)

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


def _blank(**over) -> dict:
    """Every key the mac model decodes, so no stage 404s a field."""
    out = {"masthead": None, "statusLine": None, "epigraph": None,
           "theory": None, "sheet": None, "raw": None, "plansPending": False}
    out.update(over)
    return out


def build(fen: str, *, seed: str = "", live: bool = False) -> dict:
    """MarginContent for `fen`. Three staged states (the owner's 2026-07-24
    staging, restored from backend fb4b2d7 when /content was wired up):

      1. move 1 (plies 0-1) -> the EPIGRAPH: a fact-checked authored quote,
         seeded by the SESSION so a game keeps its quote like a book keeps
         its epigraph. (The seed fallback is the DAY, never the fen — a fen
         seed changed the quote between ply 0 and ply 1, caught live.)
      2. from move 2, in book -> the THEORY card: the opening names itself
         (masthead), the AUTHORED annotation leads, and the labeled doors
         show where each next move goes.
      3. out of book -> the plans layer's sheet + raw JSON (the deep worker).

    Stages 1-2 are the INSTANT layer: authored table lookups and board
    geometry, engine-free and safe on every navigator scrub. They are also
    fully deterministic and carry no model output at all — the annotations
    are human-authored and source-checked (lucena_core.content), not
    generated per turn.
    """
    board = Board(fen)                    # validates; raises ValueError on garbage
    plies = _plies_played(fen)
    move_no = (plies // 2) + 1

    # THE AUTHORED FIELDS ARE ADDITIVE, never a short-circuit. They were
    # staged as early returns first (faithful to fb4b2d7) and that BROKE
    # THE APP: mac's MarginContent decodes only {statusLine, raw, sheet,
    # plansPending} — it has no masthead/epigraph/theory — so every move-1
    # and in-book position rendered blank. A wire contract is owned by its
    # CONSUMER; new fields ride alongside the old ones until the client
    # decodes them.
    out: dict = {}

    # move 1 — the book opens. (Seed fallback is the DAY, never the fen: a
    # fen seed changed the quote between ply 0 and ply 1.)
    if plies <= 1:
        import datetime
        q = authored.epigraph(seed or datetime.date.today().isoformat())
        out["epigraph"] = {"quote": q["quote"], "author": q["author"],
                           "source": q.get("source")}

    # in book — the authored theory card.
    name = openings.name_for(fen)
    if name:
        idea = authored.annotation_for(name)
        out["masthead"] = name
        out["theory"] = {
            "idea": _lead_sentences(idea, IDEA_SENTENCES) if idea else None,
            "doors": _doors(board, name)}

    # the plans layer — unchanged, and still runs for EVERY position, so
    # sheet/raw/plansPending keep behaving exactly as the client expects.
    key = " ".join(fen.split()[:4])
    cached = _deep_cache.get(key)
    if cached is not None:
        return _blank(statusLine=cached["statusLine"], sheet=cached["sheet"],
                      raw=cached["raw"], plansPending=cached["pending"], **out)
    if live and _pool is not None:
        with _lock:
            if key not in _inflight:
                _inflight.add(key)
                _worker.submit(_deep_job, fen)
        return _blank(statusLine="ROLLING…", plansPending=True, **out)
    return _blank(statusLine=out.get("masthead") and f"OPENING · MOVE {move_no}"
                  or f"MOVE {move_no}", **out)
