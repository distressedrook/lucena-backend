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

No engine, no LLM, no Maia here — everything is a table lookup or board
geometry, so the endpoint is safe to call on every navigator scrub.
"""

from __future__ import annotations

from lucena_core import openings
from lucena_core.board import Board
from lucena_core import content as authored

DOOR_CAP = 4          # labeled continuations shown on the theory card
IDEA_SENTENCES = 2    # authored annotations are essays; the card takes the lead


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


def build(fen: str, *, seed: str = "") -> dict:
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

    # 1. move 1 — the book opens
    if plies <= 1:
        q = authored.epigraph(seed or fen)
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

    # 3. out of book — the plans zone is a pending owner conversation; until
    # then, an honest minimal status (never an empty 404 mid-game).
    out["statusLine"] = f"MOVE {move_no}"
    out["cards"] = [{
        "id": "position", "title": "Position", "count": None,
        "sections": [{"heading": None, "rows": [
            {"text": "Out of book.", "moves": [], "squares": []},
        ]}],
    }]
    return out
