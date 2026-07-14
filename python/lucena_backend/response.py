"""Compact tool responses — the reliability contract in wire form (LLD §4.1).

Every engine-tool response carries a **piece list** (eliminates board-misreading),
the **verbatim eval/PV**, and the **fact sheet** (top 5), and stays **≤ ~300
tokens**. This module builds those pieces and enforces the ceiling; the tools in
`tools.py` assemble them and add the board side effects.
"""

from __future__ import annotations

import json

from lucena_engine.board import Board
from lucena_engine.evalmodel import Score, win_pct_from_score

MAX_TOKENS = 400   # per-response ceiling; raised from 300 to fit the positional read
_PIECE_ORDER = "KQRBNP"


def error(code: str, detail: str) -> dict:
    """The deterministic error shape — never a stack trace into the chat."""
    return {"error": code, "detail": detail}


def piece_list(board: Board) -> str:
    """Compact two-line roster: 'White: Ke1,Qd1,…\\nBlack: …' (fewest tokens)."""
    white, black = [], []
    for p in board.piece_list():
        (white if p.color == "white" else black).append(p)
    key = lambda p: (_PIECE_ORDER.index(p.piece), p.square)
    fmt = lambda ps: ",".join(f"{p.piece}{p.square}" for p in sorted(ps, key=key))
    return f"White: {fmt(white)}\nBlack: {fmt(black)}"


def eval_block(score: Score) -> dict:
    """Eval from the scored side's POV: `{cp, win_pct}` (win_pct 1 dp)."""
    return {
        "cp": score.to_ceiled_cp(),
        "win_pct": round(win_pct_from_score(score), 1),
    }


_MATERIAL_VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
_VALUE_ORDER = ["Q", "R", "B", "N", "P"]     # heaviest first, for reading order
_PIECE_WORD = {"Q": "queen", "R": "rook", "B": "bishop", "N": "knight", "P": "pawn"}
_COUNT_WORD = {2: "two", 3: "three", 4: "four", 5: "five",
               6: "six", 7: "seven", 8: "eight"}


def material(board: Board) -> dict:
    """Deterministic material balance so the coach never *counts* (LLMs miscount —
    they'll claim "up a rook" with rooks on both sides). Returns each side's point
    total, the net (White POV, in pawns), and a plain-English `standing` the coach
    reads verbatim — the *exact* imbalance ("up a rook for two pawns"), never a
    number or a fuzzy bucket it has to interpret or hedge around."""
    white = black = 0
    counts = {"white": {pc: 0 for pc in _VALUE_ORDER},
              "black": {pc: 0 for pc in _VALUE_ORDER}}
    for p in board.piece_list():
        white += _MATERIAL_VALUE.get(p.piece, 0) if p.color == "white" else 0
        black += _MATERIAL_VALUE.get(p.piece, 0) if p.color == "black" else 0
        if p.piece in _PIECE_WORD:
            counts[p.color][p.piece] += 1
    net = white - black
    return {"white": white, "black": black, "net": net, "standing": _standing(counts)}


def _pieces_phrase(surplus: list[tuple[str, int]]) -> str:
    """"a rook", "two pawns", "a rook and a pawn", "a rook, a knight, and a pawn"."""
    parts = []
    for pc, n in surplus:
        word = _PIECE_WORD[pc]
        parts.append(f"a {word}" if n == 1 else f"{_COUNT_WORD.get(n, str(n))} {word}s")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _standing(counts: dict) -> str:
    """The exact material imbalance in plain English. Cancels an even knight↔bishop
    trade (both minors, both 3) so a N-for-B swap reads as level rather than
    cluttering the phrase, then names precisely what each side has extra — a clean
    surplus ("up a rook"), the exchange ("up the exchange"), or an imbalance ("up a
    rook for two pawns")."""
    w, b = counts["white"], counts["black"]
    dw = {pc: max(0, w[pc] - b[pc]) for pc in _VALUE_ORDER}
    db = {pc: max(0, b[pc] - w[pc]) for pc in _VALUE_ORDER}
    for a, c in (("N", "B"), ("B", "N")):   # a minor-for-minor swap is even; cancel it
        t = min(dw[a], db[c])
        dw[a] -= t
        db[c] -= t
    w_up = [(pc, dw[pc]) for pc in _VALUE_ORDER if dw[pc] > 0]
    b_up = [(pc, db[pc]) for pc in _VALUE_ORDER if db[pc] > 0]
    if not w_up and not b_up:
        return "material is even"
    if not b_up:
        return f"White is up {_pieces_phrase(w_up)}"
    if not w_up:
        return f"Black is up {_pieces_phrase(b_up)}"
    # both sides have surplus -> an imbalance; lead with whoever is ahead on points
    pts = (sum(_MATERIAL_VALUE[pc] * n for pc, n in w_up)
           - sum(_MATERIAL_VALUE[pc] * n for pc, n in b_up))
    is_exchange = lambda more, less: more == [("R", 1)] and less in ([("N", 1)], [("B", 1)])
    if pts == 0:
        return (f"material is level but imbalanced — White has {_pieces_phrase(w_up)} "
                f"for Black's {_pieces_phrase(b_up)}")
    lead, comp, side = (w_up, b_up, "White") if pts > 0 else (b_up, w_up, "Black")
    if is_exchange(lead, comp):
        return f"{side} is up the exchange"
    return f"{side} is up {_pieces_phrase(lead)} for {_pieces_phrase(comp)}"


def pv_san(fen: str, pv: list[str], max_plies: int = 6) -> list[str]:
    out, b = [], Board(fen)
    for uci in pv[:max_plies]:
        try:
            out.append(b.san(uci))
            b = b.apply(uci)
        except Exception:
            break
    return out


def fact_wire(fact) -> dict:
    """Response form of a Fact — id/kind/squares/text/concept_id (drops the
    internal salience/provenance to save tokens)."""
    return {
        "id": fact.id, "kind": fact.kind, "squares": list(fact.squares),
        "text": fact.text, "concept_id": fact.concept_id,
    }


def facts_to_board(facts, *, style: str = "analysis") -> tuple[list[dict], list[dict]]:
    """Turn facts into board arrows/highlights, each tagged with its `fact_id`.

    A two-square fact draws an arrow `squares[0]→squares[1]`; a single-square
    fact a highlight on that square."""
    arrows, highlights = [], []
    for f in facts:
        if len(f.squares) >= 2:
            arrows.append({"from": f.squares[0], "to": f.squares[1],
                           "style": style, "fact_id": f.id})
        elif f.squares:
            highlights.append({"square": f.squares[0], "style": style, "fact_id": f.id})
    return arrows, highlights


def estimate_tokens(obj) -> int:
    """Rough token count of a response (≈ 4 chars/token)."""
    return (len(json.dumps(obj, ensure_ascii=False)) + 3) // 4


def enforce_budget(response: dict, max_tokens: int = MAX_TOKENS) -> dict:
    """Trim a response to the token ceiling: shorten PVs, then drop the
    lowest-salience facts, until it fits. Facts are already ≤5 and PVs ≤6, so
    this rarely bites — it is a hard guarantee, not the primary control."""
    if estimate_tokens(response) <= max_tokens:
        return response
    # 1) trim PVs to 4 plies
    for line in response.get("lines", []):
        line["pv_san"] = line.get("pv_san", [])[:4]
    # 2) drop facts from the tail (lowest salience — they arrive salience-sorted)
    while estimate_tokens(response) > max_tokens and response.get("facts"):
        response["facts"].pop()
    # 3) shed positional-read `features` under pressure, least-salient term first,
    # so the piece roster and all five standings (the essential read) survive and
    # only the extra citable specifics are trimmed.
    pos = response.get("positional")
    if pos:
        ranked = sorted((n for n, t in pos["terms"].items() if "features" in t),
                        key=lambda n: abs(pos["terms"][n]["cp"]))
        for n in ranked:
            if estimate_tokens(response) <= max_tokens:
                break
            del pos["terms"][n]["features"]
    return response
