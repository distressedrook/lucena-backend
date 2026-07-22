"""Moved from lucena-engine (core-migration Phase 5, 2026-07-23): assembly of
positional + tactical + eval material into the coach's briefing is backend
composition, not engine truth.

The coach's grounded briefing — deterministic natural-language analysis.

`build_analysis(board, engine)` assembles ONE readable, number-bearing briefing
from all the raw calculated material: the engine eval, the material balance, the
five-term positional read, and the tactical fact sheet. It is *a function of the
raw facts* — no LLM in its production, same board ⇒ same briefing.

It is the coach's INPUT, not its output: the coach reads it to understand the
position, then interprets it into player-facing coaching (translating numbers to
meaning). So — unlike a beat — the briefing may carry things the coach must never
say aloud: centipawns, win%. It may be surfaced in the UI or not; that is a
display choice, not part of what it is.

Every line is one grounded statement in plain English (all White's point of view,
so the coach never has to flip perspective). The tactical facts arrive already in
natural language from the fact sheet; the positional standings already read as
sentences; this module stitches them into a briefing and prepends the eval.
"""

from __future__ import annotations

from lucena_engine.evalmodel import win_pct_from_score
from .facts import build_fact_sheet  # the lucena-tactics shim
from lucena_core.positional import analyze_positional


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _eval_line(board, score) -> str:
    """Turn + verdict + the raw numbers (White POV), mate-aware."""
    stm = board.side_to_move
    turn = "White" if stm == "white" else "Black"
    if score.is_mate:
        if score.mate == 0:
            return f"{turn} to move and is checkmated."
        # `mate` is signed from the side-to-move POV: >0 means the mover delivers it
        mover = turn if score.mate > 0 else ("Black" if stm == "white" else "White")
        m = abs(score.mate)
        if m <= 5:
            return f"{turn} to move; {mover} has a forced mate in {m}."
        return f"{turn} to move; {mover} is easily winning."   # too far out to recite a count
    wp_stm = win_pct_from_score(score)
    white_wp = wp_stm if stm == "white" else 100 - wp_stm
    white_cp = score.to_ceiled_cp() if stm == "white" else -score.to_ceiled_cp()
    d = white_wp - 50
    if abs(d) <= 5:
        verdict = "the position is roughly equal"
    else:
        leader = "White" if d > 0 else "Black"
        aw = white_wp if d > 0 else 100 - white_wp
        band = ("is completely winning" if aw >= 90 else
                "is winning" if aw >= 75 else
                "is clearly better" if aw >= 60 else
                "is slightly better")
        verdict = f"{leader} {band}"
    return (f"{turn} to move; {verdict} "
            f"(eval {white_cp / 100:+.1f}, White {round(white_wp)}%).")


# Salience gates for the briefing (cp, White POV). Stricter than positional._LEAD_THRESHOLD (25):
# the briefing decides what to even STATE, so a near-zero term never becomes a line at all.
_SHEET_THRESHOLD = 50    # |cp| a term must clear to be worth stating
_COMP_EVAL = 100         # |eval| <= 1.00 pawn: the position is, on balance, level
_COMP_HUGE = 150         # ...yet one dimension is worth >~1.5 pawns — that gap is being COMPENSATED

_TERM_LABEL = {"material": "material", "king_safety": "king safety",
               "activity": "piece activity", "pawns": "pawn structure", "center": "the centre"}


def _white_cp(board, score) -> int:
    """The eval in centipawns, White's point of view (mate → a large signed sentinel)."""
    if score.is_mate:
        white_wins = (board.side_to_move == "white") == (score.mate > 0)
        return 100_000 if white_wins else -100_000
    cp = score.to_ceiled_cp()
    return cp if board.side_to_move == "white" else -cp


def assemble_analysis(board, best_score, pos, facts) -> list[str]:
    """Pure assembly of the briefing from already-computed pieces — the eval
    `best_score` (Score), the `analyze_positional` dict `pos`, and the fact-sheet
    list `facts`. No engine call, so the MCP layer can reuse facts it already
    built (for the board arrows) instead of recomputing the sheet.

    The five positional terms are RANKED by magnitude and only the salient ones are stated: a flat
    position clears nothing and says so once, instead of five 'even / safe / balanced / healthy'
    filler lines. The strongest term is named the 'Main factor'. When the eval is level DESPITE a big
    one-dimension gap, the opposing factors are surfaced as the other side's 'compensation' (a gambit);
    when the eval is decisive they are merely secondary context, not compensation."""
    terms = pos["terms"]
    lines = [_eval_line(board, best_score)]
    eval_cp = _white_cp(board, best_score)
    ranked = [t for t in sorted(terms, key=lambda k: -abs(terms[k]["cp"]))
              if abs(terms[t]["cp"]) >= _SHEET_THRESHOLD]
    if not ranked:
        lines.append("Nothing sharp positionally — the position is roughly balanced across the board.")
    else:
        top = ranked[0]
        lead_white = terms[top]["cp"] > 0
        other = "Black" if lead_white else "White"
        lines.append(f"Main factor — {_cap(terms[top]['standing'])}.")
        opposing = [t for t in ranked[1:] if (terms[t]["cp"] > 0) != lead_white]
        if abs(eval_cp) <= _COMP_EVAL and abs(terms[top]["cp"]) >= _COMP_HUGE and opposing:
            lines.append(f"{other}'s compensation: "
                         + "; ".join(terms[t]["standing"] for t in opposing) + ".")
        else:
            for t in ranked[1:]:                     # eval decisive → the rest is secondary context
                lines.append(_cap(terms[t]["standing"]) + ".")
    # The opening is CONTEXT, not a tactic. Joining every fact into one "Tactics:" line told the model
    # that "This is the Ruy Lopez." is a tactical observation about the position, alongside a hanging
    # queen — so it gets its own line, and the Tactics line keeps its original meaning.
    opening = [f for f in facts if f.kind == "opening"]
    tactics = [f for f in facts if f.kind != "opening"]
    if opening:
        lines.append("Opening: " + " ".join(f.text for f in opening))
    if tactics:
        lines.append("Tactics: " + "; ".join(f.text for f in tactics) + ".")
    return lines


def build_analysis(board, engine, *, nodes: int | None = None,
                   movetime_ms: int | None = None) -> list[str]:
    """The grounded briefing for `board` as a list of natural-language statements.

    Deterministic given the engine limit (pass `nodes=` for tests, `movetime_ms=`
    for production — the same split as `Engine.analyse` / `build_fact_sheet`).
    Order: eval, material, the four strategic terms (leads first), then the
    tactical facts joined into one line. Empty tail sections are omitted."""
    analysis = engine.analyse(board.fen, multipv=1, nodes=nodes, movetime_ms=movetime_ms)
    pos = analyze_positional(board)
    facts = build_fact_sheet(board, engine, nodes=nodes, movetime_ms=movetime_ms)
    return assemble_analysis(board, analysis.best.score, pos, facts)
