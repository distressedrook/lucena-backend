"""Motif: undermine the defender.

A move undermines a defender when it removes or attacks the ONLY piece holding another enemy piece up.
Two forms of the one motif, both derived from the board's attacker/defender sets — never guessed:

  REMOVE — the move CAPTURES the sole defender of an enemy piece V, so V is now undefended (and falls
           if it is also attacked). E.g. Rxc3 takes the knight that alone defended the a2 bishop.
  ATTACK — the moved piece now ATTACKS a piece G that is the sole defender of enemy piece V, and G is
           undefended or winnable (SEE ≥ 0), so the threat is real. E.g. Rxc5 attacks the c3 knight,
           the only defender of the a2 bishop.

When the captured guard is worth MORE than the piece it defended, winning the guard is the point and
the undermining is surfaced as a secondary note (no 'The point:' marker) rather than the lead.
"""
from lucena_engine.board import Board

from ._pieces import value, word

_DEFENDABLE = {"N", "B", "R", "Q"}          # a piece worth defending (V) — pawns aren't the point


def _sole_defender(board, victim_sq: str) -> str | None:
    """The square of the ONE piece defending `victim_sq`, or None if it has zero or several defenders."""
    try:
        ds = board.defenders(victim_sq)
    except Exception:
        return None
    return ds[0] if len(ds) == 1 else None


def _undermined_victim(board, guard_sq: str, opp: str):
    """The enemy piece (worth ≥ a knight) that `guard_sq` is the SOLE defender of, or None. This is the
    'remove the guard' geometry: the piece on guard_sq is the only thing holding that victim up."""
    for v in board.piece_list():
        if v.color != opp or v.square == guard_sq or v.piece.upper() not in _DEFENDABLE:
            continue
        if _sole_defender(board, v.square) == guard_sq:
            return v
    return None


def undermines_defender(pre_fen: str | None, uci: str | None) -> str | None:
    """Derive the causal point of a move that undermines a defender, or None. Colour-absolute (the
    coach's PERSPECTIVE anchor maps the colour onto 'you'). A sentence beginning 'The point of this
    move:' is meant to LEAD the verdict; an 'It also …' sentence is a secondary note."""
    if not pre_fen or not uci:
        return None
    try:
        pre = Board(pre_fen)
        after = pre.apply(uci)
        from_sq, dest = uci[:2].lower(), uci[2:4].lower()
        mover = next((p for p in pre.piece_list() if p.square == from_sq), None)
        if mover is None:
            return None
        opp = "black" if mover.color == "white" else "white"
        opp_cap = opp.capitalize()

        # REMOVE — did the move capture a piece that was the sole defender of a valuable enemy piece?
        took = next((p for p in pre.piece_list() if p.square == dest and p.color == opp), None)
        if took is not None:
            v = _undermined_victim(pre, dest, opp)
            if v is not None:
                # V had exactly one defender (the captured piece) and the opponent hasn't moved, so V is
                # provably undefended now; if it is also attacked it is free → it falls.
                falls = bool(after.attackers(v.square, mover.color))
                where = f"{opp_cap}'s {word(v.piece)} on {v.square}"
                if value(v.piece) >= value(took.piece):
                    tail = f"{where}, which now falls" if falls else f"{where}, leaving it undefended"
                    return f"The point of this move: it removes the only defender of {tail}."
                # the bigger capture is the point — the undermining is a SECONDARY note (no marker).
                return f"It also leaves {where} loose — its only defender is now gone."

        # ATTACK — does the moved piece now attack the sole defender of a valuable enemy piece?
        for g in after.piece_list():
            if g.color != opp or g.piece.upper() == "K":   # a king-guard is a check — a different motif
                continue
            if dest not in after.attackers(g.square, mover.color):
                continue
            v = _undermined_victim(after, g.square, opp)
            if v is None:
                continue
            winnable = (not after.defenders(g.square)) or after.see(dest + g.square) >= 0
            if not winnable:
                continue
            # A THREAT, not a proven win — after this it is the OPPONENT's move and they may reinforce
            # the guard or add a defender to V. Phrase the pressure honestly, don't assert the win.
            return (f"The point of this move: it attacks the {word(g.piece)} on {g.square}, the only "
                    f"defender of {opp_cap}'s {word(v.piece)} on {v.square} — threatening to win it and "
                    f"leave that {word(v.piece)} loose.")
        return None
    except Exception:
        return None
