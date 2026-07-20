"""Motif: undermine the defender.

A move undermines a defender when it removes or attacks the ONLY piece holding another enemy piece up.
Two forms of the one motif, both derived from the board's attacker/defender sets — never guessed:

  REMOVE — the move CAPTURES the sole defender of an enemy piece V, so V is now undefended (and falls
           if it is also attacked). E.g. Rxc3 takes the knight that alone defended the a2 bishop.
  ATTACK — the moved piece now ATTACKS a piece G that is the sole defender of enemy piece V, and G is
           undefended or winnable (SEE ≥ 0), so the threat is real. E.g. Rxc5 attacks the c3 knight,
           the only defender of the a2 bishop.

Attack-the-defender is a FORK: the opponent saves the guard OR the piece it defends, not both. So —
  ATTACK — the move threatens the guard while it still stands: the opponent can't save both, ONE falls.
  REMOVE (favourable capture) — you already TOOK the guard, resolving the fork in favour of that piece;
           the point is simply that you won it. The other piece does NOT also fall (the opponent had a
           move in between and can rescue it), so single-ply must not claim it does.
  REMOVE (sacrifice) — you gave material specifically to strip the defender, so the now-undefended
           victim is the point.
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


def _attack_form(after, dest: str, mover, opp: str, opp_cap: str) -> str | None:
    """ATTACK: the moved piece now attacks the sole defender of a valuable enemy piece — the fork. The
    guard still stands, so the opponent chooses which of the two to give up; single-ply states the
    dilemma, never 'wins both'."""
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
        # The FORK: the guard is attacked while it still holds V up, so the opponent must give up one
        # or the other — but NOT both to the same move. Honest single-ply framing is the dilemma
        # ('can't save both'), never 'wins both' (which piece falls is the opponent's choice next).
        return (f"The point of this move: it attacks the {word(g.piece)} on {g.square} — the only "
                f"defender of {opp_cap}'s {word(v.piece)} on {v.square} — so {opp_cap} cannot save "
                f"both; one of them falls.")
    return None


def _setup(pre_fen: str | None, uci: str | None):
    """Shared board/piece setup for the two forms. Returns (pre, after, dest, mover, opp, opp_cap) or None."""
    if not pre_fen or not uci:
        return None
    pre = Board(pre_fen)
    after = pre.apply(uci)
    from_sq, dest = uci[:2].lower(), uci[2:4].lower()
    mover = next((p for p in pre.piece_list() if p.square == from_sq), None)
    if mover is None:
        return None
    opp = "black" if mover.color == "white" else "white"
    return pre, after, uci, dest, mover, opp, opp.capitalize()


def attacks_defender(pre_fen: str | None, uci: str | None) -> str | None:
    """JUST the ATTACK-the-defender FORK (the guard still stands → 'can't save both'). Exposed separately
    so the coach can prefer this HONEST dilemma over a multi-ply reasoner that would over-certainly report
    the single line where the opponent gives up one particular piece. None if the move isn't such a fork."""
    try:
        s = _setup(pre_fen, uci)
        if s is None:
            return None
        pre, after, _uci, dest, mover, opp, opp_cap = s
        return _attack_form(after, dest, mover, opp, opp_cap)
    except Exception:
        return None


def undermines_defender(pre_fen: str | None, uci: str | None) -> str | None:
    """Derive the causal point of a move that undermines a defender, or None. Colour-absolute (the
    coach's PERSPECTIVE anchor maps the colour onto 'you'). REMOVE (a favourable capture of the guard)
    leads with the piece won; a SACRIFICE leads with the now-loose victim; otherwise the ATTACK fork."""
    try:
        s = _setup(pre_fen, uci)
        if s is None:
            return None
        pre, after, uci, dest, mover, opp, opp_cap = s

        # REMOVE — did the move capture a piece that was the sole defender of a valuable enemy piece?
        took = next((p for p in pre.piece_list() if p.square == dest and p.color == opp), None)
        if took is not None:
            v = _undermined_victim(pre, dest, opp)
            if v is not None:
                # V had exactly one defender (the captured piece) and the opponent hasn't moved, so V is
                # provably undefended now; if it is also attacked it is free → it falls.
                falls = bool(after.attackers(v.square, mover.color))
                where = f"{opp_cap}'s {word(v.piece)} on {v.square}"
                try:
                    won_the_guard = pre.see(uci) >= 0        # a FAVOURABLE capture of the guard, not a sac
                except Exception:
                    won_the_guard = value(v.piece) < value(took.piece)
                if won_the_guard:
                    # You TOOK the guard for material — that resolves the fork in favour of this piece.
                    # The point is simply that you won it. Do NOT claim V also falls: the opponent moves
                    # next and can rescue it (single-ply can't see that counterplay). It's one OR the other.
                    return f"The point of this move: it wins {opp_cap}'s {word(took.piece)} on {dest}."
                # A SACRIFICE to pull the defender — here the undefended V is the point, not the trade.
                tail = f"{where}, which now falls" if falls else f"{where}, leaving it undefended"
                return f"The point of this move: it removes the only defender of {tail}."

        # ATTACK — the moved piece now attacks the sole defender of a valuable enemy piece (the fork).
        return _attack_form(after, dest, mover, opp, opp_cap)
    except Exception:
        return None
