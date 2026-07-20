"""Multi-ply reasoning over a LINE (the engine's principal variation).

The single-ply motif reasoners (`undermine`) label ONE move; they can only ever say a piece is
"threatened", because they don't see the opponent's reply. A verified PRINCIPAL VARIATION does — it
already contains the opponent's best defense — so walking it tells us what the player ACTUALLY wins.

`describe_plan` turns that into the plan: the TARGET the player's moves win, and — when the played
move undermines that target's sole defender — the causal chain, now stated as CERTAIN ("cannot be held
and falls") instead of the single-ply hedge ("threatening to"). Spoiler-safe: it names the target
piece, never a future move.

Two guards keep it honest, because a piece merely *touched* in the line is not a piece *won*:
  (1) ABSOLUTE final material — measured on the actual final board (never the swing DURING the line). The
      mover must both NET material over the line (final > pre) AND end up AHEAD (final > 0), and the target
      square must never be recaptured by the opponent later in the PV. Together this rejects a plain trade
      (Rxd8, Kxd8 — count spikes then returns), a sacrifice into an attack (final < 0 — the point is the
      attack, not the incidental pawn), and a trade-down from a surplus (final < pre — up a rook, then
      rook-for-knight still "wins" nothing). We credit only a piece the player captures AND keeps while
      genuinely gaining.
  (2) The engine CP (lenient floor) — the material test rejects the losing sac; cp only catches the
      rarer "even on the board but the position is lost": require the engine to agree the mover is at
      least clearly better (≥ +100 mover-POV), no more.
"""
from lucena_engine.board import Board

from ._pieces import word

from .undermine import _sole_defender

# Standard exchange values — NOT the compressed `_pieces.PIECE_VAL` (R=4, N=2), which distorts a
# net-material balance across a line (a rook-for-knight trade reads as -2 instead of -2 pawns' worth of
# the real -5+3). The peak-material walk needs real pawn-equivalents to land on the right resolving ply.
_STD = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}


def _v(piece: str) -> int:
    return _STD.get((piece or "").upper(), 0)


def _material_diff(board, me: str) -> int:
    """Absolute material on `board`, mover-POV pawn-equivalents (mine − theirs), kings excluded."""
    mine = sum(_v(p.piece) for p in board.piece_list() if p.color == me)
    theirs = sum(_v(p.piece) for p in board.piece_list() if p.color != me)
    return mine - theirs


def pv_san_to_uci(fen: str | None, pv_san: list | None) -> list:
    """Convert a SAN principal variation to UCIs by walking it from `fen`. Stops at the first move that
    won't parse/apply (a truncated or illegal tail) rather than raising — the prefix is still usable."""
    if not fen or not pv_san:
        return []
    out = []
    try:
        b = Board(fen)
    except Exception:
        return []
    for san in pv_san:
        try:
            u = b.uci(san)
            b = b.apply(u)
        except Exception:
            break
        out.append(u)
    return out


def _first_move_undermines(pre, uci: str, target_sq: str, me: str) -> bool:
    """Does the just-played move REMOVE or ATTACK the sole defender of the piece on `target_sq`?"""
    guard = _sole_defender(pre, target_sq)
    if guard is None:
        return False
    dest = uci[2:4].lower()
    if dest == guard:                       # captured the defender outright
        return True
    try:
        after = pre.apply(uci)
        return dest in after.attackers(guard, me)   # the moved piece now attacks the defender
    except Exception:
        return False


def describe_plan(pre_fen: str | None, pv: list | None, cp: int | None = None) -> str | None:
    """`pv`: the engine's principal variation as UCIs (player, opponent, player, …). `cp`: the engine's
    eval of the played move, MOVER-POV centipawns (from `evaluate`'s `eval.cp`) — when given, it gates
    the claim so a losing sacrifice can't masquerade as a win. Returns the plan's point, or None.
    Colour-absolute; names the target, never a future move."""
    if not pre_fen or not pv:
        return None
    try:
        pre = Board(pre_fen)
        me = pre.side_to_move
        opp_cap = "Black" if me == "white" else "White"
        pre_diff = _material_diff(pre, me)  # mover-POV material BEFORE the line (may already be a surplus)
        b = pre
        caps = []                           # (by_player, piece, square, ply-index) of every capture
        for i, u in enumerate(pv):
            by_player = b.side_to_move == me
            dest = u[2:4].lower()
            cap = next((p for p in b.piece_list() if p.square == dest and p.color != b.side_to_move), None)
            if cap:
                caps.append((by_player, cap.piece, dest, i))
            b = b.apply(u)
        # ABSOLUTE material on the final board, mover-POV. The mover must both NET material over the line
        # (final > pre — else a pre-existing surplus makes a losing rook-for-knight trade look like a win)
        # AND end up AHEAD (final > 0 — else it's winning material while still losing, i.e. not the point).
        # This alone rejects a plain trade (net 0), a sac into an attack (final < 0), and a trade-down from
        # a surplus (final < pre) — the material swing during the line is never trusted.
        final_diff = _material_diff(b, me)
        if final_diff <= pre_diff or final_diff <= 0:
            return None
        # a piece the PLAYER captured BEYOND the played move (ply > 0 — an immediate ply-0 win is the
        # single-ply reasoner's job) and KEPT: the opponent never recaptures on that square later. That
        # excludes the transient spike of a plain trade, where the opponent takes the square straight back.
        kept = [(piece, sq) for by_player, piece, sq, ply in caps
                if by_player and ply > 0
                and not any((not bp2) and sq2 == sq and ply2 > ply for bp2, _, sq2, ply2 in caps)]
        if not kept:
            return None
        # CP gate (lenient secondary guard): the final-material test already rejects a losing sac; cp
        # only catches the rarer "up on the board but the position is lost" — require the engine to agree
        # the mover is at least clearly better, no more.
        if cp is not None and cp < 100:
            return None
        piece, sq = max(kept, key=lambda c: _v(c[0]))
        target = f"{opp_cap}'s {word(piece)} on {sq}"
        dest0 = pv[0][2:4].lower()
        recaptured = any((not bp) and s2 == sq and ply2 > 0 for bp, _, s2, ply2 in caps)
        # The win lands on the played move's OWN square via a recapture (you take on `sq`, the opponent
        # recaptures there, you win THAT piece). This is a trade-and-win, NOT an undermine — and at
        # verdict time `sq` still shows what the move just captured, so "the {piece} on {sq}" reads as a
        # contradiction. Name the SEQUENCE instead so it stays legible.
        if sq == dest0 and recaptured:
            return (f"The point of this move: it wins material on {sq} — after {opp_cap} recaptures "
                    f"there, the {word(piece)} cannot be held and falls.")
        # A true undermine: the played move pulls the target's sole defender on a DIFFERENT square, so
        # the target falls. State it CERTAIN (the PV already contains the opponent's best defense).
        if _first_move_undermines(pre, pv[0], sq, me):
            return (f"The point of this move: it undermines the only defender of {target} — it cannot "
                    f"be held, and falls.")
        return f"The point of this move: it wins {target}."
    except Exception:
        return None
