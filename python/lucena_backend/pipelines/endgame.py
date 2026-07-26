"""The endgame read — deterministic facts for the phase the plans layer declines.

`lucena-plans` is calibrated on middlegames and refuses endgames by design
(`plans.service.is_endgame`), which is correct: its plan families, corpus lifts
and verify contracts were all measured on middlegame positions. The consequence
was that a report went silent exactly where a technical game is decided. On the
game that prompted this module, all six of White's errors were in the endgame
and the walkthrough had nothing to say about any of them.

So this is a deliberately small, separate vocabulary with a different contract.

  IT STATES FACTS, NOT PLANS. Every line is a geometric property of the
  position a reader can verify on the board — which king is nearer the pawns,
  whether a king is inside the square of a passer, which side of its passer a
  rook stands on. No lift numbers, no engine verification, no "engine
  confirmed" tag, because none of these have been through the corpus work that
  earns those words. Facts do not need them; plans would.

  IT NAMES TERMS WITHOUT DEFINING THEM (the coach-voice ruling): "opposite-
  coloured bishops", "White's rook is in front of the passer" — not a lecture
  on why the rook belongs behind it.

  IT SAYS NOTHING RATHER THAN SOMETHING VAGUE. Each reader returns only what
  it can actually establish; an endgame with no passers simply has no passer
  line.
"""

from __future__ import annotations

import chess

from lucena_core.geometry import passers, side_rank

# Non-pawn material per side, the same table `plans.service` gates on, so
# "endgame" means one thing across the product.
_NPM = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3}


def _npm(b: chess.Board, side: bool) -> int:
    return sum(v * len(b.pieces(pt, side)) for pt, v in _NPM.items())


def _colour_of(sq: int) -> bool:
    """True for a light square."""
    return bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[sq])


def material_type(b: chess.Board) -> str:
    """What kind of endgame this is, by the pieces actually on the board.

    Opposite-coloured bishops are called out by name because they change how
    the position should be judged more than any other material fact here."""
    def has(pt):
        return len(b.pieces(pt, chess.WHITE)), len(b.pieces(pt, chess.BLACK))
    q, r, bi, n = has(chess.QUEEN), has(chess.ROOK), has(chess.BISHOP), has(chess.KNIGHT)
    minors = (bi[0] + n[0], bi[1] + n[1])

    if not any(q + r + bi + n):
        return "pawn endgame"
    if bi == (1, 1) and not (q[0] or q[1] or n[0] or n[1]):
        wb = next(iter(b.pieces(chess.BISHOP, chess.WHITE)))
        bb = next(iter(b.pieces(chess.BISHOP, chess.BLACK)))
        opp = _colour_of(wb) != _colour_of(bb)
        if not (r[0] or r[1]):
            return ("opposite-coloured bishops" if opp
                    else "same-coloured bishop endgame")
        return ("rooks and opposite-coloured bishops" if opp
                else "rook and bishop endgame")
    if not (q[0] or q[1] or minors[0] or minors[1]) and (r[0] or r[1]):
        return "rook endgame"
    if not (r[0] or r[1] or minors[0] or minors[1]):
        return "queen endgame"
    if not (q[0] or q[1] or r[0] or r[1]):
        return "minor-piece endgame"
    return ""


def _dist(a: int, c: int) -> int:
    """King moves between two squares."""
    return max(abs(chess.square_file(a) - chess.square_file(c)),
               abs(chess.square_rank(a) - chess.square_rank(c)))


def king_centralisation(b: chess.Board) -> dict:
    """How central each king stands — and nothing more than that.

    A first version measured mean distance to the whole pawn mass, on the
    theory that a king next to the pawns beats a king on a pretty central
    square. Measured on a real bishop endgame it was useless: with pawns split
    across both wings every king scores ~3.0 and the reading never fired once
    in sixteen plies, including while one king sat on d4 and the other wandered
    to f6. Averaging over a scattered mass cancels exactly the difference it
    was supposed to find.

    So this is the textbook measure, and the wording is held to what it
    actually establishes: "the more centralised", never "the more active".
    Centralisation is a fact about the board; activity is a judgement that
    depends on where the play is, and we have not measured that."""
    wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
    if wk is None or bk is None:
        return {}

    def off_centre(sq: int) -> float:
        return max(abs(chess.square_file(sq) - 3.5),
                   abs(chess.square_rank(sq) - 3.5))

    w, k = off_centre(wk), off_centre(bk)
    # a full square apart is a real difference; anything less is noise
    lead = None
    if abs(w - k) >= 1.0:
        lead = "white" if w < k else "black"
    return {"white": round(w, 1), "black": round(k, 1), "leader": lead,
            "white_square": chess.square_name(wk),
            "black_square": chess.square_name(bk)}


def _in_square(b: chess.Board, sq: int, side: bool) -> bool:
    """Is the defending king inside the square of this passer?

    The classic rule, with the two corrections that make it true rather than
    approximately true: a pawn still on its home rank takes one move less
    (the double step), and the defender gets an extra tempo when it is their
    move."""
    f = chess.square_file(sq)
    promo_rank = 7 if side == chess.WHITE else 0
    promo = chess.square(f, promo_rank)
    steps = abs(promo_rank - chess.square_rank(sq))
    if side_rank(sq, side) == 1:          # still on its starting rank
        steps -= 1
    dk = b.king(not side)
    if dk is None:
        return False
    d = _dist(dk, promo)
    if b.turn != side:                     # defender to move: one tempo in hand
        d -= 1
    return d <= steps


def passer_facts(b: chess.Board) -> list[dict]:
    """Every passed pawn, how far it has come, and whether it can be caught."""
    out = []
    for side, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        for sq in passers(b, side):
            rank = side_rank(sq, side) + 1        # 1..8 from the owner's side
            behind = _rook_behind(b, sq, side)
            out.append({
                "side": name,
                "square": chess.square_name(sq),
                "rank": rank,
                "caught": _in_square(b, sq, side),
                "rook_behind": behind,
                "protected": _protected(b, sq, side),
            })
    out.sort(key=lambda p: -p["rank"])
    return out


def _protected(b: chess.Board, sq: int, side: bool) -> bool:
    """Defended by one of its own pawns."""
    return any(s in b.pieces(chess.PAWN, side) for s in b.attackers(side, sq))


def _rook_behind(b: chess.Board, sq: int, side: bool) -> str | None:
    """Which rook, if any, stands behind this passer on its file — the side
    of the pawn a rook is on is the one endgame fact about rooks worth stating
    (Tarrasch); named, not explained."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    for colour, who in ((side, "own"), (not side, "enemy")):
        for rk in b.pieces(chess.ROOK, colour):
            if chess.square_file(rk) != f:
                continue
            rr = chess.square_rank(rk)
            behind = rr < r if side == chess.WHITE else rr > r
            if behind:
                return who
    return None


def bishop_facts(b: chess.Board) -> list[dict]:
    """A bishop's own pawns on its colour — the endgame form of a bad bishop.

    Counted only against the OWN pawns it is stuck behind, which is the version
    that bites in a technical position — and PASSERS are excluded. A passed
    pawn sharing the bishop's colour is the asset being escorted, not an
    obstruction; counting it turned a winning bishop-and-passer into a longer
    bad-bishop complaint on the game that prompted this module."""
    out = []
    for side, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        bs = list(b.pieces(chess.BISHOP, side))
        if len(bs) != 1:
            continue
        col = _colour_of(bs[0])
        free = set(passers(b, side))
        own = [p for p in b.pieces(chess.PAWN, side)
               if _colour_of(p) == col and p not in free]
        if not own:
            continue
        out.append({"side": name, "square": chess.square_name(bs[0]),
                    "on_light": col, "own_pawns_on_colour": len(own),
                    "squares": sorted(chess.square_name(p) for p in own)})
    return out


def read(fen: str) -> dict | None:
    """The endgame read for one position, or None when it is not an endgame."""
    b = chess.Board(fen)
    if _npm(b, chess.WHITE) > 13 or _npm(b, chess.BLACK) > 13:
        return None
    return {
        "type": material_type(b),
        "kings": king_centralisation(b),
        "passers": passer_facts(b),
        "bishops": bishop_facts(b),
    }


def sentences(r: dict | None) -> list[str]:
    """The read as reader-facing lines. Facts only — no advice, no definitions.

    Colours are absolute (White/Black), never "you"/"your opponent": the
    perspective-flip bug class this repo has hit before comes from relative
    wording, and a report is read by both players anyway."""
    if not r:
        return []
    out = []
    if r.get("type"):
        out.append(r["type"].capitalize() + ".")

    k = r.get("kings") or {}
    if k.get("leader"):
        who = k["leader"].capitalize()
        other = "Black" if who == "White" else "White"
        sq = k["white_square"] if k["leader"] == "white" else k["black_square"]
        osq = k["black_square"] if k["leader"] == "white" else k["white_square"]
        out.append(f"{who}'s king is the more centralised, on {sq} against "
                   f"{other}'s on {osq}.")

    for p in r.get("passers") or []:
        who = p["side"].capitalize()
        bits = [f"{who} has a passed pawn on {p['square']} (rank {p['rank']})"]
        if p["protected"]:
            bits.append("protected by a pawn")
        if p["rook_behind"] == "own":
            bits.append("with its own rook behind it")
        elif p["rook_behind"] == "enemy":
            bits.append("with the enemy rook behind it")
        bits.append("the defending king is inside its square" if p["caught"]
                    else "the defending king is outside its square")
        out.append(", ".join(bits) + ".")

    for bf in r.get("bishops") or []:
        who = bf["side"].capitalize()
        colour = "light" if bf["on_light"] else "dark"
        out.append(f"{who}'s bishop on {bf['square']} shares its colour with "
                   f"{bf['own_pawns_on_colour']} of its own pawns "
                   f"({', '.join(bf['squares'])}).")
    return out
