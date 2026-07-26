"""The move-quality ladder — the familiar Brilliant/Great/Best/.../Blunder
labels, derived entirely from OUR OWN numbers.

`lucena_engine.evalmodel` deliberately speaks lichess-verbatim glyphs (5
classes: ! ?! ? ?? and OK) because those thresholds are the ones our corpus work
was calibrated against, and the public engine should not drift from them. A
reader, though, wants the finer ladder every chess site now uses. This module is
that presentation layer, and it is a REFINEMENT of the engine's glyphs, never a
contradiction: every label maps back onto the same win-percentage drop.

    BOOK         still in the opening book — no judgement is offered at all
    BRILLIANT    a sound sacrifice that is not the obvious move (is_brilliant)
    GREAT        the only move that holds: 2nd best is far worse, and it was found
    BEST         the engine's first choice
    EXCELLENT    not the top move, but nothing measurable was lost
    GOOD         a small, ordinary imprecision
    INACCURACY   the engine's ?! band
    MISTAKE      the engine's ?  band
    MISS         a win% gift was on the table and this move handed it back
    BLUNDER      the engine's ?? band

Two of these are OURS rather than a copy of anyone's product. BOOK is real book
membership from `lucena_core.openings`, not a hardcoded move list, so it ends
exactly where THIS game left theory. MISS is defined against the gift actually
on the board on the previous ply, which is what makes it bankable as a tactic
the player did not see (v1 aim 2) rather than just another mistake.
"""

from __future__ import annotations

from enum import Enum

from lucena_engine.evalmodel import (BLUNDER as _WP_BLUNDER, MISTAKE as
                                     _WP_MISTAKE, DUBIOUS as _WP_DUBIOUS)

# Anything at or under this is indistinguishable from the best move at the
# depth we searched — calling it an error would be false precision.
_EXCELLENT = 2.0

# A MISS needs a real gift on the previous ply and a real part of it returned.
# Both bars are win-%, the same currency as every other rung.
MISS_GIFT = 12.0
MISS_RETURN = 7.0


class MoveClass(str, Enum):
    BOOK = "book"
    BRILLIANT = "brilliant"
    GREAT = "great"
    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    MISS = "miss"
    BLUNDER = "blunder"


#: Reader-facing symbol per class. Kept beside the enum so a renderer cannot
#: invent its own and drift from the ladder.
SYMBOL = {
    MoveClass.BOOK: "○",        # ○
    MoveClass.BRILLIANT: "!!",
    MoveClass.GREAT: "!",
    MoveClass.BEST: "✓",        # ✓
    MoveClass.EXCELLENT: "✓",
    MoveClass.GOOD: "",
    MoveClass.INACCURACY: "?!",
    MoveClass.MISTAKE: "?",
    MoveClass.MISS: "×",        # ×
    MoveClass.BLUNDER: "??",
}

#: Classes that count as an error when tallying a player's patterns.
ERRORS = {MoveClass.INACCURACY, MoveClass.MISTAKE, MoveClass.BLUNDER,
          MoveClass.MISS}


def classify_move(*, drop: float, played_is_best: bool, in_book: bool,
                  engine_class: str, gift_wp: float = 0.0) -> MoveClass:
    """One ply -> one rung.

    `drop` is the win-% the move shed against the engine's best (>= 0).
    `engine_class` is gamepass's own string, so BRILLIANT and GREAT are
    inherited from the audited detectors rather than re-derived here.
    `gift_wp` is what the OPPONENT handed over on the previous ply.
    """
    if in_book:
        return MoveClass.BOOK
    if engine_class == "brilliant":
        return MoveClass.BRILLIANT
    if engine_class == "only_move":
        return MoveClass.GREAT
    # A MISS outranks the plain error labels: "you had this and let it go" is
    # the more useful sentence, and it is the one we can bank.
    if gift_wp >= MISS_GIFT and drop >= MISS_RETURN:
        return MoveClass.MISS
    if drop >= _WP_BLUNDER:
        return MoveClass.BLUNDER
    if drop >= _WP_MISTAKE:
        return MoveClass.MISTAKE
    if drop >= _WP_DUBIOUS:
        return MoveClass.INACCURACY
    if played_is_best:
        return MoveClass.BEST
    if drop <= _EXCELLENT:
        return MoveClass.EXCELLENT
    return MoveClass.GOOD


def accuracy(drops: list[float]) -> float:
    """A 0-100 accuracy score from the per-move win-% drops.

    The per-move curve is the published lichess mapping from win-% loss to
    accuracy (103.1668 * exp(-0.04354 * loss) - 3.1669), which pairs with the
    win-% model `evalmodel` already uses — a different curve on top of that
    model would make the number mean nothing.

    They are combined with the HARMONIC mean, and the reason is the whole
    point of the statistic. Chess is not scored per move; one move can lose
    the game, and an average that lets forty accurate moves bury it is
    measuring the wrong thing. Measured on the arithmetic mean this function
    used first: forty clean moves followed by one game-losing 40-point blunder
    scored 97.9 — the blunder cost 2.1 points. The harmonic mean charges 12.5,
    because it is dominated by the smallest term, which is exactly the
    behaviour "accuracy" should have.

    Corroboration on ONE game, not the reason and not a compatibility claim:
    the harmonic mean gave 76.0 / 70.8 where Chess.com reported 75.3 / 71.3,
    while the arithmetic mean read 86.1 / 84.4. That is consistent with having
    picked a similar aggregator on this sample. It is n=1 against an
    unpublished formula, so it establishes nothing about agreement in general
    — do not treat it as a promise that our numbers track theirs.

    Known limitation, stated rather than hidden: any per-move mean still
    dilutes with game length — the same blunder scores better inside a longer
    game. Fixing that needs volatility weighting (weighting each move by how
    much was actually at stake), which needs its own calibration work.
    """
    import math
    if not drops:
        return 100.0
    vals = [max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * max(0.0, d))
                         - 3.1669)) for d in drops]
    # floor each term: a mated-in-one move scores 0, and 1/0 is undefined.
    # The floor sets how harshly a single catastrophe can pull the whole
    # number down; 1.0 keeps it severe without letting one move zero the game.
    return round(len(vals) / sum(1.0 / max(v, 1.0) for v in vals), 1)
