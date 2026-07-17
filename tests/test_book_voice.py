"""book_voice._book_route — the freeform opening-narration cadence policy (pure, no LLM/engine).

`_book_route` is a pure function: the cadence is tested by REPLAYING real openings through it and
asserting the exact fire sequence. This is where the coarsening regressions are pinned (the Open
Sicilian going silent, the shallow-family hand-off). Salvaged from the retired test_opening_narration
when the Orchestrator was deleted; the book-route logic itself is unchanged.
"""

from __future__ import annotations

from lucena_backend.coaching.book_voice import (
    _book_route, _is_swing, _NARRATE, _ENDBOOK, _COACH, _OFF_BOOK_AT, _MIN_FAMILY_SIZE,
)
from lucena_backend.coaching.freeform import _mover
from lucena_engine.board import Board
from lucena_engine import openings


def _fens(*ucis: str) -> list:
    """Replay a line from the start position, returning [start, ...after each ply] — through the REAL
    board core (not FEN literals), so the test stays pinned to the core's en-passant convention that
    keys the opening table."""
    b = Board(START)
    out = [START]
    for u in ucis:
        b = b.apply(u)
        out.append(b.fen)
    return out


def _routes(ucis: list, swing: bool = False) -> list:
    """The route for each ply of a line, as (uci, route, name)."""
    fens = _fens(*ucis)
    return [(ucis[i - 1],) + _book_route(fens[: i + 1], swing)[:2] for i in range(1, len(fens))]

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# -- 1. the cadence ------------------------------------------------------------------------------

RUY = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]
NAJDORF = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]
QGD = ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]
FRENCH = ["e2e4", "e7e6", "d2d4", "d7d5"]


def test_a_coarser_name_is_never_the_name_we_use():
    """The regression that killed `deepest()`: the table re-attaches COARSER names deeper in a line.

    Najdorf `d6` names "Sicilian Defense: Modern Variations", then `d4` re-names the position plain
    "Sicilian Defense". A last-wins walk would announce the LESS specific name it just moved past —
    the coach audibly forgetting what it said one ply ago.

    Note what this asserts and what it does NOT. These plies DO narrate — every in-book ply does, and
    an earlier version of this test asserting silence here is exactly how the Open Sicilian went mute.
    The fold's job was never to decide whether to speak; it decides what to CALL the opening. So: the
    name carried must stay the specific one, and no `prev` may be passed (there is no delta — the
    reader was never told anything new to delta from).
    """
    for line, ply, keep in ((NAJDORF, "d2d4", "Sicilian Defense: Modern Variations"),
                            (QGD, "f8e7", "Queen's Gambit Declined: Modern Variation"),
                            (FRENCH, "d7d5", "French Defense: Normal Variation")):
        routes = {u: (r, n) for u, r, n in _routes(line)}
        route, name = routes[ply]
        assert route == _NARRATE, f"{ply} went silent — a coarsening is not a reason to say nothing"
        assert name == keep, (
            f"{ply} would be announced as {name!r}, which is COARSER than the name already used "
            f"({keep!r}) — the coach reads as having forgotten what it just said"
        )


def test_a_genuine_branch_fires():
    """The other half: a real branch into a named variation MUST fire, or the fold is just a mute."""
    ruy = {u: r for u, r, _ in _routes(RUY)}
    assert ruy["f1b5"] == _NARRATE, "Bb5 enters the Ruy Lopez and must be narrated"
    assert ruy["a7a6"] == _NARRATE, "a6 enters the Morphy Defense — a genuine refinement, not a coarsening"


def test_an_unnamed_ply_inside_the_book_is_sticky_not_off_book():
    """A line that dips out of the table for a ply stays in its opening — it has not left the book."""
    routes = _routes(RUY)
    assert all(r != _ENDBOOK for _, r, _ in routes), \
        f"the Ruy Lopez mainline announced end-of-book: {[(u, r) for u, r, _ in routes]}"
    assert all(r == _NARRATE for _, r, _ in routes), \
        f"a ply inside the book did not narrate: {[(u, r) for u, r, _ in routes]}"


def test_the_first_move_narrates():
    """The design goal, at its smallest: 1.e4 gets a paragraph, and it knows what to call it."""
    r = _routes(["e2e4", "e7e5"])
    assert r[0][1] == _NARRATE and r[0][2] == "King's Pawn Game"


def test_the_open_sicilian_is_never_silent():
    """The regression that killed the silent case, pinned on the exact line that exposed it.

    Real session: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 produced THREE consecutive beats of nothing. The
    table names positions rather than lines and re-attaches the coarse "Sicilian Defense" at 3.d4, so
    the folded name stopped changing — and the router read "the name is not news" as "nothing happened
    here", through one of the most important branches in chess.

    A repeat name is a fact about the TABLE. It was never evidence about the position.
    """
    routes = _routes(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"])
    silent = [u for u, r, _ in routes if r != _NARRATE]
    assert not silent, f"the Open Sicilian went quiet at {silent}"
    # ...and the coarsening is still suppressed as a NAME: d4 must not be announced as plain
    # "Sicilian Defense" one ply after "Modern Variations" was used. It narrates with no `prev`.
    by_move = {u: (r, n) for u, r, n in routes}
    assert by_move["d2d4"] == (_NARRATE, "Sicilian Defense: Modern Variations")


# -- 2. off-book ---------------------------------------------------------------------------------

# Enters the book at ply 1 (Anderssen's Opening) and then wanders straight out of it — measured, not
# assumed: a line that was NEVER named exercises a different branch (see the None case below).
WANDER = ["a2a3", "a7a6", "h2h3", "h7h6", "a3a4", "h6h5", "a1a3", "a8a7"]


def test_off_book_announces_exactly_once_for_a_shallow_family():
    """WANDER enters "Anderssen's Opening" at ply 1 — a shallow family (family_size < _MIN_FAMILY_SIZE,
    see test_family_size_reads_the_table_not_a_guess) — so the hand-off fires at the TRANSITION (ply
    2, the first ply after the confirmed position), not at the unrelated psn == _OFF_BOOK_AT
    threshold three plies later. Still exactly once: the point of this test is that the two
    mechanisms (early transition vs. threshold) don't BOTH fire for the same departure — see
    `test_family_size_gates_the_off_book_threshold_too` for the threshold half of that guarantee.
    """
    routes = [r for _, r, _ in _routes(WANDER)]
    assert routes.count(_ENDBOOK) == 1, f"expected exactly one end-of-book, got {routes}"
    assert routes.index(_ENDBOOK) == 1, \
        f"a shallow family should hand off at the transition (ply 1, 0-indexed), not later: {routes}"
    assert all(r == _COACH for r in routes[2:]), \
        f"every ply after the hand-off must be normal coaching: {routes}"


def test_family_size_gates_the_off_book_threshold_too():
    """The threshold mechanism (`psn == _OFF_BOOK_AT`) still works standalone for a DEEP family that
    goes off-book via ordinary quiet moves (no swing, no shallow-family early exit ever fires) — the
    original off-book case, unchanged. Fixture: the Ruy Lopez main line (234 table rows), then four
    plies of a legal but uncatalogued rook shuffle.
    """
    ruy_then_wander = list(RUY) + ["a2a3", "a8a7", "a1a2", "a7a8"]
    fens = _fens(*ruy_then_wander)
    routes = [_book_route(fens[: i + 1], False)[0] for i in range(1, len(fens))]
    assert routes.count(_ENDBOOK) == 1, f"expected exactly one end-of-book, got {routes}"
    assert routes[len(RUY):].index(_ENDBOOK) == _OFF_BOOK_AT - 1, \
        f"a deep family should hand off exactly at the threshold, not earlier: {routes}"


def test_a_line_that_was_never_in_the_book_never_announces():
    """A drill or a pasted midgame FEN has no book to leave. `plies_since_named` returns None and the
    router must fall through to normal coaching — announcing "you have left opening theory" to someone
    who was handed a rook endgame is nonsense."""
    midgame = ["8/5k2/8/8/8/8/5K2/4R3 w - - 0 1"] * 8
    for i in range(1, len(midgame)):
        route, _, _ = _book_route(midgame[: i + 1], swing=False)
        assert route == _COACH, f"a never-named line routed {route!r} at ply {i}"


def test_off_book_fires_at_the_threshold_not_before():
    fens = _fens(*RUY)
    named_through = [i for i in range(1, len(fens))
                     if _book_route(fens[: i + 1], False)[0] != _ENDBOOK]
    assert len(named_through) == len(fens) - 1, "the Ruy went off-book; the threshold is too tight"
    assert _OFF_BOOK_AT == 4, "the threshold is a measured heuristic — changing it needs the measurement"


# -- 3. the swing fork ---------------------------------------------------------------------------

def test_a_repeat_name_narrates_the_move_and_passes_no_delta():
    """1...e5 adds no new name (both plies are "King's Pawn Game"). It still narrates — the subject is
    the MOVE — but `prev` must not be passed: there is no name delta to write, and handing one over
    invites a paragraph re-explaining a name the reader already has."""
    loud, name, prev = _book_route(_fens("e2e4", "e7e5"), swing=False)
    assert loud == _NARRATE
    assert name == "King's Pawn Game", "the name is still passed as CONTEXT"
    assert prev is None, "an unchanged name must not be passed as a delta"


def test_a_swing_off_book_is_normal_coaching_not_narration():
    assert _book_route(_fens(*WANDER), swing=True)[0] == _COACH, \
        "a blunder outside the book is a blunder, not theory to explain"


def test_a_swing_in_the_sticky_unnamed_zone_is_also_normal_coaching():
    """Regression, caught live: 3.h4 then 3...Nh6 in a Vienna Game the table had already lost track
    of (psn > 0, still inside the sticky tolerance) got narrated as the move being "a recognized
    theoretical attempt" — for a move that was simply a mistake. `psn is None` (WANDER, above) is a
    DIFFERENT code path from this one: here the line WAS recently in the book, so `_book_route`
    reaches the `psn < _OFF_BOOK_AT` branch and has a stale name sitting right there to reach for.
    Only `psn == 0` — the table confirming the position the move actually landed on — earns the
    "it's theory, not a blunder" voice.
    """
    vienna_then_wander = ["e2e4", "e7e5", "b1c3", "b8c6", "h2h4", "g8h6"]
    fens = _fens(*vienna_then_wander)
    psn = openings.plies_since_named(fens)
    assert 0 < psn < _OFF_BOOK_AT, f"fixture must land in the sticky-unnamed zone, got psn={psn}"
    assert _book_route(fens, swing=True)[0] == _COACH, \
        "a swing off the exact (unconfirmed) position must not get the theory voice"
    assert _book_route(fens, swing=False)[0] == _NARRATE, \
        "a non-swing move in the sticky zone still narrates — only the theory-excuse voice is gated"


def test_a_shallow_family_gets_the_hand_off_immediately_not_normal_coaching():
    """Regression, caught live in two stages. FIRST: 1.e4 h5 ("Goldsmith Defense") 2.a4 — the table
    has exactly TWO rows under "Goldsmith Defense", and none is 2.a4, yet at psn=1 the sticky
    tolerance narrated it as book: "Strong players typically respond to this position..." with the
    same confidence a real Ruy Lopez continuation would earn. Routing that to plain _COACH (an
    earlier version of this fix) closed that hole but opened a second one: the coach now stopped
    sounding like a book without ever SAYING so — the "we're out of theory" hand-off was still gated
    on the unrelated psn == _OFF_BOOK_AT threshold, still 3 plies away. `family_size` is a table
    property that does not change no matter how the game continues, so there is nothing left to
    wait FOR — the hand-off belongs at the exact ply the position one back (psn_prev) was still
    confirmed in the table, not later.

    SECOND ply (2...b5) pins the no-relatch shape: the hand-off must not fire again just because
    psn keeps climbing — `test_family_size_gates_the_off_book_threshold_too` pins the sharper case
    of that (psn actually reaching _OFF_BOOK_AT for the same shallow family).
    """
    goldsmith_then_a4 = ["e2e4", "h7h5", "a2a4"]
    fens = _fens(*goldsmith_then_a4)
    psn = openings.plies_since_named(fens)
    assert 0 < psn < _OFF_BOOK_AT, f"fixture must land in the sticky-unnamed zone, got psn={psn}"
    name = openings.book_name(fens)
    assert name == "Goldsmith Defense"
    assert openings.family_size(name) < _MIN_FAMILY_SIZE, (
        f"fixture opening must be shallow to exercise the gate, got {openings.family_size(name)} rows"
    )
    route, hand_off_name, _ = _book_route(fens, swing=False)
    assert route == _ENDBOOK, "a shallow family running out at the exact transition must hand off, not go quiet"
    assert hand_off_name == "Goldsmith Defense", "the hand-off must summarise the opening that just ended"

    fens_next = _fens(*goldsmith_then_a4, "b7b5")
    assert _book_route(fens_next, swing=False)[0] == _COACH, \
        "the ply AFTER the hand-off must not re-announce — the book already ended once"


def test_family_size_reads_the_table_not_a_guess():
    """`family_size` counts real rows, so a well-covered opening keeps the sticky tolerance and a
    one-shot entry doesn't — pinned on values that would silently drift if the table is rebuilt."""
    assert openings.family_size("Ruy Lopez: Morphy Defense") >= 100
    assert openings.family_size("Goldsmith Defense") == 2
    assert openings.family_size(None) == 0
    assert openings.family_size("") == 0
    assert openings.family_size("Not A Real Opening Name") == 0


def test_is_swing_reads_the_engines_own_classes():
    assert _is_swing({"class": "dubious"}) and _is_swing({"class": "blunder"})
    assert not _is_swing({"class": "ok"}) and not _is_swing({"class": "brilliant"})
    assert not _is_swing({}) and not _is_swing(None)


def test_mover_is_the_side_to_move_of_the_position_played_from():
    assert _mover(START) == "White"
    assert _mover(_fens("e2e4")[1]) == "Black"

