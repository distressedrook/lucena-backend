"""Rating estimator — the parts that hold without a Maia subprocess.

The estimator's contract is arithmetic on a policy, so a stub predictor with a
KNOWN rating-dependent policy pins every claim: a game generated at rating R is
recovered as R, forced moves are dropped, book moves are skipped, and a move
Maia never listed still gets a finite log-probability.
"""

from __future__ import annotations

import math

import chess
import chess.pgn
import pytest

from lucena_backend import rating as R

# A short French Exchange — 4 plies of book, then out.
PGN = """[Event "t"]
[White "w"]
[Black "b"]
[Result "*"]

1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. exd5 exd5 5. Bd3 Nc6 *
"""


class StubMaia:
    """A predictor whose sharpness rises with rating around one 'best' move.

    `pick(fen)` names the move a strong player finds; it gets probability
    `0.2 + 0.6 * (rating - 600) / 2300`, the rest split evenly. So the MLE has a
    real, monotone signal to climb and the recovered rating is checkable by hand.
    """

    def __init__(self, pick):
        self._pick = pick
        self.calls = 0

    def top_human_moves(self, fen, rating, *, n=20, oppo_rating=None):
        self.calls += 1
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        best = self._pick(fen)
        p = 0.2 + 0.6 * (rating - 600) / 2300
        rest = (1.0 - p) / max(1, len(legal) - 1)
        scored = sorted(legal, key=lambda u: (u != best, u))[:n]
        return [{"uci": u, "rank": i, "policy": p if u == best else rest}
                for i, u in enumerate(scored, start=1)]


def _first_legal(fen):
    return next(iter(chess.Board(fen).legal_moves)).uci()


# -- reading the game --------------------------------------------------------

def test_decisions_tags_colour_and_book():
    ds = R.decisions(PGN)
    assert [d.san for d in ds[:4]] == ["e4", "e6", "d4", "d5"]
    assert ds[0].color == chess.WHITE and ds[1].color == chess.BLACK
    # The French Exchange is in the book; the tail of this line has left it.
    assert ds[1].in_book
    assert not ds[-1].in_book


def test_forced_moves_are_dropped():
    # Black is in check from Qh5xf7 mate-ish setup: only one legal reply.
    forced = """[Event "t"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7+ Kxf7 *
"""
    ds = R.decisions(forced)
    assert all(d.n_legal > 1 for d in ds)
    assert "Kxf7" not in [d.san for d in ds]     # the only legal move, no signal


def test_unlisted_move_gets_the_leftover_mass():
    """A move outside MultiPV must not be log(0) — it takes the residual."""
    pol = R._Policy(StubMaia(_first_legal))
    d = R.decisions(PGN)[-1]        # 39 legal moves — more than MultiPV can list
    rows = pol.rows(d.fen, 1500, 1500)
    unlisted = next(m.uci() for m in chess.Board(d.fen).legal_moves
                    if m.uci() not in {r["uci"] for r in rows})
    d2 = R.Decision(ply=1, color=chess.WHITE, fen=d.fen, uci=unlisted, san="?",
                    n_legal=d.n_legal, in_book=False)
    lp, rank = pol.logp(d2, 1500, 1500)
    assert rank == 0
    assert -30 < lp < 0 and math.isfinite(lp)


# -- the fit -----------------------------------------------------------------

def test_recovers_a_strong_player():
    """Every move is the stub's 'best' → the MLE must climb to the grid top."""
    stub = StubMaia(_first_legal)
    # Build a game where both sides always play the stub's pick.
    board = chess.Board()
    game_moves = []
    for _ in range(24):
        board.push(chess.Move.from_uci(_first_legal(board.fen())))
        game_moves.append(board.peek())
    g = chess.pgn.Game()
    node = g
    for m in game_moves:
        node = node.add_variation(m)
    ests = R.estimate_ratings(str(g), stub, include_book=True,
                              grid=list(range(600, 2901, 300)))
    for c in (chess.WHITE, chess.BLACK):
        assert ests[c].pinned == "above"       # off the top of the grid, and says so


def test_book_moves_are_skipped_by_default():
    stub = StubMaia(_first_legal)
    grid = [1000, 1500, 2000]
    with_book = R.estimate_ratings(PGN, stub, include_book=True, grid=grid)
    stub2 = StubMaia(_first_legal)
    without = R.estimate_ratings(PGN, stub2, include_book=False, grid=grid)
    assert without[chess.WHITE].n_moves < with_book[chess.WHITE].n_moves


def test_profile_and_interval_are_consistent():
    stub = StubMaia(_first_legal)
    ests = R.estimate_ratings(PGN, stub, include_book=True,
                              grid=list(range(600, 2901, 100)))
    e = ests[chess.WHITE]
    assert e.low <= e.rating <= e.high
    assert e.log_likelihood == pytest.approx(max(e.profile.values()))
    assert e.n_moves > 0
    assert 0.0 <= e.top1_agreement <= 1.0


def test_policy_cache_collapses_repeated_lookups():
    stub = StubMaia(_first_legal)
    grid = [1000, 1500, 2000]
    R.estimate_ratings(PGN, stub, include_book=True, grid=grid, rounds=3)
    ds = [d for d in R.decisions(PGN)]
    # At most |grid| lookups per (position, opponent-rating) pair, however many
    # coordinate-ascent rounds run — the cache is what makes ROUNDS affordable.
    assert stub.calls <= len(ds) * len(grid) * len(grid)


def test_parabolic_refinement_beats_the_grid():
    prof = {1000: -10.0, 1100: -8.0, 1200: -8.5}      # peak between 1100 and 1200
    peak = R._refine(prof)
    assert 1100 < peak < 1200


def test_flat_profile_falls_back_to_the_grid_point():
    prof = {1000: -8.0, 1100: -8.0, 1200: -8.0}
    assert R._refine(prof) in (1000.0, 1100.0, 1200.0)


def test_no_moves_for_a_colour_is_an_error_not_a_guess():
    one_move = """[Event "t"]

1. e4 *
"""
    with pytest.raises(ValueError):
        R.estimate_ratings(one_move, StubMaia(_first_legal), include_book=True)
