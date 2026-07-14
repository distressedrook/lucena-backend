"""Black-box tests for lucena_engine.poisoned_line_detector
(contract: docs/contracts/M-poisoned-line-detector.md).

Written from the contract ONLY. Nothing under python/lucena/mcp/ matching
*poisoned_line_detector*, python/lucena/mcp/nuance.py, or tests/unit/test_nuance.py
was read, opened, grepped, or run to discover an expected value. The module this
file imports does not exist yet at this path (M-poisoned-line-detector is not
implemented) -- these tests encode the promised public shape and are NOT expected
to pass yet. That's the point of contract-first testing (TESTING-WORKFLOW.md):
tests define correctness before the code exists, so a real implementation is
built against them, not the other way around.

`engine` and `maia` are fully faked (duck-typed per the contract's "Dependencies
the caller injects" section) -- no real Stockfish, no real Maia subprocess.
`Line`/`Analysis`/`Score` are the REAL dataclasses from lucena_engine (the
contract explicitly types Analysis.best.score / Analysis.lines[i].score as
lucena_engine.evalmodel.Score), so FakeEngine builds real Line/Analysis objects;
only the *engine itself* -- the process, the search -- is faked.

Every FEN/UCI-legality claim below was independently verified with python-chess
in scratch (never imported here -- GPL hygiene, CLAUDE.md: python-chess may only
be imported under tools/ and tests/, and this file doesn't need it at import
time since every position is hardcoded and pre-verified). Move-sequence comments
document that verification.

Numeric expectations for win%/spike are computed from the SAME win_pct(cp)
formula the M2 contract already documents and lucena_engine exports (reused here
via win_pct, exactly as test_evalmodel.py does), not hand-derived, so exact
approx() anchors are reproducible:
    win_pct(800)  ~ 95.0058   win_pct(100) ~ 59.1026   spike(800,100) ~ 35.90
    win_pct(20)   ~ 51.8402   win_pct(0)   = 50.0
    win_pct(250)  ~ 71.5148   win_pct(-1000, i.e. 100-95.0058 style) ~ 4.9942

Sections:
  * fakes         -- FakeEngine / FakeMaia, fen-keyed, raise loudly on an
                     unmapped fen so an unexpected internal query is visible.
  * find_poisoned_lines -- shape, tiers/boundaries, dedup, ordering, edge cases.
  * find_practical_tries -- shape, floor_cp exclusion, trap_count.
  * worked example -- the contract's own integration oracle, skipped (needs a
                     real engine + a policy-emitting Maia, out of scope here).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import pytest

from lucena_engine import Analysis, Line, Score, win_pct
from lucena_engine.poisoned_line_detector import find_poisoned_lines, find_practical_tries


# ============================================================================
# Fakes
# ============================================================================


class FakeEngine:
    """Fen-keyed fake for `engine.analyse(fen, *, multipv, nodes) -> Analysis`
    and `engine.new_game() -> None`. `table` maps fen -> list of
    (score_kwargs, pv) pairs, already best-first; `analyse` truncates to the
    requested multipv (or returns everything if multipv >= what's stored).
    A fen this scenario didn't anticipate raises AssertionError instead of
    silently fabricating a result -- deliberate: if a real implementation
    queries a fen we didn't expect, that's exactly the kind of divergence
    TESTING-WORKFLOW.md wants surfaced, not masked.
    """

    def __init__(self, table):
        self._table = table
        self.new_game_calls = 0
        self.queried_fens = []

    def new_game(self):
        self.new_game_calls += 1

    def analyse(self, fen, *, multipv=1, nodes=None, movetime_ms=None, depth=None):
        self.queried_fens.append(fen)
        try:
            spec = self._table[fen]
        except KeyError:
            raise AssertionError(
                f"FakeEngine.analyse called on a fen this scenario didn't wire: {fen!r}"
            )
        lines = [Line(rank=i + 1, score=Score(**sc), pv=pv) for i, (sc, pv) in enumerate(spec)]
        return Analysis(fen=fen, lines=lines[:multipv] if multipv else lines)


class FakeMaia:
    """Fen-keyed fake for `maia.top_human_moves(fen, rating, *, n) -> list[dict]`.
    `table` maps fen -> list of {"uci":..., "policy":...} dicts, most-likely
    first (per contract: "policy" is a REAL probability, not derived from rank).
    """

    def __init__(self, table):
        self._table = table
        self.queried = []

    def top_human_moves(self, fen, rating, *, n=6):
        self.queried.append((fen, rating, n))
        try:
            moves = self._table[fen]
        except KeyError:
            raise AssertionError(
                f"FakeMaia.top_human_moves called on a fen this scenario didn't wire: {fen!r}"
            )
        return moves[:n]


# A tactic node's win%/spike anchors, reused everywhere a "one winning idea"
# node is needed: best=cp800 (~95.0%), second=cp100 (~59.1%), spike ~35.9 --
# comfortably clears win_bar=68 / spike=20 defaults without landing near either
# boundary (boundary logic is tested separately, on `poisoned`, not on these).
def tactic_lines(idea_uci, second_uci):
    return [({"cp": 800}, [idea_uci]), ({"cp": 100}, [second_uci])]


def non_tactic_lines(best_uci, second_uci):
    return [({"cp": 20}, [best_uci]), ({"cp": 0}, [second_uci])]


# ============================================================================
# Shared positions (each verified independently with python-chess in scratch;
# comments record the move sequence + legality check, never imported here).
# ============================================================================

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# checkmate: White to move, mated (Fool's mate). Reused from test_mcp.py's
# CHECKMATE_FEN convention -- White has zero legal moves.
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"

# -- "shallow" scenario: seed IS the fatal move (deep=False) -----------------
# 1. d4 (White's seed, SAN "d4") -> Black to move: FEN_A1. idea = ...Nf6 (SAN
#    "Nf6"), legal there. fatal == seed == "d2d4". Root's own best (avoids
#    context) = e2e4 (the "solution"), which != fatal -> avoids True.
FEN_A1 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
SEED_A_UCI = "d2d4"
IDEA_A_UCI = "g8f6"          # SAN "Nf6" at FEN_A1
SOLUTION_UCI = "e2e4"        # legal at START; excluded as a seed

# -- "deep" scenario: seed's own reply is quiet; the FATAL move comes later --
# 1. e3 (seed, SAN "e3") e5 (Black/opponent's Stockfish-best reply) 2. Bc4!?
#    (mover's *second* turn, greedy top-1 continuation, SAN "Bc4", this is the
#    fatal move) -- Black to move at FEN_B3, idea = ...Qh4 (SAN "Qh4", legal:
#    the d8-h4 diagonal is open once e7 has vacated to e5). Stockfish's own
#    move at FEN_B2 (the fatal node) = Nc3 (!= Bc4 -> avoids True).
FEN_B1 = "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1"        # after e2e3
FEN_B2 = "rnbqkbnr/pppp1ppp/8/4p3/8/4P3/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"     # after e2e3 e7e5
FEN_B3 = "rnbqkbnr/pppp1ppp/8/4p3/2B5/4P3/PPPP1PPP/RNBQK1NR b KQkq - 1 2"    # after ... Bc4
SEED_B_UCI = "e2e3"
OPP_REPLY_B_UCI = "e7e5"
FATAL_B_UCI = "f1c4"          # SAN "Bc4" at FEN_B2
IDEA_B_UCI = "d8h4"           # SAN "Qh4" at FEN_B3
STOCKFISH_OWN_B2_UCI = "b1c3"  # SAN "Nc3" at FEN_B2, != fatal -> avoids True

# -- dedup scenario: two DIFFERENT seeds -- "Nf3" first (path P1: g1f3, e7e5,
#    then mover's second move e2e3 is the fatal move) vs "e3" first (path Q,
#    reusing FEN_B1/FEN_B2 from the deep scenario: e2e3, e7e5, then mover's
#    second move g1f3 is the fatal move) -- TRANSPOSE into the exact same
#    fatal-node board (pawn e3 + knight f3 + black pawn e5; placement/side/
#    castling/ep identical), differing only in halfmove clock (0 vs 1, since
#    the last move played differs -- a pawn move resets the clock, a knight
#    move doesn't). This is a genuine transposition, not a coincidental SAN
#    string collision -- the contract dedups on normalized position identity.
FEN_P1_1 = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"          # after g1f3
FEN_P1_2 = "rnbqkbnr/pppp1ppp/8/4p3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq e6 0 2"       # after g1f3 e7e5
FEN_P1_3 = "rnbqkbnr/pppp1ppp/8/4p3/8/4PN2/PPPP1PPP/RNBQKB1R b KQkq - 0 2"       # after g1f3 e7e5 e2e3
FEN_Q3 = "rnbqkbnr/pppp1ppp/8/4p3/8/4PN2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"         # after e2e3 e7e5 g1f3 -- SAME position as FEN_P1_3, clock differs

# -- stalemate scenario: seed leads straight to a stalemated opponent, before
#    any tactic node is evaluated -- "a rollout that ends the game (mate/
#    stalemate reached) before a tactic node contributes nothing."
STALE_ROOT = "7k/8/6K1/5Q2/8/8/8/8 w - - 0 1"
STALE_SEED_UCI = "f5f7"   # Qf7 -> "7k/5Q2/6K1/8/8/8/8/8 b - - 1 1", confirmed
                          # stalemate (0 legal moves) for Black via python-chess.


def make_shallow_engine(idea_uci=IDEA_A_UCI, avoids_cp=250):
    """ROOT + seed d2d4 -> immediate (non-deep) tactic at FEN_A1."""
    return FakeEngine(
        {
            START: [({"cp": avoids_cp}, [SOLUTION_UCI])],  # avoids/drop context
            FEN_A1: tactic_lines(idea_uci, "b8c6"),
        }
    )


def make_shallow_maia(policy, extra_root_moves=None):
    root_moves = [{"uci": SOLUTION_UCI, "policy": 0.9}, {"uci": SEED_A_UCI, "policy": policy}]
    if extra_root_moves:
        root_moves = root_moves + extra_root_moves
    return FakeMaia({START: root_moves})


# -- balanced-root gate (material_floor) roots -------------------------------
# Static (mover-relative) material verified with python-chess in scratch:
#   ROOT_EVEN  ->  0   ROOT_DOWN1 -> -1   ROOT_DOWN2 -> -2
#   ROOT_DOWN_ROOK -> -5   ROOT_BLACK_DOWN2 -> -2 (Black to move -> negated).
# At the default material_floor=-1, a root with mover-relative material < -1 is
# gated (empty result, no seeds examined); == -1 (down exactly a pawn) or 0
# (even) is NOT gated.
ROOT_EVEN = START
ROOT_DOWN1 = "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1"       # White down 1 pawn
ROOT_DOWN2 = "rnbqkbnr/pppppppp/8/8/8/8/2PPPPPP/RNBQKBNR w KQkq - 0 1"        # White down 2 pawns
ROOT_DOWN_ROOK = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1"    # White down a rook
ROOT_BLACK_DOWN2 = "rnbqkbnr/2pppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"  # Black to move, Black down 2 pawns


# -- deep-scenario helpers (fatal move is a mover-ply PAST the seed -> deep=True) --
# Reuses the FEN_B1/B2/B3 deep scenario. Because deep temptations bypass the
# nuance filter (kept at any non-noise tier), these let a test set `poisoned` to
# any value -- including well below shallow_keep_prob -- and still exercise the
# noise/common tier boundaries. `poisoned = seed_policy * fatal_policy * 100`;
# with the default fatal_policy=1.0, `poisoned == seed_policy * 100`.
def make_deep_engine():
    return FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_B1: non_tactic_lines(OPP_REPLY_B_UCI, "d7d5"),
            FEN_B2: [({"cp": 10}, [STOCKFISH_OWN_B2_UCI])],
            FEN_B3: tactic_lines(IDEA_B_UCI, "d8g5"),
        }
    )


def make_deep_maia(seed_policy, fatal_policy=1.0):
    return FakeMaia(
        {
            START: [{"uci": SEED_B_UCI, "policy": seed_policy}],
            FEN_B2: [{"uci": FATAL_B_UCI, "policy": fatal_policy}],
        }
    )


# ============================================================================
# find_poisoned_lines
# ============================================================================


def test_has_poisoned_line_true_basic():
    # policy 0.50 -> poisoned 50.0 > shallow_keep_prob (40.0): a SHALLOW trap this
    # common survives the nuance filter, so has_poisoned_line stays True with a
    # single deep=False temptation. (Slice 2: a shallow trap at <=40% is now
    # dropped; the original 0.12 policy would no longer produce a temptation.)
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.50)
    out = find_poisoned_lines(
        START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2, plies=10
    )
    assert out["fen"] == START
    assert out["has_poisoned_line"] is True
    assert len(out["temptations"]) == 1
    t = out["temptations"][0]
    assert t["fatal"] == "d4"
    assert t["idea"] == "Nf6"
    assert t["seeds"] == ["d4"]
    assert t["deep"] is False


def test_has_poisoned_line_false_when_no_seed_reaches_tactic():
    # ROOT + one quiet seed (a3); its only opponent-node is deliberately
    # non-tactic, and plies=1 caps the rollout right there.
    engine = FakeEngine(
        {
            START: [({"cp": 0}, [SOLUTION_UCI])],
            "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1": non_tactic_lines(
                "b8c6", "g8f6"
            ),
        }
    )
    maia = FakeMaia({START: [{"uci": "a2a3", "policy": 0.15}]})
    out = find_poisoned_lines(START, engine, maia, rating=1500, k=1, plies=1)
    assert out["has_poisoned_line"] is False
    assert out["temptations"] == []


def test_has_poisoned_line_false_checkmate_root_no_legal_moves():
    engine = FakeEngine({})
    maia = FakeMaia({})
    out = find_poisoned_lines(CHECKMATE_FEN, engine, maia, rating=1500)
    assert out["has_poisoned_line"] is False
    assert out["temptations"] == []


def test_fields_present_with_correct_types_and_ranges():
    # policy 0.50 -> poisoned 50.0 > 40.0 keep threshold: the shallow trap survives
    # the nuance filter so there IS a temptation whose fields we can inspect.
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.50)
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2)
    t = out["temptations"][0]
    for key in ("idea", "fatal", "avoids", "drop", "poisoned", "tier"):
        assert key in t, f"missing field {key!r}"
    assert isinstance(t["seeds"], list) and all(isinstance(s, str) for s in t["seeds"])
    assert isinstance(t["opp_win_pct"], float)
    assert isinstance(t["spike"], float)
    assert isinstance(t["deep"], bool)
    assert isinstance(t["avoids"], bool)
    assert isinstance(t["drop"], float)
    assert isinstance(t["poisoned"], float)
    assert 0.0 <= t["poisoned"] <= 100.0
    assert t["tier"] in ("uncommon", "common")
    # avoids/drop context: Stockfish's own move at the fatal node (START) is
    # the solution (e4) != fatal (d4) -> avoids True; drop = win%(stockfish's
    # move) - win%(what's left after fatal, i.e. mover's win% at the tactic
    # node = 100 - opp_win_pct).
    assert t["avoids"] is True
    expected_drop = win_pct(250) - (100 - win_pct(800))
    assert t["drop"] == pytest.approx(expected_drop, abs=0.05)
    assert t["drop"] >= 20.0  # contract: clears on its own once a node exists
    # contract: "engine.new_game() -> None (called before each analyse)".
    assert engine.new_game_calls > 0


def test_rounding_is_exact_not_merely_approximate():
    # Contract: opp_win_pct/spike/drop are round(x, 1); poisoned is round(x, 2).
    # A buggy implementation that returns an unrounded float would pass any
    # pytest.approx(..., abs=0.05)-style check but fail this one.
    # policy 0.412345 -> poisoned 41.2345 -> rounds to 41.23, and 41.23 > 40.0 so
    # the shallow trap survives the nuance filter and there is a temptation to
    # inspect. (Original 0.123456 -> 12.35 would now be dropped as sub-threshold.)
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.412345)
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2)
    t = out["temptations"][0]
    assert t["opp_win_pct"] == round(t["opp_win_pct"], 1)
    assert t["spike"] == round(t["spike"], 1)
    assert t["drop"] == round(t["drop"], 1)
    assert t["poisoned"] == round(t["poisoned"], 2)
    assert t["poisoned"] == round(0.412345 * 100, 2)  # 41.23, not 41.2345 or 41.2


def test_no_noise_tier_value_ever_appears():
    # A noise-level line (d2d4, poisoned 0.50 < noise_bar 1.0) coexists in ONE
    # call with a genuine survivor (deep e2e3, poisoned 10.0). The contract is
    # explicit there is no "noise" tier string in the output: the noise line is
    # simply ABSENT, and the survivor carries a real tier -- so a non-empty
    # result never leaks a "noise" label. (Previously this test had only the
    # filtered line, making it vacuous -- nothing could carry any tier.)
    engine = FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_A1: tactic_lines(IDEA_A_UCI, "b8c6"),          # d2d4's noise-level shallow tactic
            FEN_B1: non_tactic_lines(OPP_REPLY_B_UCI, "d7d5"),
            FEN_B2: [({"cp": 10}, [STOCKFISH_OWN_B2_UCI])],
            FEN_B3: tactic_lines(IDEA_B_UCI, "d8g5"),          # e2e3's surviving deep tactic
        }
    )
    maia = FakeMaia(
        {
            START: [
                {"uci": SOLUTION_UCI, "policy": 0.9},
                {"uci": SEED_A_UCI, "policy": 0.005},   # poisoned 0.50 -> below noise_bar, filtered
                {"uci": SEED_B_UCI, "policy": 0.20},    # deep: 0.20*0.50=10.0 -> kept
            ],
            FEN_B2: [{"uci": FATAL_B_UCI, "policy": 0.50}],
        }
    )
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=3)
    assert out["has_poisoned_line"] is True
    assert len(out["temptations"]) == 1       # the noise line is absent, not labeled
    tiers_seen = {t["tier"] for t in out["temptations"]}
    assert tiers_seen <= {"uncommon", "common"}
    assert "noise" not in tiers_seen


@pytest.mark.parametrize(
    "policy, noise_bar, common_bar, expected_present, expected_tier",
    [
        (0.03, 3.0, 10.0, True, "uncommon"),      # poisoned 3.00 == noise_bar -> included, uncommon
        (0.0299, 3.0, 10.0, False, None),         # poisoned 2.99 < noise_bar -> excluded entirely
        (0.05, 1.0, 5.0, True, "common"),         # poisoned 5.00 == common_bar -> common
        (0.0499, 1.0, 5.0, True, "uncommon"),     # poisoned 4.99 < common_bar -> uncommon
    ],
)
def test_tier_boundary_logic(policy, noise_bar, common_bar, expected_present, expected_tier):
    # Uses a DEEP scenario (fatal one mover-ply past the seed) so the nuance
    # filter (shallow_keep_prob) never applies -- a deep temptation is kept at
    # any non-noise tier regardless of `poisoned`. That isolates the noise_bar/
    # common_bar tier boundaries, which is this test's actual subject. With
    # fatal_policy=1.0, poisoned == policy * 100, so the boundary values are the
    # same as before. (Slice 2: a shallow fixture here would be dropped outright
    # by the nuance filter for every one of these sub-40% poisoned values.)
    engine = make_deep_engine()
    maia = make_deep_maia(seed_policy=policy)
    out = find_poisoned_lines(
        START,
        engine,
        maia,
        rating=1500,
        solution_uci=SOLUTION_UCI,
        k=1,
        noise_bar=noise_bar,
        common_bar=common_bar,
    )
    if not expected_present:
        assert out["temptations"] == []
        assert out["has_poisoned_line"] is False
        return
    assert len(out["temptations"]) == 1
    t = out["temptations"][0]
    assert t["tier"] == expected_tier
    assert t["poisoned"] == pytest.approx(policy * 100, abs=0.01)


def test_dedup_by_tactic_sums_poisoned_and_collects_seeds():
    # Two genuinely different seeds, "Nf3" (path P1) and "e3" (path Q, reusing
    # FEN_B1/FEN_B2), TRANSPOSE into the identical fatal-node position (see
    # FEN_P1_3/FEN_Q3 comment) -- one merged temptation. `poisoned` = sum of
    # both paths' own poisoned (each alone is "uncommon"; summed, "common").
    # avoids/drop must come from path P1 specifically (listed FIRST at root,
    # per contract precedent: first-discovered seed's avoids/drop/opp_win_pct/
    # spike/deep win; later-merging seeds only contribute to poisoned/seeds) --
    # path Q is wired with a DIFFERENT stockfish-best-move/cp at its own
    # fatal node precisely so a wrong implementation (e.g. last-seed-wins, or
    # some average) would be caught rather than accidentally matching.
    engine = FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_P1_1: non_tactic_lines("e7e5", "d7d5"),
            FEN_B1: non_tactic_lines("e7e5", "d7d5"),
            FEN_P1_3: tactic_lines(IDEA_B_UCI, "d8g5"),
            FEN_Q3: tactic_lines(IDEA_B_UCI, "d8g5"),
            FEN_P1_2: [({"cp": 10}, ["b1c3"])],   # avoids/drop context, path P1 (Nf3 first)
            FEN_B2: [({"cp": 5}, ["d2d4"])],       # avoids/drop context, path Q -- deliberately
                                                   # different move+cp so "first seed wins" is
                                                   # actually being tested, not coincidentally true
        }
    )
    maia = FakeMaia(
        {
            START: [
                {"uci": SOLUTION_UCI, "policy": 0.9},
                {"uci": "g1f3", "policy": 0.15},   # path P1, listed first -> "discovered first"
                {"uci": "e2e3", "policy": 0.10},   # path Q, listed second
            ],
            FEN_P1_2: [{"uci": "e2e3", "policy": 0.40}],  # path P1's fatal move
            FEN_B2: [{"uci": "g1f3", "policy": 0.30}],    # path Q's fatal move
        }
    )
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=3)
    assert len(out["temptations"]) == 1
    t = out["temptations"][0]
    assert t["fatal"] == "e3"
    assert t["idea"] == "Qh4"
    # discovery order, NOT sorted -- Nf3's path was listed first at root.
    assert t["seeds"] == ["Nf3", "e3"]
    expected_poisoned = 0.15 * 0.40 * 100 + 0.10 * 0.30 * 100  # 6.0 + 3.0 = 9.0
    assert t["poisoned"] == pytest.approx(expected_poisoned, abs=0.01)
    assert t["tier"] == "common"  # 9.0 >= common_bar(5.0), though each path alone is < 5.0
    # avoids/drop: path P1's own fatal node (FEN_P1_2 said stockfish's best is
    # Nc3, cp=10) -- NOT path Q's (FEN_B2 said d2d4, cp=5).
    assert t["avoids"] is True  # Nc3 != e3
    expected_drop = win_pct(10) - (100 - win_pct(800))
    assert t["drop"] == pytest.approx(expected_drop, abs=0.05)


def test_ordering_deep_first_then_poisoned_descending():
    # Combine the shallow (poisoned 50.0, deep=False) and deep (poisoned 2.5,
    # deep=True) scenarios in ONE call. Naive poisoned-descending order would
    # put shallow first (50.0 > 2.5); the contract mandates deep-first
    # regardless of magnitude. The shallow seed's policy is 0.50 (poisoned 50.0
    # > shallow_keep_prob 40.0) so it survives the nuance filter -- otherwise it
    # would be dropped and there'd be nothing to order it against.
    engine = FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_A1: tactic_lines(IDEA_A_UCI, "b8c6"),
            FEN_B1: non_tactic_lines(OPP_REPLY_B_UCI, "d7d5"),
            FEN_B2: [({"cp": 10}, [STOCKFISH_OWN_B2_UCI])],
            FEN_B3: tactic_lines(IDEA_B_UCI, "d8g5"),
        }
    )
    maia = FakeMaia(
        {
            START: [
                {"uci": SOLUTION_UCI, "policy": 0.9},
                {"uci": SEED_A_UCI, "policy": 0.50},   # shallow: poisoned 50.0 (survives nuance)
                {"uci": SEED_B_UCI, "policy": 0.05},   # deep: 0.05*0.50=2.5
            ],
            FEN_B2: [{"uci": FATAL_B_UCI, "policy": 0.50}],
        }
    )
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=3)
    assert len(out["temptations"]) == 2
    assert out["temptations"][0]["deep"] is True
    assert out["temptations"][0]["fatal"] == "Bc4"
    assert out["temptations"][0]["poisoned"] == pytest.approx(2.5, abs=0.01)
    assert out["temptations"][1]["deep"] is False
    assert out["temptations"][1]["fatal"] == "d4"
    assert out["temptations"][1]["poisoned"] == pytest.approx(50.0, abs=0.01)


def test_maia_none_returns_empty_shape():
    engine = make_shallow_engine()
    out = find_poisoned_lines(START, engine, None, rating=1500)
    assert out == {"fen": START, "has_poisoned_line": False, "temptations": []}


def test_engine_none_returns_empty_shape():
    maia = make_shallow_maia(policy=0.12)
    out = find_poisoned_lines(START, None, maia, rating=1500)
    assert out == {"fen": START, "has_poisoned_line": False, "temptations": []}


def test_both_none_returns_empty_shape():
    out = find_poisoned_lines(START, None, None, rating=1500)
    assert out == {"fen": START, "has_poisoned_line": False, "temptations": []}


def test_illegal_fen_raises_value_error():
    engine = FakeEngine({})
    maia = FakeMaia({})
    with pytest.raises(ValueError):
        find_poisoned_lines("not a real fen at all", engine, maia, rating=1500)


def test_missing_policy_fails_loud_not_silent():
    # A Maia move dict without "policy" (the pre-wrapper production shape) must RAISE,
    # not silently default to 0 -- a 0 would zero every `poisoned` product and make the
    # whole feature quietly return "no poisoned lines", indistinguishable from a clean
    # position. The seed-policy read is the first place it's needed.
    engine = make_shallow_engine()
    maia = FakeMaia({START: [{"uci": SEED_A_UCI, "rank": 1}]})   # note: no "policy" key
    with pytest.raises(ValueError, match="policy"):
        find_poisoned_lines(START, engine, maia, rating=1500, k=2)


def test_solution_uci_never_appears_as_a_seed_or_temptation():
    # The solution is listed by Maia with a high policy but must be skipped.
    # No FakeEngine entry exists for "the solution's own rollout" -- if the
    # implementation doesn't skip it, the very next engine.analyse call on an
    # unmapped fen raises AssertionError, surfacing the bug loudly.
    # policy 0.50 keeps the shallow temptation present (poisoned 50.0 > 40.0) so
    # the "solution absent from a non-empty result" check is meaningful.
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.50)
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2)
    fatals = [t["fatal"] for t in out["temptations"]]
    seeds_flat = [s for t in out["temptations"] for s in t["seeds"]]
    assert "e4" not in fatals
    assert "e4" not in seeds_flat


def test_solution_uci_omitted_gives_same_result_as_passing_it():
    # Contract: "results are unchanged if omitted -- the solution's own rollout
    # doesn't reach a tactic node against itself." Wire an engine entry for the
    # solution's own rollout too (a quiet, non-tactic continuation) so BOTH
    # variants (with/without solution_uci) can run to completion and be
    # compared, rather than one of them erroring on a missing fixture entry.
    engine = FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_A1: tactic_lines(IDEA_A_UCI, "b8c6"),
            # the solution's (e2e4) own rollout: Black replies e7e5, quiet, no tactic.
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": non_tactic_lines(
                "e7e5", "c7c5"
            ),
        }
    )
    maia = make_shallow_maia(policy=0.50)  # keeps the shallow trap present in BOTH variants
    # plies=1 -- the solution's own rollout only needs to reach its one (non-tactic) opponent
    # check; capping plies here avoids needing a second-level maia entry for a walk that was
    # never going anywhere tactically regardless of how far it continued.
    with_solution = find_poisoned_lines(
        START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2, plies=1
    )
    without_solution = find_poisoned_lines(START, engine, maia, rating=1500, k=2, plies=1)
    assert with_solution == without_solution


def test_plies_cutoff_suppresses_a_tactic_that_would_otherwise_be_found():
    # The SAME deep scenario (seed e3, fatal Bc4 two mover-plies later) found at
    # plies=10 (enough to reach it) but suppressed at plies=1 (cut off before
    # the mover's second move even happens) -- proves plies actually bounds the
    # walk rather than being a cosmetic no-op parameter.
    engine = FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_B1: non_tactic_lines(OPP_REPLY_B_UCI, "d7d5"),
            FEN_B2: [({"cp": 10}, [STOCKFISH_OWN_B2_UCI])],
            FEN_B3: tactic_lines(IDEA_B_UCI, "d8g5"),
        }
    )
    maia = FakeMaia(
        {
            START: [{"uci": SEED_B_UCI, "policy": 0.20}],
            FEN_B2: [{"uci": FATAL_B_UCI, "policy": 0.50}],
        }
    )
    out_enough_plies = find_poisoned_lines(START, engine, maia, rating=1500, k=1, plies=10)
    assert out_enough_plies["has_poisoned_line"] is True

    out_cut_short = find_poisoned_lines(START, engine, maia, rating=1500, k=1, plies=1)
    assert out_cut_short["has_poisoned_line"] is False
    assert out_cut_short["temptations"] == []


def test_rollout_ending_in_stalemate_before_tactic_contributes_nothing():
    engine = FakeEngine({})  # deliberately no entry for the post-stalemate fen
    maia = FakeMaia({STALE_ROOT: [{"uci": STALE_SEED_UCI, "policy": 0.5}]})
    out = find_poisoned_lines(STALE_ROOT, engine, maia, rating=1500, k=1, plies=4)
    assert out["has_poisoned_line"] is False
    assert out["temptations"] == []


def test_determinism_repeated_calls_return_equal_dict():
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.50)  # keep a real (surviving) temptation to compare
    a = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2)
    b = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2)
    assert a == b


def test_stop_on_first_returns_the_first_temptation_found():
    # policy 0.50 -> the single shallow trap survives the nuance filter and is a
    # KEPT temptation, which is what stop_on_first stops at.
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.50)
    out = find_poisoned_lines(
        START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2, stop_on_first=True
    )
    assert out["has_poisoned_line"] is True
    assert len(out["temptations"]) == 1
    assert out["temptations"][0]["fatal"] == "d4"


def test_stop_on_first_stops_at_the_first_seed_in_maia_order_not_all_of_them():
    # Two DIFFERENT seeds, each independently reaching its own tactic, supplied
    # in correctly policy-descending order (a well-behaved maia -- per
    # contract, "ranked most-likely first" is the injected maia's obligation;
    # find_poisoned_lines is not required to defensively re-sort a misbehaving
    # fake). d2d4 (higher policy) is listed first, e2e3 (lower policy) second.
    # stop_on_first must return exactly ONE temptation -- d2d4's -- not both,
    # and not e3/Bc4's. d2d4's seed policy is 0.50 (poisoned 50.0 > 40.0) so its
    # shallow trap is KEPT: stop_on_first stops at the first KEPT trap, and this
    # first one already qualifies. (A separate test covers the case where the
    # first seed's trap is DROPPED and the scan must continue to a later one.)
    engine = FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_A1: tactic_lines(IDEA_A_UCI, "b8c6"),
            FEN_B1: non_tactic_lines(OPP_REPLY_B_UCI, "d7d5"),
            FEN_B2: [({"cp": 10}, [STOCKFISH_OWN_B2_UCI])],
            FEN_B3: tactic_lines(IDEA_B_UCI, "d8g5"),
        }
    )
    maia = FakeMaia(
        {
            START: [
                {"uci": SOLUTION_UCI, "policy": 0.9},
                {"uci": SEED_A_UCI, "policy": 0.50},    # listed first, higher policy, KEPT
                {"uci": SEED_B_UCI, "policy": 0.05},    # listed second, lower policy
            ],
            FEN_B2: [{"uci": FATAL_B_UCI, "policy": 0.50}],
        }
    )
    out = find_poisoned_lines(
        START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=3, stop_on_first=True
    )
    assert len(out["temptations"]) == 1
    assert out["temptations"][0]["fatal"] == "d4"  # the first seed's tactic; e3/Bc4 never reached


# ============================================================================
# Balanced-root gate (material_floor) -- new in slice 2
# ============================================================================


def test_material_floor_gates_root_down_more_than_a_pawn_no_seeds_examined():
    # Mover down a whole rook (mover-relative -5 < material_floor default -1):
    # the balanced-root gate fires FIRST, returning the empty result WITHOUT
    # examining any seed. Empty engine/maia tables would raise AssertionError if
    # the implementation queried them -- proving "no seeds examined": neither
    # maia.top_human_moves nor engine.analyse is ever called.
    engine = FakeEngine({})
    maia = FakeMaia({})
    out = find_poisoned_lines(ROOT_DOWN_ROOK, engine, maia, rating=1500)
    assert out == {"fen": ROOT_DOWN_ROOK, "has_poisoned_line": False, "temptations": []}
    assert maia.queried == []          # no seed selection happened
    assert engine.queried_fens == []   # no rollout / analysis happened
    assert engine.new_game_calls == 0


def test_material_floor_down_two_pawns_is_gated():
    # Mover-relative -2 < -1 -> gated, same empty result, no seeds examined.
    engine = FakeEngine({})
    maia = FakeMaia({})
    out = find_poisoned_lines(ROOT_DOWN2, engine, maia, rating=1500)
    assert out == {"fen": ROOT_DOWN2, "has_poisoned_line": False, "temptations": []}
    assert maia.queried == []


def test_material_floor_black_to_move_uses_mover_relative_sign():
    # Black to move and Black down two pawns -> mover-relative material is -2
    # (negated for the side to move), so the gate fires. This pins that the sign
    # is taken from the SIDE TO MOVE, not from White's absolute point count.
    engine = FakeEngine({})
    maia = FakeMaia({})
    out = find_poisoned_lines(ROOT_BLACK_DOWN2, engine, maia, rating=1500)
    assert out == {"fen": ROOT_BLACK_DOWN2, "has_poisoned_line": False, "temptations": []}
    assert maia.queried == []


def test_material_floor_down_exactly_a_pawn_is_not_gated():
    # Mover-relative -1 is NOT < material_floor (-1); the position passes the
    # gate and seeds ARE examined. An empty maia seed list means no temptation is
    # found, but the KEY observable is that maia.top_human_moves WAS called at the
    # root (the gate let the scan begin) -- the boundary is inclusive of "down
    # exactly a pawn."
    engine = FakeEngine({})
    maia = FakeMaia({ROOT_DOWN1: []})
    out = find_poisoned_lines(ROOT_DOWN1, engine, maia, rating=1500)
    assert out["has_poisoned_line"] is False
    assert out["temptations"] == []
    assert maia.queried != []            # gate passed: seed selection ran
    assert maia.queried[0][0] == ROOT_DOWN1


def test_material_floor_even_root_is_not_gated():
    # Even material (mover-relative 0) passes; seeds examined.
    engine = FakeEngine({})
    maia = FakeMaia({ROOT_EVEN: []})
    out = find_poisoned_lines(ROOT_EVEN, engine, maia, rating=1500)
    assert out["has_poisoned_line"] is False
    assert out["temptations"] == []
    assert maia.queried != []


def test_material_floor_caller_override_loosens_the_cutoff():
    # A caller-supplied material_floor changes where the gate fires. With
    # material_floor=-5, the down-a-rook root (mover-relative -5) is NOT gated
    # (-5 is not < -5) -- seeds ARE examined -- whereas at the default -1 the same
    # root was gated (see test_material_floor_gates_...). Proves the cutoff is the
    # parameter, not a hardcoded -1.
    engine = FakeEngine({})
    maia = FakeMaia({ROOT_DOWN_ROOK: []})
    out = find_poisoned_lines(ROOT_DOWN_ROOK, engine, maia, rating=1500, material_floor=-5)
    assert out["has_poisoned_line"] is False
    assert out["temptations"] == []
    assert maia.queried != []            # gate passed at the looser floor


def test_material_floor_caller_override_tightens_the_cutoff():
    # With material_floor=0, a root down exactly one pawn (mover-relative -1 < 0)
    # is now gated, even though it passed at the default -1. Same root, opposite
    # outcome purely from the caller's floor -> the cutoff is caller-controlled.
    engine = FakeEngine({})
    maia = FakeMaia({})
    out = find_poisoned_lines(ROOT_DOWN1, engine, maia, rating=1500, material_floor=0)
    assert out == {"fen": ROOT_DOWN1, "has_poisoned_line": False, "temptations": []}
    assert maia.queried == []


# ============================================================================
# Nuance filter (shallow_keep_prob) -- new in slice 2
# ============================================================================


def test_shallow_trap_at_or_below_keep_prob_is_dropped():
    # A SHALLOW trap (deep=False, fatal move IS the seed) that clears noise_bar
    # but whose poisoned is <= shallow_keep_prob (default 40.0) is discarded --
    # "a rare one-move blunder, not a special line." policy 0.30 -> poisoned 30.0.
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.30)
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2)
    assert out["has_poisoned_line"] is False
    assert out["temptations"] == []


def test_shallow_trap_above_keep_prob_is_kept():
    # The same SHALLOW trap with poisoned > 40.0 (policy 0.45 -> 45.0) survives.
    engine = make_shallow_engine()
    maia = make_shallow_maia(policy=0.45)
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=2)
    assert out["has_poisoned_line"] is True
    assert len(out["temptations"]) == 1
    t = out["temptations"][0]
    assert t["deep"] is False
    assert t["fatal"] == "d4"
    assert t["poisoned"] == pytest.approx(45.0, abs=0.01)


def test_shallow_keep_prob_boundary_is_strictly_greater_than():
    # The keep rule is `poisoned > shallow_keep_prob`, strict. poisoned exactly
    # 40.0 (policy 0.40) is DROPPED; 40.01 (policy 0.4001) is KEPT.
    engine = make_shallow_engine()
    out_at = find_poisoned_lines(
        START, engine, make_shallow_maia(policy=0.40), rating=1500,
        solution_uci=SOLUTION_UCI, k=2,
    )
    assert out_at["has_poisoned_line"] is False
    assert out_at["temptations"] == []

    out_over = find_poisoned_lines(
        START, engine, make_shallow_maia(policy=0.4001), rating=1500,
        solution_uci=SOLUTION_UCI, k=2,
    )
    assert out_over["has_poisoned_line"] is True
    assert len(out_over["temptations"]) == 1


def test_deep_trap_with_low_poisoned_bypasses_the_nuance_filter():
    # A DEEP trap (fatal one mover-ply past the seed) is kept at ANY non-noise
    # tier regardless of how rare it is: poisoned ~3.0% (well below the 40.0
    # shallow keep threshold) is KEPT precisely because it's deep. This is the
    # asymmetry the nuance filter encodes -- length is its own nuance.
    engine = make_deep_engine()
    maia = make_deep_maia(seed_policy=0.03)  # poisoned 0.03 * 1.0 * 100 = 3.0
    out = find_poisoned_lines(START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=1)
    assert out["has_poisoned_line"] is True
    assert len(out["temptations"]) == 1
    t = out["temptations"][0]
    assert t["deep"] is True
    assert t["fatal"] == "Bc4"
    assert t["poisoned"] == pytest.approx(3.0, abs=0.01)
    assert t["poisoned"] <= 40.0  # far under shallow_keep_prob, yet kept


def test_shallow_keep_prob_caller_override_changes_the_cutoff():
    # A caller-supplied shallow_keep_prob moves the shallow cutoff. The SAME
    # shallow trap (policy 0.30 -> poisoned 30.0):
    #   * default (40.0): 30.0 <= 40.0 -> dropped
    #   * shallow_keep_prob=10.0: 30.0 > 10.0 -> kept
    #   * shallow_keep_prob=60.0: 30.0 <= 60.0 -> dropped
    engine = make_shallow_engine()

    kept = find_poisoned_lines(
        START, engine, make_shallow_maia(policy=0.30), rating=1500,
        solution_uci=SOLUTION_UCI, k=2, shallow_keep_prob=10.0,
    )
    assert kept["has_poisoned_line"] is True
    assert len(kept["temptations"]) == 1
    assert kept["temptations"][0]["deep"] is False

    dropped = find_poisoned_lines(
        START, engine, make_shallow_maia(policy=0.30), rating=1500,
        solution_uci=SOLUTION_UCI, k=2, shallow_keep_prob=60.0,
    )
    assert dropped["has_poisoned_line"] is False
    assert dropped["temptations"] == []


def test_stop_on_first_skips_a_dropped_shallow_and_returns_a_later_deep_trap():
    # The slice-2 stop_on_first change: it stops at the first KEPT temptation,
    # not merely the first trap FOUND. The most-human-likely seed (d2d4, listed
    # first) reaches a SHALLOW trap whose poisoned (0.20 -> 20.0) is dropped by
    # the nuance filter; a later, less-likely seed (e2e3) reaches a DEEP trap that
    # IS kept. A "stop at first trap found" implementation would return empty
    # (or stop at the dropped shallow); the contract requires the scan to continue
    # and return the deep trap, with has_poisoned_line True.
    engine = FakeEngine(
        {
            START: [({"cp": 250}, [SOLUTION_UCI])],
            FEN_A1: tactic_lines(IDEA_A_UCI, "b8c6"),          # d2d4's shallow trap
            FEN_B1: non_tactic_lines(OPP_REPLY_B_UCI, "d7d5"),
            FEN_B2: [({"cp": 10}, [STOCKFISH_OWN_B2_UCI])],
            FEN_B3: tactic_lines(IDEA_B_UCI, "d8g5"),          # e2e3's deep trap
        }
    )
    maia = FakeMaia(
        {
            START: [
                {"uci": SOLUTION_UCI, "policy": 0.9},
                {"uci": SEED_A_UCI, "policy": 0.20},   # first, higher policy: SHALLOW, poisoned 20.0 -> dropped
                {"uci": SEED_B_UCI, "policy": 0.15},   # later, lower policy: DEEP -> kept
            ],
            FEN_B2: [{"uci": FATAL_B_UCI, "policy": 0.50}],
        }
    )
    out = find_poisoned_lines(
        START, engine, maia, rating=1500, solution_uci=SOLUTION_UCI, k=3, stop_on_first=True
    )
    assert out["has_poisoned_line"] is True
    assert len(out["temptations"]) == 1
    t = out["temptations"][0]
    assert t["deep"] is True
    assert t["fatal"] == "Bc4"   # the later, deep, KEPT trap -- not d2d4's dropped shallow
    assert "d4" not in t["seeds"]


# ============================================================================
# FakeMaia self-check: the contract's "policy" requirement, not rank
# ============================================================================


def test_fake_maia_fixture_returns_uci_and_policy_keys():
    # Meta-check on our own fixtures: every dict returned by the injected
    # maia's top_human_moves must carry BOTH "uci" and "policy" (a real
    # probability, in [0, 1]) -- the exact gap slice 1's rank-only signal left
    # open, per the contract's "Why rank alone was never enough."
    maia = make_shallow_maia(policy=0.12)
    moves = maia.top_human_moves(START, 1500, n=2)
    assert len(moves) == 2
    for m in moves:
        assert "uci" in m and "policy" in m
        assert isinstance(m["policy"], float)
        assert 0.0 <= m["policy"] <= 1.0


# ============================================================================
# find_practical_tries
# ============================================================================

# My candidates from START:
#   d2d4 -> D  (= FEN_A1) -- Black's own seed (Nf6) walks into a tactic for
#             White at D_TACT; opponent.trap_count == 1.
#   b2b3 -> D2 -- quiet, Black has no seeds -> opponent.trap_count == 0.
#   h2h4 -> D3 -- deliberately bad for White (my_cp far below floor_cp) ->
#             excluded from `tries` regardless of trap_count.
D = FEN_A1
D_TACT = "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2"  # after d2d4 g8f6
D2 = "rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1"        # after b2b3
D3 = "rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq h3 0 1"        # after h2h4

# e2e4's resulting position + its (shallow, sub-threshold) tactic node -- reused
# from the ordering test's already-verified FENs.
E = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"        # after e2e4
E_TACT = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"  # after e2e4 e7e5

# -- balanced-root gate at a candidate's RESULTING position -------------------
# Custom EVEN root (mover-relative 0, White to move; material verified with
# python-chess in scratch). White has two candidate moves:
#   Qxa8 (a3a8) -> captures a rook: the OPPONENT (Black, to move) is left at
#                 mover-relative -5 -> find_poisoned_lines gates -> trap_count 0.
#   h2h3        -> quiet: Black stays even (0) -> not gated, and is wired with a
#                 surviving shallow trap -> trap_count 1.
PT_GATE_ROOT = "r5k1/3q2pp/8/8/2r5/Q7/6PP/3R1RK1 w - - 0 1"
PT_GATE_CAPTURE = "a3a8"   # Qxa8 -- leaves opponent down a rook
PT_GATE_QUIET = "h2h3"     # quiet -- leaves opponent even
PT_RG = "Q5k1/3q2pp/8/8/2r5/8/6PP/3R1RK1 b - - 0 1"                       # after Qxa8 (Black mover-rel -5)
PT_RQ = "r5k1/3q2pp/8/8/2r5/Q6P/6P1/3R1RK1 b - - 0 1"                     # after h2h3 (Black mover-rel 0)
PT_RQ_TACT = "r5k1/6pp/8/3q4/2r5/Q6P/6P1/3R1RK1 w - - 1 2"                # after h2h3 d7d5 (tactic node)


def make_practical_tries_fakes():
    engine = FakeEngine(
        {
            D: [({"cp": 10}, ["d7d5"]), ({"cp": 5}, ["c7c5"])],   # avoids/drop + my_cp context
            D_TACT: tactic_lines("d1d3", "b1c3"),                  # the tactic White wins
            D2: [({"cp": 0}, ["d7d5"])],
            D3: [({"cp": 400}, ["d7d5"])],  # Black much better -> my_cp = -400, < floor_cp
        }
    )
    maia = FakeMaia(
        {
            # policy 0.50 -> poisoned 50.0 > shallow_keep_prob (40.0): the shallow
            # trap survives find_poisoned_lines's nuance filter so d2d4's opponent
            # really is left with one temptation (trap_count 1). (Slice 2: 0.20
            # would now be filtered out, leaving trap_count 0.)
            D: [{"uci": "g8f6", "policy": 0.50}],
            D2: [],
            D3: [],
        }
    )
    return engine, maia


def test_find_practical_tries_shape_and_fields():
    engine, maia = make_practical_tries_fakes()
    out = find_practical_tries(
        START, engine, maia, rating=1500, candidates=["d2d4", "b2b3", "h2h4"], floor_cp=-250, k=1
    )
    assert out["fen"] == START
    assert isinstance(out["tries"], list)
    uci_seen = {tr["move_uci"] for tr in out["tries"]}
    assert "d2d4" in uci_seen
    for tr in out["tries"]:
        assert set(tr) >= {"move_san", "move_uci", "my_cp", "trap_count", "top_spike", "opponent"}
        assert isinstance(tr["my_cp"], int)
        assert isinstance(tr["trap_count"], int)
        assert isinstance(tr["opponent"], dict)
        assert "has_poisoned_line" in tr["opponent"]
        assert "temptations" in tr["opponent"]


def test_find_practical_tries_trap_count_and_riddled_opponent():
    # Contract: "a move that leaves the opponent no temptations" is excluded from `tries`
    # entirely (a practical try must actually pose a problem) -- b2b3 (trap_count 0) must be
    # ABSENT, not present-with-trap_count-0.
    engine, maia = make_practical_tries_fakes()
    out = find_practical_tries(
        START, engine, maia, rating=1500, candidates=["d2d4", "b2b3"], floor_cp=-250, k=1
    )
    by_uci = {tr["move_uci"]: tr for tr in out["tries"]}
    assert by_uci["d2d4"]["trap_count"] == 1
    assert by_uci["d2d4"]["opponent"]["has_poisoned_line"] is True
    assert "b2b3" not in by_uci


def test_find_practical_tries_floor_cp_excludes_bad_candidates():
    engine, maia = make_practical_tries_fakes()
    out = find_practical_tries(
        START,
        engine,
        maia,
        rating=1500,
        candidates=["d2d4", "b2b3", "h2h4"],
        floor_cp=-250,
        k=1,
    )
    uci_seen = {tr["move_uci"] for tr in out["tries"]}
    # h2h4's my_cp (~-400) is below floor_cp=-250 -> excluded on the floor rule.
    # b2b3 stays above the floor but poses no problem (0 temptations) -> excluded on that rule.
    assert "h2h4" not in uci_seen
    assert "b2b3" not in uci_seen
    assert "d2d4" in uci_seen


def test_find_practical_tries_maia_none_or_engine_none_empty_shape():
    # Contract now pins this exactly (symmetric with find_poisoned_lines):
    # {"fen": fen, "tries": []}.
    engine, maia = make_practical_tries_fakes()
    out_no_maia = find_practical_tries(START, engine, None, rating=1500, candidates=["d2d4"])
    out_no_engine = find_practical_tries(START, None, maia, rating=1500, candidates=["d2d4"])
    assert out_no_maia == {"fen": START, "tries": []}
    assert out_no_engine == {"fen": START, "tries": []}


def test_find_practical_tries_ordering_and_top_spike_value():
    # Two candidates, both riddling the opponent: d2d4 leaves exactly one
    # temptation (top_spike = that temptation's own spike); a second candidate
    # e2e4 leaves TWO temptations (both walking into an open-diagonal Qh5)
    # with a higher top_spike among them -- trap_count descending should put
    # e2e4 first even though it's evaluated second in the candidates list.
    engine = FakeEngine(
        {
            D: [({"cp": 10}, ["d7d5"]), ({"cp": 5}, ["c7c5"])],
            D_TACT: tactic_lines("d1d3", "b1c3"),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": [
                ({"cp": 10}, ["d7d5"]), ({"cp": 5}, ["c7c5"])
            ],
            # e2e4's Black replies: two DIFFERENT seeds both walking into a
            # Qh5 idea (the diagonal is open since the e-pawn already moved),
            # one with a bigger spike than D_TACT's.
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": tactic_lines(
                "d1h5", "b1c3"
            ),
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2": [
                ({"cp": 900}, ["d1h5"]), ({"cp": 50}, ["b1c3"])
            ],
        }
    )
    maia = FakeMaia(
        {
            # all seed policies > 0.40 so every shallow trap survives the nuance
            # filter -- otherwise d2d4 (1 trap) and e2e4 (2 traps) would both
            # collapse to trap_count 0 and there'd be nothing to order.
            D: [{"uci": "g8f6", "policy": 0.50}],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": [
                {"uci": "e7e5", "policy": 0.60}, {"uci": "d7d5", "policy": 0.50}
            ],
        }
    )
    out = find_practical_tries(
        START, engine, maia, rating=1500, candidates=["d2d4", "e2e4"], floor_cp=-250, k=2
    )
    assert [tr["move_uci"] for tr in out["tries"]] == ["e2e4", "d2d4"]  # trap_count 2 before 1
    by_uci = {tr["move_uci"]: tr for tr in out["tries"]}
    assert by_uci["d2d4"]["trap_count"] == 1
    d4_spike = by_uci["d2d4"]["opponent"]["temptations"][0]["spike"]
    assert by_uci["d2d4"]["top_spike"] == pytest.approx(d4_spike, abs=0.01)


def test_find_practical_tries_candidates_none_defaults_to_my_own_maia_top_k():
    # candidates=None -> my own Maia top-k moves at `fen` become the candidate
    # set (same seed-selection call find_poisoned_lines makes at the root).
    engine, _ = make_practical_tries_fakes()
    maia = FakeMaia(
        {
            START: [{"uci": "d2d4", "policy": 0.30}],
            # opponent's seed policy 0.50 keeps d2d4's one trap alive (poisoned
            # 50.0 > 40.0), so d2d4 actually appears in `tries`.
            D: [{"uci": "g8f6", "policy": 0.50}],
        }
    )
    out = find_practical_tries(START, engine, maia, rating=1500, floor_cp=-250, k=1)
    assert {tr["move_uci"] for tr in out["tries"]} <= {"d2d4"}
    assert maia.queried[0] == (START, 1500, 1)


def test_find_practical_tries_trap_count_counts_only_both_filter_survivors():
    # Slice 2: trap_count counts temptations that survived find_poisoned_lines's
    # BOTH keep-filters -- now the nuance filter too. e2e4 leaves the opponent
    # with only a SHALLOW sub-threshold trap (seed e7e5, poisoned 20.0 <= 40.0),
    # which is dropped -> trap_count 0 -> e2e4 is excluded from `tries` entirely.
    # d2d4 leaves a surviving trap (policy 0.50 -> poisoned 50.0) -> included with
    # trap_count 1. Both resulting positions are even (not gated), isolating the
    # nuance filter as the sole reason e2e4 drops out.
    engine = FakeEngine(
        {
            D: [({"cp": 10}, ["d7d5"]), ({"cp": 5}, ["c7c5"])],   # d2d4 my_cp + fatal-node context
            D_TACT: tactic_lines("d1d3", "b1c3"),                  # d2d4's surviving trap
            E: [({"cp": -10}, ["e7e5"])],                          # e2e4 my_cp + fatal-node context
            E_TACT: tactic_lines("d1h5", "b1c3"),                  # e2e4's shallow (dropped) trap
        }
    )
    maia = FakeMaia(
        {
            D: [{"uci": "g8f6", "policy": 0.50}],   # survives nuance filter
            E: [{"uci": "e7e5", "policy": 0.20}],   # poisoned 20.0 <= 40.0 -> dropped
        }
    )
    out = find_practical_tries(
        START, engine, maia, rating=1500, candidates=["d2d4", "e2e4"], floor_cp=-250, k=1
    )
    by_uci = {tr["move_uci"]: tr for tr in out["tries"]}
    assert by_uci["d2d4"]["trap_count"] == 1
    assert by_uci["d2d4"]["opponent"]["has_poisoned_line"] is True
    assert "e2e4" not in by_uci   # its only trap was filtered -> zero survivors -> excluded


def test_find_practical_tries_balanced_root_gate_applies_at_resulting_position():
    # Slice 2: the balanced-root gate applies at each candidate's RESULTING
    # position. Qxa8 leaves the opponent (Black, to move) down a rook
    # (mover-relative -5 < material_floor -1) -> find_poisoned_lines returns the
    # empty result for that position -> trap_count 0 -> Qxa8 is excluded from
    # `tries`. The quiet h2h3 leaves Black even (not gated) with a surviving
    # shallow trap -> included. Same root, and Qxa8's exclusion is due purely to
    # the material gate at its resulting position (not the floor_cp rule: winning
    # a rook makes my_cp strongly positive, well above floor_cp).
    engine = FakeEngine(
        {
            PT_RG: [({"cp": -900}, ["g8h8"])],       # my_cp = +900 (I just won a rook), above floor
            PT_RQ: [({"cp": -20}, ["a8b8"])],        # my_cp = +20; also the shallow fatal-node context
            PT_RQ_TACT: tactic_lines("a3a8", "a3a7"),  # h2h3's surviving trap (idea/second are legal here)
        }
    )
    maia = FakeMaia(
        {
            # PT_RG is gated before any maia call -> no entry needed; an entry
            # here would never be reached (the gate short-circuits first).
            PT_RQ: [{"uci": "d7d5", "policy": 0.50}],  # shallow, poisoned 50.0 -> survives
        }
    )
    out = find_practical_tries(
        PT_GATE_ROOT,
        engine,
        maia,
        rating=1500,
        candidates=[PT_GATE_CAPTURE, PT_GATE_QUIET],
        floor_cp=-250,
        k=1,
    )
    by_uci = {tr["move_uci"]: tr for tr in out["tries"]}
    assert PT_GATE_CAPTURE not in by_uci          # gated at its resulting position -> 0 traps -> excluded
    assert by_uci[PT_GATE_QUIET]["trap_count"] == 1
    assert by_uci[PT_GATE_QUIET]["opponent"]["has_poisoned_line"] is True


# ============================================================================
# Worked example (the contract's own deterministic integration oracle)
# ============================================================================


@pytest.mark.skip(
    reason=(
        "Requires a real Engine(threads=1) AND a Maia that emits a real 'policy' "
        "probability per move. Production MaiaEngine talks to maia3-uci, which "
        "does not currently emit policy on the wire (contract: 'that wiring is "
        "out of scope for this contract'). Kept here, skipped, as the documented "
        "regression oracle to un-skip once a policy-emitting Maia path exists."
    )
)
def test_worked_example_from_contract():
    fen = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"
    # from lucena_engine import Engine
    # from lucena_engine.maia import MaiaEngine  # would need real policy support
    # with Engine(threads=1) as engine, MaiaEngine() as maia:
    #     out = find_poisoned_lines(fen, engine, maia, rating=1500, solution_uci="b5c4")
    #     assert out["has_poisoned_line"] is True
    #     t = next(t for t in out["temptations"] if t["idea"] == "hxg3")
    #     assert t["fatal"] == "Nxg3"
    #     assert "Nxe4" in t["seeds"]
    #     assert t["deep"] is True
    #     assert t["avoids"] is True
    #     assert t["drop"] >= 20.0
    #     assert t["tier"] == "common"
    #     assert "bxc4" not in [tt["fatal"] for tt in out["temptations"]]
