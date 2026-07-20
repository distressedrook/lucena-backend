"""Multi-ply plan reasoner (`reasoning/line.py`): walk the engine PV → the target the move actually
wins, verified (not merely "threatened"). Fixtures are REAL Lichess positions with the engine's PV and
mover-POV cp captured — the same cases the 150-puzzle scale run verified at 100% against an INDEPENDENT
final-board material check (not the reasoner's own walk)."""
from lucena_backend.reasoning import describe_plan, pv_san_to_uci
from lucena_backend.reasoning.line import _v

# Real fired cases (FEN of the mover's position, engine PV as UCIs, mover-POV cp).
WINS_QUEEN = (
    "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25",
    ["e6e7", "b2b1", "b3c1", "b1c1", "h6c1", "g3d6", "e7d7", "d6f8"], 740,
)
UNDERMINE_KNIGHT = (
    "r4rk1/p1R2pb1/2p2p1p/2q1pQ2/4P3/2N5/PPP2PPP/R5K1 b - - 0 15",
    ["c5b6", "c7d7", "b6b2", "a1d1", "b2c3", "h2h4", "g8h8", "d1d6"], 232,
)
# Qxc4 takes the rook; Black recaptures on c4 with the queen; Nxc4 wins it. The win lands on the
# PLAYED move's own square via a recapture — a sequence, NOT an undermine.
RECAPTURE_QUEEN = (
    "4r1k1/2q2pbp/p2p2p1/3Qp3/P1r5/2n1P3/1B1N1PPP/R2R2K1 w - - 6 26",
    ["d5c4", "c7c4", "d2c4", "c3d1", "a1d1", "e8b8"], 416,
)


def test_wins_target_names_the_piece_that_falls():
    fen, pv, cp = WINS_QUEEN
    out = describe_plan(fen, pv, cp)
    assert out == "The point of this move: it wins Black's queen on c1."


def test_undermine_plan_states_the_fall_as_certain():
    fen, pv, cp = UNDERMINE_KNIGHT
    out = describe_plan(fen, pv, cp)
    # the causal chain, stated CERTAIN (the PV already holds the opponent's best defense)
    assert out == ("The point of this move: it undermines the only defender of White's knight on c3 "
                   "— it cannot be held, and falls.")


def test_recapture_win_reads_as_a_sequence_not_an_undermine():
    # When the target is won ON the played move's own square (Qxc4 … Qxc4 … Nxc4), naming "the queen on
    # c4" reads as a contradiction at verdict time (c4 shows the rook you just took). Bridge it instead.
    fen, pv, cp = RECAPTURE_QUEEN
    out = describe_plan(fen, pv, cp)
    assert out == ("The point of this move: it wins material on c4 — after Black recaptures there, "
                   "the queen cannot be held and falls.")
    assert "undermine" not in out          # a trade-and-win must not masquerade as an undermine


def test_cp_floor_drops_the_claim_when_the_engine_disagrees():
    # Same line, but tell it the position is only equal (cp below the +100 floor): claim nothing.
    fen, pv, _ = UNDERMINE_KNIGHT
    assert describe_plan(fen, pv, cp=40) is None


def test_cp_none_does_not_gate():
    # No eval supplied (e.g. a caller without evaluate) → the peak math alone decides; still fires.
    fen, pv, _ = WINS_QUEEN
    assert describe_plan(fen, pv, cp=None) == "The point of this move: it wins Black's queen on c1."


def test_no_pv_or_fen_is_none():
    assert describe_plan(None, ["e2e4"]) is None
    assert describe_plan("8/8/8/8/8/8/8/K6k w - - 0 1", None) is None
    assert describe_plan("8/8/8/8/8/8/8/K6k w - - 0 1", []) is None


def test_no_material_won_is_none():
    # A quiet king shuffle wins nothing → no plan.
    assert describe_plan("8/8/8/4k3/8/8/8/4K3 w - - 0 1", ["e1e2", "e5e4", "e2e1", "e4e5"]) is None


def test_even_trade_is_not_a_won_piece():
    # Rxd8, Kxd8 — the running material count spikes to +5 then gives it right back. Reading a transient
    # PEAK would falsely call this even rook trade "a won rook"; the absolute final balance is level.
    fen, pv = "3r4/1p2k3/8/8/8/8/P6P/3R2K1 w - - 0 1", ["a2a4", "b7b6", "d1d8", "e7d8"]
    assert describe_plan(fen, pv) is None
    assert describe_plan(fen, pv, cp=150) is None


def test_trade_down_from_a_surplus_is_not_a_win():
    # White is already up a rook, then trades that rook for a knight (Rxc3 … Bxe3, recapture on a DIFFERENT
    # square so the target square isn't retaken). Final material is still +2 purely from the old surplus,
    # but the move NET-LOST material — requiring final > pre (not just final > 0) rejects the false claim.
    fen, pv = "6k1/7p/8/8/5b2/2n5/7P/R1R3K1 w - - 0 1", ["a1b1", "h7h6", "c1c3", "h6h5", "c3e3", "f4e3"]
    assert describe_plan(fen, pv, cp=300) is None
    assert describe_plan(fen, pv) is None


def test_sac_into_attack_does_not_claim_the_incidental_pawn():
    # White sacrifices into a mating attack (down ~4 on material) and the line grabs two pawns on the way.
    # The material SWING is +2, but White is not AHEAD — the point is the attack, not the pawn. With a
    # positive cp (the attack IS winning), only the absolute-material guard stops the false "wins a pawn".
    fen = "2rb2k1/p4ppp/1p1P4/8/3Qq1N1/7P/PP2r3/K5R1 w - - 5 29"
    assert describe_plan(fen, ["g4h6", "g8f8", "d4g7", "f8e8", "g7f7"], cp=600) is None


def test_pv_san_to_uci_walks_the_line():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert pv_san_to_uci(start, ["e4", "e5", "Nf3"]) == ["e2e4", "e7e5", "g1f3"]


def test_pv_san_to_uci_stops_at_an_unparseable_tail():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # a garbage token truncates the conversion at the valid prefix rather than raising
    assert pv_san_to_uci(start, ["e4", "zz9", "Nf3"]) == ["e2e4"]
    assert pv_san_to_uci(None, ["e4"]) == []
    assert pv_san_to_uci(start, None) == []


def test_standard_values_not_compressed():
    # the peak math must read real pawn-equivalents (rook 5, knight 3), not the compressed grounding scale
    assert (_v("R"), _v("N"), _v("Q"), _v("P")) == (5, 3, 9, 1)
