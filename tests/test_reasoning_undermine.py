"""The undermine-the-defender reasoner (lucena_backend.reasoning.undermine). Pure board derivation —
no LLM, no engine search — so these are deterministic. Validated against the real Lichess DB
separately (the coverage survey); these lock the two forms + the gap-2 secondary note + no-false-fire.
"""
from __future__ import annotations

from lucena_backend.reasoning.undermine import attacks_defender, undermines_defender


def test_attacks_defender_fires_only_for_the_fork_not_a_capture():
    # The public ATTACK-only entry (coach prefers it over the multi-ply plan). It fires when the move
    # THREATENS the guard (Rxc5), and stays silent when the move CAPTURES a piece (Rxc3) — that's the
    # REMOVE form's job, and the plan/recapture reasoners must keep those cases.
    assert "cannot save both" in attacks_defender(
        "r4rk1/p2p2p1/3Np3/2p3R1/5p2/2n2P1P/b1P3PB/R5K1 w - - 0 2", "g5c5")
    assert attacks_defender("1r3rk1/p2p2p1/3Np3/2R5/5p2/2n2P1P/b1P3PB/R5K1 w - - 1 3", "c5c3") is None
    assert attacks_defender(None, "a1a2") is None


def test_attack_form_is_the_fork_dilemma():
    # Rxc5 (g5->c5) lands the rook ATTACKING the c3 knight, the ONLY defender of the a2 bishop. The guard
    # still stands, so this is the fork: Black can't save both — one of them falls (Black chooses which).
    out = undermines_defender("r4rk1/p2p2p1/3Np3/2p3R1/5p2/2n2P1P/b1P3PB/R5K1 w - - 0 2", "g5c5")
    assert out is not None
    assert "attacks the knight on c3" in out
    assert "only defender of Black's bishop on a2" in out
    assert "cannot save both" in out             # the dilemma, NOT 'wins both'


def test_remove_form_leads_with_the_captured_piece_only():
    # Rxc3 (c5->c3) CAPTURES the knight (a clean win, SEE +). Taking it RESOLVES the fork in favour of the
    # knight — that is the point. The bishop does NOT also fall (Black moves next: Rb1+ rescues it), so
    # single-ply must not claim it: it's one piece OR the other, never both.
    out = undermines_defender("1r3rk1/p2p2p1/3Np3/2R5/5p2/2n2P1P/b1P3PB/R5K1 w - - 1 3", "c5c3")
    assert out is not None
    assert out == "The point of this move: it wins Black's knight on c3."
    assert "bishop" not in out and "falls" not in out    # no over-claim about the second piece
    assert "attacks" not in out                          # REMOVE form, not the attack form


def test_remove_form_names_only_the_piece_you_took():
    # Rxc5 (a5->c5) wins the ROOK (a clean capture) which was the sole defender of the c4 knight. You took
    # the rook — that's the point; the knight is the road not taken, so it is not claimed to fall.
    out = undermines_defender("4k3/8/8/R1r5/2n5/8/8/4K3 w - - 0 1", "a5c5")
    assert out is not None
    assert out == "The point of this move: it wins Black's rook on c5."
    assert "knight" not in out and "falls" not in out


def test_none_when_no_sole_guard_motif():
    # A victim with TWO defenders is not 'undermined' by attacking one; a plain capture isn't the motif.
    assert undermines_defender("4k3/8/8/8/8/2n1n3/8/3RK3 w - - 0 1", "d1d3") is None
    assert undermines_defender("k7/2K5/1P6/8/7p/1rR4p/7P/8 w - - 0 7", "c3h3") is None   # mate puzzle
    assert undermines_defender(None, "a1a2") is None
