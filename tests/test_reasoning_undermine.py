"""The undermine-the-defender reasoner (lucena_backend.reasoning.undermine). Pure board derivation —
no LLM, no engine search — so these are deterministic. Validated against the real Lichess DB
separately (the coverage survey); these lock the two forms + the gap-2 secondary note + no-false-fire.
"""
from __future__ import annotations

from lucena_backend.reasoning.undermine import undermines_defender


def test_attack_form():
    # Rxc5 (g5->c5) lands the rook attacking the c3 knight, the ONLY defender of the a2 bishop.
    # The reasoner derives the causal point: attack the guard -> the defended piece can't be held.
    out = undermines_defender("r4rk1/p2p2p1/3Np3/2p3R1/5p2/2n2P1P/b1P3PB/R5K1 w - - 0 2", "g5c5")
    assert out is not None
    assert "attacks the knight on c3" in out
    assert "only defender of Black's bishop on a2" in out
    assert "threatening to win it" in out       # a THREAT, not a categorical 'cannot save both'


def test_remove_form():
    # Rxc3 (c5->c3) CAPTURES that same knight — the sole defender of the a2 bishop — which then falls.
    out = undermines_defender("1r3rk1/p2p2p1/3Np3/2R5/5p2/2n2P1P/b1P3PB/R5K1 w - - 1 3", "c5c3")
    assert out is not None
    assert "removes the only defender of Black's bishop on a2" in out
    assert "which now falls" in out             # a2 is also attacked (by Ra1) -> it drops
    assert "attacks" not in out                 # this is the REMOVE form, not the attack form


def test_secondary_note_when_the_bigger_capture_is_the_point():
    # Rxc5 (a5->c5) wins the ROOK; that rook was also the sole defender of the c4 knight. Winning the
    # rook is the point, so the undermining is a SECONDARY note (no 'The point:' marker) — not dropped
    # (gap 2), not mis-led-with.
    out = undermines_defender("4k3/8/8/R1r5/2n5/8/8/4K3 w - - 0 1", "a5c5")
    assert out is not None
    assert not out.startswith("The point"), "the bigger capture is the point, not the undermining"
    assert "also leaves Black's knight on c4 loose" in out


def test_none_when_no_sole_guard_motif():
    # A victim with TWO defenders is not 'undermined' by attacking one; a plain capture isn't the motif.
    assert undermines_defender("4k3/8/8/8/8/2n1n3/8/3RK3 w - - 0 1", "d1d3") is None
    assert undermines_defender("k7/2K5/1P6/8/7p/1rR4p/7P/8 w - - 0 7", "c3h3") is None   # mate puzzle
    assert undermines_defender(None, "a1a2") is None
