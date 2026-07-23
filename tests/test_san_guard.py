"""The SAN-on-the-wire guard (grounding.san_guard) — added 2026-07-24 after
the live leak "**c7c6** was just played". Every move-bearing prompt param
routes through it; a UCI-shaped token raises AT ASSEMBLY, so the violation
can never reach a beat again."""
import pytest

from lucena_backend.coaching.grounding import san_guard
from lucena_backend.coaching.mode_prompts import ReadPrompt


@pytest.mark.parametrize("ok", [
    None, "", "c6", "e4", "exd5", "Qxd5", "Nf3", "O-O", "O-O-O", "e8=Q+",
    "Qh4#", "Rad1", "Nbd2", "a1=N", "1. e4 d5 2. exd5 Qxd5",  # SAN history line
])
def test_san_passes(ok):
    assert san_guard(ok) == ok


@pytest.mark.parametrize("bad", [
    "c7c6", "e2e4", "a7a8q", "g1f3",
    "1. e4 d5 2. exd5 d8d5",          # one UCI token hiding in a history line
    "the move b1c3 develops",         # UCI inside prose
])
def test_uci_raises(bad):
    with pytest.raises(ValueError, match="UCI leaked"):
        san_guard(bad)


def test_read_prompt_rejects_uci_played():
    # the exact live leak, now impossible
    with pytest.raises(ValueError, match="c7c6"):
        ReadPrompt.prompt(facts="material is even", played="c7c6")


def test_read_prompt_accepts_san_played():
    assert "c6 was just played" in ReadPrompt.prompt(facts="f", played="c6")


def test_multi_value_form():
    with pytest.raises(ValueError):
        san_guard("c6", "e2e4")
    assert san_guard("c6", "Nf3") == ("c6", "Nf3")
