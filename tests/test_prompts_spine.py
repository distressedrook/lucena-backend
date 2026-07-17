"""New-spine prompt STRUCTURE — the input we hand the model, never its prose.

We never assert on generated text ("did it write a good paragraph" is ungroundable). We assert on the
assembled system prompt: that the shared grounding-discipline rule is present where it must be, that
the perspective is right per mode, and that the spoil-safe framings (trap warn names no move, reveal
frames a human trap) are actually in the instruction. These are the wordings that drifted / hallucinated
in live testing, so they get a structural guard.
"""

from __future__ import annotations

from lucena_backend.coaching.mode_prompts import (
    FreeformPrompt, PositionQueryPrompt, ReadPrompt, VerdictPrompt, TrapPrompt,
)

# Stable substrings of the shared fragments (grounding.py). If these move, update here deliberately.
NO_INVENTION = "Do NOT name a tactical motif"
FREEFORM_VOICE = "there is NO 'you'"
COACH_VOICE = "address the player as 'you'"


# -- the shared no-invention rule reaches every grounded prompt -------------------------------------

def test_verdict_carries_no_invention_rule_both_ways():
    assert NO_INVENTION in VerdictPrompt.system(correct=True)
    assert NO_INVENTION in VerdictPrompt.system(correct=False)


def test_read_and_positionquery_carry_no_invention_rule():
    assert NO_INVENTION in ReadPrompt.system()
    assert NO_INVENTION in PositionQueryPrompt.system(freeform=True)
    assert NO_INVENTION in PositionQueryPrompt.system(freeform=False)


def test_trap_carries_no_invention_rule_both_ways():
    assert NO_INVENTION in TrapPrompt.system(reveal=True)
    assert NO_INVENTION in TrapPrompt.system(reveal=False)


# -- perspective split: freeform names colours, coach says 'you' ------------------------------------

def test_read_is_freeform_perspective():
    s = ReadPrompt.system()
    assert FREEFORM_VOICE in s and COACH_VOICE not in s


def test_verdict_is_coach_perspective():
    s = VerdictPrompt.system(correct=True)
    assert COACH_VOICE in s and FREEFORM_VOICE not in s


def test_positionquery_perspective_follows_the_flag():
    assert FREEFORM_VOICE in PositionQueryPrompt.system(freeform=True)
    assert COACH_VOICE in PositionQueryPrompt.system(freeform=False)


# -- trap spoil-safety + framing -------------------------------------------------------------------

def test_trap_warn_names_no_move():
    s = TrapPrompt.system(reveal=False)
    assert "WITHOUT naming the move" in s, "the warn must not leak the trap move"
    assert "must not guess" in s


def test_trap_reveal_frames_a_human_trap_not_the_best_move():
    s = TrapPrompt.system(reveal=True)
    assert "many players" in s and "NOT the engine's best" in s


# -- verdict role bodies ---------------------------------------------------------------------------

def test_verdict_right_names_the_idea_and_hides_the_continuation():
    s = VerdictPrompt.system(correct=True)
    assert "RIGHT move" in s
    assert "Do NOT" in s and "continue the line" in s, "the right verdict must not spoil the next move"


def test_verdict_wrong_explains_the_flaw_without_the_solution():
    s = VerdictPrompt.system(correct=False)
    assert "NOT the answer" in s and "WITHOUT naming the correct move" in s


# -- freeform classifier surface -------------------------------------------------------------------

def test_freeform_prompt_declares_its_four_intents():
    sys = FreeformPrompt.system()
    for intent in ("reject", "general", "needs_grounding", "dispatch_coach"):
        assert intent in sys, f"FreeformPrompt lost the {intent!r} intent"


def test_freeform_prompt_embeds_the_turn_text():
    p = FreeformPrompt.prompt(text="what is a fork?", convo=None)
    assert "what is a fork?" in p


def test_positionquery_grounds_only_on_the_given_facts():
    p = PositionQueryPrompt.prompt(text="why not Rd8?", facts="F1: rook is loose")
    assert "why not Rd8?" in p and "F1: rook is loose" in p
