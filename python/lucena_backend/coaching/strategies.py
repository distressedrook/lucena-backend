"""Adjudication strategies (LLD §3, LLD-C) — the Strategy pattern for scoring a bit.

Contract is bool-only and uniform across every strategy:

    async adjudicate(inp, grounding, spec, prog) -> (was_this_input_right, new_progress)

Two meanings carried separately, NEVER conflated:
  - the returned **bool**            = "was THIS input right" → drives the feedback / why-wrong beat.
  - **new_progress.cleared**         = "is the WHOLE bit done" → drives `Lesson.solved`.

Deterministic strategies (`move_line`, `move_exact`) are trivially async. `free_text` is
LLM-graded via an injected `grade_fn` (kept out of this module so the strategies stay
unit-testable with a stub, and so this file has no LLM/prompt dependency).

Grounded-or-guarded: the deterministic verdict is CALCULATED here; the LLM (free_text)
only interprets against a grounded rubric, never sets truth on its own.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Awaitable, Callable, Protocol

from ..grounding_tools.drill import DrillState, _same_move
from .bits import BitProgress, BitSpec


class Strategy(Protocol):
    async def adjudicate(self, inp, grounding, spec: BitSpec, prog: BitProgress,
                         history: list | None = None) -> tuple[bool, BitProgress, dict | None]: ...
    # Returns (was_this_input_right, new_progress, effects). `effects` carries board changes for a
    # move strategy — {"board": fen_after_reply, "plies": move_line} — so the handler updates the
    # board (incl. the opponent's auto-played reply); None for non-board strategies (free_text).


def _move_of(inp) -> tuple[str | None, str | None]:
    """(uci, san) from a move Input — san optional (the app sends uci; the handler may attach san)."""
    return getattr(inp, "uci", None), getattr(inp, "san", None)


class MoveLineStrategy:
    """A whole forcing-line tree, walked by `DrillState`. ONE bit wraps the ENTIRE tree — every
    branch (opponent defenses, nested re-branches) is handled inside the walker; the bit clears when
    `walker.finished`. Deterministic. `strategy_state` is `DrillState.to_state()`; the board move
    line has its ONE home in the session history, so it is NOT carried here (matches drill.py)."""

    async def adjudicate(self, inp, grounding, spec, prog, history=None):
        tree = spec.params["tree"]
        # `line=history` is REQUIRED, not optional: the walker's move line is not in `to_state()` (its
        # ONE home is the session history), so a restore without it defaults the line to just [root].
        # Missing it corrupted the line on the NEXT move — a backtrack truncated the wrong prefix and
        # the recorded history came out scrambled (Rxc3 landed at ply 1, the first move lost).
        walker = (DrillState.restore(tree, prog.strategy_state, line=history)
                  if prog.strategy_state else DrillState(tree))
        uci, san = _move_of(inp)
        result = walker.play(uci or "", san)
        new_prog = replace(prog, attempts=prog.attempts + 1,
                           strategy_state=walker.to_state(), cleared=walker.finished)
        # Board effects: the position AFTER the player's move + the walker's auto-played opponent
        # reply, plus the full move line — so the handler moves the board (and shows the reply).
        effects = {"board": result.get("board"), "plies": result.get("plies"),
                   "reply": result.get("reply"),   # the opponent's auto-played reply → its own beat
                   "await_continue": result.get("await_continue")}   # branch solved, sibling held for Continue
        return bool(result.get("correct")), new_prog, effects

    async def continue_branch(self, spec, prog, history=None):
        """Walk the next sibling defence — the DEFERRED backtrack, run when the player clicks Continue.
        Restores the walker, pops the held sibling, and returns (new_progress, effects) to paint the
        board. `effects` is None when nothing was pending."""
        walker = DrillState.restore(spec.params["tree"], prog.strategy_state, line=history)
        result = walker.continue_branch()
        new_prog = replace(prog, strategy_state=walker.to_state(), cleared=walker.finished)
        if result is None:
            return new_prog, None
        effects = {"board": result.get("board"), "plies": result.get("plies"),
                   "reply": result.get("reply")}
        return new_prog, effects


class MoveExactStrategy:
    """A single expected move (a degenerate `move_line`). Matches on SAN first, uci fallback — the
    same canonical rule as `DrillState`. Single-shot: `cleared = correct`."""

    async def adjudicate(self, inp, grounding, spec, prog, history=None):
        uci, san = _move_of(inp)
        correct = _same_move(san, uci, spec.params.get("expect_san"), spec.params.get("expect_uci"))
        new_prog = replace(prog, attempts=prog.attempts + 1, cleared=correct)
        return correct, new_prog, None


# grade_fn(text, spec, grounding) -> (correct, extras) — injected so this file needs no LLM/prompt dep.
GradeFn = Callable[[str, BitSpec, object], Awaitable[tuple[bool, dict]]]


class FreeTextStrategy:
    """A plain-English answer, LLM-graded against the bit's rubric + grounding (same discipline as
    today's `GradePrompt`). Single-shot: `cleared = correct`. The grading call is injected."""

    def __init__(self, grade_fn: GradeFn):
        self._grade = grade_fn

    async def adjudicate(self, inp, grounding, spec, prog, history=None):
        text = getattr(inp, "text", "") or ""
        correct, _extras = await self._grade(text, spec, grounding)
        new_prog = replace(prog, attempts=prog.attempts + 1, cleared=correct)
        return correct, new_prog, None


def build_registry(grade_fn: GradeFn | None = None) -> dict[str, Strategy]:
    """The strategy registry the coach handler dispatches through. `grade_fn` is required only if a
    `free_text` bit is actually adjudicated; a `move_set` (future) is registered when built."""
    reg: dict[str, Strategy] = {
        "move_line": MoveLineStrategy(),
        "move_exact": MoveExactStrategy(),
    }
    if grade_fn is not None:
        reg["free_text"] = FreeTextStrategy(grade_fn)
    return reg
