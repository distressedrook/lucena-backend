"""The Bit — one opaque, self-contained adjudicated task inside a Lesson (LLD §3, LLD-C).

A bit is ATOMIC to the Lesson but arbitrarily complex inside: a whole forcing-line
puzzle is ONE `move_line` bit whose `DrillState` walker handles every branch
internally; a bit clears when the walker finishes. Multiple bits exist only for
multi-modal lessons (e.g. explain-the-plan → find-the-move → why-does-it-win).

The content/progress split mirrors `drill.py`'s tree/walker split:
  - `BitSpec`     — authored OR derived-from-position, reusable, never per-session.
  - `BitProgress` — the only per-session runtime state; `strategy_state` is OPAQUE
                    (for a `move_line` bit it is literally `DrillState.to_state()`).
  - `Bit`         — a lightweight runtime VIEW pairing spec+progress+index, built on
                    demand by `Lesson.current_bit()`; never persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BitSpec:
    """Content for one bit — shared, reusable, position-keyed (lives in the library)."""
    strategy: str                       # "move_line" | "move_exact" | "free_text" | "move_set"
    params: dict = field(default_factory=dict)   # strategy-specific: {"tree": …} for move_line;
                                                 #   {"expect_san"/"expect_uci"} for move_exact;
                                                 #   {"rubric": …} for free_text
    challenge: str | None = None        # authored task text, or None → LLM generates at present-time
    grounding_req: list = field(default_factory=list)   # facts + visibility tier (§4)
    required: bool = True               # counts toward `Lesson.solved` (§2.1)
    concept_id: str | None = None       # optional per-bit mastery override (default: lesson-level)


@dataclass
class BitProgress:
    """Per-player runtime state for one bit — the only part that is persisted per session."""
    cleared: bool = False
    attempts: int = 0
    strategy_state: dict | None = None  # OPAQUE to the Lesson; DrillState.to_state() for move_line


@dataclass
class Bit:
    """A runtime VIEW over one bit — spec + progress + its index in the Lesson. Constructed on
    demand (`Lesson.current_bit()`), never stored. Handler ergonomics: `bit.spec`, `bit.progress`,
    `bit.index`."""
    spec: BitSpec
    progress: BitProgress
    index: int
