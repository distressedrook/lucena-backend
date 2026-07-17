"""The Lesson — the coach-mode item (LLD §2, LLD-C).

A generic, item-bound, durable unit (a puzzle is one `type` of Lesson). Coach mode is a
mostly-stateless interaction OVER durable Lesson records. Two orthogonal axes:
  - `state` (LIVE activity): active | suspended | open.
  - `meta`  (DURABLE verdict): None | "solved", STORED write-once — set exactly once at
    conclusion (with the mastery bank) and never mutated back. Stored, not derived, so it
    survives a re-attempt/reset of bit-progress and is cheap to query across lessons.

Content/progress split mirrors `bits.py`/`drill.py`: `LessonSpec` is shared library
content; `LessonProgress` is the per-player durable record; `Lesson` pairs them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .bits import Bit, BitProgress, BitSpec

# Live-activity states (the `state` axis).
ACTIVE = "active"
SUSPENDED = "suspended"
OPEN = "open"
_STATES = frozenset({ACTIVE, SUSPENDED, OPEN})

# Durable verdicts (the `meta` axis). None = unfinished.
SOLVED = "solved"


@dataclass
class LessonSpec:
    """Content — shared, reusable, position-keyed (lives in the library)."""
    id: str
    type: str                           # puzzle | endgame | midgame | opening
    fen: str                            # starting position
    bits: list[BitSpec]
    motif: list[str] = field(default_factory=lambda: ["user_generated"])
    concept_id: str | None = None       # session-level mastery concept (grain: lesson-level default)


@dataclass
class LessonProgress:
    """Per-player, DURABLE record — the new Lesson-progress store (§7)."""
    lesson_id: str
    state: str                          # active | suspended | open
    bits: list[BitProgress]
    meta: str | None = None             # STORED write-once verdict: None | "solved"
    currentBit: int = 0
    # The chat this lesson is ACTIVE in. "Active" is per-CHAT, not per-user: a puzzle being solved in
    # one chat must NOT force coach mode in another. resolve_mode matches state==active AND meta is
    # None AND chat_id == the current chat. None when not chat-bound (open/suspended/solved).
    chat_id: str | None = None


@dataclass
class Lesson:
    """Runtime pairing of shared content + per-player progress."""
    spec: LessonSpec
    progress: LessonProgress

    # -- verdict ---------------------------------------------------------------
    @property
    def solved(self) -> bool:
        """The STORED verdict — read, never recomputed. Durable across re-attempts/resets."""
        return self.progress.meta == SOLVED

    def _all_required_cleared(self) -> bool:
        """The TRIGGER check that FIRES the conclusion write (distinct from `solved`, which reads
        the stored result). True when every `required` bit's progress is cleared."""
        return all(bp.cleared
                   for bs, bp in zip(self.spec.bits, self.progress.bits) if bs.required)

    def mark_solved(self) -> None:
        """Write the durable verdict ONCE (at conclusion, alongside the mastery bank). Idempotent —
        a second call is a no-op, so the write-once invariant holds even if reached twice."""
        if self.progress.meta is None:
            self.progress.meta = SOLVED

    # -- bit navigation --------------------------------------------------------
    def current_bit(self) -> Bit:
        """The live bit as a VIEW (spec+progress+index) at `currentBit`."""
        i = self.progress.currentBit
        return Bit(spec=self.spec.bits[i], progress=self.progress.bits[i], index=i)

    def set_bit_progress(self, index: int, prog: BitProgress) -> None:
        self.progress.bits[index] = prog

    def advance_currentBit(self) -> bool:
        """Move to the next bit. Returns False when there is no next bit (the last one just cleared)."""
        if self.progress.currentBit + 1 < len(self.spec.bits):
            self.progress.currentBit += 1
            return True
        return False


def puzzle_lesson_id(fen: str) -> str:
    """Position-keyed id, so re-pasting the same puzzle reuses its computed spec (cache hit)."""
    return "pos_" + hashlib.md5(fen.encode("utf-8")).hexdigest()[:12]


def puzzle_spec(fen: str, tree: dict, motif: list[str] | None = None) -> LessonSpec:
    """Wrap a computed forcing-line `tree` as a one-bit puzzle LessonSpec (calculation, not authoring).
    Shared by freeform's paste-time cache and coach's create-on-solve so they agree on the id/shape."""
    return LessonSpec(
        id=puzzle_lesson_id(fen), type="puzzle", fen=fen,
        motif=(motif or ["user_generated"]),
        bits=[BitSpec(strategy="move_line", params={"tree": tree},
                      challenge="Find the winning continuation.")],
    )
