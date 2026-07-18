"""Presenting a drill resets its walker to the root — the stale-mid-line bug.

A left-unfinished multi-ply drill stores a mid-line walker state (`strategy_state`). Re-presented, the
board shows the root but the walker was deeper, so the student's correct FIRST move is adjudicated
against the wrong node and read as a blunder — no "Solved", no Retry (it wasn't scored wrong, just
confusingly). Presenting must reset the walker so board and walker agree at the top.
"""

from __future__ import annotations

import asyncio

from lucena_backend.coaching.coach import CoachHandler
from lucena_backend.coaching.lesson import Lesson, LessonSpec, LessonProgress, ACTIVE
from lucena_backend.coaching.bits import BitSpec, BitProgress
from lucena_backend.coaching.loop import Input


class _Store:
    """Minimal store for the present path — records saves and beats; empty tree ⇒ no board writes."""
    def __init__(self, lesson=None):
        self._lesson = lesson
        self.saved = []
        self.beats = []

    def active_lesson(self):
        return self._lesson

    def save_lesson_progress(self, prog):
        self.saved.append(prog)

    def append_beats(self, beats):
        self.beats.extend(beats)

    def publish_status(self, text):
        pass


def _lesson(strategy_state):
    spec = LessonSpec(id="p", type="puzzle", fen="8/8/8/8/8/8/8/8 w - - 0 1",
                      bits=[BitSpec(strategy="move_line", params={"tree": {}}, challenge="Find it.")])
    prog = LessonProgress(lesson_id="p", state=ACTIVE, chat_id="c1",
                          bits=[BitProgress(cleared=False, attempts=1, strategy_state=strategy_state)])
    return Lesson(spec, prog)


def _handler(store):
    return CoachHandler(ctx=object(), store=store, llm=object(), model="stub", ground=object())


# a walker mid-line, exactly as a prior unfinished attempt leaves it
STALE = {"current": ["after", ["def", 1]], "solved": 1, "finished": False,
         "stack": [], "counters": {"correct": 1, "wrong": 0, "backtrack": 0}}


def test_reset_clears_a_stale_mid_line_walker():
    store = _Store()
    lesson = _lesson(STALE)
    assert lesson.current_bit().progress.strategy_state, "precondition: the walker is mid-line"
    _handler(store)._reset_drill(lesson, lesson.current_bit())
    assert lesson.current_bit().progress.strategy_state is None, "walker must reset to the root"
    assert lesson.current_bit().progress.attempts == 0, "a fresh attempt starts clean"
    assert store.saved, "the reset must be persisted"


def test_reset_leaves_a_fresh_walker_alone():
    store = _Store()
    lesson = _lesson(None)
    _handler(store)._reset_drill(lesson, lesson.current_bit())
    assert lesson.current_bit().progress.strategy_state is None
    assert not store.saved, "a fresh drill needs no reset and no wasted write"


def test_handle_present_resets_the_walker_end_to_end():
    lesson = _lesson(STALE)
    store = _Store(lesson)
    asyncio.run(_handler(store).handle(Input(kind="present")))
    assert lesson.current_bit().progress.strategy_state is None, "presenting must start the drill fresh"
    assert store.beats, "the challenge should still be delivered"
