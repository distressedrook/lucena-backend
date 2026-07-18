"""The two-beat drill move: a correct mid-line move voices the verdict (what YOU did) AND the
opponent's auto-played reply (what THEY did). A wrong move — or a line-ending correct move that
leaves no reply — voices only the verdict. The reply move is surfaced by the walker through the
strategy effects; here we stub the strategy and the LLM so the test pins the beat WIRING, not prose.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from lucena_backend.coaching.coach import CoachHandler
from lucena_backend.coaching.lesson import Lesson, LessonSpec, LessonProgress, ACTIVE
from lucena_backend.coaching.bits import BitSpec, BitProgress
from lucena_backend.coaching.loop import Input


class _Store:
    def __init__(self, lesson):
        self._lesson = lesson
        self.beats = []

    def active_lesson(self):
        return self._lesson

    def save_lesson_progress(self, prog):
        pass

    def append_beats(self, beats):
        self.beats.extend(beats)

    def publish_status(self, text):
        pass

    # board effects land here — no-ops for the test
    def write_history(self, plies):
        pass

    def write_board(self, fen):
        pass


class _Strat:
    """Adjudicates every move correct, mid-line (not cleared), and hands back an opponent reply."""
    def __init__(self, reply):
        self._reply = reply

    async def adjudicate(self, inp, grounding, spec, prog):
        new_prog = replace(prog, attempts=prog.attempts + 1, cleared=False)
        return True, new_prog, {"board": None, "plies": None, "reply": self._reply}


def _lesson():
    spec = LessonSpec(id="p", type="puzzle", fen="8/8/8/8/8/8/8/8 w - - 0 1",
                      bits=[BitSpec(strategy="move_line", params={"tree": {}}, challenge="Find it.")])
    prog = LessonProgress(lesson_id="p", state=ACTIVE, chat_id="c1",
                          bits=[BitProgress(cleared=False, attempts=0)])
    return Lesson(spec, prog)


def _handler(store, reply):
    h = CoachHandler(ctx=object(), store=store, llm=object(), model="stub", ground=object())
    h._strategies = type("R", (), {"get": lambda self, k: _Strat(reply)})()
    # stub grounding + both LLM-voiced beats so we assert wiring, not model prose
    h._ground_for_bit = lambda bit: _async(None)
    h._verdict_text = lambda inp, g, correct, color: _async("VERDICT")
    h._reply_text = lambda reply, color: _async(f"REPLY:{reply['san']}")
    return h


def _async(v):
    async def _c():
        return v
    return _c()


REPLY = {"san": "cxd5", "uci": "c6d5", "from_fen": "1k5r/4q3/1pp5/3QNp2/6p1/P5P1/1P3P2/4RK2 b - - 0 1"}


def _texts(store):
    return ["".join(s["text"] for s in b["segments"]) for b in store.beats]


def test_correct_midline_move_pushes_verdict_then_reply():
    store = _Store(_lesson())
    h = _handler(store, REPLY)
    inp = Input(kind="move", uci="d8d5", san="Qxd5", fen="1k5r/4q3/1pp5/3bNp2/6p1/P5P1/1P3P2/3QRK2 w - - 0 1")
    asyncio.run(h.handle(inp))
    assert _texts(store) == ["VERDICT", "REPLY:cxd5"], "a correct mid-line move must voice BOTH beats, in order"


def test_no_reply_means_a_single_verdict_beat():
    store = _Store(_lesson())
    h = _handler(store, None)          # walker returned no reply (line ended / wrong move)
    inp = Input(kind="move", uci="d8d5", san="Qxd5", fen="1k5r/4q3/1pp5/3bNp2/6p1/P5P1/1P3P2/3QRK2 w - - 0 1")
    asyncio.run(h.handle(inp))
    assert _texts(store) == ["VERDICT"], "with no reply, only the verdict is voiced"
