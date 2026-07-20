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
        self._history = [{"n": 0, "san": None, "uci": None, "fen": "8/8/8/8/8/8/8/8 w - - 0 1"}]

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

    async def adjudicate(self, inp, grounding, spec, prog, history=None):
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
    h._verdict_text = lambda inp, g, correct, color, bit=None: _async("VERDICT")
    h._reply_text = lambda reply, color: _async(f"REPLY:{reply['san']}")
    return h


def _async(v):
    async def _c():
        return v
    return _c()


REPLY = {"san": "cxd5", "uci": "c6d5", "from_fen": "1k5r/4q3/1pp5/3QNp2/6p1/P5P1/1P3P2/4RK2 b - - 0 1"}


def _texts(store):
    return ["".join(s["text"] for s in b["segments"]) for b in store.beats]


FEN = "1k5r/4q3/1pp5/3bNp2/6p1/P5P1/1P3P2/3QRK2 w - - 0 1"


def test_correct_midline_move_echoes_you_then_reply_no_verdict():
    # v1: a RIGHT move gets NO auto verdict (the app adds a local praise). Only the move bubble (with the
    # ✓ badge) and the opponent's auto-played reply speak — the reply is game-progress, not interpretation.
    store = _Store(_lesson())
    h = _handler(store, REPLY)
    asyncio.run(h.handle(Input(kind="move", uci="d1d5", san="Qxd5", fen=FEN)))
    assert _texts(store) == ["Played Qxd5", "REPLY:cxd5"]     # NO "VERDICT", NO "— takes the bishop"
    you = store.beats[0]
    assert you["kind"] == "you" and you["correct"] is True and you["move"] == "Qxd5"


def test_no_reply_emits_only_the_move_bubble():
    # A right move with no reply (line ended) → only the ✓ bubble; the verdict is gone from the hot path.
    store = _Store(_lesson())
    h = _handler(store, None)
    asyncio.run(h.handle(Input(kind="move", uci="d1d5", san="Qxd5", fen=FEN)))
    assert _texts(store) == ["Played Qxd5"]


def test_explain_regenerates_the_wrong_move_verdict_on_demand():
    # v1 'Why?': the wrong-move verdict is no longer auto-shown — kind='explain' (the Why? click)
    # regenerates the SAME wrong-move explanation on demand, said as one beat, and emits NO move bubble.
    store = _Store(_lesson())
    h = _handler(store, None)
    asyncio.run(h.handle(Input(kind="explain", uci="d1d5", san="Qxd5", fen=FEN)))
    assert _texts(store) == ["VERDICT"]


def test_new_line_reply_announces_the_backtrack_deterministically():
    # A sibling-defence backtrack surfaces a reply flagged new_line; _reply_text must announce it
    # (no LLM — it's deterministic) so the board jump reads as a fresh challenge, not a glitch.
    from lucena_backend.coaching.coach import CoachHandler
    h = CoachHandler(ctx=object(), store=object(), llm=object(), model="stub", ground=object())
    reply = {"san": "Rxh3", "uci": "b3h3", "new_line": True,
             "from_fen": "k7/2K5/1P6/8/7p/1r5R/7P/8 b - - 0 7"}
    text = asyncio.run(h._reply_text(reply, "white"))
    assert "Your opponent tries 7... Rxh3" in text and "Find the win again" in text


def test_normalize_promotion_defaults_bare_uci_to_queen():
    # A bare 'b7b8' is read by the engine as b8=N (knight) — the promotion bug. Default it to queen;
    # leave suffixed / non-promotion moves untouched.
    from lucena_backend.coaching.coach import _normalize_promotion
    FEN = "8/kPK5/8/8/8/8/8/8 w - - 0 1"
    assert _normalize_promotion(FEN, "b7b8") == "b7b8q"      # queen, not knight
    assert _normalize_promotion(FEN, "b7b8q") == "b7b8q"     # already chosen — untouched
    assert _normalize_promotion(FEN, "b7b8n") == "b7b8n"     # a real underpromotion is kept
    assert _normalize_promotion("8/8/8/8/8/8/4K3/4k3 w - - 0 1", "e2e3") == "e2e3"  # not a promotion
