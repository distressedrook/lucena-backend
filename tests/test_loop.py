"""ConversationLoop — routing, mode resolution, the loop-mediated seam, and the turn-boundary status.

Pure control-flow: the handlers are fakes that record what they were handed and return a chosen
Outcome, so this tests the LOOP's decisions (which handler, which transition, re-route once, status
lifecycle) without any LLM, engine, or DB. The seam is the whole point of the redesign — handlers
never call each other; the loop reads their Outcome and re-routes.

No pytest-asyncio in this project, so each test drives the coroutine with asyncio.run (as the rest of
the suite does).
"""

from __future__ import annotations

import asyncio
import contextlib

from lucena_backend.coaching.loop import (
    ConversationLoop, Input, Mode, Handled, Suspend, Open, EnterCoach,
)
from lucena_backend.coaching import lesson as _lesson


class FakeStore:
    def __init__(self):
        self.active = None                 # the "active lesson" resolve_mode keys on
        self.suspended = None              # a parked drill a move can resume
        self.status_calls = []             # every publish_status(text)
        self.state_changes = []            # every set_lesson_state(id, state)

    @contextlib.contextmanager
    def bound(self, sid):
        yield

    def publish_status(self, text):
        self.status_calls.append(text)

    def active_lesson(self):
        return self.active

    def suspended_lesson(self):
        return self.suspended

    def set_lesson_state(self, lesson_id, state):
        self.state_changes.append((lesson_id, state))
        if state == "active" and self.suspended is not None:   # mimic resume: suspended → active
            self.active, self.suspended = self.suspended, None


class _Lesson:
    class spec:  # noqa: N801 - a stand-in exposing .spec.id
        id = "L1"


class FakeHandler:
    """Records the inputs it saw and returns a scripted sequence of Outcomes."""
    def __init__(self, outcomes=None):
        self.seen = []
        self._outcomes = list(outcomes or [Handled()])

    async def handle(self, inp):
        self.seen.append(inp)
        return self._outcomes.pop(0) if self._outcomes else Handled()


class FakeCoach(FakeHandler):
    def __init__(self, outcomes=None, enter_ok=True, store=None):
        super().__init__(outcomes)
        self.enter_ok = enter_ok
        self.store = store
        self.entered = []

    async def enter(self, outcome):
        self.entered.append(outcome)
        if self.enter_ok and self.store is not None:
            self.store.active = _Lesson()      # entering activates a lesson → next resolve = COACH
        return self.enter_ok


def _run(store, freeform, coach, inp):
    asyncio.run(ConversationLoop(store=store, freeform=freeform, coach=coach).handle_input("c1", inp))


def test_no_lesson_routes_to_freeform():
    s = FakeStore(); ff = FakeHandler(); co = FakeCoach(store=s)
    _run(s, ff, co, Input(kind="text", text="hi"))
    assert len(ff.seen) == 1 and len(co.seen) == 0


def test_a_move_resumes_a_suspended_drill_and_routes_to_coach():
    # A what-if suspended the drill; the player then plays a real move. It must resume the drill and be
    # adjudicated by coach — not fall through to freeform (the "broken state": the solution was narrated
    # instead of scored).
    s = FakeStore(); s.suspended = _Lesson()          # parked drill, nothing active
    ff = FakeHandler(); co = FakeCoach(store=s)
    _run(s, ff, co, Input(kind="move", uci="d1d5"))
    assert ("L1", "active") in s.state_changes, "the move must resume the suspended drill"
    assert len(co.seen) == 1 and len(ff.seen) == 0, "the move is adjudicated by coach, not freeform"


def test_a_text_turn_does_not_resume_a_suspended_drill():
    # Only a move signals 'back to solving'; a further text turn stays in the freeform excursion.
    s = FakeStore(); s.suspended = _Lesson()
    ff = FakeHandler(); co = FakeCoach(store=s)
    _run(s, ff, co, Input(kind="text", text="tell me more"))
    assert s.state_changes == [], "a text turn must not resume the drill"
    assert len(ff.seen) == 1 and len(co.seen) == 0


def test_active_lesson_routes_to_coach():
    s = FakeStore(); s.active = _Lesson()
    ff = FakeHandler(); co = FakeCoach(store=s)
    _run(s, ff, co, Input(kind="move", uci="e2e4"))
    assert len(co.seen) == 1 and len(ff.seen) == 0


def test_handled_does_not_reroute():
    s = FakeStore(); ff = FakeHandler([Handled()]); co = FakeCoach(store=s)
    _run(s, ff, co, Input(kind="text", text="hi"))
    assert len(ff.seen) == 1 and len(co.seen) == 0


def test_entercoach_success_reroutes_present_into_coach():
    s = FakeStore()
    ff = FakeHandler([EnterCoach(type="puzzle", source={"kind": "current"})])
    co = FakeCoach(store=s, enter_ok=True)
    _run(s, ff, co, Input(kind="position", fen="…"))
    assert len(co.entered) == 1, "coach.enter must be called for EnterCoach"
    assert len(co.seen) == 1 and co.seen[0].kind == "present", "loop must re-route a synthetic 'present'"


def test_entercoach_failure_stays_in_freeform():
    s = FakeStore()
    ff = FakeHandler([EnterCoach(type="puzzle", source={"kind": "current"})])
    co = FakeCoach(store=s, enter_ok=False)
    _run(s, ff, co, Input(kind="position", fen="…"))
    assert co.entered and len(co.seen) == 0, "a failed enter must not re-route into coach"


def test_suspend_marks_state_and_reroutes_then_to_freeform():
    s = FakeStore(); s.active = _Lesson()
    then = Input(kind="text", text="what if Nf3")
    co = FakeCoach([Suspend(then=then)], store=s)
    ff = FakeHandler()
    # Suspend clears the active binding so the re-route lands in freeform (as set_lesson_state does live).
    def _suspend_clears(lesson_id, state):
        s.state_changes.append((lesson_id, state))
        if state == _lesson.SUSPENDED:
            s.active = None
    s.set_lesson_state = _suspend_clears
    _run(s, ff, co, Input(kind="text", text="what if Nf3"))
    assert ("L1", _lesson.SUSPENDED) in s.state_changes
    assert len(ff.seen) == 1 and ff.seen[0] is then, "the what-if must re-route into freeform"


def test_open_marks_state_and_does_not_reroute():
    s = FakeStore(); s.active = _Lesson()
    co = FakeCoach([Open()], store=s); ff = FakeHandler()
    _run(s, ff, co, Input(kind="text", text="stop"))
    assert ("L1", _lesson.OPEN) in s.state_changes
    assert len(ff.seen) == 0, "Open ends the turn; no re-route"


def test_reroute_happens_at_most_once():
    # freeform → EnterCoach → coach re-route; even if coach returns a further non-Handled outcome,
    # the loop must NOT re-route a second time (rerouted=True short-circuits).
    s = FakeStore()
    ff = FakeHandler([EnterCoach(type="puzzle", source={"kind": "current"})])
    co = FakeCoach([Open()], store=s, enter_ok=True)   # would transition again if re-routed
    _run(s, ff, co, Input(kind="position", fen="…"))
    assert len(co.seen) == 1, "the loop re-routed more than once"
    assert ("L1", _lesson.OPEN) not in s.state_changes, "a second transition was applied"


def test_status_is_one_thinking_then_one_clear_across_a_reroute():
    # The 'Uh oh' fix: status is owned at the turn boundary — exactly one 'Thinking…' and one None,
    # even though the turn spans a freeform routing hand-off AND a coach present.
    s = FakeStore()
    ff = FakeHandler([EnterCoach(type="puzzle", source={"kind": "current"})])
    co = FakeCoach(store=s, enter_ok=True)
    _run(s, ff, co, Input(kind="position", fen="…"))
    assert s.status_calls == ["Thinking…", None], f"status toggled per-handler: {s.status_calls}"


def test_status_cleared_even_when_a_handler_raises():
    s = FakeStore()
    class Boom(FakeHandler):
        async def handle(self, inp):
            raise RuntimeError("kaboom")
    raised = False
    try:
        _run(s, Boom(), FakeCoach(store=s), Input(kind="text", text="x"))
    except RuntimeError:
        raised = True
    assert raised, "the handler error must propagate"
    assert s.status_calls == ["Thinking…", None], "status must be cleared in finally on error"


def test_walk_routes_to_freeform_and_never_to_coach():
    # Walking a variation is a freeform read of a sideline move — never a drill answer. The loop must
    # hand it to freeform even when a lesson is active (upstream already gates it silent during a drill;
    # if a walk reaches the loop, it is always a freeform read, never adjudicated by coach).
    s = FakeStore(); s.active = _Lesson()
    ff = FakeHandler(); co = FakeCoach(store=s)
    _run(s, ff, co, Input(kind="walk", fen="…", san="Nf3"))
    assert len(ff.seen) == 1 and ff.seen[0].kind == "walk", "a walk must go to freeform"
    assert len(co.seen) == 0, "a walk must never reach coach"


def test_walk_does_not_reroute():
    # Even if freeform returned a transition Outcome, a walk is terminal — the loop must not re-route it.
    s = FakeStore()
    ff = FakeHandler([EnterCoach(type="puzzle", source={"kind": "current"})])
    co = FakeCoach(store=s, enter_ok=True)
    _run(s, ff, co, Input(kind="walk", fen="…", san="Nf3"))
    assert len(co.entered) == 0 and len(co.seen) == 0, "a walk must not trigger a re-route into coach"


def test_resolve_mode_is_pure_state():
    s = FakeStore()
    loop = ConversationLoop(store=s, freeform=FakeHandler(), coach=FakeCoach(store=s))
    assert loop._resolve_mode() is Mode.FREEFORM
    s.active = _Lesson()
    assert loop._resolve_mode() is Mode.COACH
