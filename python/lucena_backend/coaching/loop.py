"""The conversation loop — the single entry point and mode router (LLD §1, LLD-A).

ONE `handle_input(session, input)` replaces today's transport-split (`turn`→run_turn vs
`move`→coach_move) — that split was the root of the MovePrompt/GradePrompt/CoachPrompt
drift. `resolve_mode` is pure state (no LLM): a Lesson that is `active AND meta is None`
→ coach, else freeform.

Seam is LOOP-MEDIATED: handlers never call each other. Each returns a typed `Outcome`;
the loop reads it and re-routes within the same turn (at most once — no ping-pong). Every
mode transition lives HERE, the single place lesson `state` mutates for a transition.

Output is two channels: beats are published to the store by the handlers (side effect,
streamed to the app); the `Outcome` is returned to this loop (control flow only).
`handle_input` returns None — fire-and-forget over the WS, as `run_turn` does today.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from . import lesson as _lesson


class Mode(Enum):
    FREEFORM = auto()
    COACH = auto()


@dataclass
class Input:
    """A unified turn input. `kind` decides interpretation; the mode decides meaning (a `move` is
    explain-in-freeform OR adjudicate-in-coach). httpserver builds this from the WS message and does
    the FEN-detection, so `kind == "position"` is already a known-real position."""
    kind: str                       # "text" | "move" | "position"
    text: str | None = None
    uci: str | None = None
    san: str | None = None          # attached server-side for move adjudication (app sends uci)
    fen: str | None = None


# -- handler outcomes (the loop-mediated seam) -----------------------------------------------
class Outcome:
    """Base — a handler's control-flow result (never player-visible; beats went to the store)."""


class Handled(Outcome):
    """Turn done; nothing to re-route."""


@dataclass
class Suspend(Outcome):
    """coach → freeform: a what-if mid-bit. Re-run `then` in freeform (which explores + nudges back)."""
    then: Input


class Open(Outcome):
    """coach → freeform: an explicit 'stop'. No re-route; the lesson is marked open."""


@dataclass
class EnterCoach(Outcome):
    """freeform → coach: create/resume a Lesson and enter coach mode. `source` says how to obtain the
    Lesson (current-position | new(theme) | resume(open item)); `then` is usually None → the coach
    entry action (present the first bit) runs."""
    type: str
    motif: list[str] | None = None
    source: dict | None = None
    then: Input | None = None


class ConversationLoop:
    """Holds shared deps + the two handlers, routes every turn, and owns transitions.

    The store must provide:
      - `active_lesson() -> Lesson | None`  (the lesson with state==active AND meta is None)
      - `set_lesson_state(lesson_id, state)` (persist a transition)
      - (lesson creation/resume is consumed by `_apply(EnterCoach)` via the store/library)
    """

    def __init__(self, *, store, freeform, coach):
        self.store = store
        self.freeform = freeform
        self.coach = coach

    async def handle_input(self, session_id: str, inp: Input) -> None:
        with self.store.bound(session_id):
            # The working status is owned HERE, at the turn boundary — one "Thinking…" for the whole
            # turn, cleared once when it fully resolves (after any re-route). Per-handler toggling
            # cleared it between a routing hand-off and the handler that actually speaks, so the app saw
            # working→idle with zero beats and flashed its "Uh oh, I didn't catch that" safety net — a
            # routing call is not expected to produce a beat; the turn as a whole is.
            self.store.publish_status("Thinking…")
            try:
                await self._route(inp)
            finally:
                self.store.publish_status(None)

    async def _route(self, inp: Input, *, rerouted: bool = False) -> None:
        handler = self.coach if self._resolve_mode() is Mode.COACH else self.freeform
        outcome = await handler.handle(inp)
        if rerouted or isinstance(outcome, Handled):
            return
        nxt = await self._apply(outcome)
        if nxt is not None:
            await self._route(nxt, rerouted=True)          # re-route AT MOST once

    def _resolve_mode(self) -> Mode:
        live = self.store.active_lesson()
        return Mode.COACH if live is not None else Mode.FREEFORM

    async def _apply(self, outcome: Outcome) -> Input | None:
        """Apply a transition (the ONLY place lesson state mutates for a mode change). Returns the
        Input to re-route into the now-current mode, or None to end the turn. Async because lesson
        CREATION (EnterCoach) does engine work (build the forcing-line tree) off-thread."""
        live = self.store.active_lesson()
        if isinstance(outcome, Suspend):
            if live is not None:
                self.store.set_lesson_state(live.spec.id, _lesson.SUSPENDED)
            return outcome.then                             # → freeform explores the what-if
        if isinstance(outcome, Open):
            if live is not None:
                self.store.set_lesson_state(live.spec.id, _lesson.OPEN)
            return None                                     # freeform is home next turn; no re-route
        if isinstance(outcome, EnterCoach):
            if not await self.coach.enter(outcome):         # create/resume + ACTIVATE; no beats/LLM
                return None                                 # not drillable / failed → stay in freeform
            # Re-route into coach with a synthetic `present` so the entry action (present the first
            # bit) runs INSIDE the awaited flow — never a fire-and-forget task.
            return outcome.then or Input(kind="present")
        return None
