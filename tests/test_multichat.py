"""Phase 1 oracle — many chat sessions in ONE process must not leak into each other.

These tests MUST FAIL on main. They encode the two SEPARATE defects the multi-chat refactor fixes;
if either passes before the refactor, the model of that bug is wrong.

  test_two_chats_do_not_leak  -> defect 1, the PUBLISH side.
      `StateStore._subscribers` (state.py:226) is a FLAT set of queues with no session tagging, so
      `_publish` fans every mutation out to every connected WS client. Asserted at the SOCKET, not
      the store: a write-side fix alone must not make this pass.

  test_late_read_race         -> defect 2, the WRITE side.
      `StateStore._current` (state.py:236) is a process-global cursor. `Orchestrator.run_turn`
      (orchestrator.py:182) and `QuickCoach.explain` (quick.py:38) accept a `session_id` and never
      use it; turn/explain run as BACKGROUND tasks (`_SLOW`, httpserver.py:133) and re-resolve the
      cursor LATE, at write time. The turn must START in chat A (via the existing POST /session, the
      only cursor API main has) and only then have the cursor moved under it — otherwise the test
      passes/fails for the wrong reason and is a false oracle.

Vocabulary (do not conflate — see CLAUDE.md):
  chat session  = a coaching conversation; MANY per user; the `session` table + `_live[sid]`.
  active chat   = which chat a user currently has open; ONE per user; `_current` + `is_active`.
  login session = an authenticated user; does not exist yet (Phase 3).
"""

import asyncio
import os
import queue
import shutil
import threading
import time

import pytest

from lucena_backend.llm import Completion, Usage
from lucena_backend.httpserver import build_app

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# build_app unconditionally constructs a real local Engine() for ground_ctx (httpserver.py), so
# these need a Stockfish binary even though the LLM is stubbed.
_have_sf = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_sf, reason="no stockfish")

CHAT_A = "chat-aaaa"
CHAT_B = "chat-bbbb"
A_SAYS = "AAA-only-for-chat-a-AAA"


class _StubLLM:
    """Deterministic 'tell' verdict carrying a distinctive marker, so we can prove WHICH chat it
    landed in.

    `entered` fires the moment the turn reaches the LLM call; `gate` parks it there. Together they
    give a deterministic interleaving with no sleeps: the caller can know the turn is in-flight and
    bound to its starting chat before it perturbs anything.
    """

    def __init__(self, text=A_SAYS, gate: "threading.Event | None" = None,
                 entered: "threading.Event | None" = None):
        self._text = text
        self._gate = gate
        self._entered = entered

    async def generate(self, messages, opts):
        import json
        if self._entered is not None:
            self._entered.set()
        if self._gate is not None:
            # Park the turn mid-flight. A threading.Event awaited off-loop — NOT asyncio.Event:
            # TestClient drives the app on its own loop in another thread, and asyncio.Event.set()
            # across threads will not reliably wake the waiter (it would hang).
            await asyncio.to_thread(self._gate.wait)
        payload = {"mode": "tell", "text": self._text}
        return Completion(text=json.dumps(payload), json=payload, model="stub",
                          usage=Usage(1, 1, 2))


def _texts(live) -> str:
    """Every segment of text banked in one chat's _Live, flattened."""
    out = []
    for beat in getattr(live, "beats", []) or []:
        for seg in beat.get("segments", []) or []:
            out.append(seg.get("text", ""))
    return " | ".join(out)


def _drain_ready(ws) -> None:
    while ws.receive_json()["type"] != "ready":
        pass


def _open_chat(ws, sid: str) -> None:
    """Bind THIS socket to a chat. Phase 1 adds this message; on main it falls through
    `_dispatch`'s if/elif chain silently."""
    ws.send_json({"type": "open_chat", "session_id": sid})


def _wait_for_marker(store, marker: str, timeout: float = 90.0) -> "str | None":
    """Poll every chat's document until the coaching beat is banked ANYWHERE; return the chat id it
    landed in (or None on timeout).

    Deliberately NOT a blocking `ws.receive_json()` loop: if the turn errors, the fallback beat
    carries no marker and a socket read would block forever.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        for sid, live in list(store._live.items()):
            if marker in _texts(live):
                return sid
        time.sleep(0.25)
    return None


class _SocketTap:
    """Records everything a socket receives, from the moment it is attached.

    Attached BEFORE the event under test is triggered, so nothing can be missed: the question is
    only whether a message is ever delivered, never whether we started listening in time.

    Bounded by a REAL deadline (a worker thread + queue), not a loop around a blocking receive:
    `ws.receive_json()` blocks forever precisely when the leak is FIXED, since a correctly isolated
    socket is sent nothing at all. There is also no self-echo to bound on — `position` does not
    publish (`set_board_view` -> `_report_board` -> `_persist_view`, no `_publish`). The pump is a
    daemon; closing the socket ends it.
    """

    def __init__(self, ws):
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._ws = ws
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            while True:
                self._q.put(self._ws.receive_json())
        except Exception:          # socket closed / test finished
            pass

    def types_until(self, marker: str, within: float) -> list:
        """The `type` of every message received, up to and including the one carrying `marker`.

        Returns [] if the marker never arrives. Lets a test assert ORDER, not just delivery.
        """
        seen: list = []
        deadline = time.time() + within
        while time.time() < deadline:
            try:
                m = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            seen.append(m.get("type"))
            for beat in (m.get("appended") or m.get("beats") or []):
                for seg in beat.get("segments", []) or []:
                    if marker in (seg.get("text") or ""):
                        return seen
        return []

    def saw_marker(self, marker: str, within: float) -> bool:
        """True as soon as `marker` shows up in anything this socket received."""
        deadline = time.time() + within
        while time.time() < deadline:
            try:
                m = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            for beat in (m.get("appended") or m.get("beats") or []):
                for seg in beat.get("segments", []) or []:
                    if marker in (seg.get("text") or ""):
                        return True
        return False


@requires_engine
def test_two_chats_do_not_leak(tmp_path):
    """Defect 1 (publish side): socket B must not RECEIVE chat A's coaching beat.

    FAILS ON MAIN: `_subscribers` is a flat set, so `_publish` delivers A's beat to every connected
    socket, B included.
    """
    from fastapi.testclient import TestClient

    app = build_app(home=str(tmp_path), llm=_StubLLM(), model="stub")
    client = TestClient(app)
    store = app.state.ctx.store

    # Materialise both chats; leave the cursor on A so main's single-cursor path runs A's turn in A.
    # This test is about the FAN-OUT, so the write must land correctly even on main.
    client.post("/session", json={"session_id": CHAT_B})
    client.post("/session", json={"session_id": CHAT_A})

    with client.websocket_connect("/ws") as ws_a, client.websocket_connect("/ws") as ws_b:
        _drain_ready(ws_a)
        _drain_ready(ws_b)
        _open_chat(ws_a, CHAT_A)
        _open_chat(ws_b, CHAT_B)

        # Tap B BEFORE A's turn runs, so a leaked publish cannot be missed by starting late.
        tap_b = _SocketTap(ws_b)

        ws_a.send_json({"type": "position", "fen": STARTPOS})
        ws_a.send_json({"type": "turn", "text": "what should I think about here?"})

        # Barrier on the STORE, not on A's socket. A's socket is not a usable barrier here: the
        # store keeps ONE `_loop` (state.py:227), set by whichever subscriber called subscribe()
        # last. TestClient gives each websocket_connect its own portal + loop, so `_publish`'s
        # call_soon_threadsafe schedules every queue put on B's loop and A never wakes. (Benign in
        # production — one uvicorn loop — but it makes A's socket unobservable under this harness.)
        assert _wait_for_marker(store, A_SAYS) == CHAT_A, \
            "chat A's turn never produced its coaching beat in chat A"

        # The beat is banked; its _publish is the very next statement in append_beats(), and B's tap
        # has been recording since before the turn started. If the fan-out leaks, it lands here.
        leaked = tap_b.saw_marker(A_SAYS, within=10)

    assert not leaked, (
        "LEAK: chat B's socket received chat A's coaching beat — _publish fanned it out to every "
        "subscriber instead of only chat A's"
    )


@requires_engine
def test_late_read_race(tmp_path):
    """Defect 2 (write side): a turn parked in the LLM must write to the chat it STARTED in, even if
    the global cursor moves under it.

    The turn is started with the cursor on chat A (POST /session — the only cursor API main has), so
    on main it genuinely begins in A. Only once the stub confirms the turn has reached the LLM is
    the cursor moved to B. That makes the interleaving deterministic and the failure attributable:
    the beat leaves chat A ONLY because the write re-resolves the cursor late.

    FAILS ON MAIN: on release, the beat is written into chat B.
    """
    from fastapi.testclient import TestClient

    entered, gate = threading.Event(), threading.Event()
    app = build_app(home=str(tmp_path), llm=_StubLLM(gate=gate, entered=entered), model="stub")
    client = TestClient(app)
    store = app.state.ctx.store

    client.post("/session", json={"session_id": CHAT_B})   # materialise B
    client.post("/session", json={"session_id": CHAT_A})   # cursor on A -> the turn STARTS in A

    with client.websocket_connect("/ws") as ws_a:
        _drain_ready(ws_a)
        _open_chat(ws_a, CHAT_A)

        ws_a.send_json({"type": "position", "fen": STARTPOS})
        ws_a.send_json({"type": "turn", "text": "park me in the llm"})

        # Deterministic barrier: the turn is now inside the LLM call, having started in chat A.
        assert entered.wait(90), "the turn never reached the parked LLM call"

        client.post("/session", json={"session_id": CHAT_B})   # move the cursor UNDER the parked turn
        gate.set()                                             # release it

        landed = _wait_for_marker(store, A_SAYS)

    assert landed is not None, "the parked turn never produced its beat at all"
    assert landed == CHAT_A, (
        f"LATE READ: the parked turn started in {CHAT_A!r} but re-resolved the global cursor at "
        f"write time and wrote into {landed!r}"
    )


@requires_engine
def test_open_chat_delivers_the_new_chats_snapshot_to_this_socket(tmp_path):
    """Switching chats must land on the SOCKET, immediately — not on the next reconnect.

    Regression: events are addressed to a chat's subscribers, so a REST call that changes the active
    chat is invisible to an already-connected socket. The socket has to move itself (`open_chat`), and
    the server must RETARGET IT BEFORE publishing the new chat's snapshot — publish first and this
    socket is not in the subscriber set yet, the snapshot goes to nobody, and the app keeps showing
    the old chat until it reconnects. That is exactly what "new session only appears after a refresh"
    looks like.
    """
    from fastapi.testclient import TestClient

    app = build_app(home=str(tmp_path), llm=_StubLLM(), model="stub")
    client = TestClient(app)
    store = app.state.ctx.store

    client.post("/session", json={"session_id": CHAT_A})
    with client.websocket_connect("/ws") as ws:
        _drain_ready(ws)
        # Bank something distinctive in B so B's snapshot is recognisable when it arrives.
        with store.bound(CHAT_B):
            store.open_chat(CHAT_B)
            store.append_beats([{"kind": "say", "stops": False,
                                 "segments": [{"text": "B-ONLY-SNAPSHOT"}]}])

        tap = _SocketTap(ws)
        _open_chat(ws, CHAT_B)          # the socket moves itself
        assert tap.saw_marker("B-ONLY-SNAPSHOT", within=20), \
            "the socket never received the new chat's snapshot — it would only appear on reconnect"


def test_open_chat_baseline_precedes_any_delta_for_the_new_chat(tmp_path):
    """The socket's FIRST event for the chat it just opened must be `reset` — never a delta.

    Regression on the fix above. Retargeting the socket into B and THEN publishing B's snapshot fixes
    "the snapshot goes to nobody", but leaves a smaller window: between the retarget and the publish
    this socket is already a subscriber of B, so a concurrent writer on B (a background coach beat on
    the chat being opened — routine, not exotic) can enqueue a DELTA ahead of the reset+snapshot. The
    socket then applies a B delta against its A replica, or drops it as a version gap and shows a
    stale board. `open_chat_for` closes it by doing the retarget and the baseline as one locked step.

    The interleaving is forced, not slept for: `snapshot` is where `open_chat_for` is mid-operation
    with the locks held, so the intruding writer is launched from there. It blocks on `_wlock` until
    the baseline is enqueued — which is the invariant under test. If the lock were dropped (or the
    steps split again), the intruder's beat lands first and the order assert fails.
    """
    from fastapi.testclient import TestClient

    app = build_app(home=str(tmp_path), llm=_StubLLM(), model="stub")
    client = TestClient(app)
    store = app.state.ctx.store

    client.post("/session", json={"session_id": CHAT_A})

    intruder_ran = threading.Event()
    intruder: "list[threading.Thread]" = []

    def _intrude():
        # A concurrent writer on the chat being opened. Binds explicitly: contextvars do NOT cross a
        # raw thread, so without this it would write to `_live[""]`.
        intruder_ran.set()
        with store.bound(CHAT_B):
            store.append_beats([{"kind": "say", "stops": False,
                                 "segments": [{"text": "INTRUDER-DELTA"}]}])

    real_snapshot = store.snapshot

    def snapshot_with_intruder():
        if not intruder:                       # once — the switch under test, not later snapshots
            t = threading.Thread(target=_intrude, daemon=True)
            intruder.append(t)
            t.start()
            intruder_ran.wait(timeout=5)       # it is running; it will park on _wlock
            time.sleep(0.05)                   # give it every chance to get in front of us
        return real_snapshot()

    with client.websocket_connect("/ws") as ws:
        _drain_ready(ws)
        with store.bound(CHAT_B):
            store.open_chat(CHAT_B)
            store.append_beats([{"kind": "say", "stops": False,
                                 "segments": [{"text": "B-BASELINE"}]}])

        store.snapshot = snapshot_with_intruder
        try:
            tap = _SocketTap(ws)
            _open_chat(ws, CHAT_B)
            order = tap.types_until("INTRUDER-DELTA", within=20)
        finally:
            store.snapshot = real_snapshot
            if intruder:
                intruder[0].join(timeout=5)

    assert order, "the socket never received the intruding delta — the switch itself did not deliver"
    assert order[0] == "reset", (
        f"the socket's first event after opening chat B was {order[0]!r}, not 'reset' — a delta for "
        f"the new chat arrived before its baseline. Full order: {order}"
    )
