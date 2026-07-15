"""Client <-> backend server (B4/B5, pivoted): WebSocket for the live loop + REST for the rest.

The backend uses the in-process ToolContext (the full legacy coach: analyze/evaluate/hints, drills
via DrillState + build_line_tree, poisoned lines, Maia move-meaning) — NOT a thin gRPC path — so
drills and Maia coaching behave exactly as before. (The open/closed engine firewall is set aside for
now; ToolContext calls lucena_engine in-process. Re-splitting it over gRPC is a follow-up.)

WS `/ws`: snapshot on connect then the StateStore's deltas; client sends turn/explain/position/view/
input/move/drill. `/move` adjudicates against the live drill (opponent reply, feedback) via
ctx.play_move; `/drill` arms a forcing-line drill on a position.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse

from lucena_engine.uci import Engine
from .state import StateStore
from .db import DB
from .tools import ToolContext
from .orchestrator import Orchestrator
from .quick import QuickCoach

_DEFAULT_MODEL = os.environ.get("LUCENA_MODEL", "gemini-flash-lite-latest")


def _open_chat_bound(store, sid: str) -> None:
    """`open_chat` off the loop (it does DB writes). Runs via asyncio.to_thread, which copies the
    caller's context, so the bind it performs lands where the caller expects."""
    store.open_chat(sid)
    store._seed_start_board()


def _arm_drill(ctx, store, fen):
    """Arm a drill on a position — a deliberate APP action. It supersedes any pending Socratic probe,
    so clear the turn gate first (else the `_scoped` guard refuses with `awaiting_input`)."""
    store.set_gate(False)
    return ctx.build_and_arm_drill(fen)


def _make_maia():
    """A MaiaEngine if LUCENA_MAIA is configured, else None (poisoned lines / move-meaning off)."""
    if not os.environ.get("LUCENA_MAIA"):
        return None
    try:
        from lucena_engine.maia import MaiaEngine
        return MaiaEngine()
    except Exception:  # noqa: BLE001
        return None


def build_app(*, home: str, llm=None, model: str = _DEFAULT_MODEL,
              rating: int = 1500, engine=None) -> FastAPI:
    app = FastAPI(title="lucena-backend")
    db = DB(os.path.join(home, "lucena"))
    store = StateStore(home, db=db)
    eng = engine or Engine()                       # in-process Stockfish
    maia = _make_maia()
    ctx = ToolContext(eng, store, maia=maia, player_rating=rating)
    ctx._poisoned_line_engine_factory = lambda: Engine(threads=1)   # dedicated single-thread detector
    # A SEPARATE engine + ToolContext for the coach's read-only LLM grounding (evaluate/analyze). It
    # shares the store but has its own engine and its own tool lock, so the coach's multi-second
    # analysis NEVER blocks an interactive move/drill on the main ctx (which was making a Retry right
    # after a wrong move sit locked for seconds while coach_move ran). Maia off here — the main ctx owns
    # the single Maia subprocess; a second caller would corrupt its UCI stream.
    ground_ctx = ToolContext(Engine(), store, maia=None, player_rating=rating)
    orch = Orchestrator(ctx=ctx, model=model, llm=llm, ground_ctx=ground_ctx)
    quick = QuickCoach(ctx=ctx, model=model, llm=llm, ground_ctx=ground_ctx)

    app.state.store = store
    app.state.ctx = ctx

    _bg_tasks: set = set()

    def _spawn(coro, label: str = "bg", *, session_id: str) -> None:
        """Fire a coroutine as a tracked background task with error containment — so slow, fallible
        work (coaching a move via the LLM) never blocks the caller or crashes it on failure.

        `session_id` is explicit because the failure path publishes: a status clear with no chat
        bound would resolve the cursor empty (or, worse, another chat's) and wipe someone else's
        spinner.
        """
        async def _guarded():
            try:
                await coro
            except Exception as exc:  # noqa: BLE001
                with store.bound(session_id):
                    store.publish_status(None)
                print(f"[bg] {label} failed: {exc!r}", flush=True)
        task = asyncio.create_task(_guarded())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    async def _play_and_coach(uci, fen, *, session_id: str) -> dict:
        """Apply the move (deterministic: adjudicate, opponent reply, board/history — canned verdict
        suppressed) then coach it with the LLM in the background (drill-aware: right/wrong + what's
        next). The move lands instantly; the grounded coaching beat follows a few seconds later."""
        result = await asyncio.to_thread(ctx.play_move, uci, fen, push_feedback=False)
        if isinstance(result, dict) and result.get("ok") and uci:
            # The chat is captured HERE, at dispatch, and handed to the task. coach_move is detached
            # and can outlive this turn by seconds; resolving the chat when it finally writes would
            # bind it to whatever is current by then — the late-read race.
            _spawn(orch.coach_move(session_id, uci, fen, result), label="coach_move",
                   session_id=session_id)
        return result

    async def _dispatch(msg: dict, sid: str) -> None:
        t = msg.get("type")
        if t == "turn":
            await orch.run_turn(sid, msg.get("text"))
        elif t == "explain":
            await quick.explain(session_id=sid, fen=msg.get("fen"),
                                move=msg.get("move"), correct=msg.get("correct"))
        elif t == "position":
            await asyncio.to_thread(store.set_board_view, msg.get("fen"))
        elif t == "view":
            await asyncio.to_thread(store.set_view, msg)
        elif t == "input":
            await asyncio.to_thread(store.set_input, msg.get("data") or msg)
        elif t == "move":
            await _play_and_coach(msg.get("uci"), msg.get("fen"), session_id=sid)
        elif t == "drill":
            await asyncio.to_thread(_arm_drill, ctx, store, msg.get("fen"))

    async def _handle(msg: dict, sid: str) -> None:
        # The transport has no catch-all: an unhandled error here (an LLM outage, a bad move) would
        # break the WS receive loop and drop the connection mid-turn, leaving the app spinning on a
        # status that never clears. Contain it — clear the working status and surface a plain beat so
        # the player always gets an answer, and the socket stays up.
        try:
            await _dispatch(msg, sid)
        except Exception as exc:  # noqa: BLE001
            with store.bound(sid):          # the apology belongs to THIS chat, not whatever is current
                store.publish_status(None)
                store.append_beats([{
                    "kind": "say", "tone": "teach", "stops": False,
                    "segments": [{"text": "I hit a snag reaching my coaching brain just now — the position "
                                          "is still set, so try that again in a moment."}],
                }])
            print(f"[_handle] {msg.get('type')} failed: {exc!r}", flush=True)

    # Coach thinking (turn/explain) can take several seconds of LLM+engine work. Running it INLINE in
    # the receive loop blocks every other message — clicks, navigation, moves all stall until it
    # finishes, which looks like a frozen app. Those two run as background tasks so the loop keeps
    # servicing input; the fast, state-mutating messages (move/drill/position/view/input) stay inline
    # and ordered. The store's tool lock still serializes any shared-state mutation across them.
    _SLOW = {"turn", "explain"}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        # This connection's chat. `?session=<id>` picks one explicitly; otherwise fall back to the
        # durable active-chat pointer (minting one if this home has never had a chat).
        requested = websocket.query_params.get("session")
        sid = await asyncio.to_thread(store.ensure_session_id, requested)
        sub = store.subscribe(sid)
        bg: set[asyncio.Task] = set()
        try:
            with store.bound(sid):
                snap = store.snapshot()
            for channel, payload in snap:
                await websocket.send_json({"type": channel, **payload})
            await websocket.send_json({"type": "ready"})

            async def pump():
                while True:
                    # next_event drops events queued for a chat this socket has since left. The stamp
                    # is then RE-CHECKED under the gate: send_json is an await point, so an open_chat
                    # can retarget this socket between the dequeue and the send, and that dequeued
                    # event would otherwise still go out to the new chat.
                    ev_sid, ev_epoch, channel, payload = await sub.next_event()
                    async with sub.gate:
                        if not sub.is_live(ev_sid, ev_epoch):
                            continue        # retargeted while we were between dequeue and gate
                        await websocket.send_json({"type": channel, **payload})

            sender = asyncio.create_task(pump())
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") == "open_chat":
                        # Switch THIS socket to another chat: move its subscription, then bind + snapshot.
                        new_sid = msg.get("session_id")
                        if new_sid and new_sid != sid:
                            # Under the gate: waits out any in-flight send, so no event dequeued for
                            # the old chat can still reach this socket once retarget returns.
                            async with sub.gate:
                                store.retarget(sub, new_sid)
                            sid = new_sid
                            await asyncio.to_thread(_open_chat_bound, store, sid)
                        continue
                    # Bind BEFORE spawning: create_task copies the current context, so the task
                    # inherits this chat and cannot re-resolve a different one when it finally writes.
                    if msg.get("type") in _SLOW:
                        with store.bound(sid):
                            task = asyncio.create_task(_handle(msg, sid))
                        bg.add(task)
                        task.add_done_callback(bg.discard)
                    else:
                        with store.bound(sid):
                            await _handle(msg, sid)
            finally:
                sender.cancel()
                for t in bg:
                    t.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            store.unsubscribe(sub)

    # -- REST ---------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"ok": True}

    # REST has no connection to carry a chat, so these take one explicitly (`session_id` in the body /
    # `?session=`), falling back to the durable active-chat pointer. They must never run against an
    # unbound cursor.
    async def _rest_sid(requested: str | None) -> str:
        return await asyncio.to_thread(store.ensure_session_id, requested)

    @app.get("/sessions")
    async def sessions(session: str | None = None):
        from .sessions import list_sessions
        rows = await asyncio.to_thread(list_sessions, store.home, store.db)
        return {"sessions": rows, "current": await _rest_sid(session)}

    @app.get("/session")
    async def get_session():
        return {"session_id": await _rest_sid(None)}

    @app.post("/session/new")
    async def new_session():
        """'New chat' — the BACKEND mints the id, makes it the active chat (publishing reset + a
        clean snapshot to that chat's sockets), and returns it for the app to adopt."""
        return {"session_id": await asyncio.to_thread(store.new_session)}

    @app.post("/session")
    async def post_session(body: dict):
        sid = body.get("session_id")
        if not sid:
            return JSONResponse({"error": "bad_session"}, status_code=400)
        return {"session_id": await asyncio.to_thread(store.write_session_id, sid)}

    @app.post("/move")
    async def move(body: dict):
        """Play a move — adjudicated against the live drill (opponent reply + feedback beats stream
        over the WS), or freeform if no drill is armed. Returns {ok, drill, correct?, finished?}."""
        uci, fen = body.get("uci"), body.get("fen")
        if not uci:
            return JSONResponse({"error": "bad_move"}, status_code=400)
        sid = await _rest_sid(body.get("session_id"))
        with store.bound(sid):
            return await _play_and_coach(uci, fen, session_id=sid)

    @app.post("/drill")
    async def drill(body: dict):
        """Arm a forcing-line drill on a position (build the tree + detect the poisoned line)."""
        fen = body.get("fen")
        if not fen:
            return JSONResponse({"error": "bad_fen"}, status_code=400)
        # Same as /move: _arm_drill writes tree/board/drill state and publishes, so it must run
        # against a resolved chat — never the unbound "" bucket.
        sid = await _rest_sid(body.get("session_id"))
        with store.bound(sid):
            return await asyncio.to_thread(_arm_drill, ctx, store, fen)

    @app.post("/analyze")
    async def analyze(body: dict):
        on, fen = body.get("on"), body.get("fen")
        # Live analysis toggle is a follow-up; accept the call so the app doesn't 404.
        return {"ok": True, "on": bool(on), "fen": fen}

    @app.get("/config")
    async def config():
        return {"model": model, "maia": maia is not None, "rating": rating}

    return app


def serve(*, home: str, host: str = "127.0.0.1", port: int = 8766, **kw):
    import uvicorn
    uvicorn.run(build_app(home=home, **kw), host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve(home=os.environ.get("LUCENA_HOME", os.path.expanduser("~/.lucena")),
          port=int(os.environ.get("LUCENA_BACKEND_PORT", "8766")))
