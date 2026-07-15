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

    def _spawn(coro, label: str = "bg") -> None:
        """Fire a coroutine as a tracked background task with error containment — so slow, fallible
        work (coaching a move via the LLM) never blocks the caller or crashes it on failure."""
        async def _guarded():
            try:
                await coro
            except Exception as exc:  # noqa: BLE001
                store.publish_status(None)
                print(f"[bg] {label} failed: {exc!r}", flush=True)
        task = asyncio.create_task(_guarded())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    async def _play_and_coach(uci, fen) -> dict:
        """Apply the move (deterministic: adjudicate, opponent reply, board/history — canned verdict
        suppressed) then coach it with the LLM in the background (drill-aware: right/wrong + what's
        next). The move lands instantly; the grounded coaching beat follows a few seconds later."""
        result = await asyncio.to_thread(ctx.play_move, uci, fen, push_feedback=False)
        if isinstance(result, dict) and result.get("ok") and uci:
            _spawn(orch.coach_move(uci, fen, result), label="coach_move")
        return result

    async def _dispatch(msg: dict) -> None:
        t = msg.get("type")
        if t == "turn":
            await orch.run_turn(store._current, msg.get("text"))
        elif t == "explain":
            await quick.explain(session_id=store._current, fen=msg.get("fen"),
                                move=msg.get("move"), correct=msg.get("correct"))
        elif t == "position":
            await asyncio.to_thread(store.set_board_view, msg.get("fen"))
        elif t == "view":
            await asyncio.to_thread(store.set_view, msg)
        elif t == "input":
            await asyncio.to_thread(store.set_input, msg.get("data") or msg)
        elif t == "move":
            await _play_and_coach(msg.get("uci"), msg.get("fen"))
        elif t == "drill":
            await asyncio.to_thread(_arm_drill, ctx, store, msg.get("fen"))

    async def _handle(msg: dict) -> None:
        # The transport has no catch-all: an unhandled error here (an LLM outage, a bad move) would
        # break the WS receive loop and drop the connection mid-turn, leaving the app spinning on a
        # status that never clears. Contain it — clear the working status and surface a plain beat so
        # the player always gets an answer, and the socket stays up.
        try:
            await _dispatch(msg)
        except Exception as exc:  # noqa: BLE001
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
        await asyncio.to_thread(store.ensure_session_id)
        q = store.subscribe()
        bg: set[asyncio.Task] = set()
        try:
            for channel, payload in store.snapshot():
                await websocket.send_json({"type": channel, **payload})
            await websocket.send_json({"type": "ready"})

            async def pump():
                while True:
                    channel, payload = await q.get()
                    await websocket.send_json({"type": channel, **payload})

            sender = asyncio.create_task(pump())
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("type") in _SLOW:
                        task = asyncio.create_task(_handle(msg))
                        bg.add(task)
                        task.add_done_callback(bg.discard)
                    else:
                        await _handle(msg)
            finally:
                sender.cancel()
                for t in bg:
                    t.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            store.unsubscribe(q)

    # -- REST ---------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/sessions")
    async def sessions():
        from .sessions import list_sessions
        rows = await asyncio.to_thread(list_sessions, store.home, store.db)
        return {"sessions": rows, "current": store._current}

    @app.get("/session")
    async def get_session():
        return {"session_id": await asyncio.to_thread(store.ensure_session_id)}

    @app.post("/session/new")
    async def new_session():
        """'New session' — the BACKEND mints the id, makes it active (publishing reset + a clean
        snapshot over the WS), and returns it for the app to adopt."""
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
        return await _play_and_coach(uci, fen)

    @app.post("/drill")
    async def drill(body: dict):
        """Arm a forcing-line drill on a position (build the tree + detect the poisoned line)."""
        fen = body.get("fen")
        if not fen:
            return JSONResponse({"error": "bad_fen"}, status_code=400)
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
