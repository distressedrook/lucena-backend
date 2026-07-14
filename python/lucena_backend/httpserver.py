"""Client <-> backend server (B4): WebSocket for the live coaching loop + REST for the rest.

WS `/ws`: on connect, snapshot then deltas (the StateStore's change stream); client sends
`turn` / `explain` / `position` / `view` / `input`. The turn drives the orchestrator, which
grounds over the engine gRPC and pushes beats — which stream straight back down this socket.

The old MCP `serve_http` is retired; its SSE `/state` becomes this WebSocket. Drill adjudication
(`/move`) + `/analyze` + `/activity` + `/reset` still ride the in-process ToolContext and are a
follow-up (they need the tools.py -> engine-gRPC rewrite to stay behind the firewall).
"""

from __future__ import annotations

import asyncio
import os
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse

from .state import StateStore
from .db import DB
from .engine_client import EngineClient
from .orchestrator import Orchestrator
from .quick import QuickCoach

_DEFAULT_MODEL = os.environ.get("LUCENA_MODEL", "gemini-flash-lite-latest")


def build_app(*, home: str, engine=None, llm=None, model: str = _DEFAULT_MODEL,
              engine_target: str = "localhost:50051") -> FastAPI:
    app = FastAPI(title="lucena-backend")
    db = DB(os.path.join(home, "lucena"))          # path -> a Postgres schema
    store = StateStore(home, db=db)
    eng = engine or EngineClient(engine_target)
    orch = Orchestrator(store=store, engine=eng, model=model, llm=llm)
    quick = QuickCoach(store=store, engine=eng, model=model, llm=llm)
    lock = threading.RLock()                        # single-writer for app->store mutations

    def _locked(fn, *a, **k):
        with lock:
            return fn(*a, **k)

    app.state.store = store                          # exposed for tests

    async def _handle(msg: dict) -> None:
        t = msg.get("type")
        if t == "turn":
            await orch.run_turn(store._current, msg.get("text"))
        elif t == "explain":
            await quick.explain(session_id=store._current, fen=msg.get("fen"),
                                move=msg.get("move"), correct=msg.get("correct"))
        elif t == "position":
            await asyncio.to_thread(_locked, store.set_board_view, msg.get("fen"))
        elif t == "view":
            await asyncio.to_thread(_locked, store.set_view, msg)
        elif t == "input":
            await asyncio.to_thread(_locked, store.set_input, msg.get("data") or msg)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await asyncio.to_thread(_locked, store.ensure_session_id)
        q = store.subscribe()
        try:
            for channel, payload in store.snapshot():
                await websocket.send_json({"type": channel, **payload})
            await websocket.send_json({"type": "ready"})

            async def pump():                        # store deltas -> client
                while True:
                    channel, payload = await q.get()
                    await websocket.send_json({"type": channel, **payload})

            sender = asyncio.create_task(pump())
            try:
                while True:                           # client messages -> handlers
                    await _handle(await websocket.receive_json())
            finally:
                sender.cancel()
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
        return {"session_id": await asyncio.to_thread(_locked, store.ensure_session_id)}

    @app.post("/session")
    async def post_session(body: dict):
        sid = body.get("session_id")
        if not sid:
            return JSONResponse({"error": "bad_session"}, status_code=400)
        return {"session_id": await asyncio.to_thread(_locked, store.write_session_id, sid)}

    @app.get("/config")
    async def config():
        return {"model": model, "engine": engine_target}

    return app


def serve(*, home: str, host: str = "127.0.0.1", port: int = 8766, **kw):
    import uvicorn
    uvicorn.run(build_app(home=home, **kw), host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve(home=os.environ.get("LUCENA_HOME", os.path.expanduser("~/.lucena")),
          port=int(os.environ.get("LUCENA_BACKEND_PORT", "8766")))
