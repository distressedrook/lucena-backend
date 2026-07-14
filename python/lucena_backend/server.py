"""FastAPI wrapper exposing the LucenaAgent over a local `POST /turn`.

The Swift app spawns this process alongside `serve-http` and calls `/turn` whenever
the player types (free text) or acts on the board (structured event). Beats stream
to the app over `serve-http`'s `/state`; `/turn` returns a compact trace.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from .runner import LucenaAgent


class TurnRequest(BaseModel):
    session_id: str
    text: str | None = None  # None/empty => structured board event


class ExplainRequest(BaseModel):
    session_id: str
    fen: str
    move: str | None = None       # the move to explain (its refutation, if wrong)
    correct: bool | None = None   # was it the right move?


def build_app(*, mcp_url: str, home: str, model: str | None = None,
              instruction_path: str | None = None) -> FastAPI:
    app = FastAPI(title="lucena-agent")
    state: dict[str, LucenaAgent] = {}

    @app.on_event("startup")
    async def _startup() -> None:
        if os.environ.get("LUCENA_COACH_MOCK", "").lower() in ("1", "true", "yes"):
            from .mock import MockCoach
            mc = MockCoach(mcp_url=mcp_url)               # deterministic, zero-cost, no key
            state["agent"] = mc
            state["quick"] = mc                           # mock handles /explain too
        else:
            agent = LucenaAgent(
                mcp_url=mcp_url, home=home, model=model,
                instruction_path=instruction_path)
            from .quick import QuickCoach                 # lightweight on-demand 'Why?' path
            state["quick"] = QuickCoach(mcp_url=mcp_url, model=agent.model)
            if os.environ.get("LUCENA_ORCHESTRATED", "").lower() in ("1", "true", "yes"):
                from .orchestrator import Orchestrator     # Stage 0: OPEN turn deterministic,
                state["agent"] = Orchestrator(             # rest delegates to the ADK agent
                    mcp_url=mcp_url, model=agent.model, fallback=agent)
            else:
                state["agent"] = agent

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        agent = state.get("agent")
        if agent:
            await agent.aclose()

    @app.get("/health")
    async def health() -> dict:
        agent = state.get("agent")
        return {"ok": agent is not None,
                "model": getattr(agent, "model", None),
                "mcp_url": mcp_url}

    @app.post("/turn")
    async def turn(req: TurnRequest) -> dict:
        return await state["agent"].run_turn(req.session_id, req.text)

    @app.post("/explain")
    async def explain(req: ExplainRequest) -> dict:
        return await state["quick"].explain(
            session_id=req.session_id, fen=req.fen, move=req.move, correct=req.correct)

    return app


def main(argv=None) -> None:
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(prog="lucena-agent")
    ap.add_argument("--mcp-url", default=os.environ.get(
        "LUCENA_MCP_URL", "http://127.0.0.1:8765/mcp"))
    ap.add_argument("--home", default=os.environ.get(
        "LUCENA_HOME", os.path.join(os.getcwd(), "coach-home")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--model", default=os.environ.get("LUCENA_GEMINI_MODEL") or None)
    ap.add_argument("--instruction", default=os.environ.get("LUCENA_CONTRACT") or None)
    args = ap.parse_args(argv)

    app = build_app(mcp_url=args.mcp_url, home=args.home, model=args.model,
                    instruction_path=args.instruction)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
