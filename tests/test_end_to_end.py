"""B4+B5 end-to-end: a coaching turn flows client -> WS -> orchestrator -> engine gRPC ->
LLM -> state machine -> WS, with a stub LLM (no key) and a REAL engine gRPC server."""

import os
import shutil

import pytest

from lucena_backend.llm import Completion, Usage
from lucena_backend.engine_client import EngineClient
from lucena_backend.httpserver import build_app

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_have_sf = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_sf, reason="no stockfish")


class _StubLLM:
    """Deterministic: the coach 'tells' a fixed line (no key, no network)."""
    def __init__(self, payload):
        self._payload = payload

    async def generate(self, messages, opts):
        import json
        if isinstance(self._payload, str):                 # explain: plain text
            return Completion(text=self._payload, json=None, model="stub", usage=Usage(1, 1, 2))
        return Completion(text=json.dumps(self._payload), json=self._payload,  # coach: JSON verdict
                          model="stub", usage=Usage(1, 1, 2))


@pytest.fixture
def engine_server():
    from lucena_engine.server.serve import build_server
    server, engines = build_server(port=50356, threads=1)
    server.start()
    yield "localhost:50356"
    server.stop(0)
    engines.close()


@requires_engine
def test_coaching_turn_end_to_end(tmp_path, engine_server):
    from fastapi.testclient import TestClient

    app = build_app(
        home=str(tmp_path),
        engine=EngineClient(engine_server),
        llm=_StubLLM({"mode": "tell", "text": "Focus on developing your pieces toward the center."}),
        model="stub",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        while ws.receive_json()["type"] != "ready":      # drain the snapshot
            pass
        ws.send_json({"type": "position", "fen": STARTPOS})   # the board the app shows
        ws.send_json({"type": "turn", "text": "what should I think about here?"})

        beat = None
        for _ in range(30):
            m = ws.receive_json()
            if m["type"] == "beats" and m.get("appended"):
                beat = m
                break
        assert beat is not None, "no beats streamed back"
        texts = [seg["text"] for b in beat["appended"] for seg in b.get("segments", [])]
        assert any("developing your pieces" in t for t in texts), texts


@requires_engine
def test_pasted_fen_sets_the_board(tmp_path, engine_server):
    """A FEN in the player's message is DETECTED (deterministic) and set as the board — the app
    gets a `board` event with the pasted position. The LLM never sets the board."""
    from fastapi.testclient import TestClient

    tactical = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    app = build_app(home=str(tmp_path), engine=EngineClient(engine_server),
                    llm=_StubLLM({"mode": "tell", "text": "Interesting position."}), model="stub")
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        while ws.receive_json()["type"] != "ready":
            pass
        ws.send_json({"type": "turn", "text": f"what do you think about {tactical}"})
        board = None
        for _ in range(30):
            m = ws.receive_json()
            if m["type"] == "board" and m.get("fen"):
                board = m
                break
        assert board is not None, "no board event — the pasted FEN was not set"
        assert board["fen"].startswith("r1bqkbnr/pppp1ppp/2n5/1B2p3"), board["fen"]


@requires_engine
def test_explain_end_to_end(tmp_path, engine_server):
    from fastapi.testclient import TestClient

    app = build_app(
        home=str(tmp_path),
        engine=EngineClient(engine_server),
        llm=_StubLLM("That knight move drops a pawn."),  # explain: plain text
        model="stub",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        while ws.receive_json()["type"] != "ready":
            pass
        ws.send_json({"type": "explain", "fen": STARTPOS, "move": "e4", "correct": True})
        beat = None
        for _ in range(30):
            m = ws.receive_json()
            if m["type"] == "beats" and m.get("appended"):
                beat = m
                break
        assert beat is not None, "no explain beat streamed back"
