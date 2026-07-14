"""The backend's client to the open grounding engine (gRPC). This is the ONLY way
the backend reaches chess truth — it never imports engine code (open/closed firewall).
Returns plain dicts the orchestrator/flows hand to the LLM.
"""

from __future__ import annotations

import grpc
from google.protobuf.json_format import MessageToDict

from ._pb import engine_pb2 as pb
from ._pb import engine_pb2_grpc as pbg

_DEFAULT_TARGET = "localhost:50051"


def _limit(nodes=None, movetime_ms=1500, threads=1):
    if nodes is not None:
        return pb.Limit(nodes=nodes, threads=threads)
    return pb.Limit(movetime_ms=movetime_ms)


def _d(msg) -> dict:
    return MessageToDict(msg, preserving_proto_field_name=True)


class EngineClient:
    def __init__(self, target: str = _DEFAULT_TARGET, *, channel=None):
        self._channel = channel or grpc.insecure_channel(target)
        self.truth = pbg.TruthStub(self._channel)
        self.behaviour = pbg.BehaviourStub(self._channel)

    # -- the grounding surface the coaching loop uses ----------------------
    def analyze(self, fen: str, *, multipv: int = 2, nodes=None, movetime_ms: int = 1500) -> dict:
        req = pb.AnalyzeReq(fen=fen, limit=_limit(nodes, movetime_ms), multipv=multipv,
                            top_facts=5, focus=pb.FULL)
        return _d(self.truth.Analyze(req))

    def hints(self, fen: str, *, nodes=None, movetime_ms: int = 1500) -> dict:
        r = self.truth.Hints(pb.HintsReq(fen=fen, limit=_limit(nodes, movetime_ms)))
        return {"best": r.best_san, "hints": [h.text for h in r.hints]}

    def evaluate(self, fen: str, moves: list[str], *, nodes=None, movetime_ms: int = 1500) -> dict:
        req = pb.EvaluateReq(fen=fen, moves=list(moves), limit=_limit(nodes, movetime_ms))
        return _d(self.truth.Evaluate(req))

    def validate_fen(self, fen: str) -> dict:
        r = self.truth.ValidateFen(pb.Position(fen=fen))
        return {"legal": r.legal, "side_to_move": r.side_to_move, "error": r.error}

    def get_info(self) -> dict:
        return _d(self.truth.GetInfo(pb.Empty()))

    def close(self) -> None:
        self._channel.close()
