"""Orchestrator — the in-house deterministic coaching pipeline (no LLM tool-calling).

The model never decides which tool to call. The RUNNER classifies the turn, calls the MCP
tools to GROUND the facts, asks the model for ONE grounded generation, then pushes the beat
and records mastery itself. Everything the model could get wrong is done deterministically;
the model only writes prose (or a small JSON verdict) over facts it was handed.

Intent is INTERPRETED BY THE MODEL, not matched by brittle keywords: the single coaching
generation returns {mode: 'ask'|'tell', text}, so the model decides whether to lead with a
Socratic question or explain directly — robust to phrasing and typos.

Flows ported: coach (OPEN — ask by default / tell when asked), probe_answer (PROBE_ANSWER —
grade + record mastery). Every other class delegates to the ADK `fallback`.
Enable with LUCENA_ORCHESTRATED=1.
"""
from __future__ import annotations

import asyncio
import json

from .llm import make_adapter, Message, GenerateOptions, LLMAdapter

# JSON-mode flows return a small object; a permissive schema just triggers JSON output.
_JSON_OBJECT = {"type": "object"}

_COACH_SYSTEM = (
    "You are a chess coach speaking to one player. You are given the engine's grounded read of the "
    "CURRENT position, the player's message, and the engine's best move. DECIDE how to respond:\n"
    "- DEFAULT (mode='ask'): lead with ONE Socratic question that guides them toward the key idea "
    "WITHOUT revealing or naming the best move. Use this whenever they're exploring, unsure, or ask "
    "something open like 'what should I think about here?'.\n"
    "- mode='tell': ONLY when the player clearly wants to be told the answer (e.g. asks for the best "
    "move, says 'just tell me', 'show me the move', 'what should I play'). Then explain directly, "
    "naming the move and the reason.\n"
    "Ground EVERY claim only in the facts provided — never invent a piece, square, line, or number. "
    "Translate evaluations into plain words ('you're winning', 'roughly equal') — never cite win% or "
    "centipawns.\n"
    "PERSPECTIVE (critical — getting it backwards ruins the read): address the player as 'you'; they "
    "play the side to move. The OTHER colour is 'your opponent'. Every threat, attack, or plan belongs "
    "to the OPPONENT — never say the player is threatening their own pieces or defending against "
    "themselves. Name the opponent's threat when there is one. Warm, direct, one idea, no jargon walls.\n"
    "Return JSON: {\"mode\": \"ask\"|\"tell\", \"text\": string}. `text` is the question (ask) or the "
    "explanation (tell), shown to the player."
)

_GRADE_SYSTEM = (
    "You are a chess coach grading the player's answer to a question about the CURRENT position. You "
    "are given the engine's grounded analysis (best move, top moves, evals) and the player's message. "
    "Decide if the move/idea they propose is correct (matches or is among the engine's best). Give "
    "short, encouraging feedback grounded ONLY in the analysis — never invent a line or number. "
    "Respond as JSON with keys: proposed_move (string, the move they suggested or ''), correct "
    "(boolean), quality (number 0..1: 1.0 found the best cold, 0.5 close/after help, 0.3 needed the "
    "answer), feedback (string, shown to the player), concept (string, a one-word theme, or '')."
)


def _say_beat(text: str, tone: str = "teach") -> dict:
    return {"kind": "say", "tone": tone, "segments": [{"text": text}], "stops": False}


def _ask_beat(text: str, hints: list[str] | None = None) -> dict:
    b: dict = {"kind": "ask", "segments": [{"text": text}], "stops": True}   # stops -> gate locks
    if hints:
        b["hints"] = hints
    return b


class Orchestrator:
    """The deterministic coaching pipeline. Depends only on: the state machine (in-process),
    the engine gRPC client (grounding), and the LLM adapter (generation). No MCP, no engine
    imports — the open/closed firewall holds."""

    def __init__(self, *, store, engine, model: str, llm: LLMAdapter | None = None):
        self.store = store          # StateStore
        self.engine = engine        # EngineClient (gRPC)
        self.model = model
        self._llm: LLMAdapter = llm or make_adapter({"provider": "gemini", "default_model": model})
        self._last_tokens: dict | None = None

    async def run_turn(self, session_id: str, text: str | None = None) -> dict:
        self.store.read_input()                     # consume the mailbox (parity)
        if not (text and text.strip()):
            return {"ok": True, "orchestrated": False, "flow": "unhandled"}
        fen = self.store.board_view
        if self.store._gate_awaiting:               # mid-probe -> grade the answer
            return await self._probe_answer(fen, text.strip())
        return await self._coach(fen, text.strip())

    # -- flows ---------------------------------------------------------------

    async def _coach(self, fen: str | None, text: str) -> dict:
        """OPEN turn. Ground via the engine (gRPC), let the model pick ask (Socratic, default) vs
        tell, then act on the state machine. An ask attaches the grounded hint ladder + locks the gate."""
        facts = await asyncio.to_thread(self.engine.analyze, fen) if fen else {}
        hints_res = await asyncio.to_thread(self.engine.hints, fen) if fen else {}
        best = hints_res.get("best")
        hints = [h for h in (hints_res.get("hints") or []) if isinstance(h, str)][:3]

        side = (facts.get("side_to_move") or ("black" if fen and " b " in f" {fen} " else "white"))
        you = side.capitalize()
        opp = "White" if side == "black" else "Black"
        out = await self._gen_json(
            _COACH_SYSTEM,
            f"You are coaching the player, who is playing {you} (the side to move). Their opponent is "
            f"{opp}. Every threat/attack/plan belongs to {opp}, never to the player.\n\n"
            f"Player said: {text}\n\nEngine's grounded read (coach ONLY from this):\n"
            f"{json.dumps(facts)[:2000]}\n"
            f"Best move (reveal ONLY in a 'tell'): {best}\n\nRespond as JSON.")
        mode = (out.get("mode") or "ask").lower()
        body = out.get("text") or "Let's take a look at this position together."
        if mode == "tell":
            self.store.append_beats([_say_beat(body)])
        else:
            self.store.append_beats([_ask_beat(body, hints or None)])
            self.store.set_gate(True)               # the ask locks the Socratic gate
        return {"ok": True, "orchestrated": True, "flow": f"coach:{mode}",
                "tokens": self._last_tokens}

    async def _probe_answer(self, fen: str | None, text: str) -> dict:
        """The player answered a probe. Grade vs the engine's read, give feedback, unlock the gate."""
        facts = await asyncio.to_thread(self.engine.analyze, fen) if fen else {}
        verdict = await self._gen_json(
            _GRADE_SYSTEM,
            f"Player's answer: {text}\n\nEngine's grounded analysis (grade ONLY against this):\n"
            f"{json.dumps(facts)[:2000]}\n\nGrade and give feedback as JSON.")
        correct = bool(verdict.get("correct"))
        feedback = verdict.get("feedback") or "Let's look at that together."
        self.store.append_beats([_say_beat(feedback, tone="praise" if correct else "correct")])
        self.store.set_gate(False)                  # answered -> unlock
        # (mastery recording parked this cycle; the seam is here — record vs verdict.concept/quality.)
        return {"ok": True, "orchestrated": True, "flow": "probe_answer",
                "correct": correct, "tokens": self._last_tokens}

    # -- generation ----------------------------------------------------------

    async def _gen_json(self, system: str, prompt: str) -> dict:
        comp = await self._llm.generate(
            [Message("system", system), Message("user", prompt)],
            GenerateOptions(model=self.model, schema=_JSON_OBJECT, max_tokens=400, temperature=0.4),
        )
        self._stash_tokens(comp.usage)
        return comp.json or {}

    def _stash_tokens(self, usage) -> None:
        self._last_tokens = ({"input": usage.input_tokens, "output": usage.output_tokens,
                              "total": usage.total_tokens} if usage is not None else None)

    async def aclose(self) -> None:
        return None
