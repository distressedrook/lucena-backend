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

import json

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


class Orchestrator:
    def __init__(self, *, mcp_url: str, model: str, fallback=None):
        from google import genai
        from google.genai import types

        self.mcp_url = mcp_url
        self.model = model
        self.fallback = fallback
        self._genai = genai
        self._types = types
        self._client = None
        self._last_tokens: dict | None = None

    def _get_client(self):
        if self._client is None:
            self._client = self._genai.Client(http_options=self._types.HttpOptions(
                retry_options=self._types.HttpRetryOptions(
                    attempts=3, initial_delay=1.0, max_delay=8.0, exp_base=2.0,
                    jitter=1.0, http_status_codes=[429, 503])))
        return self._client

    async def run_turn(self, session_id: str, text: str | None = None) -> dict:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        handled: dict | None = None
        async with streamablehttp_client(self.mcp_url) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()

                async def call(tool: str, args: dict) -> dict:
                    res = await s.call_tool(tool, args)
                    txt = res.content[0].text if res.content else "{}"
                    try:
                        return json.loads(txt)
                    except ValueError:
                        return {"raw": txt}

                ri = await call("read_input", {})
                cls = ri.get("classification")
                if cls == "OPEN" and text and text.strip():
                    handled = await self._coach(call, ri, text.strip())
                elif cls == "PROBE_ANSWER" and text and text.strip():
                    handled = await self._probe_answer(call, ri, text.strip())

        if handled is not None:
            return handled
        if self.fallback is not None:
            return await self.fallback.run_turn(session_id, text)
        return {"ok": True, "orchestrated": False, "flow": "unhandled"}

    # -- flows ---------------------------------------------------------------

    async def _coach(self, call, ri: dict, text: str) -> dict:
        """OPEN turn. The model decides ask (Socratic, default) vs tell (explain) and generates.
        On ask we attach the grounded hint ladder from get_hints; the ask beat stops the turn."""
        fen = ri.get("board_fen")
        facts = await call("analyze_and_show", {"fen": fen, "focus": "analysis"}) if fen else {}
        hints_res = await call("get_hints", {"fen": fen}) if fen else {}
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
            await call("push_beat", {"beats": [{"kind": "say", "tone": "teach", "text": body}]})
        else:
            beat: dict = {"kind": "ask", "text": body}      # kind=ask -> stops -> gate locks
            if hints:
                beat["hints"] = hints                       # grounded ladder
            await call("push_beat", {"beats": [beat]})
        return {"ok": True, "orchestrated": True, "flow": f"coach:{mode}",
                "tokens": self._last_tokens,
                "tool_calls": ["read_input", "analyze_and_show", "get_hints", "push_beat"]}

    async def _probe_answer(self, call, ri: dict, text: str) -> dict:
        fen = ri.get("board_fen")
        facts = await call("analyze_and_show", {"fen": fen, "focus": "analysis"}) if fen else {}
        verdict = await self._gen_json(
            _GRADE_SYSTEM,
            f"Player's answer: {text}\n\nEngine's grounded analysis (grade ONLY against this):\n"
            f"{json.dumps(facts)[:2000]}\n\nGrade and give feedback as JSON.")
        correct = bool(verdict.get("correct"))
        feedback = verdict.get("feedback") or "Let's look at that together."
        await call("push_beat", {"beats": [
            {"kind": "say", "tone": "praise" if correct else "correct", "text": feedback}]})
        recorded = False
        concept = verdict.get("concept")
        quality = verdict.get("quality")
        if concept and isinstance(quality, (int, float)):
            res = await call("record_observation", {"concept": str(concept),
                                                    "quality": float(quality), "type": "probe"})
            recorded = isinstance(res, dict) and "error" not in res
        return {"ok": True, "orchestrated": True, "flow": "probe_answer",
                "correct": correct, "recorded": recorded, "tokens": self._last_tokens,
                "tool_calls": ["read_input", "analyze_and_show", "push_beat",
                               "record_observation"]}

    # -- generation ----------------------------------------------------------

    async def _gen_json(self, system: str, prompt: str) -> dict:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=400, temperature=0.4,
            response_mime_type="application/json")
        resp = await self._get_client().aio.models.generate_content(
            model=self.model, contents=prompt, config=cfg)
        self._stash_tokens(resp)
        try:
            return json.loads(resp.text or "{}")
        except (ValueError, TypeError):
            return {}

    def _stash_tokens(self, resp) -> None:
        um = getattr(resp, "usage_metadata", None)
        self._last_tokens = ({"input": getattr(um, "prompt_token_count", None),
                              "output": getattr(um, "candidates_token_count", None),
                              "total": getattr(um, "total_token_count", None)}
                             if um is not None else None)

    async def aclose(self) -> None:
        if self.fallback is not None:
            await self.fallback.aclose()
