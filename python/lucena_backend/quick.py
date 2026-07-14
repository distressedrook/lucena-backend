"""QuickCoach — the lightweight, on-demand explanation path (no agent, no contract).

The full LucenaAgent (`/turn`) sends the 23KB contract + 24 tool schemas and runs a
tool-calling loop — right for freeform chat, far too heavy for "why was that move
wrong?" on a drill. QuickCoach answers that with: one lean engine call to GROUND the
facts, a ~100-token Socratic system prompt, and ONE generation (no tools). ~30-50x
cheaper per call, and it only fires when the player taps "Why?".

Grounding still holds: the model sees ONLY the engine's facts/refutation and is told
never to invent — it interprets, it doesn't calculate.
"""
from __future__ import annotations

import json

from .llm import make_adapter, Message, GenerateOptions, LLMAdapter

_SYSTEM = (
    "You are a chess coach explaining ONE move to a student in 1-2 short sentences. "
    "Ground everything ONLY in the engine facts you are given — never invent a piece, "
    "square, line, or evaluation. If the move was wrong, explain the flaw through the "
    "refutation provided and point toward the better idea without fully giving it away; "
    "be encouraging, not just 'that's bad'. If it was right, name the idea that makes it "
    "work. Plain language, no eval numbers, one idea. This text is shown to the player."
)


class QuickCoach:
    def __init__(self, *, mcp_url: str, model: str, llm: LLMAdapter | None = None):
        self.mcp_url = mcp_url
        self.model = model
        self._llm: LLMAdapter = llm or make_adapter({"provider": "gemini", "default_model": model})

    async def explain(self, *, session_id: str, fen: str, move: str | None = None,
                      correct: bool | None = None) -> dict:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

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

                # Ground: a move → its refutation via evaluate; a bare position → the fact sheet.
                if move:
                    facts = await call("evaluate", {"fen": fen, "sans": [move]})
                else:
                    facts = await call("analyze_and_show", {"fen": fen, "focus": "analysis"})

                prompt = self._prompt(fen, move, correct, facts)
                text = await self._generate(prompt)
                await call("push_beat", {"beats": [{
                    "kind": "say",
                    "tone": "correct" if correct is False else "teach",
                    "text": text,
                }]})
        return {"ok": True, "text": text}

    def _prompt(self, fen: str, move: str | None, correct: bool | None, facts: dict) -> str:
        verdict = ("The player just played a move that is WRONG." if correct is False
                   else "The player just played the correct move." if correct
                   else "")
        return (
            f"Position (FEN): {fen}\n"
            f"Move played: {move or '(none)'}\n"
            f"{verdict}\n"
            f"Engine facts (ground your answer ONLY in this):\n{json.dumps(facts)[:1500]}\n\n"
            f"Explain, per your instructions."
        )

    async def _generate(self, prompt: str) -> str:
        comp = await self._llm.generate(
            [Message("system", _SYSTEM), Message("user", prompt)],
            GenerateOptions(model=self.model, max_tokens=200, temperature=0.4),
        )
        return comp.text

    async def aclose(self) -> None:
        pass
