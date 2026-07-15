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

import asyncio
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
    def __init__(self, *, ctx, model: str, llm: LLMAdapter | None = None, ground_ctx=None):
        self.ctx = ctx              # ToolContext (in-process coach)
        # Read-only grounding on a SEPARATE engine + lock, so "Why?" analysis never blocks a move.
        self.ground = ground_ctx or ctx
        self.store = ctx.store      # StateStore
        self.model = model
        self._llm: LLMAdapter = llm or make_adapter({"provider": "gemini", "default_model": model})

    async def explain(self, *, session_id: str, fen: str, move: str | None = None,
                      correct: bool | None = None) -> dict:
        """Bind the chat, then explain. `session_id` was accepted and ignored for the whole
        single-chat era; binding it is what keeps this turn's beats in this chat."""
        with self.store.bound(session_id):
            return await self._explain(fen=fen, move=move, correct=correct)

    async def _explain(self, *, fen: str, move: str | None = None,
                       correct: bool | None = None) -> dict:
        # Ground: a move → its refutation via evaluate; a bare position → the grounded briefing. Raise
        # the working halo for the whole call (engine eval + one generation ≈ several seconds) so the
        # app shows "thinking" instead of appearing frozen; clear it on every exit path.
        self.store.publish_status("Thinking…")
        try:
            if move:
                facts = await asyncio.to_thread(self.ground.evaluate, fen, [move])
            else:
                facts = await asyncio.to_thread(self.ground.analyze_and_show, fen,
                                                focus="analysis", board_push=False)
            prompt = self._prompt(fen, move, correct, facts)
            text = await self._generate(prompt)
            self.store.append_beats([{
                "kind": "say", "tone": "correct" if correct is False else "teach",
                "segments": [{"text": text}], "stops": False,
            }])
            return {"ok": True, "text": text}
        finally:
            self.store.publish_status(None)

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
