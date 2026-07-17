"""Shared plumbing for the two mode handlers (freeform, coach).

Holds the LLM/grounding/beat helpers both handlers use, so neither re-implements the
`_gen_json` → beats pattern (the drift this whole redesign removes). Mirrors today's
Orchestrator helpers; the handlers add only their own control flow on top.
"""

from __future__ import annotations

import asyncio
import os

from ..llm import GenerateOptions, LLMAdapter, Message

_JSON_OBJECT = {"type": "object"}


class HandlerBase:
    def __init__(self, *, ctx, store, llm: LLMAdapter, model: str, ground=None):
        self.ctx = ctx
        self.store = store
        # Read-only grounding context (separate engine/locks) so coach analysis never blocks an
        # interactive move — same split as today's Orchestrator `ground_ctx`.
        self.ground = ground or ctx
        self.model = model
        self._llm = llm

    async def _gen_json(self, system: str, prompt: str) -> dict:
        if os.environ.get("LUCENA_DEBUG_PROMPT"):
            print(f"\n===== LLM PROMPT =====\n--- SYSTEM ---\n{system}\n\n--- USER ---\n{prompt}\n"
                  f"======================", flush=True)
        comp = await self._llm.generate(
            [Message("system", system), Message("user", prompt)],
            GenerateOptions(model=self.model, schema=_JSON_OBJECT, max_tokens=400, temperature=0.4))
        if os.environ.get("LUCENA_DEBUG_PROMPT"):
            print(f"--- RESPONSE ---\n{comp.text}\n======================", flush=True)
        return comp.json or {}

    async def _ground(self, fen, *, focus="analysis") -> dict:
        """Opt-in grounding — the caller decides WHEN. Runs on the read-only ground ctx off-thread."""
        if not fen:
            return {}
        return await asyncio.to_thread(self.ground.analyze_and_show, fen,
                                       focus=focus, board_push=False)

    # -- beats (the player-visible channel; the Outcome is the other channel) ----------------------
    def _say(self, text: str, *, tone: str = "teach", stops: bool = False) -> None:
        if not text:
            return
        self.store.append_beats([{"kind": "say", "tone": tone, "stops": stops,
                                  "segments": [{"text": text}]}])

    def _status(self, text: str | None) -> None:
        self.store.publish_status(text)
