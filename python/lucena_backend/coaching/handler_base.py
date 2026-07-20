"""Shared plumbing for the two mode handlers (freeform, coach).

Holds the LLM/grounding/beat helpers both handlers use, so neither re-implements the
`_gen_json` → beats pattern (the drift this whole redesign removes). Mirrors today's
Orchestrator helpers; the handlers add only their own control flow on top.
"""

from __future__ import annotations

import asyncio
import logging
import os

from ..llm import GenerateOptions, LLMAdapter, Message

_JSON_OBJECT = {"type": "object"}
_log = logging.getLogger(__name__)


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
        try:
            comp = await self._llm.generate(
                [Message("system", system), Message("user", prompt)],
                GenerateOptions(model=self.model, schema=_JSON_OBJECT, max_tokens=400, temperature=0.4))
        except Exception as exc:  # noqa: BLE001 — an LLM outage / rate-limit (429) must never break the
            # turn: return empty so every caller falls back to its grounded text (`out.get('text') or …`,
            # the verdict loop's `_safe_verdict`). The adapter already retried transient blips; a sustained
            # quota exhaustion needs graceful degradation, not a crash that flashes the app's "Uh oh".
            _log.warning("LLM generate failed; coach degrades to grounded fallback: %s", exc)
            return {}
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
