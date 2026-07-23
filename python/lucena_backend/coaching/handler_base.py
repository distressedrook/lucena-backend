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

    async def _gen_json(self, system: str, prompt: str, *, max_tokens: int = 400,
                        temperature: float = 0.4) -> dict:
        if os.environ.get("LUCENA_DEBUG_PROMPT"):
            print(f"\n===== LLM PROMPT =====\n--- SYSTEM ---\n{system}\n\n--- USER ---\n{prompt}\n"
                  f"======================", flush=True)
        try:
            comp = await self._llm.generate(
                [Message("system", system), Message("user", prompt)],
                GenerateOptions(model=self.model, schema=_JSON_OBJECT, max_tokens=max_tokens,
                                temperature=temperature))
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

    # -- SAN on the wire (guarded seam #1: the input boundary) -------------------------------------
    def _san(self, pre_fen, inp, probe: dict | None = None) -> str:
        """Resolve the played move to SAN, deterministically — the ONE place a
        raw UCI is allowed to exist on its way into any prose/prompt. Order:
        probe san → wire san → board conversion. NEVER returns UCI: if all
        else fails it converts on the board or raises (loud beats leaked)."""
        san = (probe or {}).get("san") or inp.san
        if san:
            return san
        if inp.uci and pre_fen:
            from lucena_core.board import Board
            return Board(pre_fen).san(inp.uci)     # raises on garbage — good
        raise ValueError(f"cannot resolve SAN for move {inp.uci!r} (no pre-move fen)")

    # -- beats (the player-visible channel; the Outcome is the other channel) ----------------------
    def _say(self, text: str, *, tone: str = "teach", stops: bool = False) -> None:
        if not text:
            return
        # Guarded seam #2 (the output net): NOTHING UCI-shaped reaches a beat,
        # whatever its origin — a call-site slip, a fact leak, an LLM echo.
        from .grounding import san_guard
        san_guard(text)
        self.store.append_beats([{"kind": "say", "tone": tone, "stops": stops,
                                  "segments": [{"text": text}]}])

    def _status(self, text: str | None) -> None:
        self.store.publish_status(text)
