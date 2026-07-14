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
import os

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

_MOVE_SYSTEM = (
    "You are a chess coach reacting to the move the player JUST made in a play-along. You are given "
    "the move, the engine's verdict on it (its class, whether it was best, the refutation if it was a "
    "mistake), and the engine's grounded read of the RESULTING position. In 1-2 short sentences: name "
    "the move they played, say plainly whether it's strong/best or a mistake (grounded ONLY in the "
    "verdict), give the one key reason, then point toward what to do NEXT — a nudge, not the full "
    "answer. Ground everything ONLY in the facts — never invent a piece, square, line, or number; "
    "translate evals into plain words. PERSPECTIVE: address the player as 'you' (they made the move); "
    "the opponent is the other colour, and threats belong to the opponent. Warm and direct. "
    "THE PIECE ROSTER in the fact sheet is the COMPLETE and ONLY truth about where pieces are: every "
    "piece and square you name MUST appear in it exactly; if a piece is not in the roster, it does not "
    "exist — never invent one or move one to a square it isn't on. Return JSON: {\"text\": string}."
)


def _say_beat(text: str, tone: str = "teach") -> dict:
    return {"kind": "say", "tone": tone, "segments": [{"text": text}], "stops": False}


def _ask_beat(text: str, hints: list[str] | None = None) -> dict:
    b: dict = {"kind": "ask", "segments": [{"text": text}], "stops": True}   # stops -> gate locks
    if hints:
        b["hints"] = hints
    return b


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _eval_line(facts: dict) -> str:
    """The legacy briefing's opening verdict — turn + who's better, all White-POV, plain words."""
    stm = facts.get("side_to_move", "white")
    turn = "White" if stm == "white" else "Black"
    wp_stm = (facts.get("eval") or {}).get("win_pct", 50.0)
    white_wp = wp_stm if stm == "white" else 100 - wp_stm
    d = white_wp - 50
    if abs(d) <= 5:
        verdict = "the position is roughly equal"
    else:
        leader = "White" if d > 0 else "Black"
        aw = white_wp if d > 0 else 100 - white_wp
        band = ("is completely winning" if aw >= 90 else "is winning" if aw >= 75
                else "is clearly better" if aw >= 60 else "is slightly better")
        verdict = f"{leader} {band}"
    return f"{turn} to move; {verdict}."


def _ground(facts: dict) -> str:
    """The coach's grounded briefing — the legacy `build_analysis` shape that worked well: an eval
    verdict, the material standing, the four strategic standings (king-safety/activity/pawns/center,
    leads first) in plain English, then the tactical facts. All White-POV so the coach never flips
    perspective. Plus the exact piece ROSTER (belt-and-suspenders: the coach can verify every piece it
    names — this is the ONLY truth about where pieces are). Never a truncated JSON dump."""
    if not facts:
        return "(no position on the board yet)"
    out = [_eval_line(facts)]
    if (m := facts.get("material") or {}).get("standing"):
        out.append(_cap(m["standing"]) + ".")
    pos = facts.get("positional") or {}
    terms = pos.get("terms") or {}
    leads = pos.get("leads") or []
    strategic = ["king_safety", "activity", "pawns", "center"]
    ordered = [t for t in leads if t in strategic] + [t for t in strategic if t not in leads]
    for t in ordered:
        if st := (terms.get(t) or {}).get("standing"):
            out.append(_cap(st) + ".")
    if tac := facts.get("facts") or []:
        out.append("Tactics: " + "; ".join(f.get("text", "") for f in tac) + ".")
    pieces = facts.get("pieces") or []
    white = ",".join(sorted(p.get("piece", "") + p.get("square", "")
                            for p in pieces if p.get("color") == "white"))
    black = ",".join(sorted(p.get("piece", "") + p.get("square", "")
                            for p in pieces if p.get("color") == "black"))
    out.append(f"Pieces on the board (verify every piece you name against this) — "
               f"White: {white}; Black: {black}")
    return "\n".join(out)


def _ground_move(verdict: dict) -> str:
    """Compact grounding for a move the player made. The engine's CLASS is authoritative — a 'best'/
    'ok'/'only_move' move is GOOD (say so); only a dubious/mistake/blunder is a mistake. Only surface
    a better move / refutation when the move was actually worse than best, so an innocent move is never
    dressed up as punishable."""
    played = verdict.get("san")
    cls = verdict.get("class")
    is_mistake = cls in ("dubious", "mistake", "blunder")
    out = [f"Move played: {played}",
           f"Engine class: {cls} — AUTHORITATIVE, do not re-judge "
           f"({'a mistake' if is_mistake else 'a good move — treat it as good'})",
           f"Change in win%: {verdict.get('delta_win_pct')}"]
    best = verdict.get("best") or {}
    if is_mistake and best.get("san") and best["san"] != played:
        out.append(f"A better move was: {best['san']} ({' '.join(best.get('pv_san') or [])})")
    if is_mistake and (ref := verdict.get("refutation_pv") or []):
        out.append(f"How the opponent punishes it: {' '.join(ref)}")
    if cap := verdict.get("captured"):
        out.append(f"The move captured a {cap}")
    return "\n".join(out)


class Orchestrator:
    """The deterministic coaching pipeline. Depends only on: the state machine (in-process),
    the engine gRPC client (grounding), and the LLM adapter (generation). No MCP, no engine
    imports — the open/closed firewall holds."""

    def __init__(self, *, store, engine, model: str, llm: LLMAdapter | None = None):
        self.store = store          # StateStore
        self.engine = engine        # EngineClient (gRPC)
        self.model = model
        self.rating = int(os.environ.get("LUCENA_RATING", "1500"))   # player level for Maia traps
        self._llm: LLMAdapter = llm or make_adapter({"provider": "gemini", "default_model": model})
        self._last_tokens: dict | None = None

    async def _arm_poisoned(self, fen: str | None) -> None:
        """Run Maia poisoned-line detection for `fen` and arm/clear the board's trap slot, then
        re-publish the board so `has_poisoned_line` projects. No-op / clears if Maia is unavailable."""
        if not fen:
            return
        try:
            res = await asyncio.to_thread(self.engine.poisoned_line, fen, self.rating)
        except Exception:  # noqa: BLE001 — Maia unavailable / detection failure -> just no trap
            self.store.clear_poisoned(fen)
            return
        if res.get("has_poisoned_line") and res.get("poisoned_line"):
            self.store.set_poisoned(fen, res["poisoned_line"],
                                    meta={"fatal": res.get("fatal"), "idea": res.get("idea")})
            self.store.write_board(fen)   # re-project the board so has_poisoned_line publishes
        else:
            self.store.clear_poisoned(fen)

    async def run_turn(self, session_id: str, text: str | None = None) -> dict:
        self.store.read_input()                     # consume the mailbox (parity)
        if not (text and text.strip()):
            return {"ok": True, "orchestrated": False, "flow": "unhandled"}
        text = text.strip()
        self.store.publish_status("Thinking…")      # coach-working signal (drives the app halo)
        try:
            # Deterministic: the LLM NEVER sets the board (it hallucinates). A pasted FEN is detected
            # by the engine and set here — write_board publishes the new position, so the app re-renders.
            detected = await asyncio.to_thread(self.engine.detect_fens, text)
            if detected:
                self.store.write_board(detected[-1])
                # A pasted position starts a fresh line — the board orientation anchors on it.
                self.store.write_history([{"n": 0, "san": "", "uci": "", "fen": detected[-1]}])
                await self._arm_poisoned(detected[-1])   # is there a trap here for this player?
            fen = self.store.board_view
            if self.store._gate_awaiting:           # mid-probe -> grade the answer
                return await self._probe_answer(fen, text)
            return await self._coach(fen, text)
        finally:
            self.store.publish_status(None)         # clear it AFTER the beat is pushed

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
            f"{_ground(facts)}\n"
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

    async def coach_move(self, fen_before: str, uci: str) -> dict:
        """The player made a move. Judge it against the engine and coach the next step — the beat that
        keeps a play-along going (correct/mistake + what to do next), grounded in the fact sheet."""
        self.store.publish_status("Thinking…")
        try:
            verdict = await asyncio.to_thread(self.engine.evaluate, fen_before, [uci])
            after = self.store.board_view
            await self._arm_poisoned(after)   # trap in the resulting position?
            facts = await asyncio.to_thread(self.engine.analyze, after) if after else {}
            san = verdict.get("san") or uci
            # PERSPECTIVE: the player is whoever was to move BEFORE the move; after it, it's the
            # opponent's turn — so the fact sheet (side-to-move POV) is the OPPONENT's view.
            player = "White" if fen_before.split()[1] == "w" else "Black"
            opp = "Black" if player == "White" else "White"
            out = await self._gen_json(
                _MOVE_SYSTEM,
                f"You are coaching the {player} player. You (playing {player}) just played {san}.\n\n"
                f"Engine verdict on YOUR move — THIS is the reason it's good or bad. Judge from the "
                f"class + (if a mistake) the opponent's refutation. Do NOT use the fact sheet's "
                f"'after a pass' threats as the reason.\n{_ground_move(verdict)}\n\n"
                f"After your move it is now {opp}'s turn (your OPPONENT). The fact sheet below is from "
                f"{opp}'s side-to-move point of view: 'your' in it means {opp} (the opponent), NOT you. "
                f"A hanging piece belongs to its OWN colour — a {opp}-coloured piece is the opponent's, "
                f"a {player}-coloured piece is yours. Never tell the player their own piece hangs when "
                f"it is the opponent's.\n"
                f"Fact sheet (position after your move, {opp} to move):\n{_ground(facts)}\n\n"
                f"React as JSON.")
            body = out.get("text") or f"You played {san}."
            tone = "praise" if str(verdict.get("class")) in ("best", "ok", "only_move", "brilliant") else "correct"
            self.store.append_beats([_say_beat(body, tone=tone)])
            return {"ok": True, "orchestrated": True, "flow": "coach_move",
                    "class": verdict.get("class"), "tokens": self._last_tokens}
        finally:
            self.store.publish_status(None)

    async def _probe_answer(self, fen: str | None, text: str) -> dict:
        """The player answered a probe. Grade vs the engine's read, give feedback, unlock the gate."""
        facts = await asyncio.to_thread(self.engine.analyze, fen) if fen else {}
        verdict = await self._gen_json(
            _GRADE_SYSTEM,
            f"Player's answer: {text}\n\nEngine's grounded analysis (grade ONLY against this):\n"
            f"{_ground(facts)}\n\nGrade and give feedback as JSON.")
        correct = bool(verdict.get("correct"))
        feedback = verdict.get("feedback") or "Let's look at that together."
        self.store.append_beats([_say_beat(feedback, tone="praise" if correct else "correct")])
        self.store.set_gate(False)                  # answered -> unlock
        # (mastery recording parked this cycle; the seam is here — record vs verdict.concept/quality.)
        return {"ok": True, "orchestrated": True, "flow": "probe_answer",
                "correct": correct, "tokens": self._last_tokens}

    # -- generation ----------------------------------------------------------

    async def _gen_json(self, system: str, prompt: str) -> dict:
        if os.environ.get("LUCENA_DEBUG_PROMPT"):
            print(f"\n===== LLM PROMPT =====\n--- SYSTEM ---\n{system}\n\n--- USER ---\n{prompt}\n"
                  f"======================", flush=True)
        comp = await self._llm.generate(
            [Message("system", system), Message("user", prompt)],
            GenerateOptions(model=self.model, schema=_JSON_OBJECT, max_tokens=400, temperature=0.4),
        )
        if os.environ.get("LUCENA_DEBUG_PROMPT"):
            print(f"--- RESPONSE ---\n{comp.text}\n======================", flush=True)
        self._stash_tokens(comp.usage)
        return comp.json or {}

    def _stash_tokens(self, usage) -> None:
        self._last_tokens = ({"input": usage.input_tokens, "output": usage.output_tokens,
                              "total": usage.total_tokens} if usage is not None else None)

    async def aclose(self) -> None:
        return None
