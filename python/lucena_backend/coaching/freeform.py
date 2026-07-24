"""Freeform mode handler (LLD §1.1, LLD-B) — the home state.

A chess CONVERSATION loop: general Q&A, rejection, move-explain, position reads, the
puzzle nudge, and what-if handling. Never socratic. Dispatch by input kind:
`position`/`move` are deterministic-grounding triggers (always a grounded answer, no
classification); only `text` runs the classifier (the two-call flow).

Output is two channels: beats via the store; an `Outcome` returned to the loop.
"""

from __future__ import annotations

import asyncio
import logging

from lucena_core import openings
from lucena_core import theory

from .. import plans as _plans
from .book_voice import (_BOOK_RATING, _BOOK_REPLIES, _COACH, _ENDBOOK, _NARRATE, _book_route, _is_swing)
from .grounding import _brief, _brief_move, _numbered
from .handler_base import HandlerBase
from .lesson import puzzle_spec
from .loop import EnterCoach, Handled, Outcome
from .mode_prompts import FreeformPrompt, PositionQueryPrompt, ReadPrompt
from .prompts import EndbookPrompt, NarratePrompt

_log = logging.getLogger(__name__)


def _mover(fen) -> str:
    """The colour that just moved FROM `fen` — i.e. `fen`'s side to move."""
    return "Black" if (fen and " b " in f" {fen} ") else "White"


def _theory_text(entry: dict) -> str:
    """A Wikibooks theory entry as user-facing text: the opening name, the
    verbatim description, its named continuations, and the REQUIRED CC BY-SA
    attribution/link. Deterministic — no model touches this text (verbatim
    display keeps share-alike clear of our prose)."""
    name = entry.get("name") or "This position"
    lines = [f"**{name}**", "", (entry.get("description") or "").strip()]
    if resp := (entry.get("responses") or []):
        lines += ["", "Main continuations: " + "; ".join(resp[:5]) + "."]
    lines += ["", f"— Theory from Wikibooks (CC BY-SA): {entry['source_url']}"]
    return "\n".join(lines)


class FreeformHandler(HandlerBase):

    async def handle(self, inp) -> Outcome:
        # Working status is owned by the loop (turn boundary), not here — a routing hand-off must not
        # clear it mid-turn (that tripped the app's "Uh oh" silent-turn net).
        if inp.kind == "position":
            return await self._on_position(inp)
        if inp.kind == "move":
            return await self._on_move(inp)
        if inp.kind == "walk":
            return await self._on_walk(inp)
        return await self._on_text(inp)

    # -- deterministic-grounding triggers: ALWAYS a grounded answer, no classifier -----------------
    async def _on_position(self, inp) -> Outcome:
        # Set the board deterministically (FEN detection lives in set_board_from_paste).
        await asyncio.to_thread(self.ctx.set_board_from_paste, inp.fen or inp.text or "")
        return await self._read_position(self.store.board_view)

    async def _read_position(self, fen, *, played: str | None = None) -> Outcome:
        """THE position pipeline — paste and board-move both land here (owner
        ruling 2026-07-24: a move on the board goes through the same path as
        conversation). Route (2026-07-22): drillable → coach; quiet + out of
        book + MIDDLEGAME + equalish (|eval| <= 1.5) → the PLANS layer
        (lucena-plans sheet, presented deterministically — no LLM); everything
        else → the plain grounded read. `played` threads the just-played move
        into the plain read's prose; the in-book case never reaches here from
        the move path (the book-voice arm routes first).

        Drillable asymmetry (owner report 2026-07-24: "I was making a move,
        the system suddenly flipped and showed me a puzzle"): a PASTE of a
        tactical position is an implicit "coach me on this" → enter the
        drill. A MOVE is play in progress — and worse, post-move the side to
        move is the OPPONENT, so auto-entering flips the board and drills
        the player as the other side. So on the move path the drill is an
        INVITATION: the lesson spec is armed (a "drill it" reply enters
        instantly), a deterministic nudge says so, and the turn continues as
        a normal read.
        """
        # IN THEORY takes precedence over everything (owner: "if in theory,
        # show theory"). A book position shows its Wikibooks theory — not a
        # positional read, and not a tactical drill (which would otherwise
        # EnterCoach below and bury the theory). Verbatim + CC BY-SA
        # attribution, deterministic, no LLM. Only presented when we can
        # ATTRIBUTE it: CC BY-SA requires the source link, so an entry without
        # source_url is never shown (never quote the text uncredited).
        if (entry := theory.theory_for(fen)) is not None and entry.get("source_url"):
            self._say(_theory_text(entry))
            return Handled()
        preview = await asyncio.to_thread(self.ctx.preview_drill, fen)
        if isinstance(preview, dict) and preview.get("drillable"):
            self.store.save_lesson_spec(puzzle_spec(fen, preview["tree"]))
            if played is None:
                return EnterCoach(type="puzzle", source={"kind": "current"})
            side = "White" if " w " in f" {fen} " else "Black"
            self._say(f"Heads up — {side} has a forcing win in this position. "
                      f"Say drill it to work it out, or just keep playing.")
        if not openings.name_for(fen) and (text := await self._plans_read(fen)):
            self._say(text)
            return Handled()
        # A quiet (non-forcing) position → a freeform grounded read.
        facts = await self._ground(fen)
        out = await self._gen_json(ReadPrompt.system(),
                                   ReadPrompt.prompt(facts=_brief(facts), played=played),
                                   temperature=1.0)  # varied prose across repeated reads
        self._say(out.get("text") or ("" if played else "Let's take a look at this position."))
        return Handled()

    async def _plans_read(self, fen) -> str | None:
        """The PLANS layer: for a quiet, out-of-book MIDDLEGAME position inside the equalish band
        (|eval| <= 1.5 pawns — the zone where 'find a tactic' has no answer and the coaching value
        is a plan), roll engine lines from this position, hand (fen, pvs, rolls) to lucena-plans,
        and PRESENT the result deterministically. Returns None when any gate fails or the layer
        errors — the caller falls back to the plain grounded read; this path must never break the
        turn.

        NO LLM IS IN THIS PATH (owner ruling 2026-07-24). The read used to be narrated by
        `PlansReadPrompt`; it is now rendered by `lucena-plans` `position_read.render`. The
        motivating reason is not latency but CORRECTNESS: the reliability tier ("never mention an
        unverified plan") used to be an instruction inside a prompt — the only thing keeping an
        unconfirmed candidate away from a student was a sentence a model could drift from. It is
        now a filter in code."""
        if not fen or _plans.is_endgame(fen):
            return None
        # The eval gate reads the same cached analyse the fallback's _ground would run (focus="eval"
        # skips the fact sheet), so gating costs one search total either way.
        probe = await asyncio.to_thread(self.ground.analyze_and_show, fen,
                                        focus="eval", board_push=False)
        cp = ((probe or {}).get("eval") or {}).get("cp")
        if cp is None or abs(cp) > _plans.PLANS_CP_BAND:
            return None
        try:
            # The sanctioned out-of-tool lease (ToolContext.engine's own guidance): the roll is a
            # multi-search read on the ground pool, not a guarded tool call. The Maia leg is ALWAYS
            # on when Maia is configured (user ruling 2026-07-22: "Maia should never be off" —
            # human-typicality is part of the product, latency paid); verify degrades to the engine
            # leg only when the process has no MaiaEngine at all (LUCENA_MAIA unset).
            _pre, post = await asyncio.to_thread(
                _plans.sheet_json_for, fen, self.ground._pool, self.ctx.maia)
            return await asyncio.to_thread(_plans.render_position_read, post)
        except Exception:  # noqa: BLE001 — a missing checkout / engine hiccup degrades, never breaks
            _log.warning("plans layer unavailable; falling back to plain read", exc_info=True)
            return None

    async def _on_move(self, inp) -> Outcome:
        pre_fen = inp.fen or self.store.board_view
        # Apply the move (freeform: one ply, no reply, canned feedback suppressed).
        await asyncio.to_thread(self.ctx.play_move, inp.uci, inp.fen, push_feedback=False,
                                client_id=inp.client_id)
        fen = self.store.board_view
        # ROUTE FIRST, off a CHEAP class probe. Routing only needs the move's class (is it a swing?)
        # and its SAN — `assess_move` (~400ms) answers both. The full `evaluate` (2×1500ms) and the
        # position read (`_ground`, ~1.5s cold) are deferred to the branches that actually consume
        # them, so a normal opening move never pays either (perf pass: the move path was doing a full
        # analyze + evaluate up front on EVERY ply, cold, before the route was even known).
        probe = (await asyncio.to_thread(self.ground.assess_move, pre_fen, inp.uci)) if pre_fen else {}
        swing = _is_swing(probe)
        route, book, prev_book = _book_route(self._history_fens(), swing)
        # Ground ONLY where the branch reads it: a NARRATE on a swing, where the concession needs
        # the positional read. ENDBOOK and a normal in-book ply narrate from the opening name +
        # move line alone — no cold position search. The COACH branch grounds inside
        # _read_position (the unified pipeline) instead.
        facts = (await self._ground(fen)) if (route == _NARRATE and swing) else {}
        # Opening-book narration arm: if the played move is still in book, narrate it in the BOOK VOICE
        # (or hand off at end-of-book) rather than a plain grounded read. Reuses the whole existing
        # subsystem (book_voice routing + NarratePrompt/EndbookPrompt + the openings table + Maia
        # replies), just driven from here instead of the retired _coach_move/_narrate_move.
        if route in (_NARRATE, _ENDBOOK):
            # The full verdict (Δwin%, class, refutation the concession renders) only when there is a
            # swing to explain; otherwise the probe's san/class is all the book voice uses.
            verdict = (await asyncio.to_thread(self.ground.evaluate, pre_fen, [inp.uci])) if swing else \
                {"san": self._san(pre_fen, inp, probe), "class": probe.get("class")}
            await self._narrate_opening(route, inp, pre_fen, verdict, facts, book, prev_book)
            self._title_from_opening(book)
            return Handled()
        # else _COACH (out of book): the SAME pipeline as a conversation paste —
        # drillable → coach handoff, quiet equalish middlegame → plans read,
        # otherwise the plain grounded move-explain (owner ruling 2026-07-24).
        # SAN on the wire: the app sends UCI; the probe already converted it —
        # raw UCI must never reach prose (caught live: "**c7c6** was just played").
        return await self._read_position(fen, played=self._san(pre_fen, inp, probe))

    async def _on_walk(self, inp) -> Outcome:
        # Walking a variation: the app has ALREADY moved the shared analysis board onto a sideline move
        # (via the /view report), so — unlike `_on_move` — we do NOT re-apply it. We just ground the
        # position now on the board and read the move that reached it, exactly like a plain freeform
        # move-explain. Book narration is intentionally skipped: a variation is exploratory analysis, not
        # the game's opening line. Upstream has deduped + gated, so reaching here means "read this move".
        fen = inp.fen or self.store.board_view
        facts = await self._ground(fen)
        out = await self._gen_json(ReadPrompt.system(),
                                   ReadPrompt.prompt(facts=_brief(facts),
                                                     played=self._san(inp.fen, inp)),
                                   temperature=1.0)  # varied prose across repeated reads
        self._say(out.get("text") or "")
        return Handled()

    # -- opening-book narration (ported from the retired Orchestrator._narrate_move) ---------------
    async def _narrate_opening(self, route, inp, pre_fen, verdict, nxt, book, prev_book) -> None:
        mover = _mover(pre_fen)
        played = _numbered((verdict or {}).get("san") or self._san(pre_fen, inp), pre_fen)
        moves_so_far = self._played_line() or played
        if route == _ENDBOOK:
            prompt = EndbookPrompt.prompt(mover=mover, played=played, book=book,
                                          moves_so_far=moves_so_far, position_read=_brief(nxt))
            out = await self._gen_json(EndbookPrompt.system(), prompt)
        else:
            swing = _is_swing(verdict)
            replies = await self._book_replies()
            prompt = NarratePrompt.prompt(
                mover=mover, played=played, book=book, moves_so_far=moves_so_far, prev_book=prev_book,
                swing=swing, replies=replies, facts=_brief_move(verdict, hide_best=not swing),
                position_read=_brief(nxt) if nxt else None)
            out = await self._gen_json(NarratePrompt.system(prev_book), prompt)
        if body := (out or {}).get("text"):
            self._say(body, tone="teach")

    def _history_fens(self) -> list:
        return [f for p in (self.store._history or []) if (f := (p or {}).get("fen"))]

    def _played_line(self) -> str:
        """The game so far as "1.e4 c5 2.Nf3" — the anchor that stops narration inventing moves."""
        out = []
        for p in (self.store._history or []):
            if not (san := (p or {}).get("san")):
                continue
            parts = str((p or {}).get("fen", "")).split()
            full = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
            white_moved = (parts[1] if len(parts) > 1 else "w") == "b"
            num = full if white_moved else full - 1
            out.append(f"{num}.{san}" if white_moved else (f"{num}...{san}" if not out else san))
        return " ".join(out)

    async def _book_replies(self) -> list:
        """What strong players play in the position now on the board, as SAN — from MAIA (human
        practice = what theory IS), on `ctx` (the ground ctx has no Maia)."""
        fen = self.store.board_view
        if not fen:
            return []
        return await asyncio.to_thread(self.ctx.human_replies, fen, _BOOK_RATING, _BOOK_REPLIES)

    def _title_from_opening(self, name: str | None) -> None:
        """Title an unnamed session after the opening's FAMILY (deterministic, from the table)."""
        if name and self.store.session_unnamed:
            self.store.set_session_name(openings._path(name)[0][:60])

    # -- the ONLY classifier path --------------------------------------------------------------------
    async def _on_text(self, inp) -> Outcome:
        convo = self._recent_conversation()
        j = await self._gen_json(FreeformPrompt.system(),
                                 FreeformPrompt.prompt(text=inp.text, convo=convo))
        intent = (j.get("intent") or "general").lower()

        if intent == "needs_grounding":
            # A question ABOUT the position → grounded answer via the shared PositionQueryPrompt.
            # TODO(WhatIfPrompt): if the text names a concrete line, fold its playout into facts first.
            facts = _brief(await self._ground(self.store.board_view))
            ans = await self._gen_json(PositionQueryPrompt.system(freeform=True),
                                       PositionQueryPrompt.prompt(text=inp.text, facts=facts))
            self._say(ans.get("text") or "Let's look at the position together.")
            self._maybe_suspend_nudge()
            return Handled()

        outcome: Outcome = Handled()
        if intent == "dispatch_coach":
            coach = j.get("coach") or {}
            outcome = EnterCoach(type=coach.get("type") or "puzzle",
                                 motif=coach.get("motif"), source=coach.get("source"))
        else:  # reject | general — answered inline by the classifier (dual-role)
            self._say(j.get("text") or "Let's take a look.")

        self._maybe_suspend_nudge()
        return outcome

    # -- helpers -----------------------------------------------------------------------------------
    def _recent_conversation(self, n: int = 6) -> str | None:
        """Last `n` beats as plain lines — grounds continuity so a reply isn't a cold open, and lets the
        classifier resolve 'yes'/'this' against what was just said (OPEN-1)."""
        lines = []
        for b in (self.store._beats or [])[-n:]:
            text = "".join(s.get("text", "") for s in (b.get("segments") or []))
            if text:
                lines.append(f"{'You' if b.get('kind') == 'you' else 'Coach'}: {text}")
        return "\n".join(lines) if lines else None

    def _maybe_suspend_nudge(self) -> None:
        """One nudge back to a SUSPENDED, unsolved lesson, then stop (gated meta==null; once/suspension).
        TODO(seam): the once-per-suspension latch. Minimal version: nudge whenever a suspended lesson
        exists — refine the latch when the suspend flow is wired end-to-end."""
        # Placeholder: no-op until the suspend flow + latch are wired (avoids nagging every turn).
        return None
