"""Coach mode handler (LLD §1.2, LLD-D) — socratic, board-constrained.

A move = a bit answer; text is classified {answer, whatif, stop, general}. Adjudication is
deterministic (or LLM-graded free_text), bool-only; "why wrong" is a SEPARATE call. On the
last required bit clearing: write meta=solved (once), bank mastery (dormant hook today),
closing beat + poisoned reveal, type-aware "another?" nudge.
"""

from __future__ import annotations

import asyncio
import contextvars

from .bits import BitProgress
from .grounding import (
    _brief_move, _brief_reply, _created_threat, _deep_tactics, _draws_by_stalemate, _invented_moves,
    _node_at, _node_solutions, _numbered, _solution_moves, _why_loses, tiered_bit_grounding,
    you_move_beat)
from ..reasoning import undermines_defender
from .handler_base import HandlerBase
from .lesson import ACTIVE, LessonProgress, puzzle_lesson_id, puzzle_spec
from .loop import Handled, Open, Outcome, Suspend
from .mode_prompts import (
    CoachTurnPrompt, OpponentReplyPrompt, PositionQueryPrompt, TrapPrompt, VerdictPrompt)
from .prompts import GradePrompt
from .strategies import build_registry

# The last coach adjudication result, for a SYNCHRONOUS caller that needs it back — the REST `/move`
# endpoint the mac app uses (it drives Retry / the poisoned-line button off {drill,correct,finished}).
# A ContextVar so it is per-execution-context (never crosses chats) and propagates back through the
# awaited call (await preserves context; the WS fire-and-forget path simply never reads it).
move_result: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "coach_move_result", default=None)


def _normalize_promotion(fen: str | None, uci: str | None) -> str | None:
    """A bare promotion uci ('b7b8') is read by the engine as an UNDERpromotion to KNIGHT — so a queen
    promotion came out as 'b8=N' in the verdict. The app now sends the piece ('b7b8q'), but default a
    suffix-less pawn-to-last-rank move to QUEEN so an old client / stray input can't silently
    underpromote. Leaves any move that already carries a suffix (or isn't a promotion) untouched."""
    if not fen or not uci or len(uci) != 4:
        return uci
    try:
        from lucena_engine.board import Board
        src, dst_rank = uci[:2], uci[3]
        if dst_rank not in ("1", "8"):
            return uci
        piece = next((p for p in Board(fen).piece_list() if p.square == src), None)
        if piece and piece.piece.upper() == "P":
            return uci + "q"
    except Exception:
        pass
    return uci


def _color_to_move(fen: str) -> str | None:
    """"white"/"black" from a FEN's active-colour field, or None if unreadable. In a verdict the
    answer's FEN is the PRE-move position, so this is the player's own colour."""
    parts = (fen or "").split()
    if len(parts) < 2 or parts[1] not in ("w", "b"):
        return None
    return "white" if parts[1] == "w" else "black"


class CoachHandler(HandlerBase):

    def __init__(self, *, ctx, store, llm, model, ground=None):
        super().__init__(ctx=ctx, store=store, llm=llm, model=model, ground=ground)
        self._strategies = build_registry(grade_fn=self._grade)

    # -- turn dispatch -----------------------------------------------------------------------------
    async def handle(self, inp) -> Outcome:
        lesson = self.store.active_lesson()
        if lesson is None:                       # defensive: loop only routes here when one is active
            return Handled()
        bit = lesson.current_bit()
        # Working status is owned by the loop (turn boundary), not here — see FreeformHandler.handle.
        if inp.kind == "present":                 # synthetic entry action (from loop re-route)
            self._reset_drill(lesson, bit)        # a presented drill starts from the top
            await self._present_bit(lesson.current_bit())
            return Handled()
        if inp.kind == "move":
            return await self._adjudicate(lesson, bit, inp)
        if inp.kind == "continue":                # the player clicked Continue → walk the next branch
            return await self._continue_branch(lesson, bit)
        j = await self._gen_json(CoachTurnPrompt.system(bit),
                                 CoachTurnPrompt.prompt(text=inp.text, bit=bit,
                                                    convo=self._recent(6)))
        intent = (j.get("intent") or "general").lower()
        if intent == "answer" and bit.spec.strategy == "free_text":
            return await self._adjudicate(lesson, bit, inp)
        if intent == "whatif":
            return Suspend(then=inp)          # loop → freeform explores; one nudge back
        if intent == "stop":
            return Open()                     # loop → freeform home
        # general: a position question mid-solve → the SHARED PositionQueryPrompt, grounded on
        # solve_text() so it can't leak the solution. Answer inline, STAY active, nudge back.
        grounding = await self._ground_for_bit(bit)
        ans = await self._gen_json(PositionQueryPrompt.system(freeform=False),
                                   PositionQueryPrompt.prompt(text=inp.text,
                                                              facts=grounding.solve_text()))
        self._say(ans.get("text") or "")
        self._say("Back to it — " + (bit.spec.challenge or "your move."), tone="teach")
        return Handled()

    # -- adjudication + conclusion -----------------------------------------------------------------
    async def _adjudicate(self, lesson, bit, inp) -> Outcome:
        strat = self._strategies.get(bit.spec.strategy)
        if strat is None:
            self._say("This exercise type isn't wired yet.")
            return Handled()
        if inp.kind == "move" and inp.uci and inp.fen:              # bare promotion 'b7b8' → queen, not knight
            inp.uci = _normalize_promotion(inp.fen, inp.uci)
        if inp.kind == "move" and inp.san is None and inp.fen:      # canonical SAN for adjudication
            inp.san = self.ground.san_of(inp.fen, inp.uci) or None
        grounding = await self._ground_for_bit(bit)
        # Hand the walker the session's current move line — it lives in the document history, not in
        # the serialized walker state, so `restore` needs it to keep the line coherent across moves.
        correct, new_prog, effects = await strat.adjudicate(
            inp, grounding, bit.spec, bit.progress, history=self.store._history)
        lesson.set_bit_progress(bit.index, new_prog)
        self.store.save_lesson_progress(lesson.progress)
        self._apply_board_effects(effects)                          # move + auto-played opponent reply

        # Echo the player's OWN move as a "you played" bubble (verdict badge + clickable chip) before
        # the coach speaks — a board move never came through the turn path, so nothing else emits it.
        # Only for a board move; a typed free_text answer was already echoed by the turn handler.
        if inp.uci and inp.fen:
            self.store.append_beats([you_move_beat(inp.fen, inp.uci, inp.san, correct=correct,
                                                   client_id=inp.client_id)])

        # Record the result for a synchronous caller (REST /move) BEFORE the slow why-wrong narration,
        # so the app gets {drill,correct,finished} promptly (the board already moved via the stream).
        finished = bool(new_prog.cleared and lesson._all_required_cleared())
        # A branch is solved but sibling defences remain — HELD behind a Continue button. The app reads
        # `await_continue` from this result to surface the button; the board stays on the solution.
        await_continue = bool((effects or {}).get("await_continue")) if correct else False
        move_result.set({"drill": True, "correct": bool(correct), "finished": finished,
                         "await_continue": await_continue})

        # Two beats on a correct mid-line move: (1) the verdict — what YOU did; (2) the opponent's
        # auto-played reply — what THEY did. Independent grounding, so generate concurrently (one LLM
        # round-trip, not two) and say in order. A wrong or line-ending move has no reply.
        player_color = _color_to_move(inp.fen) if inp.fen else None
        reply = (effects or {}).get("reply") if correct else None
        if reply:
            vtext, rtext = await asyncio.gather(
                self._verdict_text(inp, grounding, correct, player_color, bit),
                self._reply_text(reply, player_color))
            self._say(vtext, tone="praise")
            self._say(rtext, tone="teach")
        elif await_continue:
            # Solved this branch — offer to walk the next sibling defence. The board STAYS on the
            # solution; the backtrack runs only when the player clicks Continue (→ inp.kind "continue").
            self._say(await self._verdict_text(inp, grounding, correct, player_color, bit), tone="praise")
            self._say("That defence is handled — but your opponent has other tries. Continue?",
                      tone="teach")
        else:
            self._say(await self._verdict_text(inp, grounding, correct, player_color, bit),
                      tone="praise" if correct else "correct")

        if new_prog.cleared:
            if finished:
                return await self._conclude(lesson)
            lesson.advance_currentBit()
            self.store.save_lesson_progress(lesson.progress)
            await self._present_bit(lesson.current_bit())
        return Handled()

    async def _continue_branch(self, lesson, bit) -> Outcome:
        """The player clicked Continue — walk the next sibling defence (the deferred backtrack). Paint
        the branch-point position + the opponent's new defence, then announce it. No-op if nothing was
        actually held (defensive: a stale/duplicate click)."""
        strat = self._strategies.get(bit.spec.strategy)
        if strat is None or not hasattr(strat, "continue_branch"):
            return Handled()
        new_prog, effects = await strat.continue_branch(bit.spec, bit.progress, history=self.store._history)
        lesson.set_bit_progress(bit.index, new_prog)
        self.store.save_lesson_progress(lesson.progress)
        finished = bool(new_prog.cleared and lesson._all_required_cleared())
        move_result.set({"drill": True, "correct": True, "finished": finished, "await_continue": False})
        if effects:
            self._apply_board_effects(effects)     # NOW paint the sibling position (the branch-point walk)
            if reply := effects.get("reply"):
                self._say(await self._reply_text(reply, _color_to_move(reply.get("from_fen"))),
                          tone="teach")
        if new_prog.cleared and finished:
            return await self._conclude(lesson)
        return Handled()

    async def _conclude(self, lesson) -> Outcome:
        lesson.mark_solved()                       # WRITE-ONCE
        self.store.save_lesson_progress(lesson.progress)
        self._bank_mastery(lesson)                 # dormant hook (mastery not wired today)
        self._say("Solved — nicely done.", tone="praise")
        # §5 moment 3: the REVEAL is a reveal_on_resolve fact surfaced through §4's tiered grounding —
        # the same mechanism that withheld it during the solve. The facts stay structured in grounding;
        # a prompt VOICES them (never `_say` the raw template — that read as a data dump).
        reveal = tiered_bit_grounding(None, self._lesson_tree(lesson)).reveal_text()
        if reveal:
            out = await self._gen_json(TrapPrompt.system(reveal=True),
                                       TrapPrompt.prompt(facts=reveal))
            self._say(out.get("text") or reveal, tone="teach")
        self._say(self._another_nudge(lesson.spec.type), tone="teach", stops=True)
        return Handled()                           # meta=solved drops it from active → next turn freeform

    @staticmethod
    def _lesson_tree(lesson) -> dict:
        for bs in lesson.spec.bits:
            if bs.strategy == "move_line" and (bs.params or {}).get("tree"):
                return bs.params["tree"]
        return {}

    # -- entry / lesson creation -------------------------------------------------------------------
    async def enter(self, outcome) -> bool:
        """Create or resume a Lesson and ACTIVATE it in THIS chat (no beats — the loop re-routes a
        synthetic `present` so the entry action runs in the awaited flow). Returns True if a lesson is
        now active, False otherwise (not drillable / failed → loop stays in freeform). LLD §8 seam."""
        src = outcome.source or {}
        if src.get("kind") == "resume" and src.get("lesson_id"):
            if self.store.get_lesson(src["lesson_id"]) is None:
                return False
            self.store.activate_lesson(src["lesson_id"])      # binds state=active + this chat_id
            return True
        return await self._create_from_current(outcome)

    async def _create_from_current(self, outcome) -> bool:
        """Wrap the CURRENT position as a one-bit puzzle Lesson: reuse the paste-time-cached spec if
        present (library, position-keyed), else compute the forcing-line tree now (calculation, not
        authoring) and cache it. Create fresh progress bound to THIS chat, activate. Not drillable →
        False (freeform keeps the position)."""
        fen = self.store.board_view
        if not fen:
            return False
        spec = self.store.get_library_spec(puzzle_lesson_id(fen))     # cache hit from the paste nudge
        if spec is None:
            preview = await asyncio.to_thread(self.ctx.preview_drill, fen)
            if not (isinstance(preview, dict) and preview.get("drillable")):
                return False
            spec = puzzle_spec(fen, preview["tree"], motif=outcome.motif)
            self.store.save_lesson_spec(spec)
        self.store.save_lesson_progress(
            LessonProgress(lesson_id=spec.id, state=ACTIVE, chat_id=self.store.current_sid,
                           bits=[BitProgress() for _ in spec.bits]))
        return True

    def _reset_drill(self, lesson, bit) -> None:
        """Presenting a drill starts it from the ROOT: clear any stale walker state so the student's
        first move is judged as the first move. A prior UNFINISHED attempt leaves `strategy_state`
        mid-line (the walker past move 1); re-presented, the board shows the root but the walker is
        deeper, so a correct first move gets adjudicated against the wrong node and read as a blunder.
        Board and walker must agree at the top. (Resuming mid-line belongs to the suspend flow, not a
        fresh presentation.)"""
        if bit.spec.strategy == "move_line" and bit.progress.strategy_state:
            lesson.set_bit_progress(bit.index, BitProgress())
            self.store.save_lesson_progress(lesson.progress)

    async def _present_bit(self, bit) -> None:
        """Deliver the bit's challenge (authored text, else a placeholder pending co-design), then the
        §5-moment-1 WARN (from §4's warn_only tier). If a trap is present, ALSO publish it to the app —
        set the store's poisoned slot + repaint — so the board carries `has_poisoned_line` and the app
        shows its "show poisoned line" button (and latches the line for reveal)."""
        tree = (bit.spec.params or {}).get("tree") or {}
        # Board to the line's ROOT — what the student sees must be the position their first move is
        # judged against (the walker was just reset to the root in `_reset_drill`; they must agree).
        root_fen = tree.get("fen") or (tree.get("root") or {}).get("fen")
        if root_fen:
            self.store.write_history([{"n": 0, "san": None, "uci": None, "fen": root_fen}])
            self.store.write_board(root_fen)
        self._say(bit.spec.challenge or "Find the best continuation here.", tone="teach", stops=True)
        if tree.get("has_poisoned_line"):
            warn = tiered_bit_grounding(None, tree).warn_text()
            if warn:
                out = await self._gen_json(TrapPrompt.system(reveal=False),
                                           TrapPrompt.prompt(facts=warn))
                self._say(out.get("text") or warn, tone="teach")
            fen = self.store.board_view
            moves = tree.get("poisoned_line_moves")
            if fen and moves:
                self.store.set_poisoned(fen, moves, tree.get("poisoned_line_meta"))
                self.store.write_board(fen)              # re-project → board.has_poisoned_line = True

    # -- helpers -----------------------------------------------------------------------------------
    async def _ground_for_bit(self, bit):
        """Tiered grounding (§4): `solve_text()` (always + warn) is all any solve-time consumer sees;
        the solution and the trap DETAIL (reveal_on_resolve) are structurally absent."""
        resp = await self._ground(self.store.board_view)
        return tiered_bit_grounding(resp, (bit.spec.params or {}).get("tree"))

    async def _grade(self, text, spec, grounding):
        # solve_text: the free_text grader sees NO reveal content (no solution leak into the grade).
        facts = grounding.solve_text() if hasattr(grounding, "solve_text") else str(grounding)
        j = await self._gen_json(GradePrompt.system(), GradePrompt.prompt(text, facts))
        return bool(j.get("correct")), j

    async def _verdict_text(self, inp, grounding, correct: bool, player_color: str | None, bit=None) -> str:
        # Symmetric feedback (right names the idea, wrong the flaw). Ground it in the PLAYED MOVE's
        # actual engine read — NOT `solve_text()`. §4 strips the solution from solve_text() so it can't
        # leak WHILE solving; but by verdict time the move is already on the board, so grounding "why
        # it's good/bad" on the move's real read is safe AND necessary — grounding the "right" case on
        # the stripped facts left the model nothing to explain from, so it invented a rationale.
        # `player_color` (the answer's PRE-move side to move) is the fixed perspective anchor — the
        # board now shows the opponent to move, so "you play the side to move" flips White/Black.
        facts = await self._move_facts(inp, correct, grounding, bit)
        attempt = inp.san or inp.uci or (inp.text or "")
        sys = VerdictPrompt.system(correct=correct, player_color=player_color)
        usr = VerdictPrompt.prompt(attempt=attempt, facts=facts)
        # DETERMINISTIC GUARD (prompt rules alone didn't hold): reject any verdict that NAMES a move
        # not in the facts / not the move played — flash-lite kept inventing one (a promotion as "Kb2",
        # a rook dance as "Ra4# mate"). Regenerate once; if it invents again, fall back to a move-free
        # sentence so a hallucinated move never reaches the player.
        for _ in range(2):
            out = await self._gen_json(sys, usr)
            text = out.get("text") or ""
            if text and not _invented_moves(text, facts, inp.san):
                return text
        return self._safe_verdict(correct, facts)

    def _safe_verdict(self, correct: bool, facts: str) -> str:
        """A move-free fallback when the model can't stop inventing a move — grounded in the eval swing
        only (no move token, so nothing to hallucinate)."""
        if correct:
            return "Right — that's the idea. Well played."
        swing = "This move throws the advantage away." if "losing" in (facts or "") \
            else "This move lets your advantage slip." if "equal" in (facts or "") \
            else "That's not the one."
        return f"Not quite — {swing[0].lower()}{swing[1:]} Look again."

    async def _reply_text(self, reply: dict, player_color: str | None) -> str:
        """Voice the opponent's auto-played reply (the second beat). Grounded on `_brief_reply` — the
        one move, its capture, check — so it states what happened without inventing a plan or motif."""
        san = reply.get("san") or ""
        # A BACKTRACK to a sibling defence: the board just jumped back to the branch point and the
        # opponent is trying a DIFFERENT defence. Announce it plainly (deterministic — no LLM, so it
        # can't hallucinate) so the jump reads as a new challenge, not a glitch (the "weird state").
        if reply.get("new_line"):
            numbered = _numbered(san, reply.get("from_fen"))
            return f"Your opponent tries {numbered}. Find the win again from here."
        facts = _brief_reply(reply.get("from_fen"), san)
        out = await self._gen_json(OpponentReplyPrompt.system(player_color=player_color),
                                   OpponentReplyPrompt.prompt(facts=facts))
        return out.get("text") or (f"Your opponent replies {san}." if san else "")

    async def _move_facts(self, inp, correct: bool, grounding, bit=None) -> str:
        """Grounding for the verdict. A MOVE answer → the engine's read of the move just played
        (`evaluate`): the full read when correct (nothing to hide, it is on the board), the refutation
        with the best move HIDDEN when wrong (explain the flaw without naming the solution). Add the
        safe positional read (the `always` tier — never the solution) for context. A TYPED answer
        (free_text, no move) has nothing to evaluate, so it keeps the solve-time facts."""
        if not inp.uci or not inp.fen:
            return grounding.solve_text() if hasattr(grounding, "solve_text") else str(grounding)
        verdict = await asyncio.to_thread(self.ground.evaluate, inp.fen, [inp.uci])
        tree = (bit.spec.params or {}).get("tree") if (bit and getattr(bit, "spec", None)) else None
        sols = _solution_moves(tree)
        after_fen = self._after_fen(inp.fen, inp.uci)

        async def _an(fen):
            return (await asyncio.to_thread(self.ground.analyze_and_show, fen,
                                            focus="analysis", board_push=False)) if fen else None
        if correct:
            # RIGHT — the POINT of the move, run CONCURRENTLY: the PRE-move linchpin / defensive resource
            # (`_deep_tactics` — why simple tries fail, the trap the win dodges) plus the decisive threat
            # the move CREATES (`_created_threat`, from the after-move read). Solution-stripped so neither
            # names the un-played next move.
            pre, post = await asyncio.gather(_an(inp.fen), _an(after_fen))
            verdict = {k: v for k, v in verdict.items() if k != "refutation_pv"}   # a best move has none
            move_read = _brief_move(verdict, hide_best=False)
            positional = "\n".join(str(x) for x in getattr(grounding, "always", []) or [])
            created = _created_threat((post or {}).get("analysis") or [], sols)
            # THE POINT: prefer the derived causal reason (the reasoner — 'attacks the only defender of
            # the a2 bishop') over the static deep-tactics defender fact; the two overlap and the derived
            # one is the causal, non-redundant version. Fall back to _deep_tactics when it doesn't fire.
            point = (undermines_defender(inp.fen, inp.uci)
                     or _deep_tactics((pre or {}).get("analysis") or [], sols))
            # Point FIRST — it is the reason the move is the move; the capture read and the generic
            # positional context are secondary and must not become the lead.
            return "\n".join(p for p in (point, move_read, created, positional or None) if p)
        # WRONG — ONLY what explains THIS failure, relevance-filtered (no motif dump):
        #   • `_brief_move` — class + eval SWING + the refutation line (the concrete why).
        #   • `_created_threat` — the decisive threat the move made, if any (from the after-move read).
        #   • `_why_loses` — the ONE instructive mechanism (a defender walked off, a guarded square) or a
        #     stalemate draw, with its 'right idea' / 'deal with X first' claims VALIDATED against the
        #     SOLUTION line: a real-but-irrelevant fork is not credited (P3). NO `_deep_tactics` here —
        #     dumping every structural motif ('g3 defends f2', 'f6 pinned') let the model force-fit an
        #     irrelevant one into a false cause; the refutation + mechanism already carry the why.
        post = await _an(after_fen)
        move_read = _brief_move(verdict, hide_best=True)
        created = _created_threat((post or {}).get("analysis") or [], sols)
        # The interpretive fork/'idea' credit is validated only when THIS position has a single best
        # move; the mechanism itself is derived regardless. Anchor on inp.fen (a wrong move can be deep
        # in a multi-ply drill), and count the solutions AT that node.
        sol_moves = _node_solutions(_node_at(tree, inp.fen))
        why = (_draws_by_stalemate(inp.fen, inp.uci, verdict.get("refutation_pv"))
               or _why_loses(inp.fen, inp.uci, verdict.get("refutation_pv"),
                             solution_ucis=sol_moves, single_solution=(len(sol_moves) == 1)))
        return "\n".join(p for p in (move_read, created, why) if p)

    @staticmethod
    def _after_fen(fen: str | None, uci: str | None) -> str | None:
        """The FEN after `uci` is played on `fen` — the board the move CREATES, for reading the threat
        it makes. Deterministic (a board apply, no engine). None if the move can't be applied."""
        from lucena_engine.board import Board
        try:
            return Board(fen).apply(uci).fen
        except Exception:
            return None

    def _apply_board_effects(self, effects) -> None:
        """Move the board to reflect a move_line bit's advance — the player's move AND the walker's
        auto-played opponent reply — plus the move line, matching the old drill behaviour. History
        BEFORE board (orientation anchors on history.first; board-first flips it for a frame)."""
        if not effects:
            return
        plies = effects.get("plies")
        if plies:
            self.store.write_history(plies)
        board = effects.get("board")
        if board:
            self.store.write_board(board)

    def _bank_mastery(self, lesson) -> None:
        """Deterministic mastery bank on solve (LLD §2.2). Dormant: ctx.mastery is None today, so this
        is a structural hook (matches _close_drill's current no-op behavior)."""
        mastery = getattr(self.ctx, "mastery", None)
        concept = lesson.spec.concept_id
        if mastery is not None and concept:
            try:
                mastery.record({"type": "recall", "concept": concept, "quality": 1.0,
                                "resolved": True, "note": "lesson solved"})
            except Exception:
                pass

    def _another_nudge(self, type_: str) -> str:
        return {"puzzle": "Want another puzzle?",
                "endgame": "Want another endgame?",
                "opening": "Want to look at another opening?",
                "midgame": "Want another middlegame position?"}.get(type_, "Want another?")

    def _recent(self, n: int = 6) -> str | None:
        lines = []
        for b in (self.store._beats or [])[-n:]:
            text = "".join(s.get("text", "") for s in (b.get("segments") or []))
            if text:
                lines.append(f"{'You' if b.get('kind') == 'you' else 'Coach'}: {text}")
        return "\n".join(lines) if lines else None
