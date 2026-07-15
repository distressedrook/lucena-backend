"""Orchestrator — the deterministic coaching pipeline over the in-process ToolContext.

Every turn: read_input (classify + ground) -> ONE LLM generation over the grounded facts -> push
the beat. The model never tool-calls, never sets the board. ToolContext (the full legacy coach) does
the grounding + drill walk + Maia; this drives it and adds the LLM's grounded voice.
"""

from __future__ import annotations

import asyncio
import re
import os

from .llm import make_adapter, Message, GenerateOptions, LLMAdapter

# A FEN-like token (a rank of piece letters/digits then a side-to-move) anywhere in pasted text.
_FEN_RE = re.compile(r"(?:[pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+\s+[wb]\b")

# Does the player's message look like it's asking about a CONCRETE move? (piece names, squares, SAN,
# capture/push verbs, "what if"). A cheap pre-filter before spending an LLM call to resolve the move.
_MOVE_Q_RE = re.compile(
    r"\b(knight|bishop|rook|queen|king|pawn|castl\w*|takes?|captur\w*|push\w*|recaptur\w*|sac\w*|"
    r"what\s+if|instead|play(s|ed|ing)?|move[sd]?|trade[sd]?|exchang\w*)\b"
    r"|\b[a-h][1-8]\b|\b[KQRBNO][-Ox]?[a-h]?[1-8]?[a-h][1-8][+#]?\b", re.I)

_EXTRACT_SYSTEM = (
    "You turn a player's what-if question into the sequence of THEIR OWN moves it proposes, in order. "
    "You are given the position and the legal moves (SAN) for the side to move. Return JSON "
    "{\"moves\": [SAN, ...]}: the player's intended moves in order — the opponent's in-between replies "
    "are NOT included (they get filled by best play). The FIRST move must be one of the legal moves "
    "listed. Use SAN. Return {\"moves\": []} if the question isn't about concrete move(s).")

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

_DRILL_SYSTEM = (
    "You are a chess coach guiding a player through a tactical drill (a forcing line). The deterministic "
    "correct/wrong/next-move feedback is already shown; YOUR job is the grounded WHY in 1-2 short "
    "sentences: from the engine's read of the CURRENT drill position, explain the idea the player should "
    "see and nudge them toward the next move WITHOUT naming it. Ground ONLY in the facts — never invent "
    "a piece, square, line, or number. Perspective: 'you' = the player (the side to move); threats "
    "belong to the opponent. Return JSON: {\"text\": string}."
)

_WRONG_SYSTEM = (
    "You are a chess coach. The player just played a WRONG move in a drill. From the engine's read of "
    "the move, explain in 1 short sentence why it doesn't work (the refutation) and encourage them to "
    "try again — WITHOUT giving away the right move. Ground ONLY in the facts. Perspective: 'you' = the "
    "player; threats belong to the opponent. Return JSON: {\"text\": string}."
)

_MOVE_SYSTEM = (
    "You are a chess coach reacting to a move the player JUST made. You are told whether they are in a "
    "drill and, if so, whether the move was CORRECT, WRONG, or SOLVED the drill; for a freeform move you "
    "get its engine class. You also get the move's grounded facts and (when relevant) the position they "
    "now face. Ground EVERY claim ONLY in those facts — never invent a piece, square, line, or number; "
    "translate evaluations to plain words, never cite win% or centipawns. Speak to the player as 'you'; "
    "the opponent is 'your opponent'; every threat belongs to the opponent. Keep it to 1-2 short "
    "sentences.\n"
    "- DRILL / CORRECT: confirm the move is right and name the idea it achieves, then point at what to "
    "look for NEXT without naming the next move.\n"
    "- DRILL / WRONG: say it isn't the move here and explain the flaw ONLY through the refutation line you "
    "are given — name the opponent's ACTUAL first refuting move (the first move of that line). Encourage "
    "another try; do NOT reveal the right move.\n"
    "- DRILL / SOLVED: celebrate finishing the forcing line and name the key idea that won.\n"
    "- FREEFORM: give the honest verdict (a strong move / fine / an inaccuracy / a mistake) grounded in "
    "the facts, say why, and if it was a mistake point toward the better idea.\n"
    "CRITICAL — do NOT invent a mechanism. Only describe what is actually in the given line: if the "
    "refutation is a queen move, do not call it a pawn push; if no piece is trapped in the line, do not "
    "say a piece is trapped; if there is no fork/pin in the line, do not name one. When you are unsure "
    "how the line works, say plainly that the engine refutes it and the position turns against you — "
    "never fill the gap with a plausible-sounding motif. Do not add generic strategic advice ('control "
    "the center', 'develop your pieces') that is not in the facts.\n"
    "Return JSON: {\"text\": string}."
)


_PIECE_WORD = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}


def _move_phrase(san: str) -> str:
    """Deterministic, plain-language nature of a SAN move — so the coach reads the refutation from
    FACTS, not by guessing the piece from a raw move-list (the confabulation that turned a queen move,
    Qe3, into an invented 'pawn push'). Names the piece and whether it captures/checks."""
    s = (san or "").rstrip("+#")
    piece = _PIECE_WORD.get(s[:1], "pawn") if s else "pawn"
    dest = s.split("x")[-1][-2:] if s else "?"
    verb = "captures on" if "x" in s else "moves to"
    tail = " with check" if san.endswith("+") else (" — checkmate" if san.endswith("#") else "")
    return f"a {piece} {verb} {dest}{tail}"


def _brief_move(v: dict, *, hide_best: bool = False) -> str:
    """Compact grounding for a played move — the engine's verdict on it. `hide_best` drops the solution
    move (used on a WRONG drill move, so the coach can't leak the answer while explaining the flaw)."""
    if not isinstance(v, dict) or v.get("error"):
        return "(no move read available)"
    out = [f"Move played: {v.get('san')}"]
    if v.get("captured"):
        out.append(f"It captures the {v['captured']}.")
    if v.get("class"):
        out.append(f"Engine class of this move: {v['class']}.")
    if not hide_best and (b := v.get("best")) and b.get("san") and b.get("san") != v.get("san"):
        out.append(f"The engine's best move here is {b['san']}.")
    if pv := v.get("refutation_pv"):
        first = pv[0]
        piece = _PIECE_WORD.get(first.rstrip("+#")[:1], "pawn")
        out.append(f"The opponent refutes it with {first} ({_move_phrase(first)}); the line then runs "
                   f"{' '.join(pv[:6])}. Explain the flaw ONLY through this line — the refuting move is "
                   f"{first}, a {piece} move, nothing else.")
    return "\n".join(out)


def _brief(resp: dict) -> str:
    """The grounded briefing to hand the model — the NL analysis lines ToolContext already produced
    (assemble_analysis), plus best move / poisoned-line note when present. Never a raw JSON dump."""
    if not isinstance(resp, dict) or resp.get("error"):
        return "(no grounded read available)"
    out = []
    lines = resp.get("analysis")
    if isinstance(lines, list) and lines:
        out.extend(lines)
    else:
        if resp.get("pieces"):
            out.append(str(resp["pieces"]))
        if (m := resp.get("material")):
            out.append(f"Material: {m.get('standing')}")
    if best := resp.get("best_san") or (resp.get("hints") or {}).get("best"):
        out.append(f"Engine best move: {best}")
    if resp.get("has_poisoned_line"):
        out.append("There is a poisoned line here — a tempting move that loses.")
    return "\n".join(str(x) for x in out) or "(no grounded read available)"


class Orchestrator:
    def __init__(self, *, ctx, model: str, llm: LLMAdapter | None = None, ground_ctx=None):
        self.ctx = ctx              # ToolContext for INTERACTIVE ops (drills, moves, arming, read_input)
        # Read-only grounding (evaluate/analyze/hints) for the LLM prompts runs on a SEPARATE engine +
        # lock so it never blocks an interactive move on `ctx`. Falls back to `ctx` if none supplied.
        self.ground = ground_ctx or ctx
        self.store = ctx.store
        self.model = model
        self._llm: LLMAdapter = llm or make_adapter({"provider": "gemini", "default_model": model})
        self._last_tokens: dict | None = None
        self._poisoned_shown: str | None = None   # latch: the poisoned line we've already narrated

    async def run_turn(self, session_id: str, text: str | None = None) -> dict:
        ctx = self.ctx
        # Echo what the player typed into the conversation as a "you" bubble (the app doesn't render it
        # locally) — so the chat reads as a dialogue, not just the coach's replies. A pasted FEN/PGN is
        # a board set-up, not a chat line, so it isn't echoed.
        if text and text.strip() and not _FEN_RE.search(text):
            self.store.append_beats([{"kind": "you", "stops": False,
                                      "segments": [{"text": text.strip()}]}])
        # Raise the working halo IMMEDIATELY (before any slow engine/LLM work) and clear it on every
        # exit path — so the app shows "thinking" the instant the turn starts, drill-arm included.
        self.store.publish_status("Thinking…")
        try:
            # Deterministic: a pasted FEN sets the board (the LLM never sets it). set_board_from_paste
            # does the FEN/PGN detection itself; we only reach for it when the text looks board-shaped.
            # Setting up a position is ALSO the moment to arm a drill — the legacy coach did this by
            # judgment; here it's deterministic: a forcing win becomes a drill, else it stays freeform.
            if text and _FEN_RE.search(text):
                self.store.publish_status("Setting up the position…")
                await asyncio.to_thread(ctx.set_board_from_paste, text)
                armed = await self._maybe_arm_drill(ctx.store.board_view)
                if armed is not None:
                    return armed

            ri = await asyncio.to_thread(ctx.read_input)
            cls = ri.get("classification")
            fen = ri.get("board_fen")
            if cls == "DRILL_WRONG":
                return await self._drill_wrong(ri, fen)
            if cls in ("DRILL_EVENT", "DRILL_POISONED_LINE"):
                return await self._drill_event(ri, fen, text)
            if cls == "DRILL_SOLVED":
                return {"ok": True, "flow": "drill_solved"}   # deterministic finish beat already shown
            if cls == "PROBE_ANSWER" and text and text.strip():
                return await self._probe_answer(fen, text.strip())
            if text and text.strip():
                return await self._coach(fen, text.strip())
            return {"ok": True, "orchestrated": False, "flow": "unhandled"}
        finally:
            self.store.publish_status(None)

    async def _maybe_arm_drill(self, fen):
        """A freshly SET position is the moment to arm a drill (the legacy coach's judgment call, now
        deterministic). If `fen` is a forcing win, build + arm the tree and post an intro beat that names
        the challenge WITHOUT the move — the app renders the drill and moves flow to /move (play_move
        adjudicates + auto-replies + Maia). Returns the turn result when a drill was armed, else None so
        the caller falls through to freeform coaching."""
        if not fen:
            return None
        self.store.set_gate(False)   # a new position supersedes any dangling probe (else _scoped refuses)
        self._poisoned_shown = None  # a new position: let its own poisoned line (if any) narrate afresh
        try:
            out = await asyncio.to_thread(self.ctx.build_and_arm_drill, fen)
        except Exception:  # noqa: BLE001 — never let arming break the turn; fall back to coaching
            return None
        if not (isinstance(out, dict) and out.get("drillable")):
            return None
        side = out.get("side_to_solve") or ("white" if " w " in f" {fen} " else "black")
        poisoned = out.get("has_poisoned_line")
        intro = (f"Here's a winning position for {side.capitalize()} — there's a forcing line. "
                 f"Find the move." + (" Calculate carefully; not every tempting move is best."
                                      if poisoned else " Your move."))
        self.store.append_beats([{"kind": "say", "tone": "teach",
                                  "segments": [{"text": intro}], "stops": False}])
        return {"ok": True, "flow": "drill_armed", "drillable": True}

    def _poisoned_trap_note(self) -> str | None:
        """Grounded one-line description of the current drill's poisoned line, from the tree (SAN line +
        Maia's motif) — no engine call. None when the drill has no trap. Fed to the coach so it surfaces
        the trap in its own voice on solve, replacing the old deterministic 'there's a poisoned line' beat."""
        tree = self.store._last_tree or {}
        if not tree.get("has_poisoned_line"):
            return None
        moves = tree.get("poisoned_line_moves") or []
        meta = tree.get("poisoned_line_meta") or {}
        san = " ".join(m.get("san") for m in moves if m.get("san"))
        parts = [f"the tempting line {san}" if san else "a tempting move that loses"]
        if meta.get("idea"):
            parts.append(f"the catch is {meta['idea']}")
        if (fatal := meta.get("fatal")) and fatal not in (meta.get("idea") or ""):
            parts.append(f"the motif is a {fatal}")
        return "; ".join(parts) + "."

    async def coach_move(self, uci: str, pre_fen: str | None, result: dict) -> dict:
        """Coach a move the player JUST played — for drills AND freeform. The move was already
        adjudicated + applied by ctx.play_move (board, opponent reply, history); this adds the LLM's
        grounded voice, TOLD the drill context so it says 'that's the right move, now look for …' or
        'not here — it runs into …, try again'. Runs after play_move (typically as a background task)."""
        in_drill = result.get("drill") is True
        correct = bool(result.get("correct"))
        finished = bool(result.get("finished"))
        wrong_drill = in_drill and not correct
        self.store.publish_status("Thinking…")
        try:
            verdict = (await asyncio.to_thread(self.ground.evaluate, pre_fen, [uci])) if pre_fen else {}
            played = (verdict or {}).get("san") or uci
            # The position they now face (after their move + the opponent's forced reply) — only useful
            # when the drill continues, so the coach can point at the next idea.
            nxt = await self._ground(self.store.board_view) if (in_drill and correct and not finished) else {}
            if in_drill and finished:
                head = f"CONTEXT: DRILL — SOLVED. You played {played}, completing the forcing line."
                # Surface the trap the player sidestepped, grounded in the tree (no template beat).
                if trap := self._poisoned_trap_note():
                    head += (f"\nThere was a poisoned line here: {trap} SURFACE this trap: after "
                             f"congratulating the solve, name the tempting move and why it loses "
                             f"(grounded ONLY in that line), and invite them to review it.")
            elif in_drill and correct:
                head = f"CONTEXT: DRILL — CORRECT. You played {played}; it is the right move and the drill advances."
            elif wrong_drill:
                head = f"CONTEXT: DRILL — WRONG. You played {played}; it is NOT the right move here."
            else:
                head = f"CONTEXT: FREEFORM move (no drill). You played {played}."
            # Maia's human-play read of the move ('a common mistake at your level', a find beyond it) —
            # returned by play_move, folded into the coach's voice here instead of its own local beat.
            # When present it's notable, so tell the coach to SURFACE it, not just consider it.
            maia = result.get("meaning")
            maia_line = ("\nMAIA NOTE — players at this level: " + maia + "\nSURFACE this in your reply: "
                         "if it's a common mistake, reassure the player it's a natural trap most players "
                         "their level fall for (not a careless slip) BEFORE explaining why it fails; if "
                         "it credits a find beyond their level, say so.\n") if maia else ""
            prompt = (f"{head}\n\nThe move's grounded facts:\n{_brief_move(verdict, hide_best=wrong_drill)}\n"
                      + maia_line
                      + (f"\nThe position you now face (after the reply):\n{_brief(nxt)}\n" if nxt else "")
                      + "\nCoach this move per your instructions. Return JSON." + self._name_hint())
            self._apply_name(out := await self._gen_json(_MOVE_SYSTEM, prompt))
            body = out.get("text")
            if body:
                good = correct or (not in_drill and (verdict or {}).get("class")
                                   in ("ok", "only_move", "good", "brilliant", "best"))
                tone = "praise" if good else ("correct" if in_drill else "teach")
                self.store.append_beats([{"kind": "say", "tone": tone,
                                          "segments": [{"text": body}], "stops": False}])
            return {"ok": True, "flow": "coach_move", "tokens": self._last_tokens}
        finally:
            self.store.publish_status(None)

    # -- flows ---------------------------------------------------------------
    async def _ground(self, fen, *, focus="analysis"):
        if not fen:
            return {}
        return await asyncio.to_thread(self.ground.analyze_and_show, fen, focus=focus, board_push=False)

    async def _hypothetical_facts(self, fen, text: str) -> str:
        """If the player asks about a concrete move OR a short line ('what if I take on e4 and then push
        d5'), resolve the sequence of THEIR moves via the LLM and PLAY IT OUT on the engine — the
        opponent's in-between replies are the engine's best, and each ply is legality-checked. Returns a
        grounded read (the line as played + the resulting evaluation) so the coach never invents where a
        line leads. '' when no concrete move is referenced. Runs on the read-only grounding engine."""
        if not fen or not _MOVE_Q_RE.search(text):
            return ""
        try:
            from lucena_engine.board import Board
            board = Board(fen)
            legal = [board.san(u) for u in board.legal_moves()]
        except Exception:  # noqa: BLE001
            return ""
        if not legal:
            return ""
        out = await self._gen_json(
            _EXTRACT_SYSTEM,
            f"Position FEN: {fen}\nLegal moves (side to move): {', '.join(legal)}\n"
            f"Player asked: {text}\nReturn JSON.")
        proposed = [m for m in (out.get("moves") or []) if isinstance(m, str)][:5]
        if not proposed:
            return ""
        # Play the player's line, best-play replies for the opponent in between; validate every ply.
        cur = board
        line: list[str] = []          # SAN in board order (player moves + engine replies)
        first_class = None
        diverged = False
        for i, san in enumerate(proposed):
            try:
                uci = cur.uci(san)    # SAN -> UCI; raises if the move isn't legal here
            except Exception:  # noqa: BLE001
                diverged = i > 0      # the imagined line stopped being legal — report how far it got
                break
            if i == 0:                # grade the player's first move (flags a plan that starts badly)
                first_class = ((await asyncio.to_thread(self.ground.evaluate, cur.fen, [san])) or {}).get("class")
            line.append(cur.san(uci))
            cur = cur.apply(uci)
            if i < len(proposed) - 1 and cur.legal_moves():   # opponent's best reply between player moves
                rep = ((await asyncio.to_thread(self.ground.get_hints, cur.fen)) or {}).get("best")
                try:
                    ruci = cur.uci(rep)
                except Exception:  # noqa: BLE001
                    break
                line.append(cur.san(ruci))
                cur = cur.apply(ruci)
        if not line:
            return ""
        verdict = await asyncio.to_thread(self.ground._live_verdict, cur.fen)
        parts = [f"Played out with best replies for the opponent, the line runs: {' '.join(line)}."]
        if diverged:
            parts.append("(The rest of the proposed line wasn't legal from there.)")
        if first_class and first_class not in ("ok", "best", "only_move", "good"):
            parts.append(f"The first move ({line[0]}) is engine-classed a {first_class}.")
        if verdict:
            parts.append(f"The resulting position is: {verdict}.")
        return "\n\nGrounded read of the line the player asked about:\n" + " ".join(parts)

    async def _coach(self, fen, text: str) -> dict:
        facts = await self._ground(fen)
        hints_res = await asyncio.to_thread(self.ground.get_hints, fen) if fen else {}
        best = (hints_res or {}).get("best")
        hints = [h for h in ((hints_res or {}).get("hints") or []) if isinstance(h, str)][:3]
        hypo = await self._hypothetical_facts(fen, text)
        side = "black" if (fen and " b " in f" {fen} ") else "white"
        you, opp = side.capitalize(), ("White" if side == "black" else "Black")
        hypo_note = ("\n\nThe player is asking what happens after a SPECIFIC move. Answer DIRECTLY "
                     "(mode='tell'), grounded ONLY in the 'If <move> is played' facts above — never "
                     "invent the resulting evaluation or a continuation that isn't shown." if hypo else "")
        out = await self._gen_json(
            _COACH_SYSTEM,
            f"You are coaching the player, who is playing {you} (the side to move). Their opponent is "
            f"{opp}. Every threat/attack/plan belongs to {opp}, never to the player.\n\n"
            f"Player said: {text}\n\nEngine's grounded read (coach ONLY from this):\n{_brief(facts)}\n"
            f"Best move (reveal ONLY in a 'tell'): {best}" + hypo + hypo_note
            + "\n\nRespond as JSON." + self._name_hint())
        self._apply_name(out)
        mode = (out.get("mode") or "ask").lower()
        body = out.get("text") or "Let's take a look at this position together."
        beat = {"kind": "ask", "segments": [{"text": body}], "stops": True}
        if mode == "tell":
            beat = {"kind": "say", "tone": "teach", "segments": [{"text": body}], "stops": False}
        elif hints:
            beat["hints"] = hints
        self.store.append_beats([beat])
        if mode != "tell":
            self.store.set_gate(True)
        return {"ok": True, "flow": f"coach:{mode}", "tokens": self._last_tokens}

    async def _probe_answer(self, fen, text: str) -> dict:
        facts = await self._ground(fen)
        verdict = await self._gen_json(
            _GRADE_SYSTEM,
            f"Player's answer: {text}\n\nEngine's grounded analysis (grade ONLY against this):\n"
            f"{_brief(facts)}\n\nGrade and give feedback as JSON.")
        correct = bool(verdict.get("correct"))
        feedback = verdict.get("feedback") or "Let's look at that together."
        self.store.append_beats([{"kind": "say", "tone": "praise" if correct else "correct",
                                  "segments": [{"text": feedback}], "stops": False}])
        self.store.set_gate(False)
        return {"ok": True, "flow": "probe_answer", "correct": correct, "tokens": self._last_tokens}

    async def _drill_event(self, ri: dict, fen, text: str | None = None) -> dict:
        # DRILL_POISONED_LINE hands over the trap as prose — but ONLY narrate it ONCE per context.
        # Latched on the payload: the classifier re-flags every turn the board sits on the poisoned
        # line, so without this the same trap narration re-posts each turn (reads as duplicate beats).
        if pl := ri.get("poisoned_line"):
            if str(pl) != self._poisoned_shown:
                self._poisoned_shown = str(pl)
                self.store.append_beats([{"kind": "say", "tone": "teach",
                                          "segments": [{"text": str(pl)[:600]}], "stops": False}])
                return {"ok": True, "flow": "drill_poisoned"}
            # Already shown: answer a follow-up question on the current board; otherwise stay quiet.
            if text and text.strip():
                return await self._coach(fen, text.strip())
            return {"ok": True, "flow": "drill_poisoned_seen"}
        facts = await self._ground(fen)
        out = await self._gen_json(
            _DRILL_SYSTEM,
            f"Current drill position — the grounded read:\n{_brief(facts)}\n\n"
            f"Coach the idea + nudge the next move as JSON.")
        body = out.get("text")
        if body:
            self.store.append_beats([{"kind": "say", "tone": "teach",
                                      "segments": [{"text": body}], "stops": False}])
        return {"ok": True, "flow": "drill_event", "tokens": self._last_tokens}

    async def _drill_wrong(self, ri: dict, fen) -> dict:
        tried = (ri.get("tried") or ri.get("data") or {})
        move = ri.get("tried") if isinstance(ri.get("tried"), str) else None
        facts = await asyncio.to_thread(self.ground.evaluate, fen, [move]) if (fen and move) else \
            await self._ground(fen)
        out = await self._gen_json(
            _WRONG_SYSTEM,
            f"The player's WRONG move, graded:\n{_brief(facts) if isinstance(facts, dict) else facts}\n\n"
            f"Explain why it fails + encourage a retry as JSON.")
        body = out.get("text")
        if body:
            self.store.append_beats([{"kind": "say", "tone": "correct",
                                      "segments": [{"text": body}], "stops": False}])
        return {"ok": True, "flow": "drill_wrong", "tokens": self._last_tokens}

    # -- session titling -----------------------------------------------------
    def _name_hint(self) -> str:
        """A prompt suffix asking the model to ALSO title the session — only while it's still the
        placeholder 'New session'. Piggybacks on the coaching JSON (no extra LLM call)."""
        if not self.store.session_unnamed:
            return ""
        return ("\n\nThis coaching session has no title yet. ALSO include a \"name\" field in your JSON: "
                "a concise 2-5 word Title-Case label GROUNDED ONLY in the facts above — describe the "
                "CONCRETE situation actually shown (the material/phase, the piece it hinges on, or the "
                "tactic present), e.g. \"Rook vs Two Pawns\", \"Knight Fork on f7\", \"Opposite-Side "
                "Castling Attack\". Do NOT name an opening, player, variation, or theme that is not "
                "evidenced in the facts — if you are unsure, describe the position plainly (e.g. "
                "\"Middlegame With Isolated Pawn\"). Plain text, no quotes.")

    def _apply_name(self, out: dict) -> None:
        """Extract the coach's proposed `name` and title the session — once, only while unnamed."""
        if self.store.session_unnamed:
            name = (out or {}).get("name")
            if isinstance(name, str) and name.strip():
                self.store.set_session_name(name)

    # -- generation ----------------------------------------------------------
    async def _gen_json(self, system: str, prompt: str) -> dict:
        if os.environ.get("LUCENA_DEBUG_PROMPT"):
            print(f"\n===== LLM PROMPT =====\n--- SYSTEM ---\n{system}\n\n--- USER ---\n{prompt}\n"
                  f"======================", flush=True)
        comp = await self._llm.generate(
            [Message("system", system), Message("user", prompt)],
            GenerateOptions(model=self.model, schema=_JSON_OBJECT, max_tokens=400, temperature=0.4))
        if os.environ.get("LUCENA_DEBUG_PROMPT"):
            print(f"--- RESPONSE ---\n{comp.text}\n======================", flush=True)
        u = comp.usage
        self._last_tokens = ({"input": u.input_tokens, "output": u.output_tokens,
                              "total": u.total_tokens} if u is not None else None)
        return comp.json or {}

    async def aclose(self) -> None:
        return None
