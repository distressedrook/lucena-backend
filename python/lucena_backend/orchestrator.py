"""Orchestrator — the deterministic coaching pipeline over the in-process ToolContext.

Every turn: read_input (classify + ground) -> ONE LLM generation over the grounded facts -> push
the beat. The model never tool-calls, never sets the board. ToolContext (the full legacy coach) does
the grounding + drill walk + Maia; this drives it and adds the LLM's grounded voice.
"""

from __future__ import annotations

import asyncio
import re
import os

from lucena_engine import openings   # pure table lookups: no engine process, no I/O per call
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

def _perspective(freeform: bool) -> str:
    """The voice block: who the coach is talking to.

    Two bodies, one function, because the answer is a property of the MODE and it appears in four
    prompts — duplicating it is how the four drift apart.

    In a DRILL the engine really does reply (drill.py plays the defence), so the player genuinely is
    one side and "you" is correct. In FREEFORM nothing replies — `play_move` applies one ply and
    stops, and the board has no side-to-move gate, so either colour is draggable. The player is
    driving both sides of an analysis board: there is no "you" to address, only White and Black.
    Saying "you played e4" there is not a style choice, it is factually wrong.
    """
    if freeform:
        return (
            "PERSPECTIVE (critical): there is NO 'you' here. The player is moving BOTH sides on an "
            "analysis board — nobody is 'the player's colour'. Name the mover: 'White stakes the "
            "centre', 'Black challenges it'. Never address anyone as 'you', never say 'your "
            "opponent', never attribute a move or plan to 'the player'. Threats and plans belong to "
            "the colour that owns them.\n"
        )
    return (
        "PERSPECTIVE (critical — getting it backwards ruins the read): address the player as 'you'; "
        "they play the side to move. The OTHER colour is 'your opponent'. Every threat, attack, or "
        "plan belongs to the OPPONENT — never say the player is threatening their own pieces or "
        "defending against themselves. Name the opponent's threat when there is one.\n"
    )


def _coach_system(freeform: bool) -> str:
    return _COACH_SYSTEM_HEAD + _perspective(freeform) + _COACH_SYSTEM_TAIL


_COACH_SYSTEM_HEAD = (
    "You are a chess coach. You are given the engine's grounded read of the "
    "CURRENT position, the player's message, and the engine's best move. DECIDE how to respond:\n"
    "- DEFAULT (mode='ask'): lead with ONE Socratic question that guides them toward the key idea "
    "WITHOUT revealing or naming the best move. Use this whenever they're exploring, unsure, or ask "
    "something open like 'what should I think about here?'.\n"
    "- mode='tell': ONLY when the player clearly wants to be told the answer (e.g. asks for the best "
    "move, says 'just tell me', 'show me the move', 'what should I play'). Then explain directly, "
    "naming the move and the reason.\n"
    "- mode='unsupported': when the player is asking you to DO or START something you cannot do — play "
    "or start a game/match, act as their opponent or a bot, run a puzzle/lesson flow, change app "
    "settings or mode, or any request that is NOT about understanding the position on the board. Do "
    "NOT coach the position in this case. Briefly acknowledge what they asked for and say you can't "
    "help with that yet, with a light apology — e.g. 'Playing a full match against a bot isn't "
    "something I can do yet — sorry!'. Keep it to one short sentence.\n"
    "Ground EVERY claim only in the facts provided — never invent a piece, square, line, or number. "
    # The example was "you're winning" — which smuggles the drill voice into a prompt that BOTH modes
    # share, underneath the perspective block that just said not to. Named colours read fine in both.
    "Translate evaluations into plain words ('White is much better', 'roughly equal') — never cite "
    "win% or centipawns.\n"
)

_COACH_SYSTEM_TAIL = (
    "Warm, direct, one idea, no jargon walls.\n"
    "Return JSON: {\"mode\": \"ask\"|\"tell\"|\"unsupported\", \"text\": string}. `text` is the question "
    "(ask), the explanation (tell), or the short apology (unsupported), shown to the player."
)

# The narration prompt. It exists as a SECOND BODY rather than a flag on _MOVE_SYSTEM because it
# inverts three of that prompt's instructions at once: length (1-2 sentences -> a paragraph), register
# (its ban on "generic strategic advice" is exactly what narration IS), and verdict ("a strong move /
# fine / an inaccuracy" — the literal source of "That's a fine move..."). Loosening _MOVE_SYSTEM to fit
# would make DRILLS start dispensing "control the center".
_NARRATE_SYSTEM_HEAD = (
    "You are annotating a chess game as it is played, in the voice of a good opening book. A move has "
    "just entered a NAMED opening, and your job is to explain what is being played — NOT to judge the "
    "move, NOT to ask a question.\n"
)

# THE CARVE-OUT. CLAUDE.md's standing invariant is that the model interprets and never calculates:
# anything it could get wrong is grounded or guarded. This is the ONE deliberate exception, and it is
# written down here and in CLAUDE.md so it stays an exception instead of leaking into "the model does
# chess now". The boundary is the whole point: opening IDEAS are stable, well-documented knowledge;
# concrete evaluations and tactics are not, and those still come only from the facts.
_NARRATE_SYSTEM_TAIL = (
    "WHAT YOU MAY DRAW ON (a deliberate exception): you MAY use your own knowledge of the NAMED "
    "opening — its ideas, typical plans, pawn structures, what each side is trying to achieve, and why "
    "people choose it. This is the one place you are trusted beyond the given facts.\n"
    "WHAT YOU MAY NOT: do not state any concrete evaluation, tactic, threat, or verdict on a specific "
    "move that is not in the facts you were given. No win%, no centipawns, no invented lines. If you "
    "are unsure whether something is true of THIS position rather than the opening in general, say it "
    "about the opening in general or leave it out.\n"
    # Caught end-to-end on the very first real narration: given only 1.e4, the model wrote "Black
    # responds with e5, a classic challenge…" — theory reported as history, for a move nobody had
    # played. It is the carve-out's failure mode exactly. Permission to explain an opening reads as
    # permission to narrate its mainline, and the two are one word apart ("usually answers" vs
    # "responds"), so the ban has to be explicit and has to name the tense.
    "TENSE — THE ONE THING THAT MAKES THIS A LIE: only the moves listed under MOVES SO FAR have "
    "actually been played. NEVER write a later move as though it happened — no \"Black responds "
    "with...\", no \"White then castles\". You are annotating a game in progress, not summarising a "
    "finished one. A typical continuation may be mentioned ONLY if it is unmistakably marked as "
    "typical rather than actual: 'Black usually answers...', 'the main line continues...'.\n"
    # The name is the trap, and it took two failed attempts to see it. The tense rule alone did not
    # work because the model was not being careless — it was being FAITHFUL. The table names the
    # position after 1.e4 "King's Pawn Game", and that name conventionally denotes 1.e4 e5; asked to
    # explain that opening, the model explained the opening the NAME means, which includes Black's
    # reply. So the instruction has to say that the name runs ahead of the game, not just that the
    # tense matters.
    "THE NAME MAY RUN AHEAD OF THE GAME: an opening name often denotes a longer sequence than has "
    "actually been played — \"King's Pawn Game\" conventionally means 1.e4 e5, but perhaps only 1.e4 "
    "is on the board. Explain what the moves ACTUALLY PLAYED do and what the side to move is now "
    "choosing between. The rest of the name's sequence has NOT happened: it is what typically "
    "follows, and must be written that way or not at all.\n"
    # Third failure mode, third instruction. Once the model could no longer state a continuation as
    # fact, it started EXPLORING one instead: given 1.e4 it picked the Caro-Kann — a defence nobody
    # had chosen — and spent half the paragraph on it. Correctly hedged, entirely irrelevant. The
    # permission to mention what typically follows needs a budget, or it becomes the subject.
    "STAY ON THE BOARD: the subject is the position IN FRONT OF YOU. Do not pick one of the opponent's "
    "possible replies and explore it — you do not know which they will choose, and a paragraph about "
    "a defence nobody has played is noise. What typically follows is worth AT MOST one clause, and "
    "only where it explains the move actually played. Never develop a line beyond that.\n"
    "LENGTH: a short paragraph — three or four sentences. This is the one place you are NOT terse.\n"
    "IF YOU ARE GIVEN A PREVIOUS OPENING NAME: the reader already had that explained. Write only what "
    "this move ADDS. If the change is a refinement within the same family, one or two sentences is "
    "plenty — do not restate the family.\n"
    "IF YOU ARE GIVEN AN ENGINE CLASS for the move: the line concedes something real, and THAT is the "
    "interesting part. Explain what it gives up and what it gets for it — the compensation, the "
    "practical bet, why the line exists despite the engine's preference. Do not scold; this is theory.\n"
    "Return JSON: {\"text\": string}."
)


def _narrate_system(prev_name: str | None = None) -> str:
    # Narration is freeform-only by construction: a drill never narrates, so the perspective is always
    # the named-mover one.
    return _NARRATE_SYSTEM_HEAD + _perspective(True) + _NARRATE_SYSTEM_TAIL


# The end of the book: an explicit hand-off, so the voice changing is legible to the player rather
# than the coach silently developing a personality.
_ENDBOOK_SYSTEM = (
    "You are annotating a chess game. The players have just left opening theory — the position is no "
    "longer in the book. Say so in ONE short sentence, plainly and without drama, and note that from "
    "here they are working it out over the board rather than following a line.\n"
    + _perspective(True) +
    "Ground ONLY in the facts given. Return JSON: {\"text\": string}."
)

# -- book routing ---------------------------------------------------------------------------------
# The four freeform routes. Named, because the choice is made once from a pure function and asserted
# in tests by name — a bare string at the branch would be a typo away from silently coaching.
_NARRATE, _ENDBOOK, _COACH, _SILENT = "narrate", "endbook", "coach", "silent"

# Plies out of the book before we say so. FOUR, not three, and it is a measured heuristic rather than
# a round number: two-unnamed-ply gaps occur at ~12% of book positions (the table names positions, not
# lines, and re-attaches names unevenly), and the probe that measured it was capped at 3 — so 3-ply
# gaps cannot be excluded and 3 has zero margin. Announcing "you have left the book" to someone still
# in the Ruy Lopez is worse than announcing it a ply late.
_OFF_BOOK_AT = 4

# The engine's own word for "this move concedes something", not a threshold of our own invention:
# `classify` already returns these for a win-% drop past _DUBIOUS (5.0) / _MISTAKE (10.0) / _BLUNDER
# (15.0), measured against the engine's best move. A BOOK move that crosses one is the gambit case —
# theory that the engine disagrees with — and that disagreement is the interesting part.
_SWING_CLASSES = frozenset({"dubious", "mistake", "blunder"})


def _is_swing(verdict: dict) -> bool:
    return ((verdict or {}).get("class") or "") in _SWING_CLASSES


def _book_route(fens: list, swing: bool) -> tuple:
    """Which voice a freeform move gets: `(route, name, prev_name)`. Pure — no engine, no LLM, no DB.

    Pure on purpose: this is the whole cadence policy, it has four interacting cases, and the way to
    test it is to replay real openings through it and assert the exact fire sequence — not to mock an
    LLM. The caller does I/O; this only decides.

    The swing trigger is INDEPENDENT of the name change. 2...exf4 accepting the King's Gambit may not
    change the name and is exactly the move worth explaining, so:

        in book + name changed          -> narrate the opening   (nothing to judge; best/class hidden)
        in book + swing                 -> explain the concession (the name is context, not news)
        in book + both                  -> one beat: name it AND say what it concedes
        in book + neither               -> silent (nothing new to say; do not fill the air)
        just left the book              -> say so, once
        off book / never in it          -> normal coaching
    """
    psn = openings.plies_since_named(fens)
    if psn is None:                       # never in the book: a drill, a pasted midgame FEN.
        return _COACH, None, None         # "you have left the book" is meaningless there.
    if psn < _OFF_BOOK_AT:
        name = openings.book_name(fens)
        prev = openings.book_name(fens[:-1]) if len(fens) > 1 else None
        changed = name is not None and name != prev
        if changed or swing:
            # `prev` is passed ONLY when the name changed — it is there to make the model write the
            # DELTA ("Nc6 -> King's Knight Opening: Normal Variation" deserves a clause, not a
            # paragraph). On a pure swing the name is unchanged, so there is no delta to write.
            return _NARRATE, name, (prev if changed else None)
        return _SILENT, name, None
    if psn == _OFF_BOOK_AT:
        # EXACTLY at the threshold, so this fires once per exit with no latch to store or resync. A
        # transposition back into the book returns psn to 0 and re-arms it — correct: the book was
        # left twice.
        return _ENDBOOK, None, None
    return _COACH, None, None


def _mover(fen) -> str:
    """The colour that just moved FROM `fen` — i.e. `fen`'s side to move."""
    return "Black" if (fen and " b " in f" {fen} ") else "White"


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

def _move_system(freeform: bool) -> str:
    return _MOVE_SYSTEM_HEAD + _perspective(freeform) + _move_system_tail(freeform)


_MOVE_SYSTEM_HEAD = (
    "You are a chess coach reacting to a move JUST made. You are told whether this is a "
    "drill and, if so, whether the move was CORRECT, WRONG, or SOLVED the drill; for a freeform move you "
    "get its engine class. You also get the move's grounded facts and (when relevant) the position that "
    "follows. Ground EVERY claim ONLY in those facts — never invent a piece, square, line, or number; "
    "translate evaluations to plain words, never cite win% or centipawns. Keep it to 1-2 short "
    "sentences.\n"
)

def _move_system_tail(freeform: bool) -> str:
    # The "unsure" fallback names a LOSER, so it has a voice too — and it is the one sentence the model
    # reaches for exactly when it is least sure, which makes it the likeliest thing to be said. Left
    # shared, it put "the position turns against you" directly underneath a perspective block that had
    # just said there is no "you" — the prompt contradicting itself in the same breath. The lesson this
    # keeps re-teaching: a voice fix is never one line, it is every line shaped like that line.
    turns_against = ("the position turns against the side that played it" if freeform
                     else "the position turns against you")
    return (
    "- DRILL / CORRECT: confirm the move is right and name the idea it achieves, then point at what to "
    "look for NEXT without naming the next move.\n"
    "- DRILL / WRONG: say it isn't the move here and explain the flaw ONLY through the refutation line you "
    "are given — name the opponent's ACTUAL first refuting move (the first move of that line). Encourage "
    "another try; do NOT reveal the right move.\n"
    "- DRILL / SOLVED: celebrate finishing the forcing line and name the key idea that won.\n"
    "- DRILL / SUSPENDED: they stepped off their own drill line to try a what-if. Answer it honestly, "
    "as for FREEFORM below, then invite them back to the drill.\n"
    "- FREEFORM: give the honest verdict (a strong move / fine / an inaccuracy / a mistake) grounded in "
    "the facts, say why, and if it was a mistake point toward the better idea.\n"
    "CRITICAL — do NOT invent a mechanism. Only describe what is actually in the given line: if the "
    "refutation is a queen move, do not call it a pawn push; if no piece is trapped in the line, do not "
    "say a piece is trapped; if there is no fork/pin in the line, do not name one. When you are unsure "
    f"how the line works, say plainly that the engine refutes it and {turns_against} — "
    "never fill the gap with a plausible-sounding motif. Do not add generic strategic advice ('control "
    # The cliché was quoted as "develop your pieces" — a second person, in a prompt that in freeform
    # has just banned one. It is only an EXAMPLE of advice not to give, so the wording is incidental
    # and the neutral form bans exactly the same thing. Cheaper to say it neutrally than to carve an
    # exception into the check that guards this.
    "the center', 'develop the pieces') that is not in the facts.\n"
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
        # `session_id` was accepted and ignored here for the whole single-chat era (the caller passed
        # store._current straight back in). Bind it: every read/write/publish below this line resolves
        # to THIS chat, including work handed to asyncio.to_thread (which copies the context).
        with self.store.bound(session_id):
            return await self._run_turn(text)

    async def _run_turn(self, text: str | None = None) -> dict:
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

    async def coach_move(self, session_id: str, uci: str, pre_fen: str | None, result: dict) -> dict:
        """Bind the chat this move was played in, then coach it. `session_id` is REQUIRED: this runs
        as a DETACHED background task and can outlive its turn, so it must not resolve the cursor at
        write time (that is the late-read race)."""
        with self.store.bound(session_id):
            return await self._coach_move(uci, pre_fen, result)

    def _played_line(self) -> str:
        """The game so far, as "1.e4 c5 2.Nf3" — the ONE fact that stops the narration inventing moves.

        Without it the prompt's only anchor is the move just played, so "explain this opening" and
        "recite this opening's mainline" look identical from inside the model — and it recited. Naming
        the whole line, and saying it IS the whole line, is what makes the tense rule checkable rather
        than aspirational.
        """
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

    def _history_fens(self) -> list:
        """The played line as FENs, for the opening lookup.

        Read from the session's own history rather than tracked separately: the book routing is then a
        pure function of state that already exists and is already persisted — nothing to keep in sync,
        nothing to resync after a restart, and no "last narrated" latch to go stale.

        KNOWN LIMIT (pre-existing, called out rather than fixed here): `play_move` APPENDS to history
        and never truncates, so a rewound-and-replayed board leaves the tail of the old line behind and
        `plies_since_named` counts against a line that was not played. The fold survives it — a wrong
        name is still a name — but end-of-book can fire early. Fixing it means truncating history on a
        rewind, which is a board-navigation change, not an opening one.
        """
        return [f for p in (self.store._history or []) if (f := (p or {}).get("fen"))]

    def _title_from_opening(self, name: str | None) -> None:
        """Title an unnamed session after the opening's FAMILY — "Ruy Lopez", not "Ruy Lopez: Morphy
        Defense, Closed".

        Deterministic, not asked of the model. `_name_hint` exists because the model is the only thing
        that can look at a position and describe it; here the answer is already KNOWN, and known
        exactly. Asking would be strictly worse in three ways: it is the most likely titling moment in
        a session (move one), so a JSON that omits `name` would silently stop titling; the model could
        invent a name the table did not give; and the narration prompt is deliberately NOT the
        fact-grounded voice, which is the voice `_name_hint`'s "grounded ONLY in the facts above"
        assumes. The family rather than the full name because it is what a player calls the game, and
        it does not read as a truncation.
        """
        if name and self.store.session_unnamed:
            self.store.set_session_name(openings._path(name)[0][:60])

    async def _narrate_move(self, route: str, played: str, pre_fen, verdict: dict, nxt: dict,
                            book: str | None, prev_book: str | None) -> dict:
        """The book voice: narrate an opening, or announce that the book has ended.

        A separate arm rather than a flag on the coaching path because it inverts that path at three
        points at once — length (1-2 sentences -> a paragraph), register (its ban on "generic
        strategic advice" is precisely what narration IS), and verdict (its "a strong move / fine / an
        inaccuracy" is the literal source of the "That's a fine move…" this replaces). Loosening the
        shared prompt to fit would make DRILLS start dispensing "control the center".
        """
        mover = _mover(pre_fen)
        if route == _ENDBOOK:
            prompt = (f"CONTEXT: {mover} just played {played}, and the line has now left opening "
                      f"theory.\n\nThe position that follows:\n{_brief(nxt)}\n"
                      "\nSay that the book has ended, per your instructions. Return JSON.")
            out = await self._gen_json(_ENDBOOK_SYSTEM, prompt)
        else:
            swing = _is_swing(verdict)
            head = (f"CONTEXT: {mover} just played {played}. This move is in a NAMED opening: {book}.\n"
                    f"MOVES SO FAR (this is the ENTIRE game; nothing else has been played): "
                    f"{self._played_line() or played}")
            if prev_book:
                head += (f"\nPREVIOUS OPENING NAME (already explained to the reader): {prev_book}. "
                         f"Write only what this move ADDS to that.")
            if swing:
                head += ("\nNOTE: the engine does NOT prefer this move — its class is below. Yet it is "
                         "theory. THAT tension is the subject: explain what the line gives up and what "
                         "it gets for it. Do not scold; this is a known opening, not a blunder.")
            # hide_best is the difference between narration and a verdict. On a quiet book move there
            # is nothing to judge, and handing over "best: d4" makes the model helpfully report it —
            # which is the exact register being killed. On a swing the class/best/drop ARE the subject.
            prompt = (f"{head}\n\nThe move's grounded facts:\n"
                      f"{_brief_move(verdict, hide_best=not swing)}\n"
                      + (f"\nThe position that follows:\n{_brief(nxt)}\n" if nxt else "")
                      + "\nNarrate this per your instructions. Return JSON.")
            out = await self._gen_json(_narrate_system(prev_book), prompt)
        if body := (out or {}).get("text"):
            self.store.append_beats([{"kind": "say", "tone": "teach",
                                      "segments": [{"text": body}], "stops": False}])
        self._title_from_opening(book)
        return {"ok": True, "flow": f"coach_move:{route}", "tokens": self._last_tokens}

    async def _coach_move(self, uci: str, pre_fen: str | None, result: dict) -> dict:
        """Coach a move the player JUST played — for drills AND freeform. The move was already
        adjudicated + applied by ctx.play_move (board, opponent reply, history); this adds the LLM's
        grounded voice, TOLD the drill context so it says 'that's the right move, now look for …' or
        'not here — it runs into …, try again'. Runs after play_move (typically as a background task)."""
        in_drill = result.get("drill") is True
        # ADJUDICATION and VOICE are two questions, and one expression was answering both — wrongly.
        # `drill` is TRI-state: True / False / "suspended" (the player stepped back inside their OWN
        # drill line to try a what-if). Only a live drill GRADES a move, so adjudication stays `is
        # True`. But a suspended drill is still a drill — they are one side, the engine resumes
        # replying the moment they return to the line — so "you" is right there too. Only an outright
        # False is freeform. Testing `is not True` for voice made suspended drills speak as freeform.
        freeform = result.get("drill") is False
        correct = bool(result.get("correct"))
        finished = bool(result.get("finished"))
        wrong_drill = in_drill and not correct
        self.store.publish_status("Thinking…")
        try:
            verdict = (await asyncio.to_thread(self.ground.evaluate, pre_fen, [uci])) if pre_fen else {}
            played = (verdict or {}).get("san") or uci
            # Which voice this move gets. Pure + decided BEFORE any further grounding, so the silent
            # case costs nothing: an in-book move that says nothing new does no analysis and makes no
            # LLM call.
            route, book, prev_book = ((_book_route(self._history_fens(), _is_swing(verdict)))
                                      if freeform else (_COACH, None, None))
            if route == _SILENT:
                return {"ok": True, "flow": "coach_move:silent"}
            # The resulting position. The freeform arm is the LOAD-BEARING part of opening narration:
            # this gate used to be drill-only, so a freeform move got no position briefing at all and
            # the opening fact — which is emitted by the fact sheet for the position AFTER the move —
            # could not reach the model by any route. It costs a full analysis per freeform ply; it
            # runs on the read-only ground ctx in a detached task, so the board never waits on it.
            nxt = ({} if route == _ENDBOOK else
                   await self._ground(self.store.board_view)
                   if ((in_drill and correct and not finished) or freeform) else {})
            if route in (_NARRATE, _ENDBOOK):
                return await self._narrate_move(route, played, pre_fen, verdict, nxt, book, prev_book)
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
            elif not freeform:
                # SUSPENDED: `drill` is neither True nor False. Adjudication skipped it (it is not a
                # live drill move), and the voice code correctly kept "you" — but the head fell through
                # to the freeform branch below and announced "CONTEXT: FREEFORM move (no drill)" under
                # a system prompt that had just said this IS a drill. The model was handed both, and
                # the tri-state's third case was the only one nobody wrote a branch for.
                head = (f"CONTEXT: DRILL — SUSPENDED. You played {played}, stepping off the drill's "
                        f"line to try a what-if. It is still your drill and it resumes when the board "
                        f"returns to it.")
                if detail := result.get("detail"):
                    head += f"\n{detail}"
            else:
                # No "you": freeform is a shared analysis board driven from both sides (see
                # ToolContext.freeform / _perspective). Name the mover instead.
                head = f"CONTEXT: FREEFORM move (no drill). {_mover(pre_fen)} played {played}."
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
                      + ((f"\nThe position that follows:\n{_brief(nxt)}\n" if freeform else
                          f"\nThe position you now face (after the reply):\n{_brief(nxt)}\n")
                         if nxt else "")
                      + "\nCoach this move per your instructions. Return JSON." + self._name_hint())
            self._apply_name(out := await self._gen_json(_move_system(freeform), prompt))
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
        # Voice follows the MODE, and typed chat is in scope for it: with no drill armed the player is
        # driving both sides, so "you are playing White" is a claim the board does not support — they
        # are as likely to play Black's next move. With a drill (live OR suspended) they really are one
        # side and "you" is correct. Same rule as the played-move path; one predicate, not two.
        freeform = self.ctx.freeform
        mover = "Black" if (fen and " b " in f" {fen} ") else "White"
        other = "White" if mover == "Black" else "Black"
        if freeform:
            frame = (f"CONTEXT: analysis board, no drill — the player moves BOTH sides. {mover} is to "
                     f"move. Do NOT address anyone as 'you' and do NOT treat either colour as the "
                     f"player's; name the colours. Threats and plans belong to whichever colour owns "
                     f"them.\n\n")
        else:
            frame = (f"You are coaching the player, who is playing {mover} (the side to move). Their "
                     f"opponent is {other}. Every threat/attack/plan belongs to {other}, never to the "
                     f"player.\n\n")
        hypo_note = ("\n\nThe player is asking what happens after a SPECIFIC move. Answer DIRECTLY "
                     "(mode='tell'), grounded ONLY in the 'If <move> is played' facts above — never "
                     "invent the resulting evaluation or a continuation that isn't shown." if hypo else "")
        out = await self._gen_json(
            _coach_system(freeform),
            frame +
            f"Player said: {text}\n\nEngine's grounded read (coach ONLY from this):\n{_brief(facts)}\n"
            f"Best move (reveal ONLY in a 'tell'): {best}" + hypo + hypo_note
            + "\n\nRespond as JSON." + self._name_hint())
        self._apply_name(out)
        mode = (out.get("mode") or "ask").lower()
        body = out.get("text") or "Let's take a look at this position together."
        if mode == "unsupported":
            # The player asked for something we can't do (play a match, act as a bot, …). Decline
            # plainly instead of coaching the position; no Socratic gate, no hints.
            self.store.append_beats([{"kind": "say", "tone": "teach",
                                      "segments": [{"text": body}], "stops": False}])
            return {"ok": True, "flow": "coach:unsupported", "tokens": self._last_tokens}
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
