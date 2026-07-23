"""Every instruction the coach sends to the LLM — system prompt AND user message — as one class per
prompt family.

Before this file, each prompt was a handful of module-level string constants (HEAD/TAIL/BODY) built
by `+`-concatenation, scattered across orchestrator.py and book_voice.py — readable one prompt at a
time, but the files became a wall of text nothing else could be found in. The first pass here only
moved the SYSTEM prompts; the user-message halves (the "CONTEXT: ...", "MOVES SO FAR: ...", label
strings assembled in each Orchestrator method) were left inline. That was the wrong line to draw —
those are exactly as much "an instruction sent to the model" as the system prompt is, just addressed
to the other message. Every family here now owns BOTH halves.

The shape: each family is a `Prompt` subclass. Its raw pieces (fixed wording, `{}`-templated where a
runtime value slots in) are PRIVATE — single-underscore, Python's actual privacy convention, same as
the rest of this package. From outside this file there are exactly two things to call:
`SomePrompt.system(...)` for the system prompt, `SomePrompt.prompt(...)` for the user message (named
differently per family where a family needs several distinct pieces assembled by the caller — e.g.
`MovePrompt.solved_head(...)` for the one case among five the caller has already branched to). Either
way, callers pass TYPED data (a FEN, a SAN, a bool, a list of replies) and get back text; no caller
outside this file ever builds a fragment of model-facing wording by hand.

(The tests in test_opening_narration.py are the one deliberate exception: they whitebox-test prompt
COMPOSITION — e.g. "does the voice leak outside the perspective block" — which needs the pieces
apart, not just the finished string. That is exactly what "private by convention" in Python is for:
internal, not walled off, and the one file that legitimately needs to reach in still can.)
"""

from __future__ import annotations

from .grounding import _perspective, _MOVE_NUMBER_RULE, _MARKDOWN_RULE, san_guard
from .book_voice import _BOOK_RATING


class Prompt:
    """Base for a composed system prompt. A subclass holds its pieces as private attributes/methods
    and exposes them through nothing but `system(...)` — every family gets the same one-method
    surface, whether or not it actually needs arguments to compose."""

    @classmethod
    def system(cls, *args, **kwargs) -> str:
        raise NotImplementedError


class NarratePrompt(Prompt):
    """The book voice: narrate an opening move as it's played.

    A separate family rather than a flag on MovePrompt because it inverts three of that prompt's
    instructions at once: length (1-2 sentences -> a paragraph), register (its ban on "generic
    strategic advice" is exactly what narration IS), and verdict ("a strong move / fine / an
    inaccuracy" — the literal source of "That's a fine move..."). Loosening MovePrompt to fit would
    make DRILLS start dispensing "control the center".
    """

    _head = (
        "You are annotating a chess game as it is played, in the voice of a good opening book. A move has "
        "just entered a NAMED opening, and your job is to explain what is being played — NOT to judge the "
        "move, NOT to ask a question.\n"
    )

    # THE CARVE-OUT. CLAUDE.md's standing invariant is that the model interprets and never calculates:
    # anything it could get wrong is grounded or guarded. This is the ONE deliberate exception, and it is
    # written down here and in CLAUDE.md so it stays an exception instead of leaking into "the model does
    # chess now". The boundary is the whole point: opening IDEAS are stable, well-documented knowledge;
    # concrete evaluations and tactics are not, and those still come only from the facts.
    _tail = (
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
        # This replaces an instruction that did not work. It used to say "stay on the board, do not explore
        # a reply" — and the model explored anyway (given 1.e4 it picked the Caro-Kann, a defence nobody
        # had played, and spent half the paragraph there). Telling a model not to fill a gap loses to the
        # gap. The fix is to stop leaving one: the replies are now HANDED OVER, measured from Maia at
        # strong-club strength, so "what comes next" is a fact to interpret rather than a blank to fill.
        # That also pulls this back inside the grounding invariant — the continuation is no longer the
        # model's recollection, it is a measurement.
        "THE REPLIES ARE GIVEN TO YOU — USE THOSE AND ONLY THOSE. The moves under TYPICAL REPLIES are what "
        "strong players actually play in the resulting position. Do NOT name any other continuation, and "
        "do NOT invent one: a reply that is not on that list is not typical, whatever you may recall.\n"
        "SHAPE — follow it in order, and write it as TWO PARAGRAPHS separated by a blank line (a literal "
        "\\n\\n in `text`) when part 2 applies — one wall of text reads as a lecture, not a coach:\n"
        "  1. WHAT THIS MOVE DOES: the idea behind the move just played — what it claims, frees, stops, or "
        "prepares. Always its own paragraph.\n"
        "  2. WHAT USUALLY COMES NEXT — ONLY IF the TYPICAL REPLIES section lists moves. After the blank "
        "line: one short lead-in sentence, then each reply as its OWN bullet (\"- 3. Qxd4, restoring "
        "material immediately\") saying in a few words what it's going for — they are the CHOICE the "
        "position poses, and a list is how a choice actually reads as a choice instead of a run-on "
        "sentence. If that section says none are available, STOP after part 1 — no second paragraph, and "
        "inventing the replies is not the fix.\n"
        "LENGTH: a short paragraph for part 1 (three or four sentences — the one place you are NOT terse), "
        "plus the bullet list for part 2 when it applies. Each bullet is one clause, not a paragraph.\n"
        "IF YOU ARE GIVEN A PREVIOUS OPENING NAME: the reader already had that explained. Write only what "
        "this move ADDS. If the change is a refinement within the same family, one or two sentences is "
        "plenty — do not restate the family.\n"
        "IF YOU ARE GIVEN AN ENGINE CLASS for the move: the line concedes something real, and THAT is the "
        "interesting part. Explain what it gives up and what it gets for it — the compensation, the "
        "practical bet, why the line exists despite the engine's preference. Do not scold; this is theory.\n"
        + _MOVE_NUMBER_RULE + _MARKDOWN_RULE +
        "Return JSON: {\"text\": string}."
    )

    @classmethod
    def system(cls, prev_name: str | None = None) -> str:
        # Narration is freeform-only by construction: a drill never narrates, so the perspective is
        # always the named-mover one. `prev_name` is accepted (not used) so callers can pass the same
        # argument shape as `Orchestrator._narrate_move` uses for the prompt BODY itself.
        return cls._head + _perspective(True) + cls._tail

    # -- user-prompt pieces ----------------------------------------------------------------------

    _context_template = "CONTEXT: {mover} just played {played}. This move is in a NAMED opening: {book}.\n"
    _moves_so_far_template = "MOVES SO FAR (this is the ENTIRE game; nothing else has been played): {moves_so_far}"
    _prev_name_template = (
        "\nPREVIOUS OPENING NAME (already explained to the reader): {prev_book}. "
        "Write only what this move ADDS to that."
    )
    _swing_note = (
        "\nNOTE: the engine does NOT prefer this move — its class is below. Yet it is "
        "theory. THAT tension is the subject: explain what the line gives up and what "
        "it gets for it. Do not scold; this is a known opening, not a blunder."
    )
    _facts_label = "\n\nThe move's grounded facts:\n{facts}\n"
    _reply_line_template = (
        "\nTYPICAL REPLIES — what players rated ~{rating} actually play in the "
        "position this move leads to, likeliest first: {replies}\n"
    )
    _no_replies_note = (
        "\nTYPICAL REPLIES: none available for this position. Do not name any "
        "continuation at all — stop after explaining the move that was played.\n"
    )
    _position_template = "\nThe position that follows:\n{position_read}\n"
    _closing = "\nNarrate this per your instructions. Return JSON."

    @classmethod
    def reply_line(cls, replies: list) -> str:
        # ALWAYS emitted, saying "none available" rather than being omitted — see the carve-out
        # above ("THE REPLIES ARE GIVEN TO YOU"): an omitted section is a gap, and a model fills
        # gaps. An explicit "none" is an instruction; an absence is an invitation.
        if not replies:
            return cls._no_replies_note
        return cls._reply_line_template.format(rating=_BOOK_RATING, replies=", ".join(replies))

    @classmethod
    def prompt(cls, *, mover: str, played: str, book: str, moves_so_far: str, prev_book: str | None,
               swing: bool, replies: list, facts: str, position_read: str | None) -> str:
        played, moves_so_far = san_guard(played, moves_so_far)   # SAN on the wire
        head = (cls._context_template.format(mover=mover, played=played, book=book)
                + cls._moves_so_far_template.format(moves_so_far=moves_so_far))
        if prev_book:
            head += cls._prev_name_template.format(prev_book=prev_book)
        if swing:
            head += cls._swing_note
        return (head
                + cls._facts_label.format(facts=facts)
                + cls.reply_line(replies)
                + (cls._position_template.format(position_read=position_read) if position_read else "")
                + cls._closing)


class EndbookPrompt(Prompt):
    """The end of the book: an explicit hand-off, so the voice changing is legible to the player
    rather than the coach silently developing a personality. Two parts, not one line: a bare
    "you're out of theory now" answers "what changed" but drops the player at the door with nothing
    — no recap of what they just played, no read of where they actually stand. This is the ONE
    moment the two voices meet: part 1 is still the book (closing it out), part 2 is the first
    breath of normal coaching.
    """

    _body = (
        "You are annotating a chess game. The players have just left opening theory — the position is no "
        "longer in the book. Write it as TWO PARAGRAPHS separated by a blank line (a literal \\n\\n):\n"
        "  1. Say plainly, without drama, that theory has ended — \"We're out of theory now.\" or similar "
        "— then in a sentence or two say what actually happened: follow the ACTUAL sequence under MOVES "
        "SO FAR in order (White's move, then Black's reply, then White's, ...) — never reorder it or "
        "describe a later move as if it caused an earlier one — and END on the specific move that just "
        "left the book (it is the LAST one in that list): what it does, and why it steps outside the "
        "line. You are given the opening's name and may draw on your own knowledge of it for colour (its "
        "ideas, typical plan, what each side was going for) — the same trusted exception as during "
        "narration — but the sequence itself and the final move are the subject, not the name alone. "
        "This is a recap of what already happened, not a new lesson.\n"
        "  2. \"Here's the read of the position now:\" (or similar), then the honest read of the CURRENT "
        "position — who stands better and why, in plain words (never win% or centipawns), grounded ONLY "
        "in the facts you are given below. From here on you are coaching, not narrating a book — this is "
        "the same honest read normal coaching would give.\n"
        + _perspective(True) + _MOVE_NUMBER_RULE + _MARKDOWN_RULE +
        "Ground part 2 ONLY in the facts given — never invent a piece, square, line, or number. Return "
        "JSON: {\"text\": string}."
    )

    @classmethod
    def system(cls) -> str:
        return cls._body

    # -- user-prompt pieces ----------------------------------------------------------------------

    _context_template = (
        "CONTEXT: {mover} just played {played}, and the line has now left opening theory. "
        "The opening that was played: {book}.\n"
    )
    _moves_so_far_template = "MOVES SO FAR (this is the ENTIRE game; nothing else has been played): {moves_so_far}\n\n"
    _position_read_template = "The read of the position now on the board:\n{position_read}\n"
    _closing = "\nWrite the hand-off per your instructions. Return JSON."

    @classmethod
    def prompt(cls, *, mover: str, played: str, book: str | None, moves_so_far: str,
               position_read: str) -> str:
        played, moves_so_far = san_guard(played, moves_so_far)   # SAN on the wire
        return (cls._context_template.format(mover=mover, played=played, book=book or "an unnamed line")
                + cls._moves_so_far_template.format(moves_so_far=moves_so_far)
                + cls._position_read_template.format(position_read=position_read)
                + cls._closing)


class GradePrompt(Prompt):
    """Grading a typed answer to a Socratic probe."""

    _body = (
        "You are a chess coach grading the player's answer to a question about the CURRENT position. You "
        "are given the engine's grounded analysis (best move, top moves, evals) and the player's message. "
        "Decide if the move/idea they propose is correct (matches or is among the engine's best). Give "
        "short, encouraging feedback grounded ONLY in the analysis — never invent a line or number.\n"
        "The player may not have typed real chess notation — coordinates ('c2d3'), engine shorthand, "
        "plain English. When 'Their move, named' is given below, that SAN is the ONLY spelling of their "
        "move you may use anywhere in your feedback or proposed_move — never quote their own typing back "
        "verbatim, and never put a move number in front of it (it hasn't been played). If it is NOT "
        "given, you cannot tell which move they mean — talk about their idea in words, and leave "
        "proposed_move ''.\n"
        "Respond as JSON with keys: proposed_move (string — 'Their move, named' verbatim if given, else "
        "''), correct (boolean), quality (number 0..1: 1.0 found the best cold, 0.5 close/after help, 0.3 "
        "needed the answer), feedback (string, shown to the player), concept (string, a one-word theme, "
        "or '')."
    )

    @classmethod
    def system(cls) -> str:
        return cls._body

    _prompt_template = (
        "Player's answer: {text}\n"
        "{named}"
        "Engine's grounded analysis (grade ONLY against this):\n{facts}\n\n"
        "Grade and give feedback as JSON."
    )

    @classmethod
    def prompt(cls, text: str, facts: str, named_move: str | None = None) -> str:
        named_move = san_guard(named_move)                       # SAN on the wire
        named = f"Their move, named: {named_move}\n" if named_move else ""
        return cls._prompt_template.format(text=text, facts=facts, named=named)


class LinePlayoutPrompt(Prompt):
    """Turns a typed what-if question into the concrete move sequence it proposes."""

    _body = (
        "You turn a player's what-if question into the sequence of THEIR OWN moves it proposes, in order. "
        "You are given the position and the legal moves (SAN) for the side to move. Return JSON "
        "{\"moves\": [SAN, ...]}: the player's intended moves in order — the opponent's in-between replies "
        "are NOT included (they get filled by best play). The FIRST move must be one of the legal moves "
        "listed. Use SAN. Return {\"moves\": []} if the question isn't about concrete move(s)."
    )

    @classmethod
    def system(cls) -> str:
        return cls._body

    _prompt_template = "Position FEN: {fen}\nLegal moves (side to move): {legal_moves}\nPlayer asked: {text}\nReturn JSON."

    @classmethod
    def prompt(cls, fen: str, legal_moves: list, text: str) -> str:
        return cls._prompt_template.format(fen=fen, legal_moves=", ".join(legal_moves), text=text)

    # -- the grounded-read block folded back into the COACH prompt afterwards --------------------
    # Not the system/user pair above (that pair only asks "what moves did the player mean") — this
    # is the SEPARATE block `Orchestrator._hypothetical_facts` returns once the proposed line has
    # been played out on the engine, which then gets concatenated straight into `CoachPrompt`'s user
    # message. It is exactly as much "an instruction sent to the model" as anything else here.

    _label = "\n\nGrounded read of the line the player asked about:\n"
    _line_template = "Played out with best replies for the opponent, the line runs: {line}."
    _diverged_note = "(The rest of the proposed line wasn't legal from there.)"
    _first_move_note = "The first move ({move}) is engine-classed a {move_class}."
    _resulting_note = "The resulting position is: {verdict}."

    @classmethod
    def grounded_read(cls, *, line: list, diverged: bool, first_class: str | None,
                       verdict: str | None) -> str:
        parts = [cls._line_template.format(line=" ".join(line))]
        if diverged:
            parts.append(cls._diverged_note)
        if first_class and first_class not in ("ok", "best", "only_move", "good"):
            parts.append(cls._first_move_note.format(move=line[0], move_class=first_class))
        if verdict:
            parts.append(cls._resulting_note.format(verdict=verdict))
        return cls._label + " ".join(parts)
