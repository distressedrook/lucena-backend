"""Prompts for the conversation spine (LLD §1, LLD-B/D/E).

The new-spine prompt families, all first-class citizens (CLAUDE.md): every instruction is
built here, never hand-assembled in the handlers.

  - FreeformPrompt      classify + answer a free-chat turn (dual-role)
  - ReadPrompt          grounded read of a quiet position / a move-explain
  - CoachTurnPrompt     classify a coach-mode text turn
  - VerdictPrompt       coach feedback on a bit answer — symmetric (right AND wrong)
  - PositionQueryPrompt answer a player's QUESTION about the current position (both modes)

`WhatIfPrompt` (the line-playout mechanism, was `ExtractPrompt`) and `GradePrompt`/`NarratePrompt`/
`EndbookPrompt` stay in `prompts.py`.

Perspective (settled): a POSITION-grounded answer names the colours in FREEFORM (there is no "you"
on a shared analysis board) but addresses the solver as "you" in COACH (they play one side) — hence
`_perspective(freeform)`. General chess KNOWLEDGE may use conversational "you" in either mode.
Spoil-control (§4): coach-side facts arrive as `solve_text()` — the solution/trap are absent, so no
prompt can leak them.
"""

from __future__ import annotations

from .grounding import _MARKDOWN_RULE, _MOVE_NUMBER_RULE, _NO_INVENTION_RULE, _perspective


class FreeformPrompt:
    """Freeform text classifier + cheap answerer (dual-role, ONE call). Classifies a free-chat turn
    and, for the cases needing no live position (reject / general knowledge), answers inline. A
    question ABOUT the position returns `needs_grounding` → the handler grounds and answers via
    PositionQueryPrompt. Output: {intent, text, coach}."""

    _RULES = (
        "Classify the message and act:\n"
        "- NOT chess-related → intent=reject; a brief polite decline in `text`.\n"
        "- A general chess question answerable from KNOWLEDGE ALONE — rules, an opening's ideas, what a "
        "tactic is, history — with no need to read the live board → intent=general; answer it in `text` "
        "(warm, direct; conversational 'you' is fine here).\n"
        "- A question ABOUT THE CURRENT POSITION — the plan here, why not a move, is a move better, what "
        "happens if I play X, is this winning → intent=needs_grounding; text=null (a grounded answerer "
        "handles it — do NOT answer from memory).\n"
        "- Asking to be COACHED / to solve / for a puzzle / to study → intent=dispatch_coach; fill "
        '`coach`: {"type": puzzle|endgame|midgame|opening, "motif": [tags] or [], "source": '
        '{"kind":"current"} to coach the position already on the board, OR '
        '{"kind":"new","theme": <tag or null>} for a fresh exercise}.\n'
    )
    _SCHEMA = (
        'Return JSON: {"intent":"reject"|"general"|"needs_grounding"|"dispatch_coach", '
        '"text": string or null (null ONLY for needs_grounding), '
        '"coach": {"type":string,"motif":[string],"source":object} or null (ONLY for dispatch_coach)}.\n'
    )

    @classmethod
    def system(cls) -> str:
        return ("You are a chess conversation assistant. Answer anything chess-related directly — you "
                "are NOT a socratic coach here, so never quiz the player.\n"
                + cls._RULES + _MOVE_NUMBER_RULE + _MARKDOWN_RULE + cls._SCHEMA)

    @classmethod
    def prompt(cls, *, text: str, convo=None) -> str:
        c = f"Recent conversation (for context — e.g. resolving a follow-up):\n{convo}\n\n" if convo else ""
        return f"{c}Player said: {text}\n\nRespond as JSON."


class PositionQueryPrompt:
    """Answers a player's QUESTION about the CURRENT position — any shape (the plan, why-not X, is Y
    better, what-if I play Z). Grounded ONLY in the facts; in coach the facts are `solve_text()` so the
    solution/trap are absent and it cannot spoil. Shared by BOTH modes — `freeform` picks the
    perspective (name the colours vs. address the solver as 'you'). Output: {text}."""

    @classmethod
    def system(cls, *, freeform: bool) -> str:
        return ("You are a chess coach answering the player's question about the position. Translate "
                "evaluations into plain words, never win% or centipawns. If the facts don't settle the "
                "question, answer what you CAN ground and say the rest isn't clear from here. 1-3 "
                "sentences.\n"
                + _NO_INVENTION_RULE + _perspective(freeform) + _MOVE_NUMBER_RULE + _MARKDOWN_RULE
                + 'Return JSON: {"text": string}.')

    @classmethod
    def prompt(cls, *, text: str, facts: str) -> str:
        return f"Player asked: {text}\nGrounded facts (answer ONLY from these):\n{facts}\n\nRespond as JSON."


class ReadPrompt:
    """The grounded read of a QUIET position or a just-played move in FREEFORM — reached only on a
    deterministic trigger (a paste that ISN'T a puzzle, or a played move that isn't book), so it never
    classifies and never faces a solution to hide. Output: {text}."""

    # Grounding-adherence has to be forceful: the placeholder version invented "a passed pawn on f7"
    # (the no-invention specifics now live in the shared _NO_INVENTION_RULE).
    _GROUND = (
        "Ground EVERY claim in a specific fact line below. Translate evaluations into plain words "
        "('White is much better', 'roughly equal') — never win% or centipawns. 2-4 sentences, one "
        "clear idea.\n"
    )

    @classmethod
    def system(cls) -> str:
        return ("You are a chess coach giving a grounded read of the position on a shared analysis "
                "board.\n" + cls._GROUND + _NO_INVENTION_RULE + _perspective(freeform=True)
                + _MOVE_NUMBER_RULE + _MARKDOWN_RULE + 'Return JSON: {"text": string}.')

    @classmethod
    def prompt(cls, *, facts: str, played: str | None = None) -> str:
        head = f"{played} was just played.\n" if played else ""
        return f"{head}Grounded facts (read ONLY from these):\n{facts}\n\nRespond as JSON."


class CoachTurnPrompt:
    """ROUTES a player's TYPED message during a coaching exercise → answer | whatif | stop | general.
    (A move arrives via the board, not here.) It only routes — the grounded answer for `general` comes
    from PositionQueryPrompt (spoil-safe), so no answer text is produced here. Output: {intent}."""

    _SCHEMA = ('Return JSON: {"intent": "answer"|"whatif"|"stop"|"general"}.\n')

    @classmethod
    def system(cls, bit) -> str:
        expects_text = getattr(getattr(bit, "spec", None), "strategy", "") == "free_text"
        answer_rule = (
            "- It is the player's ANSWER to the current task (this task wants a TYPED answer) → "
            "intent=answer.\n" if expects_text else
            "- (This task is solved by a MOVE ON THE BOARD, not by typing — a typed message is NEVER "
            "the answer here.)\n")
        return ("You are routing a player's typed message during a coaching exercise. Decide the intent:\n"
                + answer_rule +
                "- It asks 'what if <a specific move/line>' — wanting to try an alternative on the board "
                "→ intent=whatif.\n"
                "- It asks to stop / quit / do something else → intent=stop.\n"
                "- Anything else — a question about the position, a general chess question, a remark → "
                "intent=general.\n"
                + cls._SCHEMA)

    @classmethod
    def prompt(cls, *, text: str, bit=None, convo=None) -> str:
        task = getattr(getattr(bit, "spec", None), "challenge", None) or "(the current task)"
        c = f"Recent conversation:\n{convo}\n\n" if convo else ""
        return f"{c}Current task: {task}\nPlayer said: {text}\n\nRespond as JSON."


class VerdictPrompt:
    """Coach feedback on a bit answer — SYMMETRIC (right AND wrong), one grounded voice for both
    (replaces the old asymmetry: a hardcoded "That's it." on right, an LLM only on wrong). The
    adjudicator already decided the bool; this only VOICES it. Facts are `solve_text()`, so the
    solution/trap are structurally absent — it can neither spoil the next move nor invent. Output:
    {text}."""

    _RIGHT = (
        "You are a warm chess coach and the student just played the RIGHT move. Reward it and TEACH — in "
        "1–3 short sentences of plain prose, grounded ONLY in the facts. The student can already read an "
        "engine; give them the human part: WHY it works and the pattern worth remembering.\n"
        "1. Open with brief, genuine warmth and name the move's POINT — the DEEPEST reason the facts "
        "give, never the surface capture ('removes the only defender of the bishop on a2' beats 'wins "
        "the knight'). Say it ONCE — do not state a capture AND a reason that already implies it.\n"
        "2. IF A FACT BEGINS 'The point of this move:', THAT is your lead — reproduce its logic "
        "FAITHFULLY: keep its VERB ('undermines' / 'removes' / 'wins' / 'wins material after a recapture' "
        "are different — do not swap them) and its defender relationship in the SAME direction. Phrase it "
        "naturally, but do not restructure it into a different claim. A bare capture, or a generic "
        "positional note (the material count), is NOT the lead when a 'point' fact is present.\n"
        "3. If it helps the student SEE it next time, restate the point's pattern in plain words — but "
        "ONLY the pattern the facts themselves state; never add a tactical label the facts don't use. Add "
        "a second, DISTINCT point only if the facts give one (e.g. it also threatens "
        "mate — then name the mating MOVE as the facts give it ('Ra4#') and write the WORD 'mate', with "
        "the count only if the facts give it; never reduce a mate to a bare square or the king's square).\n"
        "HARD LIMITS: do NOT state or hint at the NEXT move, and do NOT continue the line past the move "
        "just played — the continuation is the un-played puzzle and naming it spoils it. Do NOT cite "
        "win% or centipawns, and invent no move, capture, threat, or motif the facts do not carry."
    )
    _WRONG = (
        "You are a warm chess coach. The student just played a move that is NOT the best one here. This is "
        "a gentle NUDGE to look again — suggestive, encouraging, never a harsh verdict. Ground EVERY claim "
        "in the facts; invent nothing. Write 2–3 SHORT sentences of plain prose (no bullets, no move-by-"
        "move dump).\n"
        "1. LEAD with THEIR move — name the move the student played and react to it warmly. If the facts "
        "credit its idea ('the right idea'), acknowledge that first; otherwise do not guess what they "
        "intended. Frame it softly ('this one runs into a snag', 'not quite — watch what comes back'), not "
        "as a blunt 'this is a blunder that turns winning into losing'.\n"
        "2. Show the snag using ONLY the opponent's IMMEDIATE reply — the FIRST move of the refutation "
        "('but then Black just answers Nxe4 and it's gone'). Do NOT skip ahead to a LATER move in the line "
        "(a queen move three moves deep is NOT what refutes their move — leading with it is the mistake we "
        "are fixing); name at most that one first reply, in words, and stop. If the facts say the move "
        "allows a real mate, say so with the mate MOVE the facts give ('runs into Ra4#') — the exact move, "
        "never a bare square or the king's square.\n"
        "3. Close with a soft, suggestive nudge to look again for something stronger. If a HINT fact is "
        "present, fold it in as that nudge (in your own words, revealing no more than it states); "
        "otherwise a short open question ('worth another look — what's the loosest enemy piece?'). Never "
        "point them at defending when the facts say they were winning.\n"
        "HARD GROUNDING RULES — breaking these is the failure we most need to avoid:\n"
        "- Never name, spell out, or hint the specific correct MOVE.\n"
        "- Never invent a move, capture, threat, tactical motif, check, or 'mate in N' the facts do not "
        "literally state, and never extend a line past where the facts end.\n"
        "- Name a captured piece ONLY as the facts name it; state a mate only if the facts do; do not "
        "guess what they intended."
    )

    @classmethod
    def system(cls, *, correct: bool, player_color: str | None = None) -> str:
        # Coach perspective anchored on the player's actual COLOUR, not "the side to move": by verdict
        # time the move is on the board, so the board shows the OPPONENT to move — "you play the side
        # to move" made the model read the board and flip White/Black. `player_color` fixes the anchor.
        return ((cls._RIGHT if correct else cls._WRONG) + "\n"
                + _NO_INVENTION_RULE + _perspective(freeform=False, player_color=player_color)
                + _MOVE_NUMBER_RULE + 'Return JSON: {"text": string}.')

    @classmethod
    def prompt(cls, *, attempt: str, facts: str) -> str:
        return f"Their move: {attempt}\nGrounded facts (the engine's read of the move they played):\n{facts}\n\nRespond as JSON."


class OpponentReplyPrompt:
    """The SECOND coaching beat after a correct drill move: what the OPPONENT just did in reply. The
    walker auto-plays the defence; this voices it so the player sees the position move, not just their
    own move adjudicated. Grounded ONLY in `_brief_reply` (the one move, its capture, check) — it
    states what happened, never invents a plan or motif the facts don't carry. The player's own move
    was already voiced by VerdictPrompt; this beat is purely the reply."""

    _BODY = (
        "In ONE short, natural sentence — a coach's aside, not a move log — say what your OPPONENT just "
        "played in reply. If it CAPTURES or gives CHECK, lead with that concrete consequence. If it's a "
        "quiet move, just name it and stop — do NOT narrate the obvious ('moving the knight to the c6 "
        "square'); the board already shows where it went. Do NOT re-explain the player's own move, judge "
        "the reply, cite win%/centipawns, invent a plan or purpose, or continue the line."
    )

    @classmethod
    def system(cls, *, player_color: str | None = None) -> str:
        return (cls._BODY + "\n" + _NO_INVENTION_RULE
                + _perspective(freeform=False, player_color=player_color)
                + _MOVE_NUMBER_RULE + 'Return JSON: {"text": string}.')

    @classmethod
    def prompt(cls, *, facts: str) -> str:
        return f"Grounded facts (the engine's read of the opponent's reply):\n{facts}\n\nRespond as JSON."


class TrapPrompt:
    """Voices the poisoned-line moments in COACH mode — the WARN before solving (a tempting move
    loses) and the REVEAL after solving (name the trap the player sidestepped and why). Replaces the
    raw template dump that surfaced `tiered_bit_grounding`'s f-strings verbatim. Grounded ONLY in the
    tiered trap facts (§4): the warn tier carries the EXISTENCE of a trap and NOT the move, so the
    warning structurally cannot leak it; the reveal tier carries the line + catch and is handed over
    only once solved. Output: {text}."""

    _WARN = (
        "There is a natural-looking move in this position that many players are tempted by but which "
        "actually LOSES. In ONE short sentence, in a coaching voice, warn the player to look carefully "
        "before committing — WITHOUT naming the move or the line (you are not given it, and must not "
        "guess it)."
    )
    _REVEAL = (
        "The player SOLVED the exercise by sidestepping a trap. Surface it: in 1-2 short sentences, "
        "name the tempting line and say plainly that it LOSES — it is a natural-looking try that many "
        "players fall for, NOT the engine's best move — then credit the player for avoiding it. State "
        "the moves and that it loses; nothing beyond that."
    )

    @classmethod
    def system(cls, *, reveal: bool) -> str:
        return ((cls._REVEAL if reveal else cls._WARN) + "\n"
                + _NO_INVENTION_RULE + _perspective(freeform=False) + _MOVE_NUMBER_RULE
                + _MARKDOWN_RULE + 'Return JSON: {"text": string}.')

    @classmethod
    def prompt(cls, *, facts: str) -> str:
        return f"Trap facts (voice ONLY these; add nothing):\n{facts}\n\nRespond as JSON."
