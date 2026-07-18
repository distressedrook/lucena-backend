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
        "The player just played the RIGHT move in this coaching exercise. Grounded ONLY in the facts and "
        "warm throughout, respond in TWO parts:\n"
        "FIRST — ONE lead sentence (no bullet): name the move, confirm it's right, and say what it "
        "achieves in the same breath (a piece won, forces mate, the winning resource the facts state). "
        "One line.\n"
        "THEN — only if the facts support more than that one point — a blank line, then a SHORT BULLETED "
        "LIST: each beat below on its OWN \"- \" bullet, one clause each, skipping any the facts don't "
        "support. If nothing beyond the lead applies, stop after the sentence (no list). Use EVERY "
        "specific insight the facts give you; do not flatten a rich point to a bare 'nice, that wins the "
        "piece'. The bullets:\n"
        "- THE THREAT CREATED — if the facts say the move CREATES A THREAT ('creates this threat: White "
        "threatens mate: Ra4#'), name it as the concrete reward. Never state a threat the facts do not "
        "name.\n"
        "- THE POINT — the instructive heart: if the facts name the position's linchpin or the resource "
        "the simple tries stumble into (a stalemate trap the win must dodge, a saving check like Rh1+, a "
        "mutually-defending pair), explain how THIS move handles it — that is WHY it is the move. If the "
        "facts name no such point, skip this bullet.\n"
        "HARD LIMITS: do NOT state or hint at the NEXT move, and do NOT continue the line past the move "
        "just played — the continuation is the un-played puzzle and naming it spoils it. Describe only "
        "what the played move itself accomplishes and why. Do NOT cite win% or centipawns."
    )
    _WRONG = (
        "The player just played a move that is NOT the answer. Grounded ONLY in the facts, staying "
        "encouraging and WITHOUT naming the correct move, respond in TWO parts:\n"
        "FIRST — ONE lead sentence (no bullet): name the move and its consequence — what it does to the "
        "position, using the swing the facts state (a winning position turned drawn or losing). One "
        "line. Do not soften a move the facts call losing into 'a small inaccuracy'.\n"
        "THEN — a blank line, then a SHORT BULLETED LIST: each beat below on its OWN \"- \" bullet, one "
        "clause each, in order, skipping any beat the facts don't support. Use EVERY specific insight "
        "the facts give you; do not reduce a rich explanation to a bare 'you lose the piece'. The "
        "bullets:\n"
        "- THE IDEA — ONLY if the facts literally credit the move's idea ('would fork the king and the "
        "queen — the right idea'), acknowledge it warmly. If the facts do NOT credit an idea, SKIP this "
        "beat entirely: do not guess what the player intended or where the piece was heading ('moving "
        "toward the centre', 'an ambitious try') — none of that is in the facts, and inventing it is a "
        "grounding error.\n"
        "- THE REFUTATION — the refuting move the facts name and the CONCRETE thing it does. If the "
        "facts say the move CREATES A THREAT ('creates this threat: White threatens mate: Ra4#'), credit "
        "it FIRST in the clause — the move is not senseless, it really does threaten that — then how the "
        "refutation defuses it. Never state a threat the facts do not name.\n"
        "- WHY IT WORKS — the facts give ONE exact mechanism (a defender was walked off; the piece moved "
        "onto a square the opponent still GUARDS; a piece was already hanging; or the move only DRAWS by "
        "STALEMATE — the opponent left with no legal move, so the win had to keep them a spare tempo). "
        "Use THAT specific reason — not a generic 'you left it undefended'; the wrong mechanism is a "
        "grounding error. If the facts say it stalemates, lead this bullet with it.\n"
        "- THE DEEPER LINE — ONLY if the refutation line the facts give continues past the first move "
        "(you grab material back, the opponent answers with a check), walk those EXACT moves to show the "
        "recovery also fails. Never invent a continuation, a fork, or a 'mate in N' not written in the "
        "line — a short line means a short bullet or none. You MAY cite a key defensive resource ONLY if "
        "the facts name it.\n"
        "- THE FIX — if the facts point at what to deal with first ('that bishop is what you must deal "
        "with first'), give that corrective nudge. Never name the solution move."
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
        "In ONE short sentence, in a coaching voice, say what your OPPONENT just played in reply — "
        "name the move and the CONCRETE thing it does from the facts (a capture, a check, where the "
        "piece went). Do NOT re-explain the player's own move, do NOT judge the reply, do NOT cite "
        "win%/centipawns, and do NOT continue the line."
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
