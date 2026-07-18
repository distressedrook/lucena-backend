"""Shared prompt plumbing: voice/perspective, move-number and PV formatting, and grounded-fact
briefings — the building blocks every coaching prompt in this package composes from. Pulled out
of orchestrator.py because every one of these is reused across multiple prompt families (coach,
narrate, endbook, drill, wrong, move) rather than belonging to any single one of them.

No Orchestrator, no LLM, no engine calls: pure functions over dicts and FEN/SAN strings only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Shared across every prompt that hands the model a move to name. The number is COMPUTED (see
# `_numbered`) and baked into the SAN wherever a move is interpolated into CONTEXT/facts text —
# this just tells the model to keep what it was given rather than strip it or roll its own.
_MOVE_NUMBER_RULE = (
    "MOVE NUMBERS: moves you were given already carry their move number — \"12. Nf3\", "
    "\"12... Nf3\" for Black. Keep that exact form when you name them; never write the bare SAN "
    "alone, and never invent a number for a move that wasn't handed to you numbered.\n"
)

# The app renders light Markdown (bold/italic inline, "- " bullet lines) — nothing else. Shared for
# the same reason as `_MOVE_NUMBER_RULE`: it belongs in every prompt whose `text` reaches the chat,
# not duplicated per prompt and inevitably left out of the next one that's added.
_MARKDOWN_RULE = (
    "FORMATTING: `text` may use light Markdown — **bold** or *italic* for emphasis, and a \"- \" "
    "bullet per line for a short list of options. Nothing else: no headers, no links, no code "
    "spans, no nested lists. Most replies need none of this — reach for a bullet list only when "
    "there are several distinct options to lay out (e.g. several typical replies), and bold only "
    "the one or two words actually worth the reader's eye.\n"
)


# The one discipline every strictly-grounded prompt needs, and the one they kept drifting on: the
# model editorializing PAST the facts. Three separate live bugs — an invented "isolated pawn" on a
# right verdict, a "pinning threat" Qe3 never made, a "knight sacrifice" that was a pawn grab — were
# all the same failure: naming a structure/motif/characterization the grounding never stated. Shared
# for the same reason as the rules above: it belongs in every grounded prompt, not re-imagined per
# prompt and left out of the next one. NOT for the opening-narration carve-out (LLD §9), which may
# speak to a named opening's ideas beyond the facts — this rule forbids exactly that.
_NO_INVENTION_RULE = (
    "GROUNDING (critical): say ONLY what the facts state. Do NOT invent a piece, square, line, "
    "structure, or evaluation. Do NOT name a tactical motif (pin, fork, skewer, discovered attack, "
    "zugzwang, …) or characterize a move (sacrifice, combination, brilliancy) unless the facts use "
    "that word — naming one they do not license is the most common way to be confidently wrong (a "
    "knight capturing a pawn wins material; it is not a sacrifice). When the facts don't say WHY, "
    "state the concrete consequence they DO give and stop — never supply a reason of your own.\n"
)


def _perspective(freeform: bool, player_color: str | None = None) -> str:
    """The voice block: who the coach is talking to.

    Two bodies, one function, because the answer is a property of the MODE and it appears in many
    prompts — duplicating it is how they drift apart.

    In a DRILL the engine really does reply (drill.py plays the defence), so the player genuinely is
    one side and "you" is correct. In FREEFORM nothing replies — `play_move` applies one ply and
    stops, and the board has no side-to-move gate, so either colour is draggable. The player is
    driving both sides of an analysis board: there is no "you" to address, only White and Black.
    Saying "you played e4" there is not a style choice, it is factually wrong.

    `player_color` names the player's actual colour (a verdict knows it — the answer's FEN is the
    pre-move position, whose side to move IS the player). Passing it replaces the fragile "you play
    the side to move" with a fixed anchor: by verdict time the move is already on the board, so the
    board shows the OPPONENT to move — a model told "you are the side to move" reads the board, sees
    the opponent's turn, and flips the whole perspective. Naming the colour outright kills that.
    """
    if freeform:
        return (
            "PERSPECTIVE (critical): there is NO 'you' here. The player is moving BOTH sides on an "
            "analysis board — nobody is 'the player's colour'. Name the mover: 'White stakes the "
            "centre', 'Black challenges it'. The words 'you', 'your', and 'yours' may NEVER appear "
            "in `text` — not even if a fact below tells you which colour the player is looking at; "
            "that fact is for YOU to resolve what THEY meant, not permission to address them "
            "directly. Say 'White's king' / 'Black's rook', never 'your king' / 'your rook'. Never "
            "attribute a move or plan to 'the player'. Threats and plans belong to the colour that "
            "owns them. Before finishing, re-read `text`: if it contains 'you', 'your', or 'yours', "
            "rewrite that sentence naming the colour instead.\n"
        )
    if player_color:
        c = player_color.capitalize()
        opp = "Black" if player_color == "white" else "White"
        you_num = "N." if player_color == "white" else "N..."
        opp_num = "N..." if player_color == "white" else "N."
        return (
            f"PERSPECTIVE (critical — getting it backwards ruins the read): YOU are the player and you "
            f"are playing {c}. Address the player as 'you'. YOUR moves are {c}'s moves, written "
            f"'{you_num}' (e.g. '5. Nf3' is White, '5... Nf3' is Black); your OPPONENT is {opp}, whose "
            f"moves are written '{opp_num}'. You have ALREADY made your move, so the board now shows "
            f"your OPPONENT to move — do NOT infer your colour from whose turn it is; you are {c}, "
            f"period. Every threat, attack, or plan by {opp} belongs to the OPPONENT — never say you "
            f"are threatening your own pieces or defending against yourself. When a line labels a move "
            f"'(you)' or '(opponent)', trust that label exactly and never swap the two.\n"
        )
    return (
        "PERSPECTIVE (critical — getting it backwards ruins the read): address the player as 'you'; "
        "they play the side to move. The OTHER colour is 'your opponent'. Every threat, attack, or "
        "plan belongs to the OPPONENT — never say the player is threatening their own pieces or "
        "defending against themselves. Name the opponent's threat when there is one.\n"
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


def _numbered(san: str | None, fen: str | None) -> str:
    """SAN prefixed with its move number, computed from the FEN the move is played FROM —
    "12. Nf3" for White, "12... Nf3" for Black. The same fullmove-number arithmetic as
    `Orchestrator._played_line`, just read forward off the PRE-move FEN (which already names the
    mover and the number) instead of backward off the post-move one.

    This is the only move number the model is ever handed, and the point of computing it here
    rather than asking the model to work it out is that it must not have to: a wrong number is
    exactly the kind of invented fact the rest of this file spends its effort grounding out.
    """
    if not san or not fen:
        return san or ""
    parts = fen.split()
    num = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
    white_to_move = (parts[1] if len(parts) > 1 else "w") == "w"
    return f"{num}. {san}" if white_to_move else f"{num}... {san}"


def _numbered_line(pv: list, fen: str | None, *, played_by_white: bool) -> list:
    """Every move of a PV, numbered — "3. axb5", "3... e6", "4. d4", ... — for the SAME reason
    `_numbered` exists: reciting an unlabeled list of moves is move-number arithmetic, and that is
    exactly the arithmetic a refutation line asks the model to do six times in a row. Caught live:
    handed the bare pv "axb5 Bb7 Nc3 a6 Nf3 e6", the model recited it as "3.axb5 3...axb5 4.Bb7
    5.Nc3 6.a6 7.Nf3 8.e6" — duplicating the first move and mislabeling every one after it. Every
    move in that line was real; only the numbering was invented, because nothing grounded it.

    `fen` is the PRE-move position (the one the already-played move, and so this whole `pv`, hangs
    off). The fullmove number only advances when Black moves, so `pv`'s own numbering falls out of
    `fen`'s number plus one adjustment for the move already played — no need to replay the line on
    a board to track it.
    """
    parts = (fen or "").split()
    if len(parts) <= 5 or not parts[5].isdigit():
        return list(pv)   # no fullmove data to ground a number in — bare SAN beats a wrong one
    num = int(parts[5]) + (0 if played_by_white else 1)
    white_to_move = not played_by_white
    out = []
    for san in pv:
        out.append(f"{num}. {san}" if white_to_move else f"{num}... {san}")
        if not white_to_move:
            num += 1
        white_to_move = not white_to_move
    return out


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
    # The engine's own move-level facts ("bxc4 wins the bishop on c4", "Nxf6+ is strong for the
    # opponent") — the concrete WHY, previously dropped so the verdict had only class + refutation to
    # reason from. Gated on `not hide_best`: on a WRONG drill move a fact can NAME the solution move, so
    # withholding the best move must withhold these too (never capitalise — 'bxc4' is a pawn capture,
    # 'Bxc4' a bishop; the same first-letter slip we just fixed in the trap voice).
    if not hide_best:
        for f in (v.get("facts") or []):
            if isinstance(f, dict) and (t := f.get("text")):
                out.append(t if t.endswith((".", "!", "?")) else t + ".")
    if not hide_best and (b := v.get("best")) and b.get("san") and b.get("san") != v.get("san"):
        out.append(f"The engine's best move here is {b['san']}.")
    if pv := v.get("refutation_pv"):
        first = pv[0]
        piece = _PIECE_WORD.get(first.rstrip("+#")[:1], "pawn")
        # Numbered here rather than handed over bare: see `_numbered_line` — reciting an unlabeled
        # PV is move-number arithmetic, and that arithmetic is exactly what was going wrong.
        numbered = _numbered_line(pv[:6], v.get("fen"), played_by_white=v.get("side_to_move") == "white")
        # The refutation is the OPPONENT's line: it opens with their punishing move, then alternates
        # (opponent, you, opponent, …). Label every move's side explicitly — the bare mixed line let
        # the model flip who's who (a live wrong-verdict read the opponent's move as the player's).
        labeled = " ".join(f"({'opponent' if i % 2 == 0 else 'you'}) {m}" for i, m in enumerate(numbered))
        out.append(f"The opponent refutes it with {numbered[0] if numbered else first} "
                   f"({_move_phrase(first)}); the line then runs {labeled}. Explain the flaw ONLY through "
                   f"this line — the refuting move is the opponent's {first}, a {piece} move, nothing else.")
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


# -- §4: grounding visibility tiers (spoil-control) -------------------------------------------------
# Grounding is not a flat bundle: each fact carries a visibility tier the dispatch honours when
# building what reaches the player during a SOLVE vs at REVEAL. The guard is STRUCTURAL — a withheld
# fact is physically absent from the solve-time text, never present-with-a-"don't-say-it" note (the
# trust-the-model failure this whole redesign removes). The poisoned line folds in HERE as one
# customer (warn_only + reveal_on_resolve), not its own subsystem.
ALWAYS = "always"
WARN_ONLY = "warn_only"
REVEAL_ON_RESOLVE = "reveal_on_resolve"


@dataclass
class TieredFacts:
    always: list = field(default_factory=list)   # safe to say any time (the positional read)
    warn: list = field(default_factory=list)     # warn_only: existence, never the detail
    reveal: list = field(default_factory=list)   # reveal_on_resolve: WITHHELD until solved

    def solve_text(self) -> str:
        """What may reach the LLM/player WHILE solving: always + warn. reveal is physically ABSENT."""
        lines = list(self.always) + list(self.warn)
        return "\n".join(str(x) for x in lines) if lines else "(no grounded read available)"

    def reveal_text(self) -> str:
        """The reveal_on_resolve content, surfaced only AT conclusion. '' when there's nothing to reveal."""
        return " ".join(str(x) for x in self.reveal)

    def warn_text(self) -> str:
        return " ".join(str(x) for x in self.warn)


def tiered_bit_grounding(resp, tree) -> TieredFacts:
    """Tier a coach bit's grounding: the positional read (`resp`, may be None) is `always`; the
    move_line `tree`'s poisoned line becomes a `warn_only` warning + a `reveal_on_resolve` detail.
    The best move is DELIBERATELY excluded (it is the solution — never fed during a solve, and the
    player has played it by reveal time), so `solve_text()` cannot leak it."""
    tf = TieredFacts()
    if isinstance(resp, dict):
        lines = resp.get("analysis")
        if isinstance(lines, list):
            tf.always.extend(str(x) for x in lines)      # positional read only — NOT the best move
    tree = tree or {}
    if tree.get("has_poisoned_line"):
        tf.warn.append("There's a tempting move in this position that actually loses — "
                       "calculate carefully.")
        moves = tree.get("poisoned_line_moves") or []
        meta = tree.get("poisoned_line_meta") or {}
        san = " ".join(m.get("san") for m in moves if m.get("san"))
        parts = [f"The tempting line {san} looks winning but loses" if san
                 else "A tempting move there loses"]
        if meta.get("idea"):
            parts.append(f"the catch is {meta['idea']}")
        if (fatal := meta.get("fatal")) and fatal not in (meta.get("idea") or ""):
            parts.append(f"the motif is a {fatal}")
        tf.reveal.append("; ".join(parts) + ".")
    return tf
