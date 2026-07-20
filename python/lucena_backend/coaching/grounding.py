"""Shared prompt plumbing: voice/perspective, move-number and PV formatting, and grounded-fact
briefings — the building blocks every coaching prompt in this package composes from. Pulled out
of orchestrator.py because every one of these is reused across multiple prompt families (coach,
narrate, endbook, drill, wrong, move) rather than belonging to any single one of them.

No Orchestrator, no LLM, no engine calls: pure functions over dicts and FEN/SAN strings only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Shared across every prompt that hands the model a move to name. The number is COMPUTED (see
# `_numbered`) and baked into the SAN wherever a move is interpolated into CONTEXT/facts text —
# this just tells the model to keep what it was given rather than strip it or roll its own.
_MOVE_NUMBER_RULE = (
    "MOVE NUMBERS: keep each move's number EXACTLY as handed to you — never add, strip, or invent one. "
    "Lines are PGN-style: White carries the number (\"12. Nf3\"); a Black move carries \"N...\" ONLY "
    "when it OPENS a line (\"12... Nf3\"), and is BARE when it follows White's move in the same line "
    "(\"5. cxd4 Qg1+\", NEVER \"5. cxd4 5... Qg1+\"). A move handed to you bare stays bare.\n"
    "MOVE TOKENS (critical): the ONLY moves you may name are the move the player just played and the "
    "moves written in the facts. NEVER write any other move — a plausible-looking move that is not "
    "there (a queen promotion narrated as \"Kb2\") is a hallucination. If you are unsure of a move's "
    "notation, write \"your move\" / \"the opponent's reply\" instead of guessing it.\n"
)

# The app renders light Markdown (bold/italic inline, "- " bullet lines) — nothing else. Shared for
# the same reason as `_MOVE_NUMBER_RULE`: it belongs in every prompt whose `text` reaches the chat,
# not duplicated per prompt and inevitably left out of the next one that's added.
_MARKDOWN_RULE = (
    "FORMATTING: `text` may use light Markdown — **bold** or *italic* for emphasis, and a \"- \" "
    "bullet per line. Nothing else: no headers, no links, no code spans, no nested lists. When your "
    "reply makes SEVERAL DISTINCT POINTS — separate observations, options, steps, or beats (the "
    "idea, the refutation, why it fails, the fix) — give each its OWN \"- \" bullet so they read as "
    "distinct, not one run-on paragraph. Keep each bullet to one clause or short sentence. A reply "
    "that makes a SINGLE point stays a plain sentence — do not bullet a lone point, and do not split "
    "one thought across bullets. Bold only the one or two words actually worth the reader's eye.\n"
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
    "structure, or evaluation. Do NOT name any tactical motif, or characterize the move, unless the "
    "facts use that exact word — naming one they do not license is the most common way to be "
    "confidently wrong (a knight capturing a pawn wins material; do not upgrade it). Do NOT claim a "
    "MATE, a CHECK, or a "
    "threat against a king that the facts do not state: if the facts describe a repetition, a trade, or "
    "winning a piece, say exactly THAT — never escalate a quiet line into an attack or a 'mate'. Do NOT "
    "attribute a PURPOSE or "
    "DIRECTION to a move the facts don't state — not 'heading for the centre', 'bringing the king to "
    "safety', 'an ambitious try', 'developing a piece'. A move to g2 is a move to g2, not a move "
    "'toward the centre'. When the facts don't say WHY, state the concrete consequence they DO give "
    "and stop — never supply a reason, plan, or flavour of your own.\n"
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
            f"period. The facts below name sides by their absolute COLOUR, never 'you'/'opponent': "
            f"'{c}' is you, '{opp}' is your opponent. So a fact reading '{opp} threatens …' is your "
            f"OPPONENT's threat, and '{c} …' is yours — map the colour to the right person and never "
            f"swap them. Every threat, attack, or plan by {opp} belongs to the OPPONENT — never say "
            f"you are threatening your own pieces or defending against yourself.\n"
        )
    return (
        "PERSPECTIVE (critical — getting it backwards ruins the read): address the player as 'you'; "
        "they play the side to move. The OTHER colour is 'your opponent'. Every threat, attack, or "
        "plan belongs to the OPPONENT — never say the player is threatening their own pieces or "
        "defending against themselves. Name the opponent's threat when there is one.\n"
    )


_PIECE_WORD = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}


def _stm_color(fen: str | None) -> str:
    """The side to move in `fen`, as an absolute colour ('White'/'Black'). Facts use colours, never
    relative 'you'/'the opponent' — relative words invert with side-to-move and flip who a threat
    belongs to; the PERSPECTIVE anchor maps the player's colour onto 'you' for the reader."""
    parts = (fen or "").split()
    return "Black" if len(parts) > 1 and parts[1] == "b" else "White"


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


def _pv_capture_victims(fen: str | None, played_san: str | None, pv: list) -> list:
    """For each capturing move in the refutation `pv`, the piece it TAKES, as a word — walked on a
    board so the coach never guesses the victim. A bare SAN ("Rxe1") names the mover and the square
    but not what stands on it; handed only that, the model invented the captured piece (a live verdict
    called the rook on e1 a 'knight'). Aligned to `pv`; None where a move is not a capture or can't be
    resolved (bad FEN, en-passant, an out-of-line SAN). `played_san` steps the board once before
    walking `pv` (the refutation hangs off the played move); pass None to walk `pv` straight from
    `fen` (e.g. a single reply already played FROM `fen`)."""
    if not fen:
        return [None] * len(pv)
    try:
        from lucena_engine.board import Board
        b = Board(fen)
        if played_san is not None:
            played = next((m for m in b.legal_moves() if b.san(m) == played_san), None)
            if played is None:
                return [None] * len(pv)
            b = b.apply(played)                   # step into the position the refutation hangs off
        victims: list = []
        for san in pv:
            uci = next((m for m in b.legal_moves() if b.san(m) == san), None)
            if uci is None:
                break                             # line diverged; leave the rest unresolved
            victim = None
            if "x" in san:
                dest = uci[2:4]
                victim = next((_PIECE_WORD.get(p.piece.upper(), "pawn") for p in b.piece_list()
                               if p.square == dest), None)   # a piece we found but can't name is a pawn
            victims.append(victim)
            b = b.apply(uci)
        return victims + [None] * (len(pv) - len(victims))
    except Exception:
        return [None] * len(pv)


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
    for i, san in enumerate(pv):
        # PGN style: White carries the number ('5. cxd4'); Black carries 'N...' ONLY when it OPENS the
        # line (nothing before it), else it is bare ('5. cxd4 Qg1+', not '5. cxd4 5... Qg1+').
        if white_to_move:
            out.append(f"{num}. {san}")
        elif i == 0:
            out.append(f"{num}... {san}")
        else:
            out.append(san)
        if not white_to_move:
            num += 1
        white_to_move = not white_to_move
    return out


# A SAN-ish move token: castling, a piece move (optionally with disambiguation/capture), a pawn
# capture, or a promotion. Deliberately NOT bare pawn pushes like "e4" — those collide with square
# names ("the rook on c3") and would false-positive; the hallucinations we must catch (Ra4#, Kc8,
# invented lines) all carry a piece letter, a capture, or a promotion.
# NB: no trailing \b — a token ending in '+'/'#' has no word boundary after it, so \b would drop the
# check/mate suffix (and the guard could never tell 'Ra4#' from 'Ra4'). A negative lookahead for a
# continuing square char is enough to avoid partial matches.
_SAN_TOKEN = re.compile(
    r"\b(?:O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][18]=[QRBN][+#]?)(?![a-h1-8])")


def _move_tokens(text: str | None) -> set[str]:
    """The SAN move tokens in a string, KEEPING the check/mate suffix (so 'Ra4#' and 'Ra4' are
    distinct — a false mate annotation is itself a hallucination)."""
    return set(_SAN_TOKEN.findall(text or ""))


def _invented_moves(output: str | None, facts: str | None, played: str | None) -> set[str]:
    """Moves the model NAMED that are NOT grounded. Two hallucinations, both caught:
      1. the MOVE itself isn't in the facts / isn't the move played (a promotion as 'Kb2').
      2. the move IS grounded but the model added a CHECK or MATE the facts never gave it ('Ra4#'
         when the facts have only 'Ra4' — a drawn rook shuffle dressed up as mate). The move's colour
         is the facts' to state; a '#' the engine didn't write is invention.
    A non-empty result means the verdict must be regenerated or dropped."""
    fact_full = _move_tokens(facts)
    if played:
        fact_full.add(played)
    fact_core = {t.rstrip("+#") for t in fact_full}
    bad = set()
    for tok in _move_tokens(output):
        core = tok.rstrip("+#")
        if core not in fact_core:
            bad.add(tok)                                   # (1) the move isn't grounded at all
        elif tok != core and tok not in fact_full:
            bad.add(tok)                                   # (2) a check/mate the facts never stated
    return bad


_MOTIF_WORDS = ("fork", "pin", "skewer", "discovered attack", "discovered check", "double check",
                "deflection", "decoy", "overload", "interference", "zwischenzug", "zugzwang",
                "windmill", "x-ray")


def _invented_motif(output: str | None, facts: str | None) -> str | None:
    """A tactical MOTIF the model NAMED that the facts never license — flash-lite dressed a bare 'mate in
    4' up as 'a fork attacking the king and the rook' when no fork exists. `_NO_INVENTION_RULE` forbids
    this in the prompt, but the model ignores it, so it is enforced deterministically here. Word-stem
    match (\\bfork catches fork/forks/forking) with a leading boundary so a substring like 'opinion'
    can't false-trigger 'pin'. Returns the offending motif or None; a hit means REGENERATE or drop."""
    o = (output or "").lower()
    f = (facts or "").lower()
    for m in _MOTIF_WORDS:
        if re.search(r"\b" + re.escape(m), o) and not re.search(r"\b" + re.escape(m), f):
            return m
    return None


def _solution_moves(tree: dict | None) -> list[str]:
    """The move(s) the puzzle expects at the root — the answer, which the deep-tactics read must NOT
    name to a solver. `solve` → its one required move; `mate` → any of the mating options."""
    root = (tree or {}).get("root") or {}
    if root.get("kind") == "solve" and root.get("expect_san"):
        return [root["expect_san"]]
    if root.get("kind") == "mate":
        return [o.get("san") for o in (root.get("options") or []) if o.get("san")]
    return []


def _fen_key(fen: str | None) -> str:
    """Position identity for matching: placement + side + castling + ep, dropping the move clocks."""
    return " ".join((fen or "").split()[:4])


def _node_at(tree: dict | None, fen: str | None) -> dict | None:
    """The player node whose position matches `fen` (clocks ignored), or None. Searches the WHOLE
    solution tree — every mating option and every sibling defense, not just the main line — so a wrong
    move made deep on a non-first branch (after a Continue plays a sibling defense) still anchors the
    fork/idea validation on the right position. None for a non-puzzle flow (no tree) or a position off
    the tree entirely. ('reply' nodes are the opponent; the player nodes are 'solve'/'mate'.)"""
    target = _fen_key(fen)
    stack = [(tree or {}).get("root")]
    for _ in range(400):                                  # bounded search (depth × branching guard)
        if not stack:
            break
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        kind = node.get("kind")
        if kind in ("solve", "mate") and _fen_key(node.get("fen")) == target:
            return node
        if kind == "solve":
            stack.append(node.get("after"))
        elif kind == "mate":
            stack.extend(o.get("then") for o in (node.get("options") or []))
        elif kind == "reply":
            stack.extend(d.get("then") for d in (node.get("defenses") or []))
        # 'done' / unknown → leaf, nothing to push
    return None


def _node_solutions(node: dict | None) -> list[str]:
    """The solution move UCIs AT a player node — one for 'solve', every mating option for 'mate'. The
    fork/'idea' validation applies ONLY when there is EXACTLY ONE (a single best move): with several
    best moves — or in a non-puzzle flow, where there is no node at all — we surface the idea rather
    than second-guess it against a solution that isn't singular."""
    if not node:
        return []
    if node.get("kind") == "solve":
        return [node["expect_uci"]] if node.get("expect_uci") else []
    if node.get("kind") == "mate":
        return [o.get("uci") for o in (node.get("options") or []) if o.get("uci")]
    return []


def _legal_sans(fen: str | None) -> set[str] | None:
    """The base SANs (trailing +/# stripped) legal in `fen`, or None if unavailable. Used to drop a
    'threat' fact naming a move the just-played move has made impossible — e.g. 'Black threatens
    Rxc3+' read from the PRE-move board, after the rook has already left c3."""
    if not fen:
        return None
    try:
        from lucena_engine.board import Board
        b = Board(fen)
        return {b.san(u).rstrip("+#") for u in b.legal_moves()}
    except Exception:
        return None


def _deep_tactics(always_lines, solution_moves, live_fen=None) -> str | None:
    """The POINT of a CORRECT move — the engine's OWN deep read (defensive resources, structural
    linchpins, the defender 'the pawn on c6 is the only defender of the bishop on d5') MINUS the one
    clause that names the solution. Now used ONLY on the right-verdict path (the wrong path was retired
    to stop the model force-fitting an irrelevant motif), so the framing is 'why THIS move is the
    move', never 'why a move fails'. The `combination` detector announces 'a forcing sequence starting
    with <answer>', which would spoil the puzzle, so any clause naming a solution move is dropped. None
    if there is no tactical line or nothing survives the strip.

    `live_fen` (optional): the position AFTER the move just played. When given, a clause whose named
    move(s) are ALL illegal there is dropped as STALE — this read is grounded on the PRE-move board,
    so on a wrong move that neutralises a threat (moving the attacked piece) it would otherwise cite a
    threat that no longer exists ('deal with Rxc3+' after the rook left c3, contradicting the
    refutation). A clause with no move (a structural fact) is always kept."""
    legal = _legal_sans(live_fen)
    for line in (always_lines or []):
        if not str(line).startswith("Tactics:"):
            continue
        body = str(line)[len("Tactics:"):].strip().rstrip(".")
        kept = []
        for c in (x.strip() for x in body.split(";")):
            if any(m and m in c for m in solution_moves):
                continue
            # A NULL-MOVE threat ('if White ignores Nc5 …' / 'after a pass, Rb1+ is strong') describes
            # what happens if the mover PASSES — never the point of a move actually played. It also
            # reads as a bare, contextless threat (a student can't tell which piece plays it, and it may
            # even be pinned/captured by the very move), so it is not surfaceable as the move's point.
            if "ignores" in c or "after a pass" in c:
                continue
            # A PRE-EXISTING static condition ('the rook on c5 is pinned to the king') is a board fact,
            # not what THIS move does — surfaced as 'the point' it read as an irrelevant fixation, move
            # after move, and it's colourless so the model flipped it to 'your rook'. Never the point.
            if "pinned" in c:
                continue
            if legal is not None:
                toks = {t.rstrip("+#") for t in _move_tokens(c)}
                if toks and not (toks & legal):   # every move it names is now illegal → stale threat
                    continue
            kept.append(c)
        if not kept:
            continue
        # A DECISIVE clause (a forced mate / mate in N) IS the point — surface ONLY it. The background
        # structural facts (a pre-existing pin, a defender relationship) are TRUE but not why THIS move
        # is the move; dumping them let the model lead with an irrelevant pin ('your rook is pinned')
        # move after move. Only when nothing is decisive do the structural features become the point.
        decisive = [c for c in kept if "mate" in c.lower()]
        if decisive:
            return "The point of this move: " + "; ".join(decisive) + "."
        # No decisive clause → the structural feature the move turns on IS the point (the fallback the
        # single-ply reasoners don't cover). State it plainly; no priming examples for the model to copy.
        return "The point of this move turns on: " + "; ".join(kept) + "."
    return None


def _created_threat(after_analysis, solution_moves) -> str | None:
    """The decisive THREAT the move just played creates, read from the AFTER-move position and named by
    absolute colour (so it can't flip perspective — the reason this was dropped before). 'White
    threatens mate: Ra4#' is instructive on BOTH verdicts: on a wrong move the coach can credit the
    threat before the refutation ('it even threatens mate, but…'); on a right move it's the reward.
    Only the loud, decisive threats (a mate threat) are surfaced — a mundane recapture isn't a
    teaching point. The threat is named by its MOVE ('Ra4#'), NOT the king's square — the prompt tells
    the model to keep the mating move (it confabulated 'a8', the king's square, from a bare 'Ra4#').
    Any clause naming a solution/continuation move is stripped, so a right-move verdict never pre-empts
    the un-played next drill move. None if no such threat."""
    for line in (after_analysis or []):
        s = str(line)
        if not s.startswith("Tactics:"):
            continue
        body = s[len("Tactics:"):].strip().rstrip(".")
        for clause in (c.strip() for c in body.split(";")):
            if "threatens mate" in clause and not any(m and m in clause for m in solution_moves):
                return f"The move just played creates this threat: {clause}."
    return None


def _fallback_hint(pre_fen: str | None, verdict: dict) -> str | None:
    """When the grounded hint LADDER is empty — a win outside `derive_hints`' fork/king-hunt/clean-
    capture scope (e.g. winning a piece via a pin) — fall back to ONE nudge derived from the best move's
    TARGET: the enemy piece it goes after, by absolute colour+square, NEVER the move itself. This keeps a
    stuck student pointed at the idea instead of getting invented filler. None when the best move is
    quiet/positional (no capture target) — then the coach asks a plain question instead."""
    best_san = ((verdict.get("best") or {}).get("pv_san") or [None])[0]
    if not best_san or not pre_fen:
        return None
    try:
        from lucena_engine.board import Board
        b = Board(pre_fen)
        dest = b.uci(best_san)[2:4].lower()
        target = next((p for p in b.piece_list()
                       if p.square == dest and p.color != b.side_to_move), None)
    except Exception:
        return None
    if target is None:                          # best move is not a capture → nothing to point at
        return None
    colour = "Black" if b.side_to_move == "white" else "White"
    word = _PIECE_WORD.get((target.piece or "").upper(), "pawn")
    return ("HINT for the student — phrase as a Socratic nudge, never the move: "
            f"{colour}'s {word} on {dest} is the piece to go after.")


def _hint_line(hints: list | None, attempts: int) -> str | None:
    """Pick the Socratic hint rung for THIS attempt from the grounded ladder (`get_hints` — vague →
    specific, each a partial reveal of the engine PV/geometry, answer-preserving by construction).
    `attempts` is the count of PRIOR wrong tries, so it 0-indexes the rung: first miss → the vaguest
    rung, escalating on each retry, capped at the most specific rung (still never the move). None when
    the line has no tactical handle (a quiet best move → the ladder is empty)."""
    if not hints:
        return None
    rung = hints[min(max(attempts, 0), len(hints) - 1)]
    text = rung.get("text") if isinstance(rung, dict) else str(rung)
    if not text:
        return None
    return ("HINT for the student — phrase as a Socratic nudge, never as the answer and never naming a "
            f"move: {text}")


def _number_full_line(pre_fen: str | None, sans: list) -> str:
    """Number a line whose FIRST move is the PLAYER's (unlike `_numbered_line`, which starts with the
    opponent's refutation) — '7. Rc1 7... Rc3+ 8. Rxc3'. Off the pre-move FEN's number + side."""
    parts = (pre_fen or "").split()
    num = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
    white = (parts[1] if len(parts) > 1 else "w") == "w"
    out = []
    for i, s in enumerate(sans):
        # PGN style: Black carries 'N...' only when it OPENS the line, else bare (see `_numbered_line`).
        if white:
            out.append(f"{num}. {s}")
        elif i == 0:
            out.append(f"{num}... {s}")
        else:
            out.append(s)
        if not white:
            num += 1
        white = not white
    return " ".join(out)


def _draws_by_stalemate(pre_fen: str | None, uci: str | None, pv: list) -> str | None:
    """Level-2 deriver: does this move THROW THE WIN by walking into a STALEMATE (a draw)? Play the
    move, then the engine's forced refutation line, and check each resulting position for stalemate —
    no legal move AND not in check. Purely deterministic (`Board.legal_moves`/`in_check`), so the coach
    GROUNDS the stalemate; it never infers it from a bare draw eval. Returns the sentence or None.

    This is the fact that explains an "everything-else-draws" endgame: e.g. Rc1 Rc3+ Rxc3 and Black,
    king boxed and pawns frozen, has no move — stalemate — while the winning move leaves the opponent a
    spare tempo. Names the stalemated colour and the line, never the solution move."""
    if not pre_fen or not uci:
        return None
    try:
        from lucena_engine.board import Board
        start = Board(pre_fen)
        played = next((start.san(m) for m in start.legal_moves() if m == uci), None)
        if played is None:
            return None
        b = start.apply(uci)

        def is_stalemate(bd) -> bool:
            return not bd.legal_moves() and not bd.in_check

        line = [played]
        if is_stalemate(b):                                   # the move stalemates on the spot
            return _stalemate_sentence(pre_fen, line, b.side_to_move)
        for san in (pv or [])[:8]:                            # …or the forced line walks into it
            u = next((m for m in b.legal_moves() if b.san(m) == san), None)
            if u is None:
                return None
            b = b.apply(u)
            line.append(san)
            if is_stalemate(b):
                return _stalemate_sentence(pre_fen, line, b.side_to_move)
        return None
    except Exception:
        return None


def _stalemate_sentence(pre_fen: str | None, sans: list, color: str) -> str:
    c = (color or "the defending side").capitalize()   # color is always b.side_to_move in practice
    return (f"This move only DRAWS by STALEMATE: after {_number_full_line(pre_fen, sans)}, {c} has NO "
            f"legal move and is not in check — that is stalemate, a draw. Every one of {c}'s pieces is "
            f"stuck, so once the material comes off there is nothing left to move. To WIN you must leave "
            f"{c} a move to make — a spare tempo — instead of freezing the position.")


def _win_band(wp: float) -> str:
    """A win% (player POV) → a plain assessment word. Coarse on purpose: the coach voices the BAND,
    never the number (reciting '19%' is not a coaching sentence, and the exact figure is noise)."""
    return ("winning" if wp >= 65 else "better" if wp >= 55 else "equal"
            if wp >= 45 else "worse" if wp >= 35 else "losing")


def _swing_phrase(after_wp, best_wp) -> str | None:
    """The move's CONSEQUENCE as a band-to-band swing, player POV — 'a winning position into a losing
    one'. `after_wp` is this move's win%, `best_wp` what the best move held. The crux a wrong-move
    refutation only implies: the numbers said losing while the static positional read still said
    'winning', so the coach softened a blunder to 'reduces your advantage'. None when the band doesn't
    change (no meaningful swing to explain)."""
    if not isinstance(after_wp, (int, float)) or not isinstance(best_wp, (int, float)):
        return None
    a, b = _win_band(after_wp), _win_band(best_wp)
    if a == b:
        return None
    return f"this move turns a {b} position into a {a} one for you"


_PIECE_VAL = {"K": 6, "Q": 5, "R": 4, "B": 3, "N": 2, "P": 1}


def _fork_names(targets) -> str:
    """Name the pieces a fork hits, most valuable first — 'the king and the queen on e7'. The king is
    the check, so it leads; it needs no square, the others carry theirs."""
    ordered = sorted(targets, key=lambda p: _PIECE_VAL.get(p.piece.upper(), 0), reverse=True)
    parts = []
    for p in ordered[:2]:                         # two is enough to read as a fork
        w = _PIECE_WORD.get(p.piece.upper(), "pawn")
        parts.append("the king" if p.piece.upper() == "K" else f"the {w} on {p.square}")
    return " and ".join(parts)


def _why_loses(pre_fen: str | None, uci: str | None, pv: list, solution_ucis: list | None = None,
               single_solution: bool = False) -> str | None:
    """Derive the INSTRUCTIVE reason a move drops material — the coaching point, not just the outcome.
    The refutation says WHAT the opponent wins ('Rxe1 wins the rook'); this says WHY it is possible.
    Deterministic, from the board's own defender/attacker sets (`Board.defenders`/`attackers`), so it
    is grounded not guessed, and it never names the solution move. Two failure modes:

      A. You moved a piece INTO a square the opponent still guards — it is recaptured (a premature
         sac / bad trade). The refuted square IS the move's destination.
      B. You moved a DEFENDER off a piece that then hangs, or ignored an already-hanging piece. The
         refuted square is a different, stationary piece of yours.

    The MECHANISM (why the material drops) is ALWAYS derived — a blunder is a blunder whether or not
    there is a puzzle. `single_solution` only gates the INTERPRETIVE extras (the fork 'right idea'
    credit and the 'deal with X first' fix): those are validated against `solution_ucis` ONLY when the
    position has exactly one best move. With several best moves — or a non-puzzle flow, where there is
    no solution to check — we don't second-guess the idea; we surface it as before.

    Returns a sentence or None (only capture refutations lose material this way). First concrete
    deriver from the tactic-reason-derivation note."""
    if not pre_fen or not uci or not pv:
        return None
    refute = pv[0]
    if "x" not in refute:
        return None                                   # only a capture refutation loses material this way
    sq = refute.rstrip("+#")[-2:]
    try:
        from lucena_engine.board import Board
        b = Board(pre_fen)
        after = b.apply(uci)
        from_sq, dest = uci[:2].lower(), uci[2:4].lower()
        moved = next((p for p in b.piece_list() if p.square == from_sq), None)
        mword = _PIECE_WORD.get((moved.piece if moved else "").upper(), "piece") if moved else "piece"
        opp = "black" if (moved and moved.color == "white") else "white"

        # CASE A — you moved this piece INTO the square where it is captured.
        if sq == dest:
            took = next((p for p in b.piece_list() if p.square == dest), None)   # what the move grabbed
            gain = f" and takes the {_PIECE_WORD.get(took.piece.upper(), 'pawn')}" if took else ""
            # Does the piece it just moved hit TWO enemy pieces at once (a fork)?
            targets = [p for p in after.piece_list()
                       if p.color == opp and dest in after.attackers(p.square, moved.color)]
            # The guard that recaptures — the one whose piece type matches the refuting move.
            want = refute[0] if refute[:1].isupper() else "P"
            guard = next((g for g in after.attackers(dest, opp)
                          if next((p.piece.upper() for p in after.piece_list() if p.square == g), "") == want),
                         None)
            # The fork "right idea" credit + the "deal with X first" fix are INTERPRETIVE. Gate them on
            # `single_solution`: only when there is one best move can we say the fork is/ isn't the point.
            #  • single solution → VALIDATE against it: a real fork can be irrelevant (P3: it forks the
            #    king and a rook, but the win just grabs a hanging queen), so the idea is earned only if
            #    the winning line uses the same fork OR first removes the guard; 'deal with X first' only
            #    if the solution deals with THAT guard.
            #  • several best moves / no puzzle → surface the idea as before (don't second-guess it).
            is_fork = len(targets) >= 2
            same_fork = deals_with_guard = False
            sol = (solution_ucis or [None])[0]
            if single_solution and sol:
                try:
                    s_from, s_dest = sol[:2].lower(), sol[2:4].lower()
                    s_mover = next((p for p in b.piece_list() if p.square == s_from), None)
                    sb = b.apply(sol)
                    if s_mover and is_fork:
                        s_hits = {p.square for p in sb.piece_list()
                                  if p.color == opp and s_dest in sb.attackers(p.square, s_mover.color)}
                        same_fork = {t.square for t in targets}.issubset(s_hits)
                    if guard:
                        deals_with_guard = (s_dest == guard) or (guard not in sb.attackers(dest, opp))
                except Exception:
                    pass
            if single_solution:
                credit_idea = is_fork and (same_fork or deals_with_guard)
                deal_first = deals_with_guard
            else:                                # several best moves / non-puzzle → old permissive behavior
                credit_idea = deal_first = is_fork
            intent = (f"Your {mword} on {dest} would fork {_fork_names(targets)} — the right idea. But "
                      if credit_idea else "")
            if guard:
                gp = next((p for p in after.piece_list() if p.square == guard), None)
                gword = _PIECE_WORD.get((gp.piece if gp else "").upper(), "pawn") if gp else "piece"
                lead = intent or f"Your {mword} moves to {dest}{gain}, but "
                if deal_first:
                    # 'before the fork wins anything' ONLY when it actually IS a fork — a non-fork move
                    # into a guarded square (single solution removes the guard first) still earns 'deal
                    # with X first', but must not claim a fork that isn't there.
                    fork_bit = "before the fork wins anything " if is_fork else ""
                    tail = f"recaptures {fork_bit}— that {gword} is what you must deal with first."
                elif credit_idea:            # same_fork: idea already credited in `intent`, don't repeat it
                    tail = f"recaptures — but the fork must come another way, not from {dest}."
                else:
                    tail = f"recaptures it, so you give up the {mword} and come out behind."
                return f"{lead}the {gword} on {guard} still guards {dest}, so {refute} {tail}"
            return (f"{intent or f'Your {mword} moves to {dest}{gain}, but '}{refute} recaptures it — "
                    f"you give up the {mword}, coming out behind on the exchange.")

        # CASE B — a stationary piece of yours is left short of defenders.
        victim = next((p for p in b.piece_list() if p.square == sq), None)
        if victim is None:
            return None
        vword = _PIECE_WORD.get(victim.piece.upper(), "pawn")
        before = set(b.defenders(sq))                 # squares of the player's pieces guarding `sq`
        still = any(p.square == sq for p in after.piece_list())
        after_def = set(after.defenders(sq)) if still else set()
        if from_sq in before and from_sq not in after_def:
            return (f"The {mword} you moved from {from_sq} was the ONLY thing defending your {vword} on "
                    f"{sq}; moving it leaves the {vword} undefended, so {refute} wins it for free."
                    if before == {from_sq} else
                    f"The {mword} you moved from {from_sq} was defending your {vword} on {sq}; moving it "
                    f"leaves the {vword} short of defenders, so {refute} wins it.")
        if not before:
            return (f"Your {vword} on {sq} was already undefended; this move does not deal with the "
                    f"threat of {refute}, which wins it.")
        return None
    except Exception:
        return None


def _brief_move(v: dict, *, hide_best: bool = False, live_fen: str | None = None) -> str:
    """Compact grounding for a played move — the engine's verdict on it. `hide_best` drops the solution
    move (used on a WRONG drill move, so the coach can't leak the answer while explaining the flaw).
    `live_fen` (the position AFTER the move) drops STALE engine facts — a 'Black threatens Rxc3+' read
    from the PRE-move board is a lie once the played move vacated c3, and contradicts the real verdict."""
    if not isinstance(v, dict) or v.get("error"):
        return "(no move read available)"
    legal = _legal_sans(live_fen)
    out = [f"Move played: {v.get('san')}"]
    if v.get("captured"):
        out.append(f"It captures the {v['captured']}.")
    if v.get("class"):
        out.append(f"Engine class of this move: {v['class']}.")
    # The eval CONSEQUENCE — the band-to-band swing from what the best move held to what THIS move
    # gives (player POV). This is the crux a wrong-move refutation only implies; naming it stops the
    # coach softening a blunder ("reduces your advantage") when the move is in fact losing. Bands only,
    # no raw numbers, and the best move's SAN is never named — safe under hide_best.
    if phrase := _swing_phrase((v.get("eval") or {}).get("win_pct"),
                               ((v.get("best") or {}).get("eval") or {}).get("win_pct")):
        out.append(phrase[0].upper() + phrase[1:] + ".")
    # The engine's own move-level facts ("bxc4 wins the bishop on c4", "Nxf6+ is strong for the
    # opponent") — the concrete WHY, previously dropped so the verdict had only class + refutation to
    # reason from. Gated on `not hide_best`: on a WRONG drill move a fact can NAME the solution move, so
    # withholding the best move must withhold these too (never capitalise — 'bxc4' is a pawn capture,
    # 'Bxc4' a bishop; the same first-letter slip we just fixed in the trap voice).
    if not hide_best:
        for f in (v.get("facts") or []):
            if isinstance(f, dict) and (t := f.get("text")):
                # A null-move threat ('if White ignores Nc5 …' / 'after a pass, Rb1+ is strong') describes
                # PASSING, not the move played — confusing as a move fact.
                if "ignores" in t or "after a pass" in t:
                    continue
                # A raw 'X is the only defender of Y' fact gets over-read into 'so Y falls' — but whether Y
                # falls depends on the opponent's reply (it's often a FORK: one piece OR the other). The
                # curated point reasoner owns defender framing with the right certainty; drop the raw fact.
                # A static 'X is pinned to Y' is likewise background, not what the move does, and colourless
                # (the model flipped it to 'your rook') — drop it too.
                if "only defender" in t or "pinned" in t:
                    continue
                # STALE: a fact naming only moves the just-played move has made impossible (a threat
                # against a piece that just moved) — it contradicts the real, post-move verdict.
                if legal is not None:
                    toks = {tk.rstrip("+#") for tk in _move_tokens(t)}
                    if toks and not (toks & legal):
                        continue
                out.append(t if t.endswith((".", "!", "?")) else t + ".")
    if not hide_best and (b := v.get("best")) and b.get("san") and b.get("san") != v.get("san"):
        out.append(f"The engine's best move here is {b['san']}.")
    if pv := v.get("refutation_pv"):
        first = pv[0]
        piece = _PIECE_WORD.get(first.rstrip("+#")[:1], "pawn")
        # Numbered here rather than handed over bare: see `_numbered_line` — reciting an unlabeled
        # PV is move-number arithmetic, and that arithmetic is exactly what was going wrong.
        numbered = _numbered_line(pv[:6], v.get("fen"), played_by_white=v.get("side_to_move") == "white")
        # Name what each capture TAKES — resolved on a board, not left for the model to guess (it
        # invented the piece on the captured square: called a rook a 'knight').
        victims = _pv_capture_victims(v.get("fen"), v.get("san"), pv[:6])
        # The refutation opens with the reply to the just-played move, then alternates sides. Label
        # every move's side with its absolute COLOUR — the bare mixed line let the model flip who's
        # who, and relative 'you'/'opponent' labels inverted when the perspective moved. player_col
        # is the side that just moved (side_to_move of the pre-move fen); the refutation starts with
        # the other colour.
        player_col = _stm_color(v.get("fen"))
        replier_col = "White" if player_col == "Black" else "Black"
        labeled = " ".join(
            f"({replier_col if i % 2 == 0 else player_col}) {m}"
            + (f" [takes the {victims[i]}]" if i < len(victims) and victims[i] else "")
            for i, m in enumerate(numbered))
        dest = first.rstrip("+#")[-2:]
        phrase = f"a {piece} captures the {victims[0]} on {dest}" if victims and victims[0] \
            else _move_phrase(first)
        last = numbered[-1] if numbered else first
        out.append(f"{replier_col} refutes it with {numbered[0] if numbered else first} "
                   f"({phrase}). The full refutation, for YOUR reference: {labeled} — it ENDS at "
                   f"{last}, and nothing exists past {last} (introduce no further move, capture, check, "
                   f"fork, or 'mate in N' beyond it; name a captured piece only as written here).")
    return "\n".join(out)


def _brief_reply(from_fen: str | None, san: str | None) -> str:
    """Grounding for the opponent's auto-played reply in a drill — a factual read of the ONE move,
    so the coach's second beat ('what the opponent did') is grounded and can't invent a piece,
    capture, or motif. Numbered off `from_fen` (the position the reply was played from), with the
    captured piece resolved on a board and check/mate flagged. Deterministic — no engine call."""
    if not from_fen or not san:
        return "(no reply read available)"
    numbered = _numbered(san, from_fen)
    piece = _PIECE_WORD.get(san.rstrip("+#")[:1], "pawn")
    dest = san.rstrip("+#")[-2:]
    victim = _pv_capture_victims(from_fen, None, [san])[0]
    mover_col = _stm_color(from_fen)                       # the side that played this reply
    victim_col = "White" if mover_col == "Black" else "Black"
    lines = [f"{mover_col} has just replied with {numbered}."]
    if victim:
        lines.append(f"It is a {piece} that captures {victim_col}'s {victim} on {dest}.")
    else:
        lines.append(f"It is a {piece} moving to {dest} (no capture).")
    if san.endswith("#"):
        lines.append("It delivers checkmate.")
    elif san.endswith("+"):
        lines.append("It gives check.")
    return "\n".join(lines)


def you_move_beat(pre_fen: str | None, uci: str | None, san: str | None,
                  *, correct: bool | None = None, client_id: str | None = None) -> dict:
    """The player's own board move echoed as a right-aligned "you" bubble — "Played Qxd5 — takes the
    knight" — so the beats column reads as a conversation, not a coach monologue. On a DRILL move
    `correct` marks the bubble with a verdict badge (green check / red cross); `move` (SAN) + `fen`
    (the position right after the move) make it a clickable chip. Restores the beat the legacy
    play_move emitted: the new coach spine adjudicates through the walker, so it must emit it here.
    Captured piece + after-move FEN are resolved on a board — never guessed."""
    after_fen = None
    if pre_fen and uci:
        try:
            from lucena_engine.board import Board
            after_fen = Board(pre_fen).apply(uci).fen
        except Exception:
            after_fen = None
    # v1: the bubble is just the move — no "— takes the X" narration (the LLM/grounding does not
    # interpret the move on the hot path; a right move gets only the ✓, a wrong move the ✗ + Why?).
    text = f"Played {san or uci}"
    beat: dict = {"kind": "you", "stops": False, "segments": [{"text": text}]}
    if correct is not None:
        beat["correct"] = bool(correct)
    if san:
        beat["move"] = san
    if after_fen:
        beat["fen"] = after_fen
    if client_id:                       # reconcile against the app's optimistic bubble
        beat["client_id"] = client_id
    return beat


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
