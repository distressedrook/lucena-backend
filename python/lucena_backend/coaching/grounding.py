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
    for san in pv:
        out.append(f"{num}. {san}" if white_to_move else f"{num}... {san}")
        if not white_to_move:
            num += 1
        white_to_move = not white_to_move
    return out


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


def _why_loses(pre_fen: str | None, uci: str | None, pv: list) -> str | None:
    """Derive the INSTRUCTIVE reason a move drops material — the coaching point, not just the outcome.
    The refutation says WHAT the opponent wins ('Rxe1 wins the rook'); this says WHY it is possible.
    Deterministic, from the board's own defender/attacker sets (`Board.defenders`/`attackers`), so it
    is grounded not guessed, and it never names the solution move. Two failure modes:

      A. You moved a piece INTO a square the opponent still guards — it is recaptured (a premature
         sac / bad trade). The refuted square IS the move's destination.
      B. You moved a DEFENDER off a piece that then hangs, or ignored an already-hanging piece. The
         refuted square is a different, stationary piece of yours.

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
            # The move's INTENT: does the piece it just moved hit TWO enemy pieces at once? That fork is
            # WHY the move tempts — the idea is sound, it fails only because the landing square is guarded.
            targets = [p for p in after.piece_list()
                       if p.color == opp and dest in after.attackers(p.square, moved.color)]
            intent = ""
            if len(targets) >= 2:
                intent = f"Your {mword} on {dest} would fork {_fork_names(targets)} — the right idea. But "
            # Name the guard that recaptures — the one whose piece matches the refuting move.
            want = refute[0] if refute[:1].isupper() else "P"
            guard = next((g for g in after.attackers(dest, opp)
                          if next((p.piece.upper() for p in after.piece_list() if p.square == g), "") == want),
                         None)
            if guard:
                gp = next((p for p in after.piece_list() if p.square == guard), None)
                gword = _PIECE_WORD.get((gp.piece if gp else "").upper(), "pawn") if gp else "piece"
                lead = intent or f"Your {mword} moves to {dest}{gain}, but "
                tail = (f"recaptures before the fork wins anything — that {gword} is what you must deal "
                        f"with first." if intent else
                        f"recaptures it, so you just give up the {mword} and come out behind.")
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
        # The refutation is the OPPONENT's line: it opens with their punishing move, then alternates
        # (opponent, you, opponent, …). Label every move's side explicitly — the bare mixed line let
        # the model flip who's who (a live wrong-verdict read the opponent's move as the player's).
        labeled = " ".join(
            f"({'opponent' if i % 2 == 0 else 'you'}) {m}"
            + (f" [takes the {victims[i]}]" if i < len(victims) and victims[i] else "")
            for i, m in enumerate(numbered))
        dest = first.rstrip("+#")[-2:]
        phrase = f"a {piece} captures the {victims[0]} on {dest}" if victims and victims[0] \
            else _move_phrase(first)
        out.append(f"The opponent refutes it with {numbered[0] if numbered else first} "
                   f"({phrase}); the line then runs {labeled}. Explain the flaw ONLY through this line — "
                   f"the refuting move is the opponent's {first}, a {piece} move, nothing else. Name a "
                   f"captured piece ONLY as written here — never guess what stands on a square.")
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
    lines = [f"The opponent (the side NOT the player) has just replied with {numbered}."]
    if victim:
        lines.append(f"It is a {piece} that captures the player's {victim} on {dest}.")
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
    captured = _pv_capture_victims(pre_fen, None, [san])[0] if (pre_fen and san) else None
    text = f"Played {san or uci}" + (f" — takes the {captured}" if captured else "")
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
