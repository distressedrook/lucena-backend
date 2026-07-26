"""PGN -> the deterministic game STORY (the v1 deliverable).

`gamepass.py` already classified every ply against the engine. This module is
the layer above it: it decides which moments are worth a reader's attention and
composes, for each one, everything the deterministic stack knows — the plans
layer (suggest -> verify -> position_read), structures, weaknesses, phase,
dynamism, initiative, king danger — into one artifact a renderer can print.

ZERO LLM TOKENS, by construction. Nothing here calls a model; every sentence in
the output is either a number the engine produced or a phrase a detector wrote.
Where the vocabulary has nothing to say, the story says less (v1 ruling).

The three moment kinds, all defined on engine numbers alone:

  TURNING POINT   the mistake/blunder ply — what it cost, what the engine
                  played instead, and how the opponent refuted it.
  MISSED          the opponent erred and the beneficiary did NOT collect: a
                  win% gift on the board that the very next move gave back.
                  This is the "bank the tactics they missed" aim.
  PLAN            a quiet position (no big swing) where the plans layer has an
                  ENGINE-CONFIRMED plan — the moat: what to DO, not what to
                  avoid. These are chosen away from the tactical moments so the
                  story alternates rather than piling every chapter on one
                  crisis.

Learner patterns (aim 3) are tallies over the above, per side: where the errors
live by phase, which motifs recur, and how many gifts each side left on the
table. A tally is not a diagnosis and the renderer must not dress it as one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

import chess

from lucena_core.board import Board
from lucena_core.pgn import parse_pgn
from lucena_core.reads import game_phase
from .gamepass import run_pass
from .moveclass import (MoveClass, SYMBOL, ERRORS, classify_move,
                        accuracy, MISS_GIFT, MISS_RETURN)

_log = logging.getLogger(__name__)

SCHEMA = "lucena/gamestory@1"

# DETERMINISM: node limits, not movetime. gamepass splats these straight into
# Engine.analyse, and the repo's split is "movetime in production, nodes in
# tests" — but a published analysis is not an interactive read: the same game
# must produce the same page every time, or two people reading the same report
# see different moves flagged.
#
# NODES ARE NOT SUFFICIENT ON THEIR OWN. Multi-threaded Stockfish is
# nondeterministic even at a fixed node count, which we measured: two runs of
# this pipeline at 300k nodes disagreed on 80 of 86 plies' evaluations and on
# three move labels. The caller MUST pass an Engine constructed with
# `threads=1` for the output to be reproducible — `lucena_engine.uci` documents
# the same pairing. `build_story` does not construct the engine, so it cannot
# enforce this; `analyse_pgn` below does.
FAST_NODES = {"nodes": 300_000}
DEEP_NODES = {"nodes": 1_500_000}

# A moment must MATTER: the position was still playable (the loser had real
# chances) and the swing was big enough to be worth a chapter. Both are
# win-percentage, which is what the reader actually feels.
_LIVE_WP = 20.0          # below this the game was already decided
_TURNING_DROP = 10.0     # win% a mistake must shed to earn a chapter
_MISSED_GIFT = 12.0      # win% handed over that the beneficiary then returned

# Plan chapters are expensive (a full engine+Maia roll each) and repetitive if
# adjacent, so they are spread across the game and capped. Not a silent cap:
# `plan_chapters_considered` reports what was skipped.
_PLAN_CHAPTERS = 4
_PLAN_MIN_GAP = 6        # plies between plan chapters
_QUIET_SWING = 6.0       # a plan chapter's position must not be mid-crisis
_MAX_MOMENTS = 8         # walkthrough chapters; the rest stay in the ply table
_MOMENT_GAP = 3          # plies between chapters (one flurry != four chapters)
# A slide worth a chapter. Two turning points' worth of ground, shed without
# ever making a move bad enough to be called a mistake.
_DRIFT_TOTAL = 20.0
_DRIFT_STEP = 5.0        # a concession must be a real inaccuracy to count
_DRIFT_MOVES = 3         # ...and one or two of them is not a slide
# The opening table names NODES, not every ply; 3 is the largest internal gap
# measured across five real mainlines, so a 4th unnamed ply means out of book.
_BOOK_GAP = 3
# Weakness terms that need a middlegame's worth of pieces to mean anything.
_MIDDLEGAME_ONLY = {"weak_color_complex", "exposed_king", "back_rank_weak"}


@dataclass
class Moment:
    kind: str                     # turning | missed | plan
    ply: int
    move_no: int
    side: str                     # "w" | "b"
    san: str
    fen_before: str
    fen_after: str
    eval_cp: int                  # mover POV, after the move
    win_pct: float
    delta_win_pct: float
    cls: str
    best_san: str = ""
    best_pv_san: list = field(default_factory=list)
    refutation_pv: list = field(default_factory=list)
    motifs: list = field(default_factory=list)
    # composed by _enrich
    phase: str = ""
    developed: dict = field(default_factory=dict)
    structure: str = ""
    character: str = ""
    weaknesses: dict = field(default_factory=dict)
    plans: dict = field(default_factory=dict)
    read: str = ""
    gift_wp: float = 0.0          # missed only: win% the opponent handed over
    uci: str = ""
    label: str = ""               # the reader-facing ladder rung
    symbol: str = ""
    character_why: str = ""
    initiative: dict = field(default_factory=dict)
    king_risk: dict = field(default_factory=dict)
    only_move: bool = False
    intent: list = field(default_factory=list)
    alignment: dict = field(default_factory=dict)
    endgame: list = field(default_factory=list)
    span_to_move: int = 0         # drift only: where the slide ended
    span_cost: float = 0.0        # drift only: win% shed across the run
    span_moves: list = field(default_factory=list)


def _label(plies: list[dict]) -> None:
    """Attach the reader-facing ladder to every ply, in place.

    Book membership is asked of the real opening table, walking the game's own
    FEN path, so "book" ends where THIS game left theory rather than at a
    fixed move number.

    THE TABLE NAMES NODES, NOT EVERY PLY, so a line in perfectly good theory
    dips out of it for a ply or two (`openings.book_name` documents the same
    stickiness). Treating the first unnamed ply as the end of book cut this
    game's French Exchange off at move 4 and reported 2 book plies for a line
    that is named through move 7. Measured across five mainlines (French
    Exchange, Ruy Morphy, Najdorf, QGD Tartakower, Four Knights) the largest
    internal gap is 3 plies.

    But tolerating a gap FORWARD grants free book plies to a player who simply
    left theory with a bad move — after 1.d4 Nh6 the next plies would go
    ungraded (Codex 2026-07-26). A gap is only *inside* theory if the line
    comes BACK, and since we hold the whole game we can just look: book runs to
    the last named ply reachable without ever exceeding _BOOK_GAP. A name that
    appears after a longer silence is a transposition, not theory this game
    was ever following, and does not re-open the book."""
    from lucena_core.openings import name_for
    named_at = [name_for(pr["fen_after"]) for pr in plies]
    last_named, gap = -1, 0
    for i, nm in enumerate(named_at):
        if nm is not None:
            if gap > _BOOK_GAP:
                break                    # already left; this is a transposition
            last_named, gap = i, 0
        else:
            gap += 1
            if gap > _BOOK_GAP:
                break
    for i, pr in enumerate(plies):
        pr["book"] = named_at[i]
        pr["in_book"] = i <= last_named
        drop = max(0.0, -pr["delta_win_pct"])
        cls = classify_move(drop=drop,
                            played_is_best=(pr["san"] == pr["best"]["san"]),
                            in_book=pr["in_book"],
                            engine_class=pr["class"],
                            gift_wp=pr.get("gift_wp", 0.0))
        pr["label"] = cls.value
        pr["symbol"] = SYMBOL[cls]


def _gifts(plies: list[dict]) -> None:
    """Record, on each ply, what the OPPONENT handed over immediately before —
    the input the MISS rung is defined against. Must run before _label."""
    for i, pr in enumerate(plies):
        prev = plies[i - 1] if i else None
        pr["gift_wp"] = round(max(0.0, -prev["delta_win_pct"]), 1) if prev else 0.0


def _side_name(headers: dict, side: str) -> str:
    return headers.get("White" if side == "w" else "Black", "?")


def _norm(fen: str) -> str:
    """Position identity for repetition: the first four FEN fields (clocks and
    move number are not part of who-stands-where)."""
    return " ".join((fen or "").split(" ")[:4])


def _ending(plies: list[dict], headers: dict, result: str) -> dict:
    """How the game actually finished.

    Written because the first quiet game analysed here ended by REPEATING the
    position and the report said nothing about it — the one fact that explains
    the result was missing while four chapters discussed castling. A draw is
    not an absence of story; how a game stops is part of it.

    Repetition is read off the board, not the headers: `Termination` says
    "drawn by agreement" even when the players were shuffling, and a claim we
    can verify beats one we are told."""
    out = {"result": result, "termination": headers.get("Termination", "")}
    if not plies:
        return out

    # Ply 0 is the position BEFORE the first move; without it a repetition of
    # the starting position is invisible and any count that includes it is one
    # short (Codex 2026-07-26). Occurrences carry (ply, move_no) so the opening
    # position reports cleanly as move 1 rather than needing a special case.
    seen: dict[str, list[tuple[int, int]]] = {}
    if plies[0].get("fen_before"):
        seen.setdefault(_norm(plies[0]["fen_before"]), []).append((0, 1))
    for pr in plies:
        seen.setdefault(_norm(pr["fen_after"]), []).append((pr["ply"], pr["move_no"]))

    final = seen.get(_norm(plies[-1]["fen_after"]), [])
    # A DECISIVE game can end in a position that happens to have occurred
    # before — someone resigns, flags, or is mated after a repeat. Calling
    # that "how it ended: repetition" would explain a resignation with the
    # wrong cause, so the reader-facing ending is drawn games only. The raw
    # signal stays available as `repeats` for anything that wants it.
    if len(final) >= 2:
        out["repeats"] = len(final)
    if len(final) >= 2 and (result or "").strip() == "1/2-1/2":
        # the cycle is what happened between two occurrences of the SAME
        # position; name the moves so a reader can see the shuffle
        (first, first_move), (last, _) = final[0], final[-1]
        cycle = [p for p in plies if first < p["ply"] <= last]
        out["repetition"] = {
            "times": len(final),
            "from_move": first_move,
            "to_move": plies[-1]["move_no"],
            "moves": [p["san"] for p in cycle],
            # a repetition with the evaluation level is a genuine standoff; one
            # with a side clearly better means somebody let a better game go
            "cp_white": plies[-1].get("cp_white", 0),
        }
    return out


def _tracks(plies: list[dict]) -> dict:
    """Per-ply ACTIVITY and INITIATIVE, White-positive, for the whole game.

    Two dimensions the eval curve cannot show. Evaluation says who is better;
    these say WHY — whose pieces are doing more work, and who is dictating.
    They routinely disagree with the eval and with each other, which is the
    interesting part: a side can be worse on the board and still holding the
    initiative, and that is a different game to play than being worse and
    passive.

    ACTIVITY is the positional term's own cp differential (mobility and
    placement against a GM-fitted baseline). Static, exact, ~6ms.

    INITIATIVE is `initiative.py`'s reading of who is dictating. Given two
    engine lines it uses the VALIDATED engine-spread basis (the mover's
    MultiPV gap — AUC 0.766 held-vs-failed); the second line's cp is all it
    needs, and gamepass already computes it at multipv=2, so the whole track
    is free of extra engine time. Where a second line does not exist (a
    forced position) it degrades to the geometry prior and says so in
    `basis`, which the renderer must not paper over.

    CAVEAT, recorded rather than buried: the AUC was measured under the
    production 4-PV contract at research node counts. This is 2 PVs at the
    story's node budget — the same basis, a noisier estimate of it.
    """
    from ..plans.service import _bootstrap
    _bootstrap()
    from lucena_core import positional
    from initiative import initiative

    act, ini, bases = [], [], {}
    for i, pr in enumerate(plies):
        # AFTER the move, so all three lines on the chart show the same
        # instant. The eval curve already plots fen_after; reading these at
        # fen_before would offset them by a ply and invite the reader to
        # attribute a swing to the wrong move.
        fen = pr.get("fen_after")
        if not fen:
            continue
        # ...and the engine's lines for THIS position are the ones recorded
        # against the NEXT ply, whose fen_before is this fen_after.
        nxt = plies[i + 1] if i + 1 < len(plies) else None
        white_to_move = (nxt or {}).get("side", "b" if pr["side"] == "w" else "w") == "w"
        try:
            terms = positional.analyze_positional(Board(fen))["terms"]
            a = int((terms.get("activity") or {}).get("cp") or 0)
        except Exception as exc:
            _log.debug("activity failed on %s: %s", fen, exc)
            a = 0

        # gamepass scores are MOVER POV; the (fen, pvs, rolls) contract is
        # White POV. Getting this backwards silently mirrors the track for
        # every Black ply, which reads as violent oscillation.
        sign = 1 if white_to_move else -1
        pvs = None
        src = (nxt or {}).get("best") or {}
        best_cp, second_cp = src.get("eval_cp"), (nxt or {}).get("second_cp")
        if best_cp is not None:
            pvs = [{"cp": sign * best_cp, "ucis": list(src.get("pv_uci") or [])}]
            if second_cp is not None:
                pvs.append({"cp": sign * second_cp, "ucis": []})
        try:
            iv = initiative(fen, pvs)
            d = float(iv.get("diff") or 0.0)
            bases[iv.get("basis", "?")] = bases.get(iv.get("basis", "?"), 0) + 1
        except Exception as exc:
            _log.debug("initiative failed on %s: %s", fen, exc)
            d = 0.0
        act.append(round(a, 1))
        ini.append(round(d, 3))
    return {"activity": act, "initiative": ini, "bases": bases}


def _opening(plies: list[dict]) -> dict:
    """The opening this game actually played, from our own table.

    `book_name` folds the whole line rather than taking the last named node,
    because the table re-attaches COARSER names deeper in — a naive last-wins
    walk reports a less specific opening than the game reached.

    It is fed only the IN-BOOK prefix. book_name is deliberately sticky, so
    handing it the whole game lets a late accidental transposition rename the
    opening after the players had long left theory — contradicting the very
    boundary _label computes (Codex 2026-07-26)."""
    from lucena_core.openings import book_name
    fens = [p["fen_after"] for p in plies if p.get("in_book")]
    name = book_name(fens)
    depth = sum(1 for p in plies if p.get("in_book"))
    return {"name": name, "plies": depth,
            "left_at_move": (depth // 2) + 1 if depth else None}


def _turning_points(plies: list[dict]) -> list[Moment]:
    """Every mistake that was both large and still live."""
    out = []
    for pr in plies:
        before_wp = pr["win_pct"] - pr["delta_win_pct"]
        drop = -pr["delta_win_pct"]
        if drop >= _TURNING_DROP and before_wp > _LIVE_WP:
            out.append(_moment("turning", pr))
    return out


def _missed(plies: list[dict]) -> list[Moment]:
    """The opponent erred and the beneficiary gave it straight back.

    Deliberately strict: the gift must be large, and the reply must return a
    real part of it. A player who stays a little worse after being handed a
    winning position is not "missing a tactic" in any sense a learner can use;
    a player who hands the whole thing back on the next move is."""
    out = []
    for i in range(len(plies) - 1):
        gift, reply = plies[i], plies[i + 1]
        given = -gift["delta_win_pct"]
        returned = -reply["delta_win_pct"]
        if given >= MISS_GIFT and returned >= MISS_RETURN:
            m = _moment("missed", reply)
            m.gift_wp = round(given, 1)
            out.append(m)
    return out


def _drift(plies: list[dict]) -> list[Moment]:
    """A game lost WITHOUT a blunder — ground shed a little at a time.

    The turning-point and missed-win readers both need one move to cross a
    bar, so a technical collapse is invisible to them. On the game that
    prompted this, White lost a level endgame across moves 40-57 in five
    separate inaccuracies of 6-9.5 win% each: not one of them reached the
    10-point turning-point bar, the walkthrough had no chapter for any of it,
    and the phase that decided the game went unmentioned.

    A drift is a run of one side's own moves that sheds _DRIFT_TOTAL between
    them while NO single move crosses the turning-point bar — if one does, it
    is a turning point and this reader stands aside for it. The moment is
    anchored at the start of the run, because "here is where it began to go
    wrong" is the useful board to show.

    The run is reported WHOLE. Emitting the moment the threshold is crossed
    truncated a five-move slide to its first three, which tells the reader the
    position was lost in fewer concessions than it was.
    """
    out = []
    for side in ("w", "b"):
        run: list[dict] = []

        def flush(run):
            if len(run) < _DRIFT_MOVES:
                return None
            # NET ground lost across the span, not the sum of per-move drops.
            # Summing double-counts: over a dozen moves of ordinary play the
            # small wobbles alone reach 40+ "points" while the position has not
            # moved at all, and that fired a slide on a game with zero errors
            # by either side, and on a side that was winning.
            first, last = run[0], run[-1]
            total = (first["win_pct"] - first["delta_win_pct"]) - last["win_pct"]
            if total < _DRIFT_TOTAL:
                return None
            m = _moment("drift", first)
            m.span_to_move = last["move_no"]
            m.span_cost = round(total, 1)
            m.span_moves = [f"{q['move_no']}"
                            f"{'.' if q['side'] == 'w' else '...'} {q['san']}"
                            for q in run]
            return m

        # The WHOLE ply stream, not just this side's moves: ground can be
        # handed back by the OPPONENT erring, and a per-side scan cannot see
        # that — it would merge two separated collapses into one continuous
        # "slide" that never happened (Codex 2026-07-27).
        for pr in plies:
            drop = -pr["delta_win_pct"]
            if pr["side"] != side:
                if drop > _QUIET_SWING:       # the opponent gave it back
                    m = flush(run)
                    if m:
                        out.append(m)
                    run = []
                continue
            # A move bad enough to be a mistake OWNS its ground — that is a
            # turning point and this reader stands aside rather than
            # double-counting it. Recovering it yourself ends the slide too.
            if drop >= _TURNING_DROP or drop < -_QUIET_SWING:
                m = flush(run)
                if m:
                    out.append(m)
                run = []
                continue
            # Only REAL concessions join a slide. A 0.5-point wobble is the
            # noise of ordinary play, not a step toward losing.
            if drop >= _DRIFT_STEP:
                run.append(pr)
            # neutral moves neither extend the span nor break it: the slide
            # runs from the first concession to the LAST, not to whatever
            # quiet move happened to follow
        m = flush(run)
        if m:
            out.append(m)
    return out


def _moment(kind: str, pr: dict) -> Moment:
    return Moment(
        kind=kind, ply=pr["ply"], move_no=pr["move_no"], side=pr["side"],
        san=pr["san"], fen_before=pr.get("fen_before", ""),
        fen_after=pr["fen_after"], eval_cp=pr["eval_cp"],
        win_pct=pr["win_pct"], delta_win_pct=pr["delta_win_pct"],
        cls=pr["class"], best_san=pr["best"]["san"],
        best_pv_san=pr["best"]["pv_san"], refutation_pv=pr["refutation_pv"],
        motifs=pr["motifs"], uci=pr["uci"], label=pr["label"],
        symbol=pr["symbol"], gift_wp=pr.get("gift_wp", 0.0),
    )


def _plan_candidates(plies: list[dict], taken: set[int],
                     want: int = _PLAN_CHAPTERS) -> list[dict]:
    """Quiet, out-of-book positions to read for plans, in PRIORITY order.

    The order matters as much as the filter. Walking the game front to back
    and keeping the first few that confirm put every plan chapter in the
    opening — on a 32-move game the chapters landed on moves 8, 11, 14 and 29,
    so three of four discussed castling and development while the middlegame
    the players actually had to solve went unread.

    So candidates are returned nearest-first to evenly spaced targets across
    the game. Selection still walks this list in order and can skip any
    position the plans layer declines, but it now reaches for the middlegame
    before the fourth opening move."""
    pool, last = [], -99
    for pr in plies:
        if pr["ply"] in taken or abs(pr["delta_win_pct"]) > _QUIET_SWING:
            continue
        # OUT of book, as the docstring says: a theory position has nothing to
        # teach about planning, and spending a plan read on one wastes the
        # budget on moves we just deliberately declined to grade. The move_no
        # bar alone missed this for long theoretical lines (Codex 2026-07-26).
        if pr.get("in_book"):
            continue
        if pr["ply"] - last < _PLAN_MIN_GAP or pr["move_no"] < 8:
            continue
        pool.append(pr)
        last = pr["ply"]
    if not pool or want <= 0:
        return pool

    lo, hi = pool[0]["ply"], pool[-1]["ply"]
    targets = [lo + (hi - lo) * (i + 0.5) / want for i in range(want)]
    ordered, left = [], list(pool)
    for t in targets:
        if not left:
            break
        pick = min(left, key=lambda p: abs(p["ply"] - t))
        left.remove(pick)
        ordered.append(pick)
    # the rest stay available as fallbacks, in game order
    return ordered + left


# What the player did NEXT — the window `move_intent` reads. A single move
# cannot be named: plan_diff's event detectors need observation plies after an
# event before they will confirm it (TAIL), which is exactly right for corpus
# auditing and useless for one move. So we name the CONTINUATION the player
# actually played, which is a better question anyway.
_INTENT_PLIES = 12
_INTENT_TAIL = 2   # loosened from plan_diff's audit default (6), deliberately:
                   # this is a DESCRIPTION of what one player then did, not a
                   # verified plan claim, and the tier tags carry the evidence.


def move_intent(fen_before: str, ucis: list[str]) -> list[str]:
    """Which plan FAMILIES the moves actually played execute.

    `plan_diff` is the retrospective namer the whole verify loop is built on:
    it reads a move sequence and says which plans that sequence carries out.
    Pointed at the continuation a player really chose, it answers "what did
    this player actually go on to DO" — deterministically, in the same
    vocabulary the suggester proposes in.

    Empty is a real answer (the moves execute no plan our grammar names) and
    must never be rendered as "the player had no plan"."""
    from ..plans.service import _bootstrap
    _bootstrap()
    if not fen_before or not ucis:
        return []
    try:
        from plan_diff import labels
        b = chess.Board(fen_before)
        mv = [chess.Move.from_uci(u) for u in ucis[:_INTENT_PLIES]]
        return sorted(labels(b, mv, horizon=_INTENT_PLIES, tail=_INTENT_TAIL))
    except Exception as exc:
        _log.debug("plan_diff failed on %s: %s", fen_before, exc)
        return []


# Families whose trigger is near-universal: almost every opening position
# offers "castle" and "finish developing", so matching one says nothing about
# whether the player understood THIS position. Crediting them produced the
# sentence "right idea, wrong move — castle kingside" about 4.Bc4, whose actual
# error was missing a free pawn. They stay in the plan menus (the repo's 2026-
# 07-22 ruling: do not fake selectivity by tightening triggers) but they cannot
# carry an alignment claim.
_UNIVERSAL = {"castle_kingside", "castle_queenside", "complete_development",
              "development"}


def _alignment(intent: list[str], plans: dict, side: str, label: str) -> dict:
    """Did the move serve a plan the position actually offered?

    This is the sentence the market does not have. Everyone can say "that was a
    blunder"; the interesting statement is *"the idea was right and the move
    was wrong"* — the player had the correct plan and botched the execution,
    which is a completely different lesson from having no plan at all.

    Only asserted when there is something to assert. `unnamed` means our
    grammar had no name for the move, NOT that the move was aimless — the
    renderer must not upgrade silence into a verdict.
    """
    tag = "white" if side == "w" else "black"
    mine = plans.get(tag) or []
    # plan_diff labels are "W:family" / "B:family" and a continuation contains
    # BOTH players' plans. Comparing the mover's menu against everything that
    # happened credits them with the opponent's ideas — filter to their side.
    pfx = "W:" if side == "w" else "B:"
    played = {i.split(":", 1)[1].split(":")[0]
              for i in intent if i.startswith(pfx)}
    if not played:
        return {"state": "unnamed"}
    fams = {f for p in mine for f in (p.get("families") or [])}
    hit = sorted((played & fams) - _UNIVERSAL)
    if not hit:
        # NOT "no plan": our menu is not exhaustive, and a plan we never
        # proposed can still be a good one. Say what they did instead.
        return {"state": "different-plan", "played": sorted(played)}
    tiers = {f: p["tier"] for p in mine for f in (p.get("families") or [])}
    best = max((tiers.get(f, "structure") for f in hit),
               key=lambda t: {"engine": 2, "human": 1, "structure": 0}[t])
    if label in {c.value for c in ERRORS}:
        return {"state": "right-idea-wrong-move", "families": hit, "tier": best}
    return {"state": "on-plan", "families": hit, "tier": best}


def _static_read(fen: str) -> dict:
    """Everything the no-engine detectors know about one position."""
    import sys
    from ..plans.service import _bootstrap
    _bootstrap()
    from structures import classify
    from weaknesses import census
    b = chess.Board(fen)
    ph = game_phase(fen)
    try:
        # classify returns (name, white_owns_it) pairs — say whose it is.
        # classify returns (name, white_owns_it) pairs, and a symmetric
        # structure is reported once per owner — say it once, with both owners.
        owners = {}
        for n, w in classify(b):
            owners.setdefault(n, set()).add("White" if w else "Black")
        struct = [f"{n} ({'/'.join(sorted(o))})" for n, o in sorted(owners.items())]
    except Exception as exc:                         # a recognizer may abstain
        _log.debug("structures failed on %s: %s", fen, exc)
        struct = []
    out = {
        "phase": ph["phase"],
        "developed": ph.get("developed", {}),
        "structure": ", ".join(struct),
        "weaknesses": {},
    }
    # MIDDLEGAME VOCABULARY STANDS DOWN IN THE ENDGAME. A "weak colour complex"
    # or a soft back rank needs pieces to exploit it; printed over a bishop
    # ending they are noise dressed as analysis, and the endgame read below
    # says the true thing instead. Weak/backward pawns and passive rooks stay:
    # those are exactly what a technical game is about.
    endgame = out["phase"] == "endgame"
    for tag, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        try:
            c = census(b, color)
        except Exception as exc:
            _log.debug("census failed on %s: %s", fen, exc)
            c = {}
        # `total` is the census's own count field, not a weakness — printing
        # it as one would put the word "total" in a reader's weakness list.
        out["weaknesses"][tag] = {
            k: v for k, v in (c or {}).items()
            if v and k != "total" and not (endgame and k in _MIDDLEGAME_ONLY)}
    return out


def _plans_read(fen: str, pool, maia) -> dict:
    """The full plans layer for one position: roll, verify, render.

    Returns {} when the layer declines the position (endgame, or the engine
    leg failed) — the story then simply has no plan chapter there, which is
    the honest outcome, not a gap to paper over."""
    from ..plans.service import sheet_json_for, render_position_read, is_endgame
    if is_endgame(fen):
        return {}
    try:
        _pre, post = sheet_json_for(fen, pool, maia)
    except Exception as exc:
        _log.warning("plans roll failed on %s: %s", fen, exc)
        return {}
    read = render_position_read(post) or ""
    sides = post.get("sides", {})
    plans = {}
    for tag in ("white", "black"):
        plans[tag] = [
            {"idea": p.get("idea", ""), "verdict": p.get("verdict"),
             "families": ([p["family"]] if p.get("family")
                          else _family_of(p.get("idea", ""))),
             "tier": _tier(p)}
            for p in sides.get(tag, {}).get("plans", [])
        ]
    a = post.get("assessment") or {}
    ini = a.get("initiative") or {}
    _lead = ini.get("leader")
    _wside = "white" if _lead == "White" else "black" if _lead == "Black" else None
    _why = ((ini.get(_wside) or {}).get("why") or {}) if _wside else {}
    kr = a.get("king_risk") or {}
    return {
        "read": read, "plans": plans,
        # `character`/`king_risk`/`initiative` live under `assessment`, not at
        # the top level — reading them from the wrong depth silently yields ""
        # everywhere, which looks like "the detector had nothing to say".
        "character": (a.get("character") or {}).get("bucket", ""),
        "character_why": (a.get("character") or {}).get("summary", ""),
        "verdict": a.get("verdict", ""),
        # `initiative` reports its holder as `leader` ("White"/"Black"/None)
        # with `mechanism` and a per-side `why` of citable board facts — not
        # the side/verdict pair a first guess assumed.
        "initiative": {
            "leader": _lead,
            "basis": ini.get("basis"),
            "mechanism": ini.get("mechanism") or [],
            "checks": _why.get("checks") or [],
            "captures": _why.get("captures") or [],
            "attacked": _why.get("attacked") or [],
            "forcing": ((ini.get(_wside) or {}).get("forcing_moves") or [])
                       if _wside else [],
        },
        "king_risk": {t: (kr.get(t) or {}) for t in ("white", "black")},
        "only_move": (a.get("only_move") or {}).get("only_move", False),
    }


# The three evidence tiers, read off the verdict exactly as position_read does
# (owner 2026-07-25). Kept in ONE place so the HTML cannot invent a fourth.
_ENGINE = {"CONFIRMED-SOUND", "CONFIRMED-SOUND-LATER"}


def _family_of(idea: str) -> list:
    """Backfill a plan's family from its own head.

    `fact_sheet` attaches `family` only to plans that carry a verify contract,
    so every structure-tier plan arrives with family=None — and an alignment
    check keyed on family would then be structurally incapable of ever matching
    one, silently reporting "different plan" for a player who followed the
    structural advice exactly. suggest.py's CANDIDATE_FAMILIES is the existing
    head->family table; use it rather than inventing a second one."""
    from ..plans.service import _bootstrap
    _bootstrap()
    try:
        from suggest import CANDIDATE_FAMILIES
    except Exception:
        return []
    head = (idea or "").upper()
    for key, fams in CANDIDATE_FAMILIES.items():
        if head.startswith(key):
            # a head can map to SEVERAL families (OUTPOST -> occupation and
            # blockade); return them all, and let the caller union.
            return sorted(fams)
    return []


def _tier(plan: dict) -> str:
    v = (plan.get("verdict") or "")
    if v in _ENGINE:
        return "engine"
    if v == "HUMAN-TYPICAL":
        return "human"
    return "structure"


def _arc(plies: list[dict]) -> list[dict]:
    """The shape of the game, segmented off the eval curve alone.

    Every ply is bucketed by who stood better and by how much; consecutive
    plies in the same bucket merge into one act. This is what turns 86 numbers
    into "Black equalised, took over, then threw it away" without anybody
    writing that sentence — the segmentation IS the narrative.
    """
    def band(cp: int) -> tuple[str, str]:
        who = "white" if cp > 0 else "black"
        a = abs(cp)
        if a < 50:
            return "level", ""
        if a < 150:
            return "edge", who
        if a < 350:
            return "clear", who
        return "winning", who

    acts = []
    for pr in plies:
        # eval_cp is mover-POV; normalise to White-POV so the curve is one line.
        cp = pr["eval_cp"] if pr["side"] == "w" else -pr["eval_cp"]
        pr["cp_white"] = cp
        kind, who = band(cp)
        if acts and acts[-1]["kind"] == kind and acts[-1]["who"] == who:
            acts[-1]["to_ply"] = pr["ply"]
            acts[-1]["to_move"] = pr["move_no"]
        else:
            acts.append({"kind": kind, "who": who, "from_ply": pr["ply"],
                         "to_ply": pr["ply"], "from_move": pr["move_no"],
                         "to_move": pr["move_no"]})
    # An act of one or two plies is eval jitter, not a chapter of the game.
    kept = [a for a in acts if a["to_ply"] - a["from_ply"] >= 2] or acts
    # Dropping a short act can leave two neighbours of the SAME kind adjacent
    # ("level, then level") and a gap in the move numbering. Merge them so the
    # arc reads as continuous prose rather than a list with holes.
    merged: list[dict] = []
    for a in kept:
        if merged and merged[-1]["kind"] == a["kind"] and merged[-1]["who"] == a["who"]:
            merged[-1]["to_ply"], merged[-1]["to_move"] = a["to_ply"], a["to_move"]
        else:
            merged.append(dict(a))
    # Close the numbering gaps the dropped jitter left behind, so the acts
    # tile the game end to end — a reader should never see the story skip
    # from "moves 1-7" to "moves 9-11" with nothing said about move 8.
    for prev, nxt in zip(merged, merged[1:]):
        prev["to_move"] = max(prev["to_move"], nxt["from_move"] - 1)
    return merged


_RANK = {"missed": 3.0, "turning": 2.0, "drift": 2.0, "plan": 1.0}


def _rank(m: Moment) -> float:
    """How much a reader should care. Swing dominates; a missed win outranks an
    equal-sized ordinary error because it is the more actionable lesson."""
    # a drift's weight is the whole slide, not the one move it is anchored on
    size = m.span_cost if m.kind == "drift" else abs(m.delta_win_pct)
    return size * _RANK.get(m.kind, 1.0) + m.gift_wp


def _patterns(plies: list[dict], moments: list[Moment], headers: dict) -> dict:
    """Aim 3: what this player tends to do. Tallies only — never a diagnosis."""
    out = {}
    for side in ("w", "b"):
        mine = [p for p in plies if p["side"] == side]
        errs = [p for p in mine if p["label"] in {c.value for c in ERRORS}]
        by_phase = {"opening": 0, "middlegame": 0, "endgame": 0}
        for p in errs:
            ph = game_phase(p["fen_after"])["phase"]
            by_phase[ph] = by_phase.get(ph, 0) + 1
        motifs = {}
        for m in moments:
            if m.side != side:
                continue
            for mo in m.motifs:
                motifs[mo["motif"]] = motifs.get(mo["motif"], 0) + 1
        out[side] = {
            "name": _side_name(headers, side),
            "moves": len(mine),
            "errors": len(errs),
            "error_rate": round(100.0 * len(errs) / max(1, len(mine)), 1),
            "by_phase": by_phase,
            "worst": max((abs(p["delta_win_pct"]) for p in errs), default=0.0),
            "missed": sum(1 for m in moments if m.kind == "missed" and m.side == side),
            "motifs": sorted(motifs.items(), key=lambda kv: -kv[1]),
            "counts": _counts(mine),
            "accuracy": accuracy([max(0.0, -p["delta_win_pct"]) for p in mine]),
            # in_book, not book: `book` is only the NAMED table nodes, so
            # counting it under-reports depth on any line with a table gap —
            # the very case _label was fixed for (Codex 2026-07-26).
            "book_plies": sum(1 for p in mine if p.get("in_book")),
        }
    return out


def _counts(plies: list[dict]) -> dict:
    """Tally by the READER's ladder, in ladder order (not alphabetical — the
    order is the ranking, and a renderer should not have to re-derive it)."""
    order = [c.value for c in MoveClass]
    c = {}
    for p in plies:
        c[p["label"]] = c.get(p["label"], 0) + 1
    return {k: c[k] for k in order if k in c}


def build_story(pgn_text: str, engine, pool, maia=None, *,
                fast_limit: dict | None = None,
                deep_limit: dict | None = None,
                plan_chapters: int = _PLAN_CHAPTERS,
                max_moments: int = _MAX_MOMENTS) -> dict:
    """PGN -> the story artifact. Blocking and slow (engine + Maia); the caller
    runs it off-thread or offline."""
    game = parse_pgn(pgn_text)
    fen_before = {p.ply: p.fen_before for p in game.plies}

    analysis = run_pass(pgn_text, engine, fast_limit=fast_limit or FAST_NODES,
                        deep_limit=deep_limit or DEEP_NODES)
    plies = analysis["plies"]
    for pr in plies:
        pr["fen_before"] = fen_before.get(pr["ply"], "")
    _gifts(plies)
    _label(plies)

    acts = _arc(plies)

    moments = _turning_points(plies) + _missed(plies) + _drift(plies)
    # A ply is one moment, not two: a missed chance that is ALSO a blunder
    # reads as the missed chance (it is the more useful sentence).
    seen, deduped = set(), []
    for m in sorted(moments, key=lambda m: (m.ply, m.kind != "missed")):
        if m.ply in seen:
            continue
        seen.add(m.ply)
        deduped.append(m)

    # RANK, then cap. A game with eighteen errors is not eighteen chapters; the
    # rest still exist in the ply table, so nothing is hidden — but the reader
    # is walked through the ones that decided the game. The cap is reported.
    # Ranked greedily with a minimum spacing, NOT just top-N. In a scrappy
    # game both sides blunder alternately in the same two-ply window, and a
    # plain top-N fills every chapter with one flurry while the actual turning
    # points earlier in the game fall off the list. Spacing keeps the
    # walkthrough spread across the game the way the game was played.
    ranked = sorted(deduped, key=_rank, reverse=True)
    chosen: list[Moment] = []
    for m in ranked:
        if len(chosen) >= max_moments:
            break
        if any(abs(m.ply - c.ply) < _MOMENT_GAP for c in chosen):
            continue
        chosen.append(m)
    dropped = len(ranked) - len(chosen)
    chosen.sort(key=lambda m: m.ply)

    # PLAN CHAPTERS MUST EARN THEIR PLACE. A quiet ply is only a candidate;
    # the plans layer decides. It declines endgames outright and can fail to
    # confirm anything, and printing "The plan on offer" over an empty read
    # would promise the reader the one thing this product exists to deliver
    # and then deliver nothing (Codex 2026-07-26). So: read FIRST, keep only
    # what carries an engine-confirmed plan, and report the attrition.
    reads: dict[str, dict] = {}          # fen -> plans read, rolled at most once

    def read_for(fen: str) -> dict:
        if fen not in reads:
            reads[fen] = _plans_read(fen, pool, maia)
        return reads[fen]

    considered = _plan_candidates(plies, seen, plan_chapters)
    tried = 0
    for pr in considered:
        if len([m for m in chosen if m.kind == "plan"]) >= plan_chapters:
            break
        tried += 1
        pl = read_for(pr["fen_before"])
        confirmed = any(p["tier"] == "engine"
                        for side in ("white", "black")
                        for p in (pl.get("plans") or {}).get(side, []))
        if pl.get("read") and confirmed:
            chosen.append(_moment("plan", pr))
    plan_shown = len([m for m in chosen if m.kind == "plan"])
    moments = sorted(chosen, key=lambda m: m.ply)

    for m in moments:
        fen = m.fen_before or m.fen_after
        st = _static_read(fen)
        m.phase, m.developed = st["phase"], st["developed"]
        m.structure, m.weaknesses = st["structure"], st["weaknesses"]
        pl = read_for(fen)
        m.plans = pl.get("plans", {})
        m.read = pl.get("read", "")
        m.character = pl.get("character", "")
        m.character_why = pl.get("character_why", "")
        m.initiative = pl.get("initiative", {})
        m.king_risk = pl.get("king_risk", {})
        m.only_move = pl.get("only_move", False)
        # the moves the player ACTUALLY went on to play, from this position
        cont = [q["uci"] for q in plies if q["ply"] >= m.ply][:_INTENT_PLIES]
        # the plans layer declines endgames by design, so the endgame read is
        # what carries a technical position — see pipelines/endgame.py
        if m.phase == "endgame":
            from .endgame import read as eg_read, sentences as eg_sentences
            m.endgame = eg_sentences(eg_read(fen))
        m.intent = move_intent(m.fen_before, cont)
        m.alignment = _alignment(m.intent, m.plans, m.side, m.label)

    return {
        "schema": SCHEMA,
        "game": analysis["game"],
        "headers": dict(game.headers),
        "plies": plies,
        "arc": acts,
        "ending": _ending(plies, dict(game.headers),
                          analysis["game"].get("result", "")),
        "opening": _opening(plies),
        "tracks": _tracks(plies),
        "moments": [asdict(m) for m in moments],
        "summary": analysis["summary"],
        "patterns": _patterns(plies, moments, game.headers),
        # No silent caps (repo rule): say what was left out of the walkthrough.
        "moments_found": len(deduped),
        "moments_dropped": dropped,
        # honest attrition: how many quiet positions we READ vs how many
        # actually carried an engine-confirmed plan worth a chapter
        "plan_chapters_considered": len(considered),
        "plan_chapters_read": tried,
        "plan_chapters_shown": plan_shown,
    }


def analyse_pgn(pgn_text: str, *, maia=None, **kw) -> dict:
    """PGN -> story, standing up the engine correctly.

    The one entry point a caller should use. It constructs the engine and pool
    single-threaded, which is what makes the analysis REPRODUCIBLE (see the
    note on FAST_NODES): the same PGN yields the same page. `build_story` stays
    injectable for callers that already hold a warm pool, and carries the
    reproducibility caveat with it.
    """
    from lucena_engine.uci import Engine
    from ..engine_io.enginepool import EnginePool
    # The pool spawns Stockfish lazily during the plan reads, so it owns
    # processes by the time build_story returns — close it on every path or a
    # batch of offline analyses leaks one engine per run (Codex 2026-07-26).
    pool = EnginePool(size=2, threads=1)
    try:
        with Engine(threads=1) as engine:
            return build_story(pgn_text, engine, pool, maia, **kw)
    finally:
        pool.close()
