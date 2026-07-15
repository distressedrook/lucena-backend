"""Curated puzzle content — selection layer for "give me a puzzle" (see docs/design/puzzle-library.md).

A pure, engine-free reader over `content/puzzles/*.jsonl`. The files reuse the poisoned-puzzle
schema (`content/special-puzzles/poisoned_special_puzzles.jsonl`): one JSON object per line with
`position_fen` (the solver to move), `themes` (space-separated Lichess tags), `rating`, and
optional `moves`/`bands` metadata. This module ONLY selects — the coach hands the chosen FEN to
`build_and_arm_drill`, which re-derives the solution deterministically. Nothing here calls the
engine, and (GPL hygiene) it never imports python-chess: FENs are validated with `lucena_board.Board`.

Selection is DETERMINISTIC (no RNG, matching the repo's determinism discipline): a fixed order over
theme filter → weakest-concept bias → rating/id tie-break, minus the ids already served this session.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from lucena_engine.board import Board

# Lichess theme token -> domain concept id. Used both to pick the
# `concept_id` build_and_arm_drill banks on solve AND to bias adaptive selection toward the
# player's weakest concept. A pragmatic hand map; unmatched themes fall through to _DEFAULT_CONCEPT.
THEME_CONCEPT = {
    # forks & double attacks
    "fork": "forks-double-attack", "doubleAttack": "forks-double-attack",
    "doubleCheck": "forks-double-attack",
    # pins, skewers, x-rays
    "pin": "pins-skewers", "skewer": "pins-skewers", "xRayAttack": "pins-skewers",
    # discovered attacks
    "discoveredAttack": "discovered-attack",
    # removing/overloading the defender
    "deflection": "removing-the-defender", "attraction": "removing-the-defender",
    "removeTheDefender": "removing-the-defender", "capturingDefender": "removing-the-defender",
    "interference": "removing-the-defender", "clearance": "removing-the-defender",
    "overloading": "removing-the-defender", "intermezzo": "removing-the-defender",
    # back rank
    "backRankMate": "back-rank",
    # hanging / trapped material
    "hangingPiece": "hanging-pieces", "trappedPiece": "hanging-pieces",
    # attacking the king (mates, sacrifices, wing attacks)
    "mate": "attack-defense-balance", "mateIn1": "attack-defense-balance",
    "mateIn2": "attack-defense-balance", "mateIn3": "attack-defense-balance",
    "mateIn4": "attack-defense-balance", "mateIn5": "attack-defense-balance",
    "sacrifice": "attack-defense-balance", "kingsideAttack": "attack-defense-balance",
    "queensideAttack": "attack-defense-balance", "exposedKing": "attack-defense-balance",
    "attackingF2F7": "attack-defense-balance", "smotheredMate": "attack-defense-balance",
    "arabianMate": "attack-defense-balance", "anastasiaMate": "attack-defense-balance",
    "bodenMate": "attack-defense-balance", "hookMate": "attack-defense-balance",
    "dovetailMate": "attack-defense-balance", "doubleBishopMate": "attack-defense-balance",
    # defence / quiet play
    "defensiveMove": "prophylaxis", "quietMove": "plan-formation", "zugzwang": "conversion",
    # endgames & conversion
    "advancedPawn": "conversion", "promotion": "conversion", "endgame": "conversion",
    "queenEndgame": "conversion", "bishopEndgame": "conversion", "knightEndgame": "conversion",
    "rookEndgame": "rook-endings", "pawnEndgame": "pawn-endings",
}

# The concept banked when a puzzle's themes match nothing above (a tactics catch-all).
_DEFAULT_CONCEPT = "tactical-signals"

# Meta/difficulty Lichess tags that describe a puzzle's SHAPE, not a teachable concept — never the
# concept a puzzle demonstrates, so puzzle_concept skips them when picking the demonstrated concept.
_META_THEMES = frozenset({
    "short", "long", "veryLong", "oneMove", "master", "masterVsMaster", "superGM",
    "crushing", "advantage", "equality", "middlegame", "opening",
})

# Broad phase tags that DO map to a concept but lose to a specific tactical motif when both are
# present — "advantage endgame fork" is a fork lesson, not a generic conversion lesson.
_BROAD_THEMES = frozenset({"endgame"})

# Selection cache, keyed by (dir, newest-mtime) so an edited/added file is picked up automatically.
_cache: dict = {}


def puzzles_dir() -> Path:
    """The curated-puzzle directory: `$LUCENA_PUZZLES`, else `<repo>/content/puzzles`. The default is
    absolute, resolved from this file's location in the repo."""
    env = os.environ.get("LUCENA_PUZZLES")
    if env:
        return Path(env)
    # .../python/lucena/mcp/puzzle_content.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "content" / "puzzles"


def _themes(puzzle: dict) -> list[str]:
    """The puzzle's Lichess theme tokens (a space-separated string in the schema)."""
    raw = puzzle.get("themes")
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return []


def puzzle_fen(puzzle: dict) -> str | None:
    """The coaching FEN — the position with the SOLVER to move. Prefer `position_fen` (after the
    schema's setup move); fall back to `fen` for hand-authored puzzles that omit it."""
    return puzzle.get("position_fen") or puzzle.get("fen")


def puzzle_id(puzzle: dict) -> str | None:
    return puzzle.get("puzzle_id") or puzzle.get("id")


def puzzle_concept(puzzle: dict) -> str:
    """The mastery concept a puzzle demonstrates: its first specific tactical theme that maps to a
    domain concept (meta/difficulty tags skipped, broad phase tags like `endgame` deferred), falling
    back to a broad tag's concept, then the tactics default."""
    matches = [(t, THEME_CONCEPT[t]) for t in _themes(puzzle)
               if t not in _META_THEMES and t in THEME_CONCEPT]
    if not matches:
        return _DEFAULT_CONCEPT
    for t, concept in matches:                       # a specific motif wins over a broad phase tag
        if t not in _BROAD_THEMES:
            return concept
    return matches[0][1]


def _valid(puzzle: dict) -> bool:
    """A row is usable iff it has an id and a legal solver-to-move FEN."""
    if not puzzle_id(puzzle):
        return False
    fen = puzzle_fen(puzzle)
    if not fen:
        return False
    try:
        Board(fen)
    except Exception:
        return False
    return True


def load_puzzles(directory=None) -> list[dict]:
    """Every usable puzzle under `directory` (default `puzzles_dir()`), read from all `*.jsonl`
    files. Malformed lines and illegal-FEN rows are skipped. Cached by newest file mtime so edits
    during a session are picked up without a restart."""
    d = Path(directory) if directory is not None else puzzles_dir()
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.jsonl"))
    mtime = max((f.stat().st_mtime for f in files), default=0.0)
    key = (str(d), mtime, len(files))
    cached = _cache.get(key)
    if cached is not None:
        return cached
    out: list[dict] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and _valid(obj):
                out.append(obj)
    _cache[key] = out
    return out


def select_puzzle(puzzles, *, theme=None, weakest=None, served=frozenset()) -> dict | None:
    """Pick one puzzle, DETERMINISTICALLY. Excludes ids in `served`. If `theme` is given, keep only
    puzzles carrying that theme token (case-insensitive). Otherwise, if `weakest` (concept ids
    ordered weakest-first) is given, bias toward puzzles whose concept the player is weakest on.
    Ties break by ascending `rating` then `puzzle_id`. Returns None if nothing is left."""
    pool = [p for p in puzzles if puzzle_id(p) not in served]
    if theme:
        t = theme.lower()
        pool = [p for p in pool if any(t == tok.lower() for tok in _themes(p))]
    if not pool:
        return None

    rank = {cid: i for i, cid in enumerate(weakest or [])}
    fallback_rank = len(rank)

    def sort_key(p):
        concept_rank = fallback_rank
        if theme is None and weakest:
            concept_rank = rank.get(puzzle_concept(p), fallback_rank)
        rating = p.get("rating")
        rating = rating if isinstance(rating, (int, float)) else 10 ** 9
        return (concept_rank, rating, str(puzzle_id(p)))

    return sorted(pool, key=sort_key)[0]
