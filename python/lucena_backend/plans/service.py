"""The plans-layer facade: gate helpers + the fact sheet, one call each.

`sheet_for(fen, pool, maia)` is the whole integration: lease an engine,
produce the rolled lines (rolls.py), hand (fen, pvs, rolls) to
lucena-plans' `build_fact_sheet`, return (sheet, opaque_id). The caller
(freeform._plans_read) owns the routing gates; the two deterministic ones
that need chess logic live here (`is_endgame`) or in lucena_engine
(`openings.name_for` for the book gate).

lucena-plans is imported from its superrepo checkout (a sibling of
backend/), not installed into the venv — its src/ dir holds flat modules whose
top-level names (suggest, verify, fact_sheet, ...) we only want on the
path deliberately, appended (never prepended) so nothing in the backend's
own environment can be shadowed. `LUCENA_PLANS_DIR` overrides the
location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lucena_core.board import Board

from .rolls import roll_engine, roll_maia

# |eval| <= this (cp, either POV) counts as equalish — the band in which a
# position is a game to be played rather than a result to be converted, so the
# chat answers a paste with the PLANS read instead of a plain grounded one.
# Raised 150 -> 250 with the sheet's own _DECISIVE_CP (owner 2026-07-26: "make
# the positional information show up when eval < |2.5|"); the two are the same
# ruling about the same band and drifting apart would mean the margin shows
# plans the chat refuses to discuss.
PLANS_CP_BAND = 250


class PlansRollError(RuntimeError):
    """The engine leg failed to produce lines — a plans-read failure, not a
    degraded sheet. Raised so the caller (freeform._plans_read) falls back to
    the plain grounded read rather than narrating an all-unverified sheet."""


def _require_engine(pvs):
    """A None engine leg means every plan comes out unverified — surface it as
    a failure (2026-07-24 fix), not a silent engine-less sheet. The Maia leg
    may still be None (verify degrades to engine); the engine leg may not."""
    if pvs is None:
        raise PlansRollError("engine roll produced no lines")
    return pvs

# Non-pawn material per side; the middlegame gate. A side at Q+minor /
# R+R+minor or less (<= 13) has entered the technical phase — with BOTH
# sides there, plans-vocabulary coaching (which is calibrated on
# middlegames) stands aside. Deliberately simple and tunable.
_NPM = {"Q": 9, "R": 5, "B": 3, "N": 3}
_ENDGAME_NPM = 13


def is_endgame(fen: str) -> bool:
    # delegates to the core phase classifier (2026-07-23 consolidation —
    # same NPM table and bar; one source of truth for "endgame")
    from lucena_core.reads import game_phase
    return game_phase(fen)["phase"] == "endgame"


def _plans_dir() -> Path:
    if env := os.environ.get("LUCENA_PLANS_DIR"):
        return Path(env)
    # backend/python/lucena_backend/plans/service.py -> superrepo root
    return Path(__file__).resolve().parents[4] / "lucena-plans"


def _bootstrap() -> None:
    d = str(_plans_dir() / "src")   # src layout (2026-07-22 restructure)
    if d not in sys.path:
        sys.path.append(d)          # append: never shadow backend deps


def sheet_for(fen: str, pool, maia=None, *, horizon: int | None = None
              ) -> tuple[str, str]:
    """(sheet, opaque_id) for one position — rolls, then builds.

    Blocking (one root search + per-line extensions + an optional Maia
    roll); call it off-thread. Raises on a missing lucena-plans checkout
    or an unusable pool — the caller treats any exception as "no plans
    read" and falls back to the plain grounded path.
    """
    kw = {"horizon": horizon} if horizon else {}
    with pool.lease() as engine:
        pvs = _require_engine(roll_engine(engine, fen, **kw))
    rolls = roll_maia(maia, fen, **kw)
    _bootstrap()
    from fact_sheet import build_fact_sheet   # lucena-plans, flat module
    return build_fact_sheet(fen, pvs, rolls)


def sheet_json_for(fen: str, pool, maia=None, *, horizon: int | None = None
                   ) -> tuple[dict, dict]:
    """(pre_verify_json, post_verify_json) for one position — the product
    path since 2026-07-24 (the text sheet is retired to research use).
    Rolls ONCE; pre is emitted from the same lines without verify_plan
    calls, post runs the verify gate. Blocking; call off-thread."""
    kw = {"horizon": horizon} if horizon else {}
    with pool.lease() as engine:
        pvs = _require_engine(roll_engine(engine, fen, **kw))
    rolls = roll_maia(maia, fen, **kw)
    _bootstrap()
    from fact_sheet import pre_verify_json, post_verify_json
    return pre_verify_json(fen, pvs, rolls), post_verify_json(fen, pvs, rolls)


def sheet_json_staged(fen: str, pool, maia=None, *, horizon: int | None = None,
                      on_pre=None) -> tuple[dict, dict]:
    """Like sheet_json_for, but hands the pre-verify artifact to `on_pre`
    the moment it exists (rolls done, no verify_plan calls yet), then runs
    the verify gate and returns (pre, post). Blocking; call off-thread."""
    kw = {"horizon": horizon} if horizon else {}
    with pool.lease() as engine:
        pvs = _require_engine(roll_engine(engine, fen, **kw))
    rolls = roll_maia(maia, fen, **kw)
    _bootstrap()
    from fact_sheet import pre_verify_json, post_verify_json
    pre = pre_verify_json(fen, pvs, rolls)
    if on_pre is not None:
        on_pre(pre)
    return pre, post_verify_json(fen, pvs, rolls)


def render_position_read(post: dict) -> str | None:
    """The post-verify sheet, presented DETERMINISTICALLY for the reader
    (owner ruling 2026-07-24 — no LLM in this path; the verified-plans
    reliability gate is a code filter now, not a prompt instruction).
    None when there is nothing worth showing; the caller falls back to
    its plain read. Pure formatting — cheap, but called off-thread with
    the roll anyway."""
    _bootstrap()
    from position_read import render     # lucena-plans, flat module
    return render(post)
