"""Roll producers for the plans layer — the backend side of the
(fen, pvs, rolls) contract.

lucena-plans (verify/fact_sheet/suggest) never rolls: it takes engine PVs
and Maia rollouts as arguments and only checks lines. The research repo
produces them by shelling to chess-lab's venv and a docker pipe
(lucena-plans/research/experiments/tools/rolls.py); HERE the backend produces the
same shapes from what it already owns — a leased pool Engine and the
in-process MaiaEngine. No subprocesses, no gRPC.

Shapes (the contract):
  pvs    [{"cp": int, "ucis": [uci, ...]}, ...]   cp White-POV, best line
         first; each line extended toward `horizon` plies with cheap
         best-move searches (same scheme as the research helper: one
         MultiPV root search, then multipv=1 extensions).
  rolls  [[uci, ...], ...]                        K gated Maia rollouts,
         the 40-60 policy gate (deviate from the top move only when
         p/(p_top+p) >= 0.40), K*=9 from the k-study.

Node counts are env-tunable and default to the CALIBRATION numbers
(1M/250k — the depths every corpus/benchmark result was measured at;
user ruling 2026-07-22: full main nodes on the live path, latency paid).
Lower them via env for a faster, noisier read — an equal line is an equal
line at any depth, the equality is just noisier.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random

from lucena_core.board import Board

_log = logging.getLogger(__name__)

# LIVE budgets (2026-07-24, owner: "13s is way too much"). The research
# calibration ran at 1M/250k; live that measured ~13s per fresh position,
# and MarginLiveView's poll is `.task(id: fen)` — it CANCELS the moment the
# position changes, so at 13s a roll never finished while you were playing
# and the plans appeared to "stop arriving". 300k/80k lands ~5s.
# TRADEOFF, stated plainly: this is a shallower search than the one the
# verify gate's lift/confirmation numbers were measured at, so a marginal
# plan may fail to confirm that would have confirmed at 1M. Restore the
# research budget with LUCENA_PLANS_MAIN_NODES=1000000 (env still wins).
MAIN_NODES = int(os.environ.get("LUCENA_PLANS_MAIN_NODES", 300_000))
EXT_NODES = int(os.environ.get("LUCENA_PLANS_EXT_NODES", 80_000))
HORIZON = 30       # covers the slow families that matter (minority 30);
                   # verify truncates per family, so fast families still
                   # read their own natural window. 40 measured +4-5s for
                   # alternation-only coverage — not worth it live.
MULTIPV = 4        # the multi-line agreeability result: ~3.4 equal plans/position
K = 9              # the k-study K*
MAIA_RATING = 2400 # the calibration model (Maia 2400) — strong-human typicality
_BAND = 0.40       # the 40-60 policy gate


def _seed(fen: str) -> int:
    """Stable across processes (unlike hash(fen), which Python randomizes)."""
    return int(hashlib.sha256(fen.encode()).hexdigest()[:8], 16)


def roll_engine(engine, fen: str, *, horizon: int = HORIZON,
                multipv: int = MULTIPV, main_nodes: int | None = None,
                ext_nodes: int | None = None) -> list[dict] | None:
    """MultiPV lines extended toward `horizon`, or None if the engine fails.

    `engine` is a leased `lucena_engine.uci.Engine` — the caller owns the
    lease (`with pool.lease() as eng`). cp is White-POV (the fact-sheet /
    verify convention); mates collapse to the ±1000 ceiling. Node counts
    default to the module constants AT CALL TIME (late-bound, so tests and
    runtime tuning can override `rolls.MAIN_NODES`).
    """
    main_nodes = MAIN_NODES if main_nodes is None else main_nodes
    ext_nodes = EXT_NODES if ext_nodes is None else ext_nodes
    try:
        a = engine.analyse(fen, multipv=multipv, nodes=main_nodes)
    except Exception:
        # visible, not silent (2026-07-24 fix): a None engine leg makes the
        # whole sheet unverified, so the caller (service.sheet_json_for)
        # treats it as a plans-read failure and falls back to the plain read.
        _log.warning("roll_engine: analyse failed for %s", fen, exc_info=True)
        return None
    white = fen.split()[1] == "w"
    pvs = []
    for ln in a.lines:
        cp = ln.score.to_ceiled_cp()
        line = list(ln.pv[:horizon])
        try:
            line = _extend(engine, fen, line, horizon, ext_nodes)
        except Exception:
            pass                     # a truncated line is still a line
        pvs.append({"cp": cp if white else -cp, "ucis": line})
    return pvs


def _extend(engine, fen: str, line: list[str], horizon: int,
            ext_nodes: int) -> list[str]:
    """Push the line to `horizon` plies with multipv=1 best-move searches
    (a root PV rarely reaches 25 plies on its own)."""
    cur = Board(fen)
    for u in line:
        cur = cur.apply(u)
    while len(line) < horizon and cur.legal_moves():
        sub = engine.analyse(cur.fen, multipv=1, nodes=ext_nodes)
        add = sub.lines[0].pv if sub.lines else []
        if not add:
            break
        for u in add:
            if len(line) >= horizon or not cur.legal_moves():
                break
            line.append(u)
            cur = cur.apply(u)
    return line


def roll_maia(maia, fen: str, *, horizon: int = HORIZON, k: int = K,
              rating: int = MAIA_RATING) -> list[list[str]] | None:
    """K gated rollouts as UCI lists, or None when Maia is absent/failing
    (verify degrades to the engine leg — the contract allows a None leg).

    The gate mirrors the research harness (gpu_bench.rollout_gated): play
    the policy top move unless a rival is genuinely contested
    (p/(p_top+p) >= 0.40), then sample among the contested set. Seeds are
    derived from the FEN, so the same position rolls the same lines in
    every process.
    """
    if maia is None:
        return None
    try:
        base = _seed(fen)
        return [_rollout(maia, fen, horizon, rating,
                         random.Random(base * 100 + i))
                for i in range(k)]
    except Exception:
        # Maia is the optional leg (verify degrades to engine), but a genuine
        # rollout failure should still be visible, not swallowed (2026-07-24).
        _log.warning("roll_maia: rollout failed for %s", fen, exc_info=True)
        return None


def _rollout(maia, fen: str, horizon: int, rating: int,
             rng: random.Random) -> list[str]:
    cur = Board(fen)
    out: list[str] = []
    while len(out) < horizon:
        legal = set(cur.legal_moves())
        if not legal:
            break
        rows = maia.top_human_moves(cur.fen, rating, n=5)
        cand = [(r["uci"], r["policy"]) for r in rows
                if r.get("uci") in legal and r.get("policy") is not None]
        if not cand:
            break                     # policy-less wrapper → no rollouts
        top_u, top_p = cand[0]
        contested = [(top_u, top_p)] + [
            (u, p) for u, p in cand[1:]
            if top_p + p > 0 and p / (top_p + p) >= _BAND]
        mv = top_u if len(contested) == 1 else \
            rng.choices([u for u, _ in contested],
                        weights=[p for _, p in contested])[0]
        out.append(mv)
        cur = cur.apply(mv)
    return out
