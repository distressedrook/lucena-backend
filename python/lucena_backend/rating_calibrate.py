"""Does the estimator recover a rating it *knows* is true? — the calibration rig.

`lucena_backend.rating` fits a rating by maximum likelihood under Maia's policy.
The obvious worry is bias: a single game is ~40 decisions, and a player who keeps
finding the natural move can look stronger than they are. The only way to answer
that is ground truth, and the one ground truth available offline is Maia itself:
sample a whole game *from* the policy at a fixed SelfElo and the true rating of
each side is known by construction.

That makes this a recovery test of a correctly-specified model — it measures the
estimator, not whether Maia's idea of a 1500 matches a real 1500. Read a bias
found here as the estimator's own; read the absolute scale as lichess-rapid,
because that is what Maia3 was trained on.

    cd backend && PYTHONPATH=python .venv/bin/python -m lucena_backend.rating_calibrate
"""

from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _cand in (_ROOT, _ROOT.parents[2] if len(_ROOT.parents) > 2 else _ROOT):
    if (_cand / ".venv-maia/bin/python").exists() and not os.environ.get("LUCENA_MAIA"):
        os.environ["LUCENA_MAIA"] = (f"{_cand}/.venv-maia/bin/python "
                                     f"{_cand}/engine/scripts/maia_policy_uci.py")

import chess                                            # noqa: E402
import chess.pgn                                        # noqa: E402

from .rating import MULTIPV, estimate_ratings           # noqa: E402

BANDS = [1100, 1400, 1700, 2000, 2300]
GAMES_PER_BAND = 2
PLIES = 80              # ~40 moves each, the length of a real club game
# Coarser than the live grid: calibration needs the peak's location, not a
# pretty profile, and every grid point is a forward pass per move.
CAL_GRID = list(range(800, 2901, 200))


def selfplay(maia, rating: int, seed: int, plies: int = PLIES) -> str:
    """One game where both sides ARE a Maia rated `rating`, moves sampled from
    the policy (not argmax — argmax would produce a player far stronger and far
    more repetitive than the rating it is labelled with)."""
    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = game.headers["Black"] = f"maia-{rating}"
    game.headers["WhiteElo"] = game.headers["BlackElo"] = str(rating)
    node = game
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        rows = maia.top_human_moves(board.fen(), rating, n=MULTIPV, oppo_rating=rating)
        moves, weights = [], []
        for r in rows:
            try:
                mv = chess.Move.from_uci(r["uci"])
            except ValueError:
                continue
            if mv in board.legal_moves:
                moves.append(mv)
                weights.append(max(float(r.get("policy", 0.0)), 1e-9))
        if not moves:
            break
        mv = rng.choices(moves, weights=weights, k=1)[0]
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = board.result(claim_draw=True)
    return str(game)


def _one(job: tuple[int, int]) -> tuple[int, int, int, int, int]:
    """(rating, seed) → (rating, seed, white_est, black_est, covered) in a fresh
    process: each needs its own Maia subprocess, and they are CPU-bound."""
    rating, seed = job
    from lucena_engine.maia import MaiaEngine
    with MaiaEngine() as maia:
        pgn = selfplay(maia, rating, seed)
        ests = estimate_ratings(pgn, maia, grid=CAL_GRID)
    w, b = ests[chess.WHITE], ests[chess.BLACK]
    covered = sum(1 for e in (w, b) if e.low <= rating <= e.high)
    return rating, seed, w.rating, b.rating, covered


def main(argv: list[str]) -> int:
    workers = int(os.environ.get("LUCENA_CAL_WORKERS", "4"))
    jobs = [(r, s) for r in BANDS for s in range(GAMES_PER_BAND)]
    print(f"{len(jobs)} self-play games over {BANDS}, {workers} workers", file=sys.stderr)

    rows: list[tuple] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_one, jobs):
            rows.append(res)
            print(f"  true {res[0]}  seed {res[1]}  →  W {res[2]}  B {res[3]}  "
                  f"(in-interval {res[4]}/2)", file=sys.stderr, flush=True)

    print("\ntrue   n   mean est   bias   covered by 95% interval")
    total_cov = total_n = 0
    for band in BANDS:
        ests = [e for r in rows if r[0] == band for e in (r[2], r[3])]
        cov = sum(r[4] for r in rows if r[0] == band)
        total_cov += cov
        total_n += len(ests)
        mean = sum(ests) / len(ests)
        print(f"{band:5d} {len(ests):3d}   {mean:7.0f}  {mean - band:+6.0f}   "
              f"{cov}/{len(ests)}")
    allpairs = [(r[0], e) for r in rows for e in (r[2], r[3])]
    bias = sum(e - t for t, e in allpairs) / len(allpairs)
    mae = sum(abs(e - t) for t, e in allpairs) / len(allpairs)
    print(f"\noverall bias {bias:+.0f}   MAE {mae:.0f}   "
          f"coverage {total_cov}/{total_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
