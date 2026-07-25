"""Estimate a player's rating from one game — maximum likelihood under Maia.

The question "how strong is this player?" is a *behaviour* question, not a
quality question, so it belongs to Maia and not to Stockfish. Maia3's policy
head is conditioned on a rating: `p(move | position, SelfElo=R, OppoElo=O)` is
a real probability (the `maia_policy_uci.py` wrapper puts it on the wire; see
`lucena_engine.maia`). That makes the estimator a textbook MLE — no heuristic
about blunder rates, no centipawn-loss table, no engine agreement percentage:

    LL(R) = Σ_i log p(move_i | fen_i, SelfElo=R, OppoElo=O)      R̂ = argmax LL

Maia3 interpolates the rating embedding linearly over [0, 5000] (`models.py
interpolate_elo`), so LL is smooth in R — a coarse grid plus a parabolic fit at
the peak gets a continuous estimate without a fine sweep.

Three details that matter:

* **Both ratings at once.** The policy is conditioned on the *opponent's*
  rating too, so White's estimate depends on Black's and vice versa. We solve
  it by coordinate ascent: fix the opponent, fit the player, alternate. It
  converges in two or three rounds on a single game.
* **Book moves carry prep, not skill.** A move played from a position still in
  the opening book says how much theory someone memorised, which is exactly the
  confound Maia can't see. Skipped by default (`--book`).
* **MultiPV caps at 20** (maia3's own limit), so a move outside the top 20 has
  no printed policy. It gets the *leftover* mass spread over the unlisted legal
  moves — the honest floor, and the only place the estimate is approximate.

The output is a profile-likelihood interval, not just a number: one game is
~40 decisions and the surface is genuinely flat. Read a ±200 band as the
signal, and the point estimate as its centre.

CLI:

    cd backend && PYTHONPATH=python .venv/bin/python -m lucena_backend.rating game.pgn
"""

from __future__ import annotations

import io
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# serve.sh's Maia auto-detect, so a bare shell behaves like the served stack.
_ROOT = Path(__file__).resolve().parents[3]
if not os.environ.get("LUCENA_MAIA") and (_ROOT / ".venv-maia/bin/python").exists():
    os.environ["LUCENA_MAIA"] = (f"{_ROOT}/.venv-maia/bin/python "
                                 f"{_ROOT}/engine/scripts/maia_policy_uci.py")

import chess                                            # noqa: E402
import chess.pgn                                        # noqa: E402

from lucena_core import openings                        # noqa: E402

# The grid the profile is computed on. Wider than any real player so the peak is
# interior (a peak at the edge means "off the scale", and we say so rather than
# silently clamping). 100-point steps: finer buys nothing, the parabola handles it.
GRID = list(range(600, 2901, 100))
MULTIPV = 20        # maia3's own cap (uci.py: "MultiPV … min 1 max 20")
ROUNDS = 3          # coordinate-ascent passes; converges in 2 on a single game
PRIOR = 1500        # the opponent rating each side is first fitted against
# 1.92 = χ²(1, 0.95)/2 — the standard profile-likelihood cutoff for a 95% interval.
LR_CUTOFF = 1.92


@dataclass(frozen=True)
class Decision:
    """One move by one player, with the position it was chosen from."""
    ply: int
    color: bool          # chess.WHITE / chess.BLACK
    fen: str
    uci: str
    san: str
    n_legal: int
    in_book: bool


@dataclass
class Estimate:
    """What we can say about one player after fitting the whole game."""
    color: bool
    rating: int                       # the MLE, parabola-refined
    low: int                          # 95% profile-likelihood interval
    high: int
    n_moves: int                      # decisions that actually carried signal
    log_likelihood: float
    profile: dict[int, float] = field(default_factory=dict)
    top1_agreement: float = 0.0       # share of moves that were Maia's #1 at R̂
    median_rank: float = 0.0
    surprises: list[tuple[str, float]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "White" if self.color == chess.WHITE else "Black"

    @property
    def pinned(self) -> str:
        """'' unless the peak sits on the edge of the grid — then the estimate is
        a bound, not a measurement, and the caller must not print it as one."""
        if not self.profile:
            return ""
        if self.rating <= min(self.profile):
            return "below"
        if self.rating >= max(self.profile):
            return "above"
        return ""


# -- reading the game --------------------------------------------------------

def decisions(pgn_text: str) -> list[Decision]:
    """Every move of the mainline, tagged with the position it was played from.

    Positions with a single legal move are dropped here: a forced move has
    probability 1 at every rating, so it contributes a constant zero to every
    LL and only dilutes the diagnostics."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("no game found in the PGN")
    board = game.board()
    out: list[Decision] = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        legal = board.legal_moves.count()
        if legal > 1:
            out.append(Decision(
                ply=ply, color=board.turn, fen=board.fen(), uci=move.uci(),
                san=board.san(move), n_legal=legal,
                in_book=openings.name_for(board.fen()) is not None,
            ))
        board.push(move)
    return out


# -- the likelihood ----------------------------------------------------------

class _Policy:
    """Maia lookups with memoisation. Coordinate ascent revisits the same
    (position, rating) pairs across rounds; one forward pass each is plenty."""

    def __init__(self, maia):
        self._maia = maia
        self._cache: dict[tuple[str, int, int], list[dict]] = {}
        self.calls = 0

    def rows(self, fen: str, self_elo: int, oppo_elo: int) -> list[dict]:
        key = (fen, self_elo, oppo_elo)
        if key not in self._cache:
            self.calls += 1
            self._cache[key] = self._maia.top_human_moves(
                fen, self_elo, n=MULTIPV, oppo_rating=oppo_elo)
        return self._cache[key]

    def logp(self, d: Decision, self_elo: int, oppo_elo: int) -> tuple[float, int]:
        """log p(the move actually played) and its rank in Maia's ordering.

        A move outside the top 20 gets the leftover probability mass shared over
        the legal moves Maia didn't print — the honest floor. Rank is reported as
        0 for those ("off the list")."""
        rows = self.rows(d.fen, self_elo, oppo_elo)
        listed = 0.0
        for r in rows:
            p = float(r.get("policy", 0.0))
            listed += p
            if r["uci"] == d.uci:
                return math.log(max(p, 1e-9)), int(r.get("rank", 0))
        unlisted = max(1, d.n_legal - len(rows))
        floor = max(1.0 - listed, 1e-6) / unlisted
        return math.log(floor), 0


def _refine(profile: dict[int, float]) -> float:
    """Parabolic interpolation through the peak and its two neighbours — the
    continuous argmax of a smooth LL sampled on a coarse grid."""
    grid = sorted(profile)
    i = max(range(len(grid)), key=lambda k: profile[grid[k]])
    if i in (0, len(grid) - 1):
        return float(grid[i])
    x0, x1, x2 = grid[i - 1], grid[i], grid[i + 1]
    y0, y1, y2 = profile[x0], profile[x1], profile[x2]
    denom = y0 - 2 * y1 + y2
    if denom >= 0:                       # not a peak (flat or a trough) — trust the grid
        return float(x1)
    step = (x1 - x0)
    return x1 + step * (y0 - y2) / (2 * denom)


def _interval(profile: dict[int, float], peak: float) -> tuple[int, int]:
    """The 95% profile-likelihood interval: every rating whose LL is within
    1.92 of the maximum, with linear interpolation to the crossing point."""
    grid = sorted(profile)
    best = max(profile.values())
    thresh = best - LR_CUTOFF
    inside = [r for r in grid if profile[r] >= thresh]
    lo, hi = inside[0], inside[-1]

    def cross(a: int, b: int) -> float:      # a inside, b outside
        ya, yb = profile[a], profile[b]
        if ya == yb:
            return float(a)
        return a + (b - a) * (ya - thresh) / (ya - yb)

    if lo != grid[0]:
        lo = cross(lo, grid[grid.index(lo) - 1])
    if hi != grid[-1]:
        hi = cross(hi, grid[grid.index(hi) + 1])
    return int(round(min(lo, peak))), int(round(max(hi, peak)))


def _fit(pol: _Policy, moves: list[Decision], oppo: int,
         grid: list[int]) -> dict[int, float]:
    """LL over the whole grid for one player, against a fixed opponent rating."""
    return {r: sum(pol.logp(d, r, oppo)[0] for d in moves) for r in grid}


def _diagnose(pol: _Policy, moves: list[Decision], rating: int, oppo: int) -> tuple:
    """Human-readable checks at the fitted rating: how often the player found
    Maia's top pick, the median rank, and the least-likely moves they played."""
    ranks, probs = [], []
    for d in moves:
        lp, rank = pol.logp(d, rating, oppo)
        ranks.append(rank if rank else MULTIPV + 1)
        probs.append((d, math.exp(lp)))
    top1 = sum(1 for r in ranks if r == 1) / len(ranks)
    srt = sorted(ranks)
    mid = len(srt) // 2
    median = float(srt[mid]) if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2
    probs.sort(key=lambda t: t[1])
    worst = [(f"{(d.ply + 1) // 2}{'.' if d.color else '...'} {d.san}", p)
             for d, p in probs[:3]]
    return top1, median, worst


def estimate_ratings(pgn_text: str, maia, *, include_book: bool = False,
                     rounds: int = ROUNDS,
                     grid: list[int] | None = None) -> dict[bool, Estimate]:
    """Fit both players jointly. Returns {chess.WHITE: Estimate, chess.BLACK: …}.

    Coordinate ascent: each side is first fitted against a 1500 opponent, then
    re-fitted against the other side's current estimate until both settle."""
    grid = list(grid or GRID)
    all_moves = decisions(pgn_text)
    by_color = {c: [d for d in all_moves if d.color == c and (include_book or not d.in_book)]
                for c in (chess.WHITE, chess.BLACK)}
    for c, ms in by_color.items():
        if not ms:
            raise ValueError(
                f"no out-of-book moves for {'White' if c else 'Black'} — "
                f"pass include_book=True to fit the opening too")

    pol = _Policy(maia)
    cur = {chess.WHITE: PRIOR, chess.BLACK: PRIOR}
    profiles: dict[bool, dict[int, float]] = {}
    for _ in range(rounds):
        moved = False
        for c in (chess.WHITE, chess.BLACK):
            # The opponent rating is snapped to the grid so the cache actually
            # hits across rounds; the policy barely moves within 100 points.
            oppo = min(grid, key=lambda r: abs(r - cur[not c]))
            prof = _fit(pol, by_color[c], oppo, grid)
            profiles[c] = prof
            peak = _refine(prof)
            if abs(peak - cur[c]) >= 25:
                moved = True
            cur[c] = peak
        if not moved:
            break

    out: dict[bool, Estimate] = {}
    for c in (chess.WHITE, chess.BLACK):
        prof = profiles[c]
        peak = _refine(prof)
        lo, hi = _interval(prof, peak)
        oppo = min(grid, key=lambda r: abs(r - cur[not c]))
        snapped = min(grid, key=lambda r: abs(r - peak))
        top1, median, worst = _diagnose(pol, by_color[c], snapped, oppo)
        out[c] = Estimate(
            color=c, rating=int(round(peak)), low=lo, high=hi,
            n_moves=len(by_color[c]), log_likelihood=max(prof.values()),
            profile=prof, top1_agreement=top1, median_rank=median, surprises=worst,
        )
    return out


# -- CLI ---------------------------------------------------------------------

def _render(est: Estimate) -> str:
    lines = [f"{est.name}: ~{est.rating}  (95% {est.low}–{est.high}, "
             f"{est.n_moves} scored moves)"]
    if est.pinned:
        lines[0] += f"  [peak {est.pinned} the grid — read as a bound]"
    lines.append(f"  played Maia's top move {est.top1_agreement * 100:.0f}% of the time, "
                 f"median rank {est.median_rank:g}")
    if est.surprises:
        odd = ", ".join(f"{m} (p={p:.3f})" for m, p in est.surprises)
        lines.append(f"  least human-typical: {odd}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    flags = {a for a in argv[1:] if a.startswith("-")}
    if not args:
        print(__doc__)
        return 1
    src = Path(args[0])
    pgn_text = src.read_text() if src.exists() else args[0]

    if not os.environ.get("LUCENA_MAIA"):
        print("LUCENA_MAIA is unset and no .venv-maia found — no predictor, no estimate.",
              file=sys.stderr)
        return 2
    from lucena_engine.maia import MaiaEngine

    with MaiaEngine() as maia:
        ests = estimate_ratings(pgn_text, maia, include_book="--book" in flags)
    for c in (chess.WHITE, chess.BLACK):
        print(_render(ests[c]))
    if "--profile" in flags:
        print("\nlog-likelihood profile (higher is better):")
        for r in sorted(ests[chess.WHITE].profile):
            print(f"  {r:5d}  W {ests[chess.WHITE].profile[r]:9.2f}   "
                  f"B {ests[chess.BLACK].profile[r]:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
