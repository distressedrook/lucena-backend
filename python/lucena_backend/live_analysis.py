"""Live engine analysis — a dedicated Stockfish deepening the current position for the Analysis
panel. Runs on its OWN engine instance (a separate Stockfish process) so continuous search never
contends with the coach's lock-shared engine.

The loop is iterative deepening by repeated one-shot `analyse(depth=d)` calls (the hash stays warm
between depths), publishing a top-N multi-PV snapshot after each depth on the `engine_lines`
channel — so the app watches the depth climb and the lines refine. Timing here is non-deterministic
(a UI readout) and never feeds the deterministic drill/test paths.
"""

from __future__ import annotations

import threading

from lucena_engine.evalmodel import win_pct_from_score
from lucena_engine import openings
from .response import pv_san


class LiveAnalyzer:
    """Deepens one position at a time on a dedicated engine, streaming `engine_lines`."""

    def __init__(self, engine, store, *, max_depth: int = 28, multipv: int = 3, pv_plies: int = 12):
        self._engine = engine
        self._store = store
        self._max_depth = max_depth
        self._multipv = multipv
        self._pv_plies = pv_plies
        self._cond = threading.Condition()
        self._fen: str | None = None
        self._on = False
        self._gen = 0                 # bumps on every target change → cancels the in-flight deepen
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="live-analysis", daemon=True)
        self._thread.start()

    def set_target(self, fen: str | None, on: bool) -> None:
        """Analyze `fen` while `on`; a change cancels any in-flight deepen and restarts from depth 1."""
        with self._cond:
            self._fen, self._on, self._gen = fen, on, self._gen + 1
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._gen += 1
            self._cond.notify_all()
        try:
            self._engine.close()
        except Exception:
            pass

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._stopped and not (self._on and self._fen):
                    self._cond.wait()
                if self._stopped:
                    return
                fen, gen = self._fen, self._gen
            self._deepen(fen, gen)

    def _deepen(self, fen: str, gen: int) -> None:
        for depth in range(1, self._max_depth + 1):
            with self._cond:
                if self._stopped or gen != self._gen:
                    return
            try:
                analysis = self._engine.analyse(fen, depth=depth, multipv=self._multipv)
            except Exception:
                return
            with self._cond:
                if self._stopped or gen != self._gen:   # target changed mid-search — drop this result
                    return
            self._publish(fen, depth, analysis)
        # Reached max depth — idle on this position until the target changes.
        with self._cond:
            while not self._stopped and gen == self._gen:
                self._cond.wait()

    def _publish(self, fen: str, depth: int, analysis) -> None:
        white = fen.split()[1] == "w"
        lines = []
        for ln in analysis.lines:
            cp = ln.score.to_ceiled_cp()            # side-to-move POV → flip to white-relative
            wp = win_pct_from_score(ln.score)
            lines.append({
                "rank": ln.rank,
                "eval_white_cp": cp if white else -cp,
                "win_pct": round(wp if white else 100 - wp, 1),
                "pv_san": pv_san(fen, ln.pv, max_plies=self._pv_plies),
            })
        self._store.publish_engine_lines({
            "fen": fen, "depth": depth, "engine": self._engine.name,
            "opening": openings.name_for(fen), "lines": lines,
        })
