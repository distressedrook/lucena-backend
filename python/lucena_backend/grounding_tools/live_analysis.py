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
from lucena_core import openings
from .response import pv_san


class LiveAnalyzer:
    """Deepens one position at a time on a dedicated engine, streaming `engine_lines`."""

    def __init__(self, engine, store, *, max_depth: int = 28, multipv: int = 3, pv_plies: int = 12,
                 positions=None):
        self._engine = engine
        self._store = store
        # positions.PositionCache | None — the fen -> lines cache. Walking BACK through a game used
        # to re-deepen every position from scratch (2026-07-26); now the last line-up published for
        # a position is republished immediately and the search resumes past it.
        self._positions = positions
        self._max_depth = max_depth
        self._multipv = multipv
        self._pv_plies = pv_plies
        self._cond = threading.Condition()
        self._fen: str | None = None
        self._sid: str | None = None  # the chat this target belongs to (see set_target)
        self._on = False
        self._gen = 0                 # bumps on every target change → cancels the in-flight deepen
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="live-analysis", daemon=True)
        self._thread.start()

    def set_target(self, fen: str | None, on: bool, *, session_id: str) -> None:
        """Analyze `fen` for chat `session_id` while `on`; a change cancels any in-flight deepen and
        restarts from depth 1.

        `session_id` is REQUIRED and travels with the target: this class deepens on its OWN raw
        `threading.Thread`, and contextvars do NOT cross a raw thread — the store's bound cursor would
        resolve empty here, so the chat must be carried explicitly and handed to publish_engine_lines.
        """
        with self._cond:
            self._fen, self._on, self._sid = fen, on, session_id
            self._gen += 1
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
                fen, gen, sid = self._fen, self._gen, self._sid
            self._deepen(fen, gen, sid)

    def _deepen(self, fen: str, gen: int, sid: str | None) -> None:
        # WHAT WE ALREADY KNOW about this position goes out first, and the search picks up past it.
        # Arrowing back through a game re-analysed every position from depth 1 (owner 2026-07-26:
        # "the entire position is being recalculated ... even when I go back") — a position already
        # taken to depth 28 now answers instantly and searches not at all.
        start = 1
        cached = self._cached(fen)
        if cached is not None:
            self._republish(cached, sid)
            start = int(cached.get("depth") or 0) + 1
        for depth in range(start, self._max_depth + 1):
            with self._cond:
                if self._stopped or gen != self._gen:
                    return
            try:
                analysis = self._engine.analyse(fen, depth=depth, multipv=self._multipv)
            except Exception:
                # A search that raises must not become a hot loop: `_run` would
                # re-enter with the same still-on target immediately. Idle on
                # this target until it changes (2026-07-26; the route also
                # rejects an unparseable FEN, this is the backstop for the rest
                # — an engine that died, a position it refuses).
                with self._cond:
                    while not self._stopped and gen == self._gen:
                        self._cond.wait()
                return
            with self._cond:
                if self._stopped or gen != self._gen:   # target changed mid-search — drop this result
                    return
            self._publish(fen, depth, analysis, sid)
        # Reached max depth — idle on this position until the target changes.
        with self._cond:
            while not self._stopped and gen == self._gen:
                self._cond.wait()

    def _cached(self, fen: str) -> dict | None:
        """The deepest line-up we have for this position, if any. A cache miss is always fine."""
        if self._positions is None:
            return None
        try:
            from ..positions import LINES
            payload = self._positions.get(fen, LINES)
        except Exception:                            # noqa: BLE001 — never break a search on a cache
            return None
        # The payload carries the fen it was computed for; a normalized-key collision (same
        # placement, different clocks) is still the same position to an engine, but the PV is
        # rendered from a fen, so republish the one that was stored.
        return payload if isinstance(payload, dict) and payload.get("lines") else None

    def _republish(self, payload: dict, sid: str | None) -> None:
        if not sid:
            return
        try:
            self._store.publish_engine_lines(payload, session_id=sid)
        except Exception:                            # noqa: BLE001
            pass

    def _store_lines(self, payload: dict) -> None:
        """Keep the line-up for next time. Written at a few depths rather than all 28: every depth
        would be ~28 database writes per position for a readout that only improves."""
        if self._positions is None:
            return
        depth = int(payload.get("depth") or 0)
        if depth < 8 or (depth % 4 and depth != self._max_depth):
            return
        try:
            from ..positions import LINES
            self._positions.put(payload["fen"], LINES, payload)
        except Exception:                            # noqa: BLE001
            pass

    def _publish(self, fen: str, depth: int, analysis, sid: str | None) -> None:
        if not sid:                                  # no chat → nobody to address; never publish blind
            return
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
        payload = {
            "fen": fen, "depth": depth, "engine": self._engine.name,
            "opening": openings.name_for(fen), "lines": lines,
        }
        self._store.publish_engine_lines(payload, session_id=sid)
        self._store_lines(payload)
