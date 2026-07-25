"""WHAT WE KNOW ABOUT A POSITION — a fen → JSON cache (2026-07-26, owner: "the entire position is
being recalculated. Upon every move even when I go back ... There needs to be a fen: positionJSON
cache").

Analysis is a pure function of the position, so it is cached BY THE POSITION and shared across
chats, activities, navigation and restarts. Two artifacts live here today:

  SHEET  the plans layer's post-verify JSON — a ~5s engine+Maia roll
  LINES  the live analyzer's deepest multi-PV snapshot — the Analysis panel's four lines

Two layers, one contract: a bounded in-memory dict in front (the same process re-visits the same
position constantly while you arrow around) and Postgres behind it (so yesterday's roll is still
there tomorrow). A cache miss is always survivable — every caller recomputes — so EVERY database
failure here is swallowed and logged, never raised: an unreachable cache must not take a read down.

Keys are NORMALIZED fens (placement + side + castling + ep). The move clocks are deliberately not
part of the key: the same position reached with a different halfmove count is the same position to
an engine, and including them fragmented the cache on every transposition.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

_log = logging.getLogger(__name__)

SHEET = "sheet"          # lucena-plans post-verify JSON
LINES = "lines"          # the live analyzer's engine_lines payload

MEM_MAX = 256            # positions held in memory per kind (a long session's worth)
DB_KEEP = 20_000         # rows kept on disk; older ones are pruned by recency
_PRUNE_EVERY = 200       # writes between prunes — a bounded table, not a per-write scan


def norm(fen: str) -> str:
    """The cache key: placement, side to move, castling, en passant. Same rule as the margin's
    own key and lucena_core's norm_fen, kept here so both callers agree byte for byte."""
    return " ".join(fen.split()[:4])


class PositionCache:
    """Memory in front, Postgres behind. Thread-safe; every DB error degrades to a miss."""

    def __init__(self, db=None, *, mem_max: int = MEM_MAX, db_keep: int = DB_KEEP):
        self._db = db
        self._mem: dict[str, OrderedDict[str, dict]] = {}
        self._lock = threading.Lock()
        self._mem_max = mem_max
        self._db_keep = db_keep
        self._writes = 0

    def get(self, fen: str, kind: str) -> dict | None:
        key = norm(fen)
        with self._lock:
            bucket = self._mem.setdefault(kind, OrderedDict())
            if key in bucket:
                bucket.move_to_end(key)                  # LRU: a re-visit is a use
                return bucket[key]
        if self._db is None:
            return None
        try:
            payload = self._db.get_position(key, kind)
        except Exception:                                # noqa: BLE001 — a miss, never a failure
            _log.warning("position cache read failed for %s/%s", key, kind, exc_info=True)
            return None
        if payload is not None:
            self._remember(key, kind, payload)
        return payload

    def put(self, fen: str, kind: str, payload: dict) -> None:
        key = norm(fen)
        self._remember(key, kind, payload)
        if self._db is None:
            return
        try:
            self._db.put_position(key, kind, payload, time.time())
            with self._lock:
                self._writes += 1
                due = self._writes % _PRUNE_EVERY == 0
            if due:
                self._db.prune_positions(self._db_keep)
        except Exception:                                # noqa: BLE001 — the memory layer still has it
            _log.warning("position cache write failed for %s/%s", key, kind, exc_info=True)

    def _remember(self, key: str, kind: str, payload: dict) -> None:
        with self._lock:
            bucket = self._mem.setdefault(kind, OrderedDict())
            bucket[key] = payload
            bucket.move_to_end(key)
            while len(bucket) > self._mem_max:
                bucket.popitem(last=False)               # oldest use goes first

    def clear(self) -> None:
        """Memory only — the durable half is pruned by recency, never wiped. For tests."""
        with self._lock:
            self._mem.clear()
