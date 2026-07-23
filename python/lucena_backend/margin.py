"""The MARGIN's content builder — JSON-inspection mode (owner, 2026-07-23).

Everything previously implemented here (epigraph / theory doors / position
card / badges / deterministic formatting) is REMOVED for now — git history at
backend fb4b2d7 holds the card builder. The margin currently shows the plans
layer's artifacts raw, pretty-printed:

  1. request lands → ONE background worker rolls the position and emits the
     PRE-VERIFY JSON the moment it exists (no verify_plan calls yet) — the
     margin shows it while `plansPending` stays true;
  2. the verify gate finishes → the POST-VERIFY JSON replaces it and
     `plansPending` drops, ending the app's polling.

Results are cached per position; the deep pass runs for the LIVE position
only (scrubs never trigger rolls — standing rule).
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from lucena_core.board import Board

_log = logging.getLogger(__name__)

# -- plumbing (configured once by httpserver) ---------------------------------
_pool = None          # EnginePool — leased per job
_maia = None          # MaiaEngine | None (plans verify degrades without it)
_deep_cache: dict[str, dict] = {}      # norm fen -> {"raw", "statusLine", "pending"}
_inflight: set[str] = set()
_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)   # ONE: plans rolls are heavy


def configure(pool, maia) -> None:
    """Called once at server build; without it the deep layer stays off and
    the margin serves nothing (tests, offline)."""
    global _pool, _maia
    _pool, _maia = pool, maia


def _cache(key: str, raw: str, status: str, pending: bool) -> None:
    with _lock:
        _deep_cache[key] = {"raw": raw, "statusLine": status, "pending": pending}
        if not pending:
            _inflight.discard(key)
        if len(_deep_cache) > 64:                 # bounded: a session's worth
            _deep_cache.pop(next(iter(_deep_cache)))


def _deep_job(fen: str) -> None:
    """Roll once; publish PRE the moment it exists, POST when verify lands.
    Every failure caches a terminal result so the app's polling terminates."""
    key = " ".join(fen.split()[:4])
    try:
        from .plans import service as _plans
        _pre, post = _plans.sheet_json_staged(
            fen, _pool, _maia,
            on_pre=lambda pre: _cache(key, json.dumps(pre, indent=2),
                                      "PRE-VERIFY · VERIFYING…", True))
        _cache(key, json.dumps(post, indent=2), "POST-VERIFY", False)
    except Exception:
        _log.warning("margin deep layer failed for %s", fen, exc_info=True)
        _cache(key, json.dumps({"error": "sheet failed — see backend log"},
                               indent=2), "ERROR", False)


def build(fen: str, *, seed: str = "", live: bool = False) -> dict:
    """MarginContent for `fen` — currently just the raw sheet JSON."""
    Board(fen)                            # validates; raises ValueError on garbage
    key = " ".join(fen.split()[:4])
    cached = _deep_cache.get(key)
    if cached is not None:
        return {"statusLine": cached["statusLine"], "raw": cached["raw"],
                "plansPending": cached["pending"]}
    if live and _pool is not None:
        with _lock:
            if key not in _inflight:
                _inflight.add(key)
                _worker.submit(_deep_job, fen)
        return {"statusLine": "ROLLING…", "raw": None, "plansPending": True}
    return {"statusLine": None, "raw": None, "plansPending": False}
