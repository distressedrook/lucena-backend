"""Thin re-export: the fact sheet moved to lucena-tactics (2026-07-23,
core-migration Phase 4). Same pattern as drill.py; the sys.path bootstrap has
ONE home, `_tactics_path`."""

from __future__ import annotations

from ._tactics_path import ensure as _ensure_tactics

_ensure_tactics()

from facts import Fact, build_fact_sheet  # noqa: E402,F401
