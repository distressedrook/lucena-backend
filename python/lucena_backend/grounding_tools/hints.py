"""Thin re-export: the hint ladder moved to lucena-tactics (2026-07-23,
core-migration Phase 4). Same pattern as drill.py."""

from __future__ import annotations

from ._tactics_path import ensure as _ensure_tactics

_ensure_tactics()

from hints import Hint, derive_hints  # noqa: E402,F401
