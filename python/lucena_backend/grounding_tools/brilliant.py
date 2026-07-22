"""Thin re-export: the sound-sacrifice (!!) classifier moved to lucena-tactics
(core-migration Phase 5, 2026-07-23). Same pattern as drill.py."""

from __future__ import annotations

from ._tactics_path import ensure as _ensure_tactics

_ensure_tactics()

from brilliant import classify_brilliant, is_brilliant, is_material_sacrifice  # noqa: E402,F401
