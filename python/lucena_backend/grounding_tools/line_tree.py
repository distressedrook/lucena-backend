"""Thin re-export: the forcing-line tree builder moved to lucena-tactics
(core-migration Phase 5, 2026-07-23). Same pattern as drill.py."""

from __future__ import annotations

from ._tactics_path import ensure as _ensure_tactics

_ensure_tactics()

from line_tree import build_line_tree, count_leaves, is_only_move  # noqa: E402,F401
