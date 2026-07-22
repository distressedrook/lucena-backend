"""Thin re-export: the drill walker moved to lucena-tactics (2026-07-22).

`DrillState` + the feedback beats are pure tree-walking coaching logic — no DB,
no session state — so they live beside the rest of the tactical vocabulary in
the lucena-tactics repo (same reasoning as the poisoned-line detector's move
the same day). Every backend import site (`from .drill import DrillState`,
`from ..grounding_tools.drill import DrillState, _same_move`) keeps working
through this shim; the sys.path bootstrap has ONE home, `_tactics_path`.
"""

from __future__ import annotations

from ._tactics_path import ensure as _ensure_tactics

_ensure_tactics()

from drill import DrillState, _same_move  # noqa: E402,F401
