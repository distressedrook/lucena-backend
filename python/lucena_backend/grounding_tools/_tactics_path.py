"""Sibling-repo bootstrap for lucena-tactics — the ONE place the path hack lives.

lucena-tactics is a private sibling repo in the superrepo tree (not a pip
package); its `src/` modules (poisoned_line_detector, drill, drill_feedback)
are imported top-level after appending that directory to sys.path. Append,
never prepend, so nothing in the backend's own environment can be shadowed.
`LUCENA_TACTICS_DIR` overrides the location (same convention as
`lucena_backend.plans._bootstrap()`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure() -> None:
    tactics_dir = Path(os.environ.get(
        "LUCENA_TACTICS_DIR",
        str(Path(__file__).resolve().parents[4] / "lucena-tactics")))
    src = str(tactics_dir / "src")
    if src not in sys.path:
        sys.path.append(src)
