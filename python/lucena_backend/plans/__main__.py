"""Print the plans fact sheet for one position — sheet only, no LLM.

    cd backend && PYTHONPATH=python .venv/bin/python -m lucena_backend.plans "<FEN>"

Rolls both legs exactly as the live path does (full main nodes; Maia if
LUCENA_MAIA is configured — serve.sh's auto-detect is replicated here) and
prints what PlansReadPrompt would be handed. Gate diagnostics go to stderr
so you can see WHY a position would or wouldn't route through the layer.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# serve.sh's Maia auto-detect, so a bare shell behaves like the served stack.
_ROOT = Path(__file__).resolve().parents[4]
if not os.environ.get("LUCENA_MAIA") and (_ROOT / ".venv-maia/bin/python").exists():
    os.environ["LUCENA_MAIA"] = (f"{_ROOT}/.venv-maia/bin/python "
                                 f"{_ROOT}/engine/scripts/maia_policy_uci.py")

from lucena_engine import openings                      # noqa: E402
from ..engine_io.enginepool import EnginePool           # noqa: E402
from .service import is_endgame, sheet_for, PLANS_CP_BAND  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    fen = sys.argv[1]

    book = openings.name_for(fen)
    err = sys.stderr
    print(f"gates: book={book or 'no'}  endgame={is_endgame(fen)}  "
          f"(live route also needs |eval| <= {PLANS_CP_BAND}cp and not drillable)", file=err)

    maia = None
    if os.environ.get("LUCENA_MAIA"):
        try:
            from lucena_engine.maia import MaiaEngine
            maia = MaiaEngine()
        except Exception as e:  # noqa: BLE001
            print(f"maia: unavailable ({e}) — engine leg only", file=err)
    else:
        print("maia: LUCENA_MAIA unset — engine leg only", file=err)

    pool = EnginePool(size=1, threads=1, hash_mb=128)
    t = time.time()
    sheet, pid = sheet_for(fen, pool, maia)
    print(f"rolled + built in {time.time() - t:.1f}s  ({pid})\n", file=err)
    print(sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
