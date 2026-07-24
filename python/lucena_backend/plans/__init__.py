"""The PLANS layer — lucena-plans wired into the backend.

The (fen, pvs, rolls) contract: lucena-plans never rolls an engine or Maia;
it only checks lines. THIS package is where the backend produces those
lines (rolls.py, from its own pool + Maia) and asks lucena-plans for the
verified fact sheet (service.py). The conversation loop gates entry
(freeform._plans_read): quiet paste, out of book, middlegame, |eval| within
the equalish band.
"""

from .service import (sheet_for, sheet_json_for, render_position_read,  # noqa: F401
                       is_endgame, PLANS_CP_BAND)
