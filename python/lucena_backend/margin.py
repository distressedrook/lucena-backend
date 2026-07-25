"""The MARGIN's content builder — authored content + the plans layer.

Staged by ply (owner 2026-07-24, "look inside /content, wire that up"):

  1. move 1 (plies 0-1) → the EPIGRAPH, from the fact-checked quote table.
  2. from move 2, in book → the THEORY card: masthead + the AUTHORED opening
     annotation + labeled doors.
  3. out of book → the plans layer's artifacts. ONE background worker rolls
     the position and emits the PRE-VERIFY JSON the moment it exists (no
     verify_plan calls yet) — shown while `plansPending` stays true; the
     verify gate then replaces it with POST-VERIFY and `plansPending` drops,
     ending the app's polling.

Stages 1-2 (the card builder) were removed in the 2026-07-23 JSON-inspection
pass and are restored here verbatim from backend fb4b2d7, now that
/content is wired through `lucena_core.content`. They are the INSTANT layer:
authored lookups + board geometry, engine-free, safe on every navigator
scrub, and carrying NO model output — the annotations are human-authored and
source-checked, not generated per turn.

Results are cached per position; the deep pass runs for the LIVE position
only (scrubs never trigger rolls — standing rule).
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from lucena_core import content as authored
from lucena_core import openings
from lucena_core import theory
from lucena_core.board import Board

_log = logging.getLogger(__name__)

IDEA_SENTENCES = 2    # authored annotations are essays; the card takes the lead

# -- plumbing (configured once by httpserver) ---------------------------------
_pool = None          # EnginePool — leased per job
_maia = None          # MaiaEngine | None (plans verify degrades without it)
_publish = None       # state.publish_margin_progress | None (loading stream)
_deep_cache: dict[str, dict] = {}      # norm fen -> {"raw", "statusLine", "pending"}
_inflight: set[str] = set()
_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)   # ONE: plans rolls are heavy


def configure(pool, maia, publish=None) -> None:
    """Called once at server build; without it the deep layer stays off and
    the margin serves nothing (tests, offline). `publish` streams pre-roll
    stages to the session's sockets (interactive loading); None = silent."""
    global _pool, _maia, _publish
    _pool, _maia, _publish = pool, maia, publish


def _cache(key: str, sheet: dict, status: str, pending: bool) -> None:
    with _lock:
        _deep_cache[key] = {"sheet": sheet, "raw": json.dumps(sheet, indent=2),
                            "statusLine": status, "pending": pending}
        if not pending:
            _inflight.discard(key)
        if len(_deep_cache) > 64:                 # bounded: a session's worth
            _deep_cache.pop(next(iter(_deep_cache)))


def _stream_preroll(fen: str, session_id: str) -> None:
    """The PRE phase, streamed (owner 2026-07-25): every feature the fast
    geometry detects goes out as a margin_progress event with its squares —
    the app cycles the highlights while the rolls grind. Failures are
    swallowed: the loading stream must never take the sheet down.

    Streams for whatever it is handed — the IN-BOOK / OUT-OF-BOOK gate lives
    upstream in build() (an in-book position returns the theory card before
    ever submitting a deep job), so by the time this runs the position is
    out of book (owner 2026-07-25: "it should stream when not in book, not
    on the phase of the game")."""
    if _publish is None or not session_id:
        return
    try:
        from .plans.service import _bootstrap
        _bootstrap()
        from preroll import features as _preroll_features
        stages = _preroll_features(fen)
        n = len(stages)
        for i, st in enumerate(stages):
            _publish({"fen": fen, "i": i, "n": n, **st}, session_id=session_id)
    except Exception:
        _log.warning("preroll stream failed for %s", fen, exc_info=True)


def _deep_job(fen: str, session_id: str = "") -> None:
    """The ROLL phase only (the PRE stream is fired upstream, from the request
    thread — see build). Roll once; cache POST when verify lands; end the
    loading stream. Every failure caches a terminal result so the app's
    polling terminates."""
    key = " ".join(fen.split()[:4])
    try:
        from .plans import service as _plans
        # No pipeline jargon in the status bar (owner: "remove the Rolling,
        # wtf is it") — the loading placeholder already says we're reading;
        # a finished sheet needs no "POST-VERIFY" header. None => no label.
        _pre, post = _plans.sheet_json_staged(
            fen, _pool, _maia,
            on_pre=lambda pre: _cache(key, pre, None, True))
        _cache(key, post, None, False)
        _publish_done(fen, session_id)
    except Exception:
        _log.warning("margin deep layer failed for %s", fen, exc_info=True)
        _cache(key, {"error": "sheet failed — see backend log"}, "ERROR", False)
        # the loading stream must END on failure too, or the app cycles
        # pre-roll highlights forever against a terminal error (Codex P1)
        _publish_done(fen, session_id)


def _publish_done(fen: str, session_id: str) -> None:
    """End the loading stream — guarded so a publisher hiccup can never
    convert a successful sheet into an error."""
    if _publish is None or not session_id:
        return
    try:
        _publish({"fen": fen, "stage": "done"}, session_id=session_id)
    except Exception:
        _log.warning("margin done-publish failed for %s", fen, exc_info=True)


# -- the INSTANT layer's card helpers (restored verbatim from backend
# fb4b2d7 — audited code, unchanged; only build()'s staging was rewired)

def _plies_played(fen: str) -> int:
    """Half-moves since the start, derived from the FEN's move counters."""
    parts = fen.split()
    fullmove = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
    white_to_move = len(parts) > 1 and parts[1] == "w"
    return (fullmove - 1) * 2 + (0 if white_to_move else 1)


def _lead_sentences(text: str, n: int) -> str:
    """The first n sentences of an authored annotation — the card's budget."""
    out, count, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1] in " \n"):
            out.append(text[start:i + 1].strip())
            count += 1
            start = i + 1
            if count >= n:
                break
    return " ".join(out) if out else text.strip()



def _blank(**over) -> dict:
    """Every key the mac model decodes, so no stage 404s a field."""
    out = {"masthead": None, "statusLine": None, "epigraph": None,
           "theory": None, "sheet": None, "raw": None, "plansPending": False}
    out.update(over)
    return out


def build(fen: str, *, seed: str = "", live: bool = False) -> dict:
    """MarginContent for `fen`. Three staged states (the owner's 2026-07-24
    staging, restored from backend fb4b2d7 when /content was wired up):

      1. move 1 (plies 0-1) -> the EPIGRAPH: a fact-checked authored quote,
         seeded by the SESSION so a game keeps its quote like a book keeps
         its epigraph. (The seed fallback is the DAY, never the fen — a fen
         seed changed the quote between ply 0 and ply 1, caught live.)
      2. from move 2, in book -> the THEORY card: the opening names itself
         (masthead), the AUTHORED annotation leads, and the labeled doors
         show where each next move goes.
      3. out of book -> the plans layer's sheet + raw JSON (the deep worker).

    Stages 1-2 are the INSTANT layer: authored table lookups and board
    geometry, engine-free and safe on every navigator scrub. They are also
    fully deterministic and carry no model output at all — the annotations
    are human-authored and source-checked (lucena_core.content), not
    generated per turn.
    """
    board = Board(fen)                    # validates; raises ValueError on garbage
    plies = _plies_played(fen)
    move_no = (plies // 2) + 1

    out: dict = {}

    # MOVE 0 (the starting position, ply 0 only) — the book opens: ONLY the
    # epigraph, nothing else (owner: "move 0, just the quote, centered" — and
    # "when I made a move it showed the quote again": the cover ends the
    # instant a move is played). An early return, so no theory/positional
    # competes with the quote. Seeded by SESSION so it never churns.
    if plies == 0:
        import datetime
        q = authored.epigraph(seed or datetime.date.today().isoformat())
        return _blank(epigraph={"quote": q["quote"], "author": q["author"],
                                "source": q.get("source")})

    # IN THEORY — the theory card. An AUTHORED annotation leads when we have
    # one (higher curation, our own prose); otherwise the verbatim Wikibooks
    # lead, shown as-is with its CC BY-SA attribution (never model-adapted).
    # Wikibooks is FEN-keyed, so it also names positions the authored table
    # skips — the card appears wherever either source knows the position.
    name = openings.name_for(fen)
    wb = theory.theory_for(fen)
    # "In theory" = a known opening NAME, or an ATTRIBUTABLE Wikibooks entry.
    # An entry without source_url can't be shown (CC BY-SA needs the credit),
    # so it is NOT theory — fall through to the positional read rather than
    # gate the sheet off behind an empty card.
    if name or (wb and wb.get("source_url")):
        idea = None
        attribution = None
        if wb and wb.get("source_url"):
            # WIKIBOOKS FIRST (owner: the wiki theory is the point — don't let
            # an authored annotation shadow it). VERBATIM: the harvest already
            # extracted only the lead paragraph, so it is shown as-is, never
            # truncated further. Always with its required CC BY-SA credit+link.
            idea = wb["description"]
            attribution = {"text": "Wikibooks · CC BY-SA",
                           "url": wb["source_url"]}
        elif name:
            # only where Wikibooks has nothing: fall back to our authored
            # annotation (lead sentences), our own prose so no attribution.
            a = authored.annotation_for(name)
            idea = _lead_sentences(a, IDEA_SENTENCES) if a else None
        out["masthead"] = name or (wb or {}).get("name")
        # continuations dropped (owner): the card is the theory prose only.
        out["theory"] = {"idea": idea, "doors": [], "attribution": attribution}
        # IN THEORY -> remove our positional stuff (owner: "if in theory,
        # remove our positional stuff"). The position is theoretical (named
        # or Wikibooks-covered); show the THEORY, never the positional read.
        # Skip the deep plans job entirely — no roll, no sheet, no badges.
        return _blank(statusLine=f"OPENING · MOVE {move_no}", **out)

    # out of book — the plans layer (unchanged; runs for every position that
    # is NOT in theory, so sheet/raw/plansPending behave as before).
    key = " ".join(fen.split()[:4])
    cached = _deep_cache.get(key)
    if cached is not None:
        return _blank(statusLine=cached["statusLine"], sheet=cached["sheet"],
                      raw=cached["raw"], plansPending=cached["pending"], **out)
    if live and _pool is not None:
        submitted = False
        with _lock:
            if key not in _inflight:
                _inflight.add(key)
                submitted = True
        if submitted:
            # PRE streams from the REQUEST thread, NOT the (serialized,
            # single-worker) roll queue — the fast geometry phase must never
            # wait behind the slow roll backlog (owner 2026-07-25: "works for
            # a couple of turns then stops" = the pre-roll was gated behind
            # the roll worker). Fires once per position, in fen order.
            # Stream BEFORE submitting the roll: an immediate/fast worker must
            # not publish `done` ahead of the PRE events (Codex P1) — so the
            # `done` the queued job emits always follows the whole PRE stream.
            _stream_preroll(fen, seed)
            try:
                _worker.submit(_deep_job, fen, seed)
            except Exception:
                with _lock:
                    _inflight.discard(key)   # let a later poll retry the roll
                _log.warning("margin roll submit failed for %s", fen,
                             exc_info=True)
        return _blank(plansPending=True, **out)      # no "ROLLING…" jargon
    return _blank(statusLine=out.get("masthead") and f"OPENING · MOVE {move_no}"
                  or f"MOVE {move_no}", **out)
