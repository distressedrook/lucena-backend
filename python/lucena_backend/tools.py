"""The Lucena tool implementations (LLD §4.2).

`ToolContext` holds the persistent Stockfish `Engine` and the `StateStore`, and
each method is one MCP tool. Methods return compact JSON-able dicts (the ≤300-
token response contract) and paint the board as a side effect of answering
(`board_push`). Errors are the deterministic `{error, detail}` shape, never
exceptions into the chat.

The transport (`server.py`) is thin glue over this; these methods are the
testable surface (LLD §9 "contract tests"). Determinism split: production uses
`movetime_ms`; tests construct the context with a `limit={"nodes": N}` override.
"""

from __future__ import annotations

import contextvars
import functools
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext as _nullcontext
from dataclasses import dataclass, field

from lucena_engine.board import Board
from lucena_engine.brilliant import is_brilliant
from lucena_engine.uci import EngineError
from lucena_engine.evalmodel import Glyph, Score, classify, win_pct_from_score
from lucena_engine.analysis import assemble_analysis
from lucena_engine.facts import build_fact_sheet
from lucena_engine.hints import derive_hints
from lucena_engine.line_tree import build_line_tree, count_leaves
from lucena_engine.positional import analyze_positional
from lucena_engine import poisoned_line_detector as _poisoned_line_detector
from . import puzzle_content
from . import response as R
from .enginepool import SingleEnginePool

# The Stockfish leased to the current guarded call, and that call's nesting depth. Both are
# ContextVars, not fields: a field would be shared by every chat running concurrently, which is the
# state the pool and the per-chat locks exist to eliminate.
#
# `_guard_depth` is a property of the CALL STACK, not of a chat — `_guarded` is sync and runs
# top-to-bottom in one thread. As a plain int it was correct only because ONE process-wide lock made
# it single-threaded; the moment locks became per-chat, two chats in `_guarded` would corrupt the
# counter and the error journal would double-log or drop silently. Nothing crashes — which is why it
# must not be split from the lock change.
_leased_engine: contextvars.ContextVar = contextvars.ContextVar("lucena_leased_engine", default=None)
_guard_depth: contextvars.ContextVar[int] = contextvars.ContextVar("lucena_guard_depth", default=0)

_POISONED_CACHE_CAP = 256      # unbounded, this grew for the life of the process


class _SyncCache(dict):
    """A dict that is safe to share across chats, and bounded.

    `find_poisoned_lines` takes the cache by reference and does its own get/set inside the search, so
    synchronisation has to live in the mapping itself rather than at the call site. The lock covers
    get/set ONLY — never a search — so two chats racing a cold position may both compute it. That is a
    duplicated search, not a wrong answer: detection is a pure function of the position.
    """

    def __init__(self, *, cap: int = _POISONED_CACHE_CAP):
        super().__init__()
        self._cap = cap
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)
            while len(self) > self._cap:
                super().pop(next(iter(self)))      # FIFO: the detector does not track recency

# Live poisoned-line-detection params — tuned for ~1-2s synchronous latency (deep combos still surface at
# plies=8; k=4 covers the top human moves; 120k nodes catches the big spikes). Full-enumeration
# offline mining can pass richer params directly to find_poisoned_lines.
_LIVE_POISONED_LINE = {"nodes": 120_000, "k": 4, "plies": 4}   # 4 moves deep is plenty for a live trap read
_POISONED_LINE_EXECUTOR = ThreadPoolExecutor(max_workers=2)   # runs detection parallel to the coach's eval

# class string per lichess glyph (shared vocabulary with gamepass)
_CLASS = {
    Glyph.OK: "ok", Glyph.ONLY_MOVE: "only_move", Glyph.DUBIOUS: "dubious",
    Glyph.MISTAKE: "mistake", Glyph.BLUNDER: "blunder",
}
# board arrow color by move class
_CLASS_STYLE = {
    "ok": "analysis", "only_move": "gold", "dubious": "correction",
    "mistake": "correction", "blunder": "correction", "brilliant": "gold",
}

# a coordinate move like e2e4 / e7e8q (to tell UCI from SAN in `advance`)
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

# Beats collapse to two acts: the coach `say`s (tells) or `ask`s (a probe that
# stops). teach/verdict/probe are kept as aliases so older callers still work.
_BEAT_ALIAS = {"say": "say", "ask": "ask", "you": "you",
               "teach": "say", "verdict": "say", "probe": "ask"}
_SAY_TONES = ("teach", "praise", "correct", "verdict")


_PIECE_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def _row_squares(fen_row: str) -> list[str | None]:
    """Expand one FEN rank ('r1b1') into 8 squares (piece char or None). Board-core truth."""
    out: list[str | None] = []
    for ch in fen_row:
        if ch.isdigit():
            out.extend([None] * int(ch))
        else:
            out.append(ch)
    return out


def _captured_piece(fen: str, uci: str) -> str | None:
    """Name the piece a move captures — 'bishop', 'pawn', … — from board-core truth, NOT the coach's
    guess (SAN like 'Qxf2' names the square, never the captured piece, so the model hallucinates it,
    usually 'pawn'). Handles en passant. Returns None when the move captures nothing."""
    try:
        rows = fen.split()[0].split("/")            # rows[0] = rank 8
        tgt_file = ord(uci[2]) - ord("a")           # 0..7
        tgt_rank = int(uci[3])                       # 1..8
        piece = _row_squares(rows[8 - tgt_rank])[tgt_file]
        if piece:
            return _PIECE_NAMES.get(piece.lower())
        # En passant: a pawn moving diagonally onto an empty square captures the passed pawn.
        src_file = ord(uci[0]) - ord("a")
        mover = _row_squares(rows[8 - int(uci[1])])[src_file]
        if mover and mover.lower() == "p" and src_file != tgt_file:
            return "pawn"
        return None
    except Exception:
        return None


def _you_beat(text: str, *, correct: bool | None = None,
              move: str | None = None, fen: str | None = None) -> dict:
    """A player-turn beat — the player's own words (or 'Played <move>' for a board move) shown as a
    right-aligned bubble, so the beats column reads as a conversation, not a coach monologue. On a
    DRILL move, `correct` marks the bubble with a verdict badge (green check / red cross) instead of a
    separate feedback beat; None (freeform / typed text) shows no badge. `move` (SAN) + `fen` (the
    position right after the move) make the move a clickable chip that snaps the board there."""
    beat: dict = {"kind": "you", "stops": False, "segments": [{"text": text}]}
    if correct is not None:
        beat["correct"] = bool(correct)
    if move:
        beat["move"] = move
    if fen:
        beat["fen"] = fen
    return beat

# Point-of-action reminder stamped on every "here's the position" result. A tool
# call cannot be structurally forced, so this is the strongest guard available:
# put the rule where the coach is actively reading (the result it just got), not
# only in a contract it read once at session start. The coach analyzing and then
# narrating in the terminal — where the player sees nothing — is a real failure
# mode; this makes "now push_beat" impossible to miss. (LLD §4.3, coach-contract
# "never coach in the terminal".)
_DELIVER = "coach this via push_beat — your terminal text is NOT shown to the player"


def _positional_block(board) -> dict:
    """Compact wire projection of the five-term positional read: every term's
    `cp` + plain `standing`, plus the citable `features` for the LEAD terms only
    (the 1-2 dimensions that actually characterise the position). Standings for
    all five give the coach the overview; features for the leads give it the
    specifics it will coach — and dropping non-lead features keeps the response
    inside the ~300-token budget."""
    p = analyze_positional(board)
    leads = set(p["leads"])
    terms = {}
    for name, t in p["terms"].items():
        entry = {"cp": t["cp"], "standing": t["standing"]}
        if name in leads and t.get("features"):
            entry["features"] = t["features"]
        terms[name] = entry
    return {"phase": p["phase"], "leads": p["leads"], "terms": terms}


def _guarded(method):
    """Catch-all so a *runtime* failure inside a tool never escapes as a raw
    traceback — the transport has no catch-all, and the deterministic
    `{error, detail}` contract is defined on this (testable) surface, not the
    plumbing. Validation paths still return their own specific codes *before*
    anything can raise; this only fires on the unexpected: an engine that
    crashes/times out (`EngineError` -> a recoverable `engine_unavailable`) or a
    genuine bug/IO error (`internal`). Both details tell the coach how to recover
    (retry, then restart the session) rather than dumping a stack trace."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Serialize tool entries PER CHAT across both surfaces (coach calls and the app's threaded
        # /move + /explore_poisoned_line endpoints). ToolContext/StateStore state — the drill walker,
        # the turn gate, seq counters, the history line — is single-threaded by design, so a coach
        # call racing an app move IN THE SAME CHAT would interleave read-modify-writes. Different
        # chats share none of that state, so they are not held against each other. RLock: nested
        # guarded calls (explore_poisoned_line → _explore_poisoned_line, play_move → assess_move)
        # re-enter fine.
        sid = self.store.current_sid if self.store is not None else ""
        with self._lock_for(sid):
            depth = _guard_depth.get()
            _guard_depth.set(depth + 1)
            # Lease an engine for the OUTERMOST guarded frame only, and hold it for the whole call:
            # callers mutate per-engine UCI options mid-call (explore_and_show sets Threads=1 and
            # restores it) and MultiPV is sticky, so the engine must not change under them. Nested
            # frames reuse the same lease — a nested lease would deadlock a pool of one and waste a
            # slot in a real pool.
            leased = _leased_engine.get()
            need_lease = leased is None and self._pool is not None
            lease_cm = self._pool.lease() if need_lease else _nullcontext(leased)
            try:
                with lease_cm as eng:
                    token = _leased_engine.set(eng) if need_lease else None
                    try:
                        result = method(self, *args, **kwargs)
                    finally:
                        if token is not None:
                            _leased_engine.reset(token)
            except EngineError as e:
                result = R.error(
                    "engine_unavailable",
                    f"Stockfish did not respond ({e}) — retry the call; if it keeps "
                    f"failing, restart the session to respawn the engine.",
                )
            except Exception as e:  # a bug or IO failure — still structured, still actionable
                result = R.error(
                    "internal",
                    f"unexpected failure in {method.__name__} ({type(e).__name__}: {e}) "
                    f"— retry; if it persists, restart the session.",
                )
            finally:
                _guard_depth.set(depth)
            # Local error journal: every error a tool hands back (validation, refusal, gate, crash)
            # is appended to <home>/errors.log with a timestamp — errors are otherwise only visible
            # inside the coach's context, which makes field debugging guesswork. Outermost guarded
            # frame only (a nested guarded call would double-log the same error). Best-effort.
            if depth == 0 and isinstance(result, dict) and "error" in result:
                self._log_error(method.__name__, result)
            return result

    return wrapper


# ---------------------------------------------------------------------------
# Request classification & the tool-scope gate (contract: docs/contracts/M-classification.md)
# ---------------------------------------------------------------------------
# The scoped tools (subject to per-class refusal); everything else is always
# allowed. `get_mastery` is always-allowed but is SESSION's grounding, so it is
# `_scoped`-decorated too — for the grounding hook only; it is never refused.
_SCOPED = frozenset({
    "analyze_and_show", "evaluate", "evaluate_and_show", "get_hints",
    "build_and_arm_drill", "get_game_analysis", "explore_and_show", "explore_poisoned_line",
})

# class -> {must_ground: <tool | None>, refused: <scoped tools not allowed here>}
_CLASS_TABLE = {
    # `explore_and_show`/`explore_poisoned_line` are refused inside a LIVE drill (revealing/walking a line
    # would hand over the answer or fight the active drill) but allowed on discussion turns.
    # `explore_poisoned_line` is ALSO allowed on DRILL_SOLVED (unlike explore_and_show) — that's its whole
    # point: the post-solve "here's the trap you dodged" payoff, explored and narrated once the drill is done.
    "DRILL_WRONG":  {"must_ground": "evaluate",
                     "refused": frozenset({"build_and_arm_drill", "explore_and_show", "explore_poisoned_line"})},
    "DRILL_EVENT":  {"must_ground": "analyze_and_show",
                     "refused": frozenset({"build_and_arm_drill", "explore_and_show", "explore_poisoned_line"})},
    "DRILL_SOLVED": {"must_ground": None,
                     "refused": frozenset({"build_and_arm_drill", "explore_and_show"})},
    # Solved, but the player has navigated INTO the poisoned line they dodged. read_input hands over the
    # whole trap (moves + Maia's motif), so no must_ground — the line is already grounded; narrate it.
    "DRILL_POISONED_LINE": {"must_ground": None,
                            "refused": frozenset({"build_and_arm_drill", "explore_and_show"})},
    "SESSION":      {"must_ground": "get_mastery",
                     "refused": frozenset({"build_and_arm_drill", "explore_and_show", "explore_poisoned_line"})},
    "GAME_REVIEW":  {"must_ground": "get_game_analysis",
                     "refused": frozenset({"build_and_arm_drill"})},
    "MOVE_EXPLORE": {"must_ground": "evaluate",
                     "refused": frozenset({"build_and_arm_drill"})},
    "PROBE_ANSWER": {"must_ground": None,
                     "refused": frozenset({"build_and_arm_drill"})},
    "OPEN":         {"must_ground": "analyze_and_show",
                     "refused": frozenset()},
}

# The standard chess starting position — the board the app's "back to the previous concept" resets to
# when the activity stack is empty (nothing to pop back to).
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Local intro beats for set_puzzle — the puzzle's "here it is, take a look" prompt is server-authored
# (never an LLM push_beat). Rotated deterministically so a second puzzle doesn't read identically.
_PUZZLE_INTROS = [
    "Here's a puzzle — take a look at the board. What do you think is the best move?",
    "New position on the board. Study it — what's the winning idea here?",
    "Try this one. What would you play?",
    "Here's another. Take your time, then make your move on the board.",
]

# The class -> lane-skill map (coach-skills design §5). read_input returns `load_skill` so the coach
# loads the right lane's choreography without guessing from descriptions. Only the mutually-exclusive
# LANES route here; cross-cutting craft (Socratic, mastery, activities, sessions) lives in the always-on
# core, so those classes carry no skill. `build_and_arm_drill` routes the drills-vs-coaching fork itself
# (its `drillable` verdict, which a turn class can't see). The map lives with the gate on purpose: skill
# routing is turn-scope (this transient state machine), not durable session-document content.
_CLASS_SKILL = {
    "DRILL_WRONG":  "drills",
    "DRILL_EVENT":  "drills",
    "DRILL_SOLVED": "drills",
    "DRILL_POISONED_LINE": "drills",
    "GAME_REVIEW":  "game-review",
    "MOVE_EXPLORE": "coaching-a-position",
    "OPEN":         "coaching-a-position",
    # SESSION / PROBE_ANSWER -> no lane skill (the core spine handles them).
}

_OOC_HINT = {
    "build_and_arm_drill": "you're not entering a fresh tactic here",
    "explore_and_show": "a live drill is on — walking the line would hand over the answer; "
                        "coach the pending event instead",
    "explore_poisoned_line": "a live drill is on — reveal the trap only after it's solved (DRILL_SOLVED)",
}


def _classify(inp: dict, was_awaiting: bool) -> str:
    """Deterministic turn class from the app's input `kind` + whether a probe was
    pending. Priority: drill_wrong -> drill -> start -> game_id -> move ->
    none(probe|open) -> OPEN (catch-all for unknown/missing/non-string kind)."""
    kind = inp.get("kind")
    if kind == "drill_wrong":
        return "DRILL_WRONG"
    if kind == "drill_solved":
        return "DRILL_SOLVED"
    if kind == "drill":
        return "DRILL_EVENT"
    if kind == "start":
        return "SESSION"
    if inp.get("game_id") or kind == "game":
        return "GAME_REVIEW"
    if kind == "move":
        return "MOVE_EXPLORE"
    if kind in (None, "none") or not isinstance(kind, str):
        return "PROBE_ANSWER" if was_awaiting else "OPEN"
    return "OPEN"


def _ooc_detail(name: str, cls: str) -> str:
    return f"{name} isn't available in {cls} — {_OOC_HINT.get(name, 'use a tool this class allows')}."


def _scoped(name: str):
    """Attach the classification gate to a tool. For a scoped tool: the awaiting
    gate fires first (precedence #1), then out-of-class refusal (#2) — but refusal
    and the grounding hook are **inert until the first `read_input`** classifies a
    turn (a session with no turn is unclassified/permissive). A successful call to a
    class's `must_ground` tool grounds the turn."""
    def deco(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if name in _SCOPED:
                if (g := self._gate()) is not None:              # #1 awaiting
                    return g
                if self._classified and name in _CLASS_TABLE[self._class]["refused"] \
                        and not self._refusal_bypassed(name, args, kwargs):
                    return R.error("out_of_class", _ooc_detail(name, self._class))  # #2
            result = method(self, *args, **kwargs)
            if self._classified and not (isinstance(result, dict) and "error" in result) \
               and name == _CLASS_TABLE[self._class]["must_ground"]:
                self._grounded = True
            return result
        return wrapper
    return deco


def _pgn_line(moves: list[dict]) -> str:
    """Render a run of view moves ({san, fen}) as PGN-numbered text — e.g. "17.Rxd6 cxd6 18.Bxd6+".
    The number + side come from each move's resulting fen (fullmove ticks after Black; the mover is
    the side NOT to move). Black-led runs and mid-run Black moves read "17…" / bare figure."""
    out: list[str] = []
    for i, m in enumerate(moves):
        san = m.get("san")
        if not san:
            continue
        parts = str(m.get("fen", "")).split()
        full = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
        white_moved = (parts[1] if len(parts) > 1 else "w") == "b"   # side-to-move is the NON-mover
        num = full if white_moved else full - 1
        if white_moved:
            out.append(f"{num}.{san}")
        elif i == 0:
            out.append(f"{num}...{san}")
        else:
            out.append(san)
    return " ".join(out)


def _branch_label(branch: dict) -> str:
    """A short "where this branches from" hint for a variation, from its mainline ply if known."""
    ply = branch.get("at_ply")
    return f"at ply {ply}" if isinstance(ply, int) else "off-line"


def _norm_fen(fen: str) -> str:
    """A position's identity ignoring clocks — placement + side + castling + ep (first 4 FEN fields).
    Lets "is the board at this move's position?" ignore halfmove/fullmove drift."""
    return " ".join(str(fen).split()[:4])


def _num(fen: str) -> tuple[int, bool]:
    """(move number, white_moved) for a move whose RESULTING position is `fen` — the mover is the
    side NOT to move; White's number is the fullmove, Black's is one less."""
    parts = str(fen).split()
    full = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
    white_moved = (parts[1] if len(parts) > 1 else "w") == "b"
    return (full if white_moved else full - 1), white_moved


def _viewing_line(items: list[dict], cursor: int) -> str:
    """The resolved line as a one-line PGN for the coach — every move numbered, the variation entry
    flagged with `→`, and the cursor move wrapped in brackets so the coach sees exactly where the
    player is. Skips the root "…" block."""
    out: list[str] = []
    for idx, m in enumerate(items):
        san = m.get("san")
        if not san:
            continue
        num, white_moved = _num(m.get("fen", ""))
        tok = f"{num}.{san}" if white_moved else f"{num}...{san}"
        if m.get("branch"):
            tok = "→ " + tok                 # the move the current line branched into
        if idx == cursor:
            tok = f"[{tok}]"                  # where the player is looking right now
        out.append(tok)
    return " ".join(out)


@dataclass
class _SessCtx:
    """One session's turn/drill context (state-machine P2). Held per Claude session id so switching
    sessions can't leak or desync the gate, the drill walker, or the analysed fact sheet — the
    "two partition scopes" fragility (§1.2). These are MCP-PRIVATE (never leave the server, §9); the
    durable Socratic gate lives in the document (a later P2 step). Defaults match the pre-P2 flat
    inits exactly, so behaviour is unchanged — only the *scope* (per-session, not per-process) is."""
    # last analysed fact sheet, so push_beat can resolve a cited [F#] -> its arrow (LLD §4.3)
    facts: dict = field(default_factory=dict)
    last_fen: str | None = None
    # (The Socratic gate `awaiting_input` is DURABLE — it lives in the session document, not here;
    #  see StateStore.set_gate / ToolContext._awaiting_input.)
    # Classification gate (M-classification): the turn's class, whether grounding is earned, whether
    # a turn has been classified. Inert until the first read_input; `grounded` starts True.
    cls: str = "OPEN"
    grounded: bool = True
    classified: bool = False
    # Same-turn read_input idempotence: True after the coach speaks; a re-read before speaking is a
    # no-op replay (can't reset grounding mid-turn).
    spoke_since_read: bool = True
    last_read_out: dict = field(default_factory=lambda: {"kind": "none", "classification": "OPEN"})
    # Server-side drill walk: the MCP owns the tree and adjudicates the app's moves.
    drill: object | None = None
    # Per-session caches/counters (were flat instance attrs on ToolContext, so they bled across a
    # session switch — the "flat side object" scope P2 was built to eliminate). Now keyed per session.
    maia_cache_key: object | None = None
    maia_cache_val: list = field(default_factory=list)
    ptxt_i: int = 0        # rotates the player-facing mistake phrasing
    pfind_i: int = 0       # rotates the player-facing "find" phrasing


class ToolContext:
    def __init__(self, engine=None, store=None, *, limit: dict | None = None, mastery=None,
                 maia=None, player_rating: int = 1500, pool=None):
        # An engine is LEASED from the pool for the duration of each guarded tool call and exposed
        # through `self.engine` (a property over a ContextVar), so all the call sites below read the
        # engine THIS call owns. `ToolContext(engine, store)` still works — a caller-supplied engine
        # becomes a pool of one.
        self._pool = pool or (SingleEnginePool(engine) if engine is not None else None)
        self.store = store
        self.mastery = mastery  # a MasteryEngine, or None (analysis-only sessions)
        self.maia = maia        # a MaiaEngine, or None — the human-move predictor (never truth)
        self.player_rating = int(player_rating)  # currentPlayerRating: whom Maia predicts for
        self._limit = limit  # tests: {"nodes": N}; prod: None -> use movetime arg
        # Bounded position-analysis cache (LRU). Analysis is a pure function of (fen, limit) — Stockfish
        # is stateless per position and version-pinned — so the coach hitting the SAME position across
        # analyze_and_show / evaluate / get_hints in one turn reuses one search instead of 2-3. Also
        # makes the eval the coach reads consistent across those tools (no movetime jitter per call).
        # SHARED across chats on purpose (a position's analysis does not depend on who asked), which is
        # a real saving — but that makes it concurrently accessed, hence its own mutex: an unsynchronised
        # OrderedDict with LRU eviction corrupts or raises under threads.
        self._analysis_cache: OrderedDict = OrderedDict()
        self._analysis_cache_lock = threading.Lock()
        # One reentrant lock PER CHAT, not one for the process: the state it protects (the drill walker,
        # the turn gate, seq counters, the history line) is per-chat, so a single lock made every user's
        # turn queue behind every other user's. Chats share no state, so they need no mutual exclusion.
        self._tool_locks: dict = {}
        self._tool_locks_meta = threading.Lock()   # guards first-touch of the dict itself
        # The turn/drill context, PARTITIONED per session id (P2) — same pattern as StateStore._live.
        # `_facts`/`_last_fen`/`_awaiting_input`/`_class`/… below are proxy properties onto the current
        # session's _SessCtx, so every call site stays unchanged.
        self._sctx: dict = {}
        # Poisoned-line detection: a DEDICATED single-threaded engine so a detect runs parallel to the coach's
        # main-engine eval without contention. Spawned LAZILY via the factory serve_http sets — on first
        # poisoned-line use, NOT at launch, so it never adds to the already-slow startup. No factory (stdio/
        # tests) → detect runs serially on the main engine. `_poisoned_line_cache` memoizes detect by position.
        self.poisoned_line_engine = None
        self._poisoned_line_engine_factory = None
        # Serializes the WHOLE detector call on the shared dedicated engine. `find_poisoned_lines` is
        # a multi-call algorithm (new_game/analyse, repeatedly): Engine's own lock makes each command
        # atomic, but not the sequence, so two detections on one engine interleave into each other's
        # searches. One process-wide tool lock used to prevent that as a side effect; per-chat locks
        # do not, so the engine needs its own.
        self._poisoned_line_engine_lock = threading.RLock()
        # Passed BY REFERENCE into find_poisoned_lines, which reads and writes it inside the search —
        # so it cannot be locked at the call boundary. A guard on get/set only: never across a search,
        # which would serialize every chat on the cache instead of on the engine.
        self._poisoned_line_cache = _SyncCache(cap=_POISONED_CACHE_CAP)

    # -- the leased engine -------------------------------------------------
    @property
    def engine(self):
        """The Stockfish leased to THIS call.

        A property over a ContextVar rather than a field, so every existing `self.engine` call site
        reads the engine its own guarded call owns, with no rewrite. A field would be one shared
        engine again — the thing the pool exists to stop.
        """
        eng = _leased_engine.get()
        if eng is not None:
            return eng
        if self._pool is None:
            return None     # ToolContext(None, store): there is genuinely no engine. Callers test
                            # `self.engine is None` to skip engine work — keep that answerable.
        # Outside a guarded call there is no lease. The pool-of-one case can still answer (the caller
        # handed us that engine and owns it); a real pool cannot, and saying so beats handing back an
        # arbitrary engine nobody has checked out.
        if isinstance(self._pool, SingleEnginePool):
            return self._pool._engine
        raise RuntimeError(
            "no engine is leased in this context: reach the engine from inside a @_guarded tool "
            "call, or take a lease explicitly with `with ctx._pool.lease() as eng:`"
        )

    def _lock_for(self, sid: str):
        with self._tool_locks_meta:
            lock = self._tool_locks.get(sid)
            if lock is None:
                lock = self._tool_locks[sid] = threading.RLock()
            return lock

    # -- per-session turn/drill context (P2): proxies onto the current session's _SessCtx --
    @property
    def _sc(self) -> _SessCtx:
        """The current session's turn/drill context, keyed by `store._current` (created on demand).
        The empty-string key is the pre-session bucket, mirroring StateStore's `_cur`."""
        sid = self.store._current
        if sid not in self._sctx:
            # First touch of this session in THIS process — e.g. after a session SWITCH, or a session
            # whose drill was armed in a prior process. The drill walker is a pure derivation of the
            # session DOCUMENT, so rebuild it from the persisted drill_state + tree now. Without this a
            # fresh _SessCtx carries drill=None, and play_move silently treats every drill move as
            # freeform — killing the live drill while its tree is still on screen (audit item 3). Insert
            # the ctx FIRST so the write inside _rehydrate_drill resolves to it (no re-entrant recursion).
            self._sctx[sid] = _SessCtx()
            self._rehydrate_drill()
        return self._sctx[sid]

    @property
    def _facts(self): return self._sc.facts
    @_facts.setter
    def _facts(self, v): self._sc.facts = v

    @property
    def _last_fen(self): return self._sc.last_fen
    @_last_fen.setter
    def _last_fen(self, v): self._sc.last_fen = v

    # The Socratic gate is DURABLE (P2b): it lives in the session document, so a restart mid-probe
    # resumes locked. Reads/writes proxy to the store (which persists on change).
    @property
    def _awaiting_input(self): return self.store._gate_awaiting
    @_awaiting_input.setter
    def _awaiting_input(self, v): self.store.set_gate(bool(v))

    @property
    def _class(self): return self._sc.cls
    @_class.setter
    def _class(self, v): self._sc.cls = v

    @property
    def _grounded(self): return self._sc.grounded
    @_grounded.setter
    def _grounded(self, v): self._sc.grounded = v

    @property
    def _classified(self): return self._sc.classified
    @_classified.setter
    def _classified(self, v): self._sc.classified = v

    @property
    def _spoke_since_read(self): return self._sc.spoke_since_read
    @_spoke_since_read.setter
    def _spoke_since_read(self, v): self._sc.spoke_since_read = v

    @property
    def _last_read_out(self): return self._sc.last_read_out
    @_last_read_out.setter
    def _last_read_out(self, v): self._sc.last_read_out = v

    @property
    def _drill(self): return self._sc.drill
    @_drill.setter
    def _drill(self, v): self._sc.drill = v

    def _log_error(self, tool: str, result: dict) -> None:
        """Append one timestamped line per tool error to <home>/errors.log — the local journal
        that makes tool failures visible outside the coach's context (where they're otherwise
        invisible for field debugging). Best-effort: logging must never break the tool path."""
        try:
            import datetime
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            detail = str(result.get("detail", ""))[:300].replace("\n", " ")
            with open(os.path.join(self.store.home, "errors.log"), "a", encoding="utf-8") as fh:
                fh.write(f"{ts}\t{tool}\t{result['error']}\t{detail}\n")
        except Exception:
            pass

    def _poisoned_line_or_false(self, fen: str, engine, *, stop_on_first: bool):
        """Run poisoned-line detection, converting ANY failure into `False` — but LOUDLY, journaled
        to errors.log. Detection is best-effort (never break the coach), yet a blanket silent
        `except` once masked a real misconfiguration: a Maia wrapper that doesn't emit `policy`
        makes `find_poisoned_lines` raise on every call, which silently zeroed `has_poisoned_line`
        for EVERY position — indistinguishable from "no trap here", with no trace. The detector's
        `_policy` guard is deliberately fail-loud; swallowing it in silence defeats that guard, so
        we log instead of muting. Returns the result dict, or None on failure."""
        if self.maia is None:
            return None
        try:
            with self._shared_engine_guard(engine):
                return _poisoned_line_detector.find_poisoned_lines(
                    fen, engine, self.maia, rating=self.player_rating,
                    stop_on_first=stop_on_first, cache=self._poisoned_line_cache,
                    **_LIVE_POISONED_LINE)
        except Exception as e:
            self._log_error("poisoned_line", {"error": "detection_failed", "detail": f"{type(e).__name__}: {e}"})
            return None

    @contextmanager
    def _shared_engine_guard(self, engine):
        """Hold the poisoned-line engine's lock for this block IFF `engine` is that shared engine.

        EVERY multi-call engine sequence that can land on the dedicated engine must go through here —
        the detector, and the concrete-line reconstruction in `_poisoned_line_moves` (a new_game +
        analyse loop). Guarding only one of them leaves the other free to interleave on the same
        process, which is the whole failure this lock exists to stop.

        A LEASED engine takes no lock: the lease is already exclusive to this call, and locking again
        would serialize chats that share nothing.
        """
        if engine is not None and engine is self.poisoned_line_engine:
            with self._poisoned_line_engine_lock:
                yield
        else:
            yield

    def _get_poisoned_line_engine(self):
        """The dedicated poisoned-line engine, spawned on first use (kept off the launch critical path).
        Returns None when no factory was set (stdio/tests) → callers fall back to the main engine."""
        # Under the meta lock: a bare check-then-set spawns two engines when two chats first need a
        # detect at once, and leaks one of them forever (nothing else holds a reference).
        with self._tool_locks_meta:
            if self.poisoned_line_engine is None and self._poisoned_line_engine_factory is not None:
                self.poisoned_line_engine = self._poisoned_line_engine_factory()
            return self.poisoned_line_engine

    def _lim(self, movetime_ms: int) -> dict:
        return self._limit or {"movetime_ms": movetime_ms}

    # Fact-sheet VERIFICATION probes (null-move threat, combination, hang/fork reconciliation) only
    # need a quick tactical check, not the deep primary read — so they get a SHORT movetime instead of
    # inheriting the main 1500ms. `build_fact_sheet` fans several of these out (one per opponent reply
    # in reconciliation), so at the full movetime they dominated tool latency (a single grounding read
    # was 4-6 sequential 1500ms searches ≈ 6-9s). Tests/eval-harness keep the reproducible `nodes`
    # limit unchanged (so fact CONTENT is identical there) — only production wall-time drops.
    _PROBE_MOVETIME_MS = 300

    def _probe_lim(self) -> dict:
        return self._limit or {"movetime_ms": self._PROBE_MOVETIME_MS}

    _ANALYSIS_CACHE_CAP = 256

    def _cached_analyse(self, fen: str, *, multipv: int = 1, **lim):
        """Analyse `fen` through the bounded position cache. The FIRST result for a (fen, limit) is
        reused for later calls — the coach hits the same position across analyze/evaluate/get_hints in
        one turn, so this collapses 2-3 full searches into one. A cached result at multipv M serves any
        request for ≤ M lines (extra lines ignored); a deeper-multipv request re-searches and upgrades
        the entry. Keyed by (fen, limit), so a deep read and a short probe stay separate — they are
        genuinely different depths. Safe because analysis is a pure function of the position (stateless
        per fen, engine version-pinned); in nodes-mode (tests) the cached value is bit-identical."""
        key = (fen, tuple(sorted(lim.items())))
        with self._analysis_cache_lock:
            hit = self._analysis_cache.get(key)
            if hit is not None and hit[0] >= multipv:
                self._analysis_cache.move_to_end(key)
                return hit[1]
        # Searched OUTSIDE the cache lock: this is the multi-second part, and holding a mutex across
        # it would serialize every chat on the cache instead of on the engine — reintroducing the
        # bottleneck the pool removes. Two chats racing the same cold position may both search; that
        # is a duplicated search, not a wrong answer (analysis is a pure function of the position),
        # and the loser simply overwrites with an identical value.
        # `new_game()` is not called here — the lease already does it on acquire.
        res = self.engine.analyse(fen, multipv=multipv, **lim)
        with self._analysis_cache_lock:
            self._analysis_cache[key] = (multipv, res)
            self._analysis_cache.move_to_end(key)
            while len(self._analysis_cache) > self._ANALYSIS_CACHE_CAP:
                self._analysis_cache.popitem(last=False)
        return res

    def _maia_top(self, fen: str, n: int = 5) -> list[dict]:
        """Maia's top-`n` likely human moves at the current rating, cached per
        (fen, rating) so one grounding call reuses the prediction across the tendency
        line and the played-move read (no double subprocess hit)."""
        if self.maia is None:
            return []
        key = (fen, self.player_rating, n)
        sc = self._sc
        if sc.maia_cache_key == key:
            return sc.maia_cache_val
        try:
            val = self.maia.top_human_moves(fen, self.player_rating, n=n)
        except Exception:
            val = []
        sc.maia_cache_key, sc.maia_cache_val = key, val
        return val

    def _maia_block(self, fen: str) -> str | None:
        """A **deterministic plain-text** read of what a player at `currentPlayerRating`
        is likely to play here — computed by us, so the coach reads a sentence, never
        raw ranks/probabilities. Pairs Maia's *behaviour* prediction with the engine's
        *truth* (the best move): what humans reach for, and whether the popular pick is
        actually right. Maia predicts; Stockfish grounds — the sentence never states an
        eval, only a move name. Absent when Maia isn't configured; never breaks a tool."""
        if self.maia is None:
            return None
        try:
            board = Board(fen)
            sans: list[str] = []
            for m in self._maia_top(fen)[:4]:
                try:
                    sans.append(board.san(m["uci"]))
                except Exception:
                    continue
            if not sans:
                return None
            best = board.san(self._cached_analyse(fen, multipv=1, **self._lim(250)).best.pv[0])
            listed = ", ".join(sans)
            if sans[0] == best:
                tail = f"their first instinct ({best}) is also the best move."
            elif best in sans:
                tail = f"the best move ({best}) is among them but not their first pick ({sans[0]})."
            else:
                tail = f"the best move ({best}) is NOT a popular pick — most reach for {sans[0]}."
            return (f"A player rated {self.player_rating} most likely plays {listed} here "
                    f"(likeliest first); {tail}")
        except Exception:
            return None

    def _add_maia(self, resp: dict, fen: str) -> None:
        """Stamp the plain-text player-tendency read onto a grounding response (after
        the budget, so it never costs a fact — same rule as `deliver`). The wire key is
        neutral (`player_tendency`): the coach reads a sentence and never learns there's
        a model called Maia behind it."""
        mb = self._maia_block(fen)
        if mb is not None:
            resp["player_tendency"] = mb

    def _played_move_read(self, fen: str, played_uci: str, cls: str, *, is_best: bool) -> str | None:
        """Deterministic plain-text calibration of the move the player just PLAYED: how
        *typical* is it at their level? A `mistake`/`blunder` that's a top-3 human pick is
        a **level-typical trap** — teach the pattern, kindly, not as a random slip; an
        error nobody at the level plays is an individual slip; the best move most players
        would miss is a genuine find. Stockfish grounds `cls`; the human prediction grounds
        how common. Source hidden, no eval numbers. `None` when there's nothing to say."""
        if self.maia is None:
            return None
        top = self._maia_top(fen)
        rank = next((m["rank"] for m in top if m["uci"] == played_uci), None)
        R = self.player_rating
        if cls in ("mistake", "blunder"):
            if rank is not None and rank <= 3:
                return (f"This is a COMMON {cls} at {R}: the #{rank} move players this level "
                        f"reach for here — a level-typical trap, not a random slip. Teach the "
                        f"pattern, and be kind: it's the natural mistake to make.")
            return (f"This {cls} is uncommon at {R} — most players this level don't play it; "
                    f"treat it as an individual slip.")
        if is_best and rank is None:
            return (f"A strong find: most players at {R} would miss this — it isn't among "
                    f"their likely moves here. Worth real praise.")
        return None

    # PLAYER-facing text (addressed to the player, no coach instructions) — this is what
    # the app shows. Distinct from `move_read`, which is the coach's private notes.
    _MISTAKE_PLAYER = (
        "Here's the thing — that's the exact move most players at your level play here. "
        "It's a natural trap, not a careless slip. Worth having a conversation with the "
        "coach about why.",
        "You're in good company: most players your rating reach for that move here. It "
        "just doesn't work — a classic trap at this level. I'd recommend talking it "
        "through with the coach.",
        "That's the most common choice at your level in this position. Tempting, but it's "
        "a trap — a good one to discuss with the coach.",
    )
    _FIND_PLAYER = (
        "Strong move — most players at your rating would miss that. Well spotted! Ask the "
        "coach what makes it work, so it sticks.",
        "Nice one — that's a find beyond your level; most players your rating don't see it. "
        "Worth a chat with the coach on why it's so strong.",
        "That's a higher-level move than your rating usually finds. Genuinely well played — "
        "the coach can show you the idea behind it.",
    )

    def _played_move_player_text(self, fen: str, played_uci: str, cls: str, *, is_best: bool) -> str | None:
        """The PLAYER-facing meaning of the move they just made (or `None` when there's
        nothing notable). A level-typical mistake → "you're in good company, it's a trap";
        a move beyond their level → real praise. Cycled; addressed to the player."""
        if self.maia is None:
            return None
        rank = next((m["rank"] for m in self._maia_top(fen) if m["uci"] == played_uci), None)
        if cls in ("mistake", "blunder") and rank is not None and rank <= 3:
            i = self._sc.ptxt_i; self._sc.ptxt_i = i + 1
            return self._MISTAKE_PLAYER[i % len(self._MISTAKE_PLAYER)]
        if is_best and rank is None:
            i = self._sc.pfind_i; self._sc.pfind_i = i + 1
            return self._FIND_PLAYER[i % len(self._FIND_PLAYER)]
        return None


    def _gate(self) -> dict | None:
        """Return an error if the flow is stopped at a probe, else None."""
        if self._awaiting_input:
            # The correct recovery here is INACTION — end the turn and wait. So name NO tool: an
            # agentic model acts on whatever verb the message hands it (even a forbidden one), and the
            # old "call read_input first" wording made the coach do exactly that, unlocking its own probe.
            return R.error(
                "awaiting_input",
                "STOP — you already asked your question, and a probe ENDS your turn. End your "
                "response now and wait for the player; take no other action this turn.",
            )
        return None

    def _poisoned_line_block(self, nres) -> dict | None:
        """The coach's compact poisoned-line read, stamped on an `analyze_and_show(poisoned_line=True)` response
        (after the budget, so it never costs a fact). WARN, don't spoil: `note` reminds the coach to
        tell the player to calculate carefully WITHOUT naming the trap during play."""
        if not nres or not nres.get("has_poisoned_line"):
            return None
        return {
            "has_poisoned_line": True,
            "temptations": [
                {"fatal": t["fatal"], "idea": t["idea"], "seeds": t["seeds"], "deep": t["deep"]}
                for t in nres["temptations"][:3]
            ],
            "note": "This position holds a poisoned line a player at this level would fall for. WARN "
                    "them to calculate carefully — do NOT name the tempting move or its refutation "
                    "during play (that defeats the point). Reveal it only after the position is resolved.",
        }

    # -- analyze_and_show (MIXED: compute the read AND paint the board) ----
    @_guarded
    @_scoped("analyze_and_show")
    def analyze_and_show(self, fen, *, multipv=2, movetime_ms=1500,
                         focus=None, board_push=True, poisoned_line=False) -> dict:
        """MIXED — analyses the position (verb) AND paints the board with the grounded
        fact arrows (mutation); the name says both. Piece list, eval, top lines, fact
        sheet, and a board repaint (no separate set_board). `focus`: None (full) | "eval"
        (skip facts) | "threats" | "positional" (five-term strategic read) | "analysis"
        (the grounded briefing you READ before coaching). `poisoned_line=True` when the
        player is about to choose a move: adds human-trap detection in parallel."""
        if (g := self._gate()) is not None:
            return g
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        if not board.legal_moves():
            return R.error("terminal_position", "no legal moves (checkmate/stalemate)")

        # Repainting to a position that isn't the active drill's = the coach set up something new.
        # Retire the stale walker + navigator NOW (before the board flips), so the strip never shows
        # the previous game and the next move is adjudicated freeform, not against a dead drill.
        if board_push and self._drill_diverged(fen):
            self._reset_drill_line(fen)

        # Kick off poisoned-line detection in PARALLEL with the eval below — but ONLY with a dedicated engine
        # (sharing self.engine across threads would interleave its analyse calls). Without one we run
        # it serially after the eval (see below). Both overlap the ~1.5s eval, so lag is ~max not sum.
        poisoned_line_fut = None
        neng = self._get_poisoned_line_engine() if (poisoned_line and self.maia is not None) else None
        if neng is not None:
            poisoned_line_fut = _POISONED_LINE_EXECUTOR.submit(
                self._poisoned_line_or_false, fen, neng, stop_on_first=True)

        lim = self._lim(movetime_ms)
        analysis = self._cached_analyse(fen, multipv=multipv, **lim)
        facts = [] if focus in ("eval", "positional") else \
            build_fact_sheet(board, self.engine, top_n=5, **self._probe_lim())
        if focus == "threats":
            facts = [f for f in facts if f.kind in ("threat", "hanging")]
        # cache for board painting (arrows auto-derive from the computed facts —
        # the coach never cites an id; see the analysis/beats design).
        self._facts = {f.id: f for f in facts}
        self._last_fen = fen

        # Resolve poisoned-line detection: the parallel result (overlapped with the eval), or a serial run now.
        nres = None
        if poisoned_line_fut is not None:
            nres = poisoned_line_fut.result()   # the helper already logs+swallows; never re-raises here
        elif poisoned_line and self.maia is not None:
            nres = self._poisoned_line_or_false(fen, self.engine, stop_on_first=True)
        has_poisoned_line = bool(nres and nres["has_poisoned_line"])
        # Load the ENTIRE poisoned line (the full trap move sequence) so the app can hold it and,
        # on reveal, show it as a variation. Only walked when a trap is actually present.
        poisoned_line_moves = None
        if has_poisoned_line and nres["temptations"]:
            poisoned_line_moves = self._poisoned_line_moves(
                fen, nres["temptations"][0], neng or self.engine)
        # Store the freeform trap DURABLY (§6.4) — on the document, keyed to this fen — never on the
        # transient board. The board's poisoned fields are a projection of this slot (see write_board),
        # so the trap survives repaints instead of evaporating. Only touch the slot when a poisoned
        # check was actually requested this call; a no-trap result clears only THIS position's slot.
        if poisoned_line:
            if has_poisoned_line and poisoned_line_moves:
                top = nres["temptations"][0]
                self.store.set_poisoned(fen, poisoned_line_moves,
                                        {"fatal": top.get("fatal"), "idea": top.get("idea")})
            else:
                self.store.clear_poisoned(fen)

        if focus == "analysis":   # the grounded NL briefing (the coach's input)
            pos = analyze_positional(board)
            resp = {
                "fen": fen, "side_to_move": board.side_to_move,
                "pieces": R.piece_list(board),
                "analysis": assemble_analysis(board, analysis.best.score, pos, facts),
            }
            R.enforce_budget(resp)
            resp["deliver"] = _DELIVER
            self._add_maia(resp, fen)
            if (nb := self._poisoned_line_block(nres)) is not None:
                resp["poisoned_line"] = nb
            if board_push:   # board auto-shows the salient facts; no coach citation
                arrows, highlights = R.facts_to_board(facts)
                self.store.write_board(fen, arrows=arrows, highlights=highlights,
                                       eval=R.eval_block(analysis.best.score))
            return resp

        resp = {
            "fen": fen, "side_to_move": board.side_to_move,
            "pieces": R.piece_list(board),
            "material": R.material(board),
            "eval": R.eval_block(analysis.best.score),
            "lines": [] if focus == "positional" else [   # a strategic read needs no PVs
                {"rank": ln.rank, "eval": R.eval_block(ln.score),
                 "pv_san": R.pv_san(fen, ln.pv)}
                for ln in analysis.lines
            ],
            "facts": [R.fact_wire(f) for f in facts],
        }
        if focus == "positional":   # the grounded five-term strategic read (no fact sheet)
            resp["positional"] = _positional_block(board)
        R.enforce_budget(resp)
        resp["deliver"] = _DELIVER   # after budget: a reminder never costs a fact
        self._add_maia(resp, fen)
        if (nb := self._poisoned_line_block(nres)) is not None:
            resp["poisoned_line"] = nb
        if board_push:
            arrows, highlights = R.facts_to_board(facts)
            self.store.write_board(fen, arrows=arrows, highlights=highlights, eval=resp["eval"])
        return resp

    # -- get_hints --------------------------------------------------------
    @_guarded
    @_scoped("get_hints")
    def get_hints(self, fen, *, movetime_ms=1500) -> dict:
        """Derive a *grounded* hint ladder for the best move — 1–3 nudges,
        vague → specific, each a partial reveal of the engine's own PV/geometry
        (never a fabricated claim). Feed the returned `text`s into a probe's
        `hints`. Returns `hints:[]` when the line has no tactical handle — then
        ask a mastery-tiered question rather than invent a hint."""
        if (g := self._gate()) is not None:
            return g
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        if not board.legal_moves():
            return R.error("terminal_position", "no legal moves (checkmate/stalemate)")

        lim = self._lim(movetime_ms)
        analysis = self._cached_analyse(fen, multipv=1, **lim)
        hints = derive_hints(board, analysis)
        resp = {
            "fen": fen,
            "best": R.pv_san(fen, analysis.best.pv[:1])[0] if analysis.best.pv else None,
            "hints": [{"rung": h.rung, "text": h.text, "squares": h.squares,
                       "provenance": h.provenance} for h in hints],
        }
        R.enforce_budget(resp)
        return resp

    @staticmethod
    def _resolve_move(board, raw: str) -> str | None:
        """Accept a move as SAN ('Nxe4') or UCI ('f6e4' / promo 'e7e8q'), returning the
        legal UCI or None. The coach passes whichever it has; don't burn a call on the
        wrong form."""
        try:
            return board.uci(raw)                 # SAN
        except Exception:
            pass
        legal = board.legal_moves()
        if raw in legal:                          # exact UCI
            return raw
        for lm in legal:                          # UCI missing its promotion suffix
            if lm.startswith(raw):
                return lm
        return None

    # -- evaluate_and_show (MIXED: classify move(s) AND paint) -----------
    @_guarded
    @_scoped("evaluate")
    def evaluate(self, fen, sans=None, *, san=None, move=None) -> dict:
        """PURE READ — evaluate candidate move(s) and hand back the read, touching NOTHING on the
        board. ONE move → the deep read (class/glyph, Δwin%, best alternative, refutation PV,
        brilliancy). SEVERAL (a list of up to 4, SAN or UCI) → a ranked comparison. This is your
        GROUNDING tool: call it whenever you just want the numbers behind a move. Use
        `evaluate_and_show` only when you want the PLAYER to see the candidates as arrows."""
        if (g := self._gate()) is not None:
            return g
        return self._evaluate(fen, sans, san=san, move=move, board_push=False)

    @_guarded
    @_scoped("evaluate_and_show")
    def evaluate_and_show(self, fen, sans=None, *, san=None, move=None, board_push=True) -> dict:
        """MIXED — like `evaluate`, but ALSO paints the candidate move(s) as arrows on the CURRENT
        board (it never advances the board past the move). Use this only to SHOW the player the
        candidates; to ground your own reasoning, use the pure `evaluate`."""
        if (g := self._gate()) is not None:
            return g
        return self._evaluate(fen, sans, san=san, move=move, board_push=board_push)

    def _evaluate(self, fen, sans, *, san, move, board_push) -> dict:
        """Shared dispatch for `evaluate` (board_push=False) and `evaluate_and_show` (board_push=True)."""
        if sans is None:
            one = san if san is not None else move
            sans = [one] if one else None
        if isinstance(sans, str):        # a bare move string → a one-move list
            sans = [sans]
        if not sans or not isinstance(sans, list) or len(sans) > 4:
            return R.error("bad_args", "pass 1-4 candidate moves as `sans`")
        if len(sans) == 1:
            return self._evaluate_one(fen, sans[0], board_push=board_push)
        return self._compare_many(fen, sans, board_push=board_push)

    def _evaluate_one(self, fen, san=None, *, move=None, board_push=True) -> dict:
        raw = san if san is not None else move
        if not raw:
            return R.error("no_move", "give the move as `san` ('Nxe4') or `move` (uci 'f6e4')")
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        uci = self._resolve_move(board, raw)
        if uci is None:
            return R.error("illegal_move",
                           f"{raw} is not a legal move for {board.side_to_move} here — "
                           f"re-check the position and whose move it is")

        lim = self._lim(1500)
        before = self._cached_analyse(fen, multipv=2, **lim)
        best_from_mover = before.best.score
        second = before.lines[1].score if len(before.lines) > 1 else None
        best_wp = win_pct_from_score(best_from_mover)

        after_board = board.apply(uci)
        refutation: list[str] = []
        if not after_board.legal_moves():
            if after_board.in_check:                 # the move delivers mate
                glyph, played_wp, played_cp = Glyph.OK, 100.0, 1000
            else:                                    # stalemate -> drawn
                glyph, _ = classify(best_from_mover, Score(cp=0),
                                    second_best_from_mover=second,
                                    played_is_best=(uci == before.best.pv[0]))
                played_wp, played_cp = 50.0, 0
        else:
            after = self._cached_analyse(after_board.fen, multipv=1, **lim)
            played_result = after.best.score
            glyph, _ = classify(best_from_mover, played_result,
                                second_best_from_mover=second,
                                played_is_best=(uci == before.best.pv[0]))
            mover_view = played_result.negated()
            played_wp = win_pct_from_score(mover_view)
            played_cp = mover_view.to_ceiled_cp()
            refutation = R.pv_san(after_board.fen, after.best.pv)

        cls = _CLASS[glyph]
        if cls in ("ok", "only_move"):   # a sound sacrifice upgrades to brilliant (!!)
            second_wp = win_pct_from_score(second) if second is not None else None
            if is_brilliant(board, uci, best_win=best_wp, played_win=played_wp,
                            second_win=second_wp, played_is_best=(uci == before.best.pv[0])):
                cls = "brilliant"
        facts = build_fact_sheet(board, self.engine, top_n=5, **self._probe_lim())
        resp = {
            "fen": fen, "side_to_move": board.side_to_move,
            "material": R.material(board),
            "san": board.san(uci), "captured": _captured_piece(fen, uci),
            "class": cls, "glyph": str(glyph.value),
            "delta_win_pct": round(played_wp - best_wp, 1),
            "eval": {"cp": played_cp, "win_pct": round(played_wp, 1)},
            "best": {
                "san": (R.pv_san(fen, before.best.pv, 1) or [""])[0],
                "pv_san": R.pv_san(fen, before.best.pv),
                "eval": R.eval_block(best_from_mover),
            },
            "refutation_pv": refutation,
            "facts": [R.fact_wire(f) for f in facts],
        }
        R.enforce_budget(resp)
        resp["deliver"] = _DELIVER   # after budget: a reminder never costs a fact
        self._add_maia(resp, fen)
        is_best = uci == before.best.pv[0]
        mr = self._played_move_read(fen, uci, cls, is_best=is_best)   # coach's private notes
        if mr is not None:
            resp["move_read"] = mr
        pm = self._played_move_player_text(fen, uci, cls, is_best=is_best)   # what the app shows
        if pm is not None:
            resp["move_meaning"] = pm
        if board_push:
            # Show the candidate as an ARROW on the CURRENT board — never advance the board past the
            # move (that reads as "the coach played my move"). Matches the multi-move path; the eval
            # bar is left on the current position (not the candidate's outcome).
            arrows = [{"from": uci[:2], "to": uci[2:4],
                       "style": _CLASS_STYLE[cls], "fact_id": None}]
            _, highlights = R.facts_to_board(facts)
            self.store.write_board(fen, arrows=arrows, highlights=highlights,
                                   caption=f"{resp['san']} {resp['glyph']}".strip())
        return resp

    # -- assess_move (read-only; for the APP as a direct MCP client) ------
    @_guarded
    def assess_move(self, fen, move, *, movetime_ms=400) -> dict:
        """A lean, **read-only** assessment of a move the player just made, at the
        current player's rating — called by the APP directly over MCP (not the coach).
        Returns `{san, class, meaning}` where `meaning` is the PLAYER-facing text (a
        level-typical mistake, or a find beyond their level) or `None` when there's
        nothing notable. No fact sheet, no state writes, never gated — quick feedback."""
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        uci = self._resolve_move(board, move)
        if uci is None:
            return R.error("illegal_move",
                           f"{move} is not legal for {board.side_to_move} here")
        lim = self._lim(movetime_ms)
        before = self._cached_analyse(fen, multipv=2, **lim)
        best = before.best.score
        second = before.lines[1].score if len(before.lines) > 1 else None
        is_best = uci == before.best.pv[0]
        after_board = board.apply(uci)
        if not after_board.legal_moves():
            glyph = Glyph.OK if after_board.in_check else classify(
                best, Score(cp=0), second_best_from_mover=second, played_is_best=is_best)[0]
        else:
            after = self._cached_analyse(after_board.fen, multipv=1, **lim)
            glyph, _ = classify(best, after.best.score,
                                second_best_from_mover=second, played_is_best=is_best)
        cls = _CLASS[glyph]
        return {"san": board.san(uci), "class": cls,
                "meaning": self._played_move_player_text(fen, uci, cls, is_best=is_best)}

    # -- get_common_mistakes (grounded "what do players my level get wrong here?") --
    @_guarded
    def get_common_mistakes(self, fen, *, movetime_ms=400) -> dict:
        """Grounded read of the likely human moves at the current player's rating, each
        **classified by the engine** — so the coach answers "what do players at my level
        get wrong here?" from data, not a guess. Maia predicts *which* moves players reach
        for; Stockfish rules on each. Returns the classified moves (likeliest first), the
        mistakes among them, and the best move. Translate for the player — never recite a
        class or a number."""
        if (g := self._gate()) is not None:
            return g
        if self.maia is None:
            return R.error("no_predictor", "the human-move predictor isn't configured")
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        if not board.legal_moves():
            return R.error("terminal_position", "no legal moves (checkmate/stalemate)")
        top = self._maia_top(fen, n=5)              # Maia SELECTS — the human-likely moves here
        if not top:
            return {"rating": self.player_rating, "best": None, "moves": [], "mistakes": []}
        lim = self._lim(movetime_ms)
        before = self._cached_analyse(fen, multipv=2, **lim)
        best_uci, best_score = before.best.pv[0], before.best.score
        best_wp = win_pct_from_score(best_score)
        second = before.lines[1].score if len(before.lines) > 1 else None
        moves = []
        for m in top:                               # Stockfish EVALUATES each selected move
            uci = m["uci"]
            try:
                san = board.san(uci)
            except Exception:
                continue
            after_board = board.apply(uci)
            refutation: list[str] = []
            if not after_board.legal_moves():
                glyph = Glyph.OK if after_board.in_check else classify(
                    best_score, Score(cp=0), second_best_from_mover=second,
                    played_is_best=(uci == best_uci))[0]
                played_wp = 100.0 if after_board.in_check else 50.0
            else:
                after = self._cached_analyse(after_board.fen, multipv=1, **lim)
                glyph, _ = classify(best_score, after.best.score,
                                    second_best_from_mover=second, played_is_best=(uci == best_uci))
                played_wp = win_pct_from_score(after.best.score.negated())
                refutation = R.pv_san(after_board.fen, after.best.pv)
            moves.append({
                "san": san, "rank": m["rank"], "class": _CLASS[glyph],
                "delta_win_pct": round(played_wp - best_wp, 1),
                "refutation": refutation,
            })
        mistakes = [e["san"] for e in moves if e["class"] in ("dubious", "mistake", "blunder")]
        return {
            "rating": self.player_rating, "side_to_move": board.side_to_move,
            "best": board.san(best_uci), "moves": moves, "mistakes": mistakes,
            "note": "Maia picked which moves players at THIS level reach for; the engine evaluated "
                    "each (class / delta / the refutation line). The `mistakes` are the "
                    "popular-but-wrong ones. Narrate for the player — translate the evals, never "
                    "recite a class or a number.",
        }

    # -- _compare_many (helper for evaluate_and_show; multi-move ranking) -
    def _compare_many(self, fen, sans, *, board_push=True) -> dict:
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        if not sans or len(sans) > 4:
            return R.error("bad_args", "pass 1-4 candidate moves")

        lim = self._lim(1500)
        best_wp = win_pct_from_score(self._cached_analyse(fen, multipv=1, **lim).best.score)

        moves = []
        for san in sans:
            uci = self._resolve_move(board, san)
            if uci is None:
                return R.error("illegal_move",
                               f"{san} is not a legal move for {board.side_to_move} here — "
                               f"re-check the position and whose move it is")
            after_board = board.apply(uci)
            if not after_board.legal_moves():
                played_wp = 100.0 if after_board.in_check else 50.0
                played_cp = 1000 if after_board.in_check else 0
            else:
                mv = self._cached_analyse(after_board.fen, multipv=1, **lim).best.score.negated()
                played_wp, played_cp = win_pct_from_score(mv), mv.to_ceiled_cp()
            moves.append({
                "san": board.san(uci), "uci": uci,
                "eval": {"cp": played_cp, "win_pct": round(played_wp, 1)},
                "delta_win_pct": round(played_wp - best_wp, 1),
            })

        moves.sort(key=lambda m: -m["eval"]["win_pct"])
        verdict = " > ".join(m["san"] for m in moves)
        resp = {"fen": fen, "side_to_move": board.side_to_move,
                "moves": moves, "verdict": verdict}
        R.enforce_budget(resp)
        resp["deliver"] = _DELIVER   # after budget: a reminder never costs a fact
        self._add_maia(resp, fen)
        if board_push:
            arrows = [{"from": m["uci"][:2], "to": m["uci"][2:4],
                       "style": "analysis" if i == 0 else "ghost", "fact_id": None}
                      for i, m in enumerate(moves)]
            self.store.write_board(fen, arrows=arrows, caption=verdict)
        return resp

    # -- explore_and_show (MIXED: walk a line AND paint it) --------------
    @_guarded
    @_scoped("explore_and_show")
    def explore_and_show(self, fen, moves, *, analyze=True, movetime_ms=1500, board_push=True) -> dict:
        """MIXED — walks a line (verb) AND repaints the board to it (mutation). Applies
        `moves` (SAN or UCI, both sides, in order) to `fen` via the board core — never from
        memory. With `analyze=True` (default) it also hands back the ENGINE's read of the end
        position: eval, best move, mate distance, PV — the deterministic answer to "what if I
        play X, then Y…?" A refutation the player missed surfaces as a grounded `mate_in` + PV,
        never invented. With `analyze=False` it just walks + repaints (no engine) — the fast
        primitive for coaching a combination one move at a time (advance by the student's move
        and the forced reply, then re-`analyze_and_show` and probe the next). Do NOT evaluate a
        line yourself; walk it here."""
        if (g := self._gate()) is not None:
            return g
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        if not isinstance(moves, list) or not (1 <= len(moves) <= 12):
            return R.error("bad_args", "pass 1-12 moves")
        played: list[str] = []
        for m in moves:
            uci = m if _UCI_RE.match(str(m)) else None
            if uci is None:
                try:
                    uci = board.uci(m)               # SAN -> UCI
                except Exception:
                    return R.error(
                        "illegal_move",
                        f"{m!r} is not legal here (it's {board.side_to_move}'s move); "
                        f"played so far: {played or '(none)'} — re-check the move order and "
                        f"whose turn it is, then re-explore")
            try:
                played.append(board.san(uci))
                board = board.apply(uci)
            except Exception:
                return R.error("illegal_move", f"{m!r} is not legal in the line")
        end_fen = board.fen
        terminal = None
        if not board.legal_moves():   # the line itself ended the game — nothing to analyse
            terminal = "checkmate" if board.in_check else "stalemate"
        # Light path: no engine read (old `advance`), or a terminal line (nothing to analyse).
        if not analyze or terminal:
            if board_push:
                self.store.write_board(end_fen, caption=" ".join(played))
            resp = {"fen": end_fen, "side_to_move": board.side_to_move, "line": played}
            if terminal:
                resp["terminal"] = terminal
            resp["deliver"] = _DELIVER
            return resp
        lim = self._lim(movetime_ms)
        analysis = self._cached_analyse(end_fen, multipv=1, **lim)
        score = analysis.best.score
        resp = {
            "fen": end_fen, "side_to_move": board.side_to_move, "line": played,
            "material": R.material(board),
            "eval": R.eval_block(score),
            "best": (R.pv_san(end_fen, analysis.best.pv, 1) or [""])[0],
            "pv_san": R.pv_san(end_fen, analysis.best.pv),
        }
        if score.is_mate:   # + = the side to move mates, - = the side to move gets mated
            resp["mate_in"] = score.mate
        R.enforce_budget(resp)
        resp["deliver"] = _DELIVER   # after budget: a reminder never costs a fact
        self._add_maia(resp, end_fen)
        if board_push:
            best_uci = analysis.best.pv[0] if analysis.best.pv else None
            arrows = ([{"from": best_uci[:2], "to": best_uci[2:4], "style": "gold", "fact_id": None}]
                      if best_uci else None)
            self.store.write_board(end_fen, arrows=arrows, caption=" ".join(played),
                                   eval=resp["eval"])
        return resp

    # -- build_and_arm_drill (MIXED: build the tree AND arm the drill) ----
    @_guarded
    @_scoped("build_and_arm_drill")
    def build_and_arm_drill(self, fen, *, concept_id=None, movetime_ms=250) -> dict:
        """MIXED — builds the forcing-line coaching TREE for `fen` (verb) AND arms the drill
        (mutation): writes `tree.json` and seeds the move history for the APP to walk. The name
        says both — this is not a read. The app drives the session (poses each only-move, checks
        it, branches every defense, backtracks); you coach the events it reports **live, one
        position at a time** (`analyze_and_show` the event's `fen`, then `push_beat`). Returns
        only a COMPACT summary (the full tree is the app's, never your context — no bulk
        authoring). A `drillable` root means it's a forcing win worth coaching; otherwise the
        position isn't a forcing line and no drill is armed. Pass `concept_id` — the mastery
        concept the win demonstrates — and the MCP banks it DETERMINISTICALLY on solve (you don't
        record_observation the drill afterward)."""
        if (g := self._gate()) is not None:
            return g
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        if not board.legal_moves():
            return R.error("terminal_position", "no legal moves (checkmate/stalemate)")
        # two-tier limits — verify must be DEEPER than discover or a combination whose
        # win sits deep won't clear the bar. tests: nodes override -> verify = 6x nodes;
        # prod: movetime, verify the full budget and discover a third of it.
        if self._limit and "nodes" in self._limit:
            n = self._limit["nodes"]
            discover, verify = {"nodes": n}, {"nodes": n * 6}
        else:
            # A drill tree is a reproducible coaching artifact — build the STRUCTURE
            # DETERMINISTICALLY (fixed node budgets + single thread), never with a
            # timing-based search that yields a different tree each run. `movetime_ms`
            # does not affect the tree structure.
            discover, verify = {"nodes": 45_000}, {"nodes": 180_000}
        # Force single-threaded ONLY for the tree build so the fixed node budget is
        # reproducible (multi-threaded search is non-deterministic even at fixed nodes).
        prior_threads = getattr(self.engine, "_threads", 1)
        force_single = prior_threads != 1
        if force_single:
            self.engine._set("Threads", 1)
        try:
            tree = build_line_tree(fen, self.engine, discover=discover, verify=verify)
        finally:
            if force_single:
                self.engine._set("Threads", prior_threads)
        root = tree["root"]
        drillable = root["kind"] in ("solve", "mate")
        first = root.get("expect_san") or (
            root["options"][0]["san"] if root.get("options") else None)
        # Only persist a tree the app can actually walk. A non-drillable root ("done"
        # — not a single-only-move win) has no moves to solve; leaving it on disk makes
        # the app present a broken drill that rejects every move as wrong. Clear it and
        # let the coach coach the position normally.
        if drillable:
            # DETERMINISTICALLY detect whether this drill hides a human trap — at build time, on the
            # single-threaded poisoned-line engine (reproducible) — so the "calculate deeply" warning and the
            # on-solve nudge don't depend on the coach remembering to pass poisoned_line=True (it doesn't
            # reliably). The flag rides the tree, so it persists for the whole drill (no board-repaint
            # flicker). Maia-less sessions just get has_poisoned_line=false.
            has_poisoned_line = False
            neng = self._get_poisoned_line_engine() or self.engine
            nres = self._poisoned_line_or_false(fen, neng, stop_on_first=True)
            if nres and nres["has_poisoned_line"] and nres["temptations"]:
                # Carry the FULL trap move sequence ({uci,san,fen}) from the drill root on the tree,
                # so the APP renders it as a variation on reveal — client-side, no server round-trip
                # and no server-side "punish drill" (that mutation duplicated the app's local render).
                # It survives solve: the tree persists until a new drill, unlike the transient board flag.
                top = nres["temptations"][0]
                moves = self._poisoned_line_moves(fen, top, neng)
                if moves:
                    has_poisoned_line = True
                    tree["poisoned_line_moves"] = moves
                    # Persist Maia's motif (why the human is tempted, why it fails) alongside the line, so
                    # a DRILL_POISONED_LINE turn can hand it over without a fresh detection call.
                    tree["poisoned_line_meta"] = {"fatal": top.get("fatal"), "idea": top.get("idea")}
            tree["has_poisoned_line"] = has_poisoned_line   # the ONE source: durable on the tree/document
            if concept_id:
                # The mastery concept this win demonstrates rides the tree (durable, survives restart),
                # so the MCP can bank it deterministically on solve — no LLM close, no lost credit.
                tree["concept_id"] = concept_id
            seq = self.store.write_tree(tree)
            self.store.clear_poisoned()   # the drill's trap lives on the tree; drop any freeform slot
            from .drill import DrillState
            self._drill = DrillState(tree)          # the MCP walks it; the app just pushes moves
            self.store.write_history(self._drill.line)   # seed the move line with ply 0 (start)
            self.store.set_drill_state(self._drill.to_state())   # persist the walker (P2c)
        else:
            # Not a forcing line — no drill to walk. Retire any prior drill AND reset the navigator
            # to this position, so a leftover line from an earlier drill stops rendering.
            self._reset_drill_line(fen)
            seq = None
        return {
            "fen": fen, "side_to_solve": tree["side_to_solve"],
            "root_kind": root["kind"],                       # solve | mate | done
            "drillable": drillable,
            "has_poisoned_line": tree.get("has_poisoned_line", False),     # trap present → the app warns "calculate deeply"
            "concept_id": concept_id if drillable else None, # banked deterministically on solve
            # The lane skill to load: a drillable position → the app-driven `drills` walk; otherwise coach
            # it freeform. This fork is the tool's verdict (drillable), which read_input's class can't see.
            "load_skill": "drills" if drillable else "coaching-a-position",
            "first_move": first,                             # None if not a forcing line
            "lines": count_leaves(root),                     # forcing lines the session covers
            "seq": seq,
        }

    def _weakest_concepts(self) -> list[str]:
        """Domain concept ids ordered WEAKEST first, for adaptive puzzle selection. Ranks by the
        derived EWMA `mastery` (ascending); a concept with no evidence yet sorts as weakest (a gap).
        Ties keep domain order (a stable sort over the ontology's own order). Empty if no mastery
        engine (analysis-only sessions) — selection then falls back to rating/id order."""
        if self.mastery is None:
            return []
        learner, _ = self.mastery.recompute()
        observed = learner.get("concepts", {})
        all_ids = [c["id"] for c in self.mastery.domain.get("concepts", []) if c.get("id")]

        def score(cid):
            m = observed.get(cid, {}).get("mastery")
            return m if m is not None else -1.0     # never observed -> weakest (an untouched gap)

        return sorted(all_ids, key=score)

    # -- set_puzzle (MIXED: pick a curated puzzle AND set it up in one call) ----
    @_guarded
    def set_puzzle(self, theme=None) -> dict:
        """MIXED — the whole "give me a puzzle" flow in ONE call. Picks a curated puzzle from
        `content/puzzles/` (deterministically: a named `theme` filters by tag; otherwise it biases
        to the player's WEAKEST concept, skipping puzzles already served this session), then sets it
        up: pushes a fresh activity frame (the puzzle is an UNRELATED position), paints the position
        WITHOUT revealing the solution, and arms the forcing-line drill (which re-derives the
        solution and detects the poisoned line). Returns a COMPACT summary — `drillable`,
        `first_move`, `has_poisoned_line`, `load_skill` (usually `drills`), plus `concept_id`/`theme`
        for context. Load the named skill and coach the walk; `pop_activity` when the player is done
        or asks for another. On an empty/exhausted deck returns a structured `no_puzzles` error."""
        if (g := self._gate()) is not None:
            return g
        puzzles = puzzle_content.load_puzzles()
        if not puzzles:
            return R.error(
                "no_puzzles",
                "no puzzle content found — add *.jsonl puzzle files (poisoned-puzzle schema) under "
                "content/puzzles/ (or point LUCENA_PUZZLES at the directory), then try again.")
        served = frozenset(self.store.served_puzzles)
        weakest = None if theme else self._weakest_concepts()
        chosen = puzzle_content.select_puzzle(puzzles, theme=theme, weakest=weakest, served=served)
        if chosen is None:
            if theme and any(theme.lower() == t.lower()
                             for p in puzzles for t in puzzle_content._themes(p)):
                detail = (f"every {theme!r} puzzle has already been served this session — "
                          "ask for a different theme, or start a new session.")
            elif theme:
                detail = (f"no {theme!r} puzzles in the deck — try another theme "
                          "(e.g. fork, pin, backRankMate) or ask for any puzzle.")
            else:
                detail = "every puzzle has already been served this session — start a new session for more."
            return R.error("no_puzzles", detail)

        fen = puzzle_content.puzzle_fen(chosen)
        concept = puzzle_content.puzzle_concept(chosen)
        pid = puzzle_content.puzzle_id(chosen)
        # An unrelated position → push a fresh frame first (contract: Activities). Paint it with the
        # engine-free set_board so the player SEES the puzzle but not the answer, THEN arm the drill.
        self.push_activity("puzzle", seed={"puzzle_id": pid, "concept_id": concept})
        painted = self.set_board(fen)
        if isinstance(painted, dict) and "error" in painted:
            self.pop_activity()                              # don't strand an empty frame
            return painted
        armed = self.build_and_arm_drill(fen, concept_id=concept)
        if isinstance(armed, dict) and "error" in armed:
            self.pop_activity()
            return armed
        self.store.mark_puzzle_served(pid)
        # The puzzle's intro is a LOCAL beat — the coach must NOT narrate the setup (no LLM push_beat
        # for "here's a puzzle"). Rotate the phrasing deterministically by how many were served before,
        # so "another one" doesn't read identically. The player answers by moving on the board.
        intro = _PUZZLE_INTROS[len(served) % len(_PUZZLE_INTROS)]
        self.store.append_beats([{"kind": "say", "stops": False, "tone": "teach",
                                  "segments": [{"text": intro}]}])
        remaining = sum(1 for p in puzzles
                        if puzzle_content.puzzle_id(p) not in served
                        and puzzle_content.puzzle_id(p) != pid)
        return {
            "ok": True,
            "puzzle_id": pid,
            "fen": fen,
            "themes": chosen.get("themes"),
            "rating": chosen.get("rating"),
            "concept_id": concept,
            "drillable": armed.get("drillable"),
            "first_move": armed.get("first_move"),
            "has_poisoned_line": armed.get("has_poisoned_line"),
            "load_skill": armed.get("load_skill"),
            "remaining": remaining,
        }

    # -- explore_poisoned_line (PURE READ: explore the trap; narrate with push_beat) --
    @_guarded
    @_scoped("explore_poisoned_line")
    def explore_poisoned_line(self, fen=None) -> dict:
        """The post-solve payoff: reveal the human TRAP the player dodged. PURE READ — it
        explores and hands back the line; it NEVER mutates the board, drill, or history (the app
        already renders the poisoned line as a local variation on reveal — a server mutation would
        just duplicate that). Call it with NO argument on a solved drill — it uses that drill's ROOT
        (where the trap is), so you never have to figure out the fen. Detects the top temptation and
        returns `poisoned_line` (the concrete `[{uci,san,fen}]` trap sequence: the greedy human line
        answered by the refutation) plus the `fatal` motif and `idea`. Narrate it with `push_beat`.
        `has_poisoned_line:false` when there's no trap to explore."""
        if (g := self._gate()) is not None:
            return g
        return self._explore_poisoned_line(fen)

    def _poisoned_line_moves(self, fen, temptation, eng, *, plies=None):
        """The ENTIRE poisoned line as a concrete move sequence — `[{uci, san, fen}]` from `fen`: the
        seed tempting move, then the greedy human continuation (Maia top-1) answered by the defender's
        best reply, through the refuting tactic (the decisive defender move ends it). `fen` is the
        position AFTER each move (the app's VarNode shape). Node-limited + deterministic, so it
        reproduces the line the detector found. Empty list if there's no temptation or the seed is
        illegal. Pure — writes no state.

        Guards itself: this is a new_game + analyse LOOP, so on the shared dedicated engine another
        chat's detector would interleave with it. Guarding here rather than at each call site means a
        new caller cannot forget (the RLock makes it free when a caller already holds the guard)."""
        if not temptation or not temptation.get("seeds") or self.maia is None:
            return []
        with self._shared_engine_guard(eng):
            return self._poisoned_line_moves_locked(fen, temptation, eng, plies=plies)

    def _poisoned_line_moves_locked(self, fen, temptation, eng, *, plies=None):
        nodes = _LIVE_POISONED_LINE["nodes"]
        depth = plies if plies is not None else _LIVE_POISONED_LINE["plies"]
        board0 = Board(fen)
        try:
            seed_uci = board0.uci(temptation["seeds"][0])
            board = board0.apply(seed_uci)
        except Exception:
            return []
        mover = board0.side_to_move                     # the human who'd fall for it
        out = [{"uci": seed_uci, "san": temptation["seeds"][0], "fen": board.fen}]
        for _ in range(depth):
            if not board.legal_moves():
                break
            if board.side_to_move != mover:             # defender — best reply (the refutation, when decisive)
                eng.new_game()
                a = eng.analyse(board.fen, multipv=1, nodes=nodes)
                uci = a.best.pv[0]
                decisive = win_pct_from_score(a.best.score) >= 90.0
            else:                                       # the human's greedy move
                tops = self.maia.top_human_moves(board.fen, self.player_rating, n=1)
                if not tops:
                    break
                uci = tops[0]["uci"]
                decisive = False
            san = board.san(uci)
            board = board.apply(uci)
            out.append({"uci": uci, "san": san, "fen": board.fen})
            if decisive:                                # the refuting tactic just landed — trap sprung
                break
        return out

    @_guarded
    def _explore_poisoned_line(self, fen=None) -> dict:
        """Ungated core of explore_poisoned_line. PURE — detects the trap and returns the concrete
        poisoned line for the coach to narrate; writes NO state (no board, tree, or history). The app
        renders the line locally as a variation, so there is nothing for the server to paint."""
        if self.maia is None:
            return R.error("no_predictor", "the human-move predictor isn't configured")
        last_tree = getattr(self.store, "_last_tree", None) or {}
        # Default (no fen): explore the last drill's ROOT, where the trap lives.
        if not fen:
            fen = last_tree.get("fen")
        if not fen:
            return R.error("no_position", "no drill to explore — nothing was just solved.")
        try:
            Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        eng = self._get_poisoned_line_engine() or self.engine
        # ONE guard across detection AND the reconstruction below: they are a single logical sequence
        # on one engine, so releasing between them would let another chat interleave in the gap.
        # (_shared_engine_guard's RLock re-enters _poisoned_line_moves' own guard.)
        with self._shared_engine_guard(eng):
            return self._explore_poisoned_line_locked(fen, eng)

    def _explore_poisoned_line_locked(self, fen, eng):
        nres = _poisoned_line_detector.find_poisoned_lines(fen, eng, self.maia, rating=self.player_rating,
                              cache=self._poisoned_line_cache, **_LIVE_POISONED_LINE)
        if not nres["has_poisoned_line"] or not nres["temptations"]:
            return {"fen": fen, "has_poisoned_line": False, "poisoned_line": [],
                    "note": "no poisoned line here to explore — coach the position normally."}
        top = nres["temptations"][0]
        moves = self._poisoned_line_moves(fen, top, eng)   # the concrete [{uci,san,fen}] trap sequence
        return {"fen": fen, "has_poisoned_line": True, "poisoned_line": moves,
                "fatal": top.get("fatal"), "idea": top.get("idea"),
                "note": "The poisoned line — the greedy human continuation answered by the refutation. "
                        "The app shows it as a variation; narrate WHY it fails with push_beat. This tool "
                        "does not touch the board or drill."}

    # -- get_game_analysis ------------------------------------------------
    @_guarded
    @_scoped("get_game_analysis")
    def get_game_analysis(self, game_id, what="summary") -> dict:
        """Digest from a precomputed `analysis.json` — never the whole file.

        `what`: `"summary"` (game + summary blocks), `"mistakes"` (only classified
        plies with motifs/refutation — the Socratic raw material), or `"ply:<n>"`
        (one ply's full record)."""
        analysis = self.store.read_analysis(game_id)
        if analysis is None:
            return R.error("game_not_found", f"no analysis for {game_id!r}")
        if what == "summary":
            return {"game": analysis.get("game"), "status": analysis.get("status"),
                    "summary": analysis.get("summary")}
        if what == "mistakes":
            bad = {"dubious", "mistake", "blunder"}
            return {"game_id": game_id, "plies": [
                p for p in analysis.get("plies", []) if p.get("class") in bad]}
        if what.startswith("ply:"):
            try:
                n = int(what.split(":", 1)[1])
            except ValueError:
                return R.error("bad_args",
                               f"malformed selector {what!r} — use ply:<n>, e.g. ply:12")
            for p in analysis.get("plies", []):
                if p.get("ply") == n:
                    return p
            return R.error("ply_not_found", f"no ply {n} in {game_id!r}")
        return R.error("bad_args",
                       f"unknown selector {what!r} — use summary | mistakes | ply:<n>")

    # -- set_board (no engine) --------------------------------------------
    @_guarded
    def set_board(self, fen, *, arrows=None, highlights=None, caption=None) -> dict:
        if (g := self._gate()) is not None:
            return g
        try:
            Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        seq = self.store.write_board(fen, arrows=arrows, highlights=highlights,
                                     caption=caption)
        return {"ok": True, "seq": seq}

    # -- set_board_from_paste (DETERMINISTIC — never an LLM-supplied position) --
    @_guarded
    def set_board_from_paste(self, text) -> dict:
        """Set the board from a position the PLAYER pasted into chat — deterministically, never
        from an LLM FEN. Detects and VALIDATES a FEN or PGN in `text` via the board core: a legal
        FEN sets the board; a PGN is reported (its game-review load is separate). This is called
        by the runner BEFORE the model runs, so the model only ever interprets a board that a
        parser — not the LLM — put there. Not gated: a fresh pasted position starts a new context.
        Returns {found: 'fen'|'pgn'|None, ...}."""
        from lucena_engine.detect import detect_fens, detect_pgn
        fens = detect_fens(text or "")
        if fens:
            fen = fens[0]
            try:
                Board(fen)
            except Exception as e:
                return R.error("illegal_fen", str(e))
            seq = self.store.write_board(fen)
            # A pasted position is a NEW game — reset the move line to just this position (ply 0), so a
            # stale move list from a prior game can't linger and disagree with the board (single source
            # of truth: board and history describe the same game).
            self.store.write_history([{"n": 0, "san": None, "uci": None, "fen": fen}])
            return {"found": "fen", "fen": fen, "seq": seq}
        game = detect_pgn(text or "")
        if game is not None:
            return {"found": "pgn", "plies": len(game.plies)}   # game-review load: TODO
        return {"found": None}

    # -- push_beat --------------------------------------------------------
    @_guarded
    def push_beat(self, beats, *, session=None) -> dict:
        """Append 1–4 coaching beats. Three kinds — the coach tells, asks, or
        echoes the player:

        - `say` — one idea of coaching. Optional `tone ∈ teach|praise|correct|
          verdict` (the app colours it; default `teach`). Never stops.
        - `ask` — one Socratic question. Always STOPS the turn (the gate locks
          until the player answers — end your turn after an `ask`). May carry
          `hints`: 1–3 nudges vague → specific, none of which is the answer.
        - `you` — the player's own words, echoed so the beats column reads as a
          conversation. Push this FIRST when responding to something the player
          typed (`{kind:'you', text:'<their message>'}`). Not a coaching claim —
          it skips the grounding gate.

        (`teach`/`verdict` are accepted as aliases for `say` with that tone;
        `probe` for `ask`.) Text is `segments:[{text, tone?}]` or a flat `text`
        string. Optional `concept_id`.

        There are NO fact ids and no citation. The board shows the salient facts
        as arrows automatically (painted by `analyze_and_show` from the computed
        facts) — the coach never cites a fact. To repaint (e.g. after walking a
        line), pass an explicit `board:{fen, arrows?, highlights?, caption?}`."""
        if (g := self._gate()) is not None:
            return g
        if not isinstance(beats, list) or not (1 <= len(beats) <= 4):
            return R.error("bad_args", "push 1-4 beats per call")
        # A pure player-echo call (`you` beats only) isn't a coaching claim — it skips the
        # grounding gate so the coach can relay what the player typed before it has grounded.
        only_you = all(_BEAT_ALIAS.get(b.get("kind") or b.get("type")) == "you" for b in beats)
        spec = _CLASS_TABLE[self._class]                 # grounding gate (precedence after awaiting)
        if not only_you and self._classified and spec["must_ground"] and not self._grounded:
            return R.error("ungrounded",
                           f"call {spec['must_ground']} before push_beat ({self._class})")
        shaped = []
        for beat in beats:
            raw = beat.get("kind") or beat.get("type")
            kind = _BEAT_ALIAS.get(raw)
            if kind is None:
                return R.error("bad_args", f"bad beat kind {raw!r} — use say | ask")
            segs = beat.get("segments")
            if not segs and beat.get("text"):            # flat-text shorthand
                segs = [{"text": beat["text"]}]
            if not isinstance(segs, list) or not segs:
                return R.error("bad_args", "beat needs `segments` or a `text` string")
            entry = {
                "kind": kind,
                "stops": kind == "ask",   # an ask ENDS the turn; a say never stops
                "segments": [{"text": s.get("text", ""),
                              **({"tone": s["tone"]} if s.get("tone") else {})}
                             for s in segs],
            }
            if kind == "say":             # tone colours the say (teach/praise/correct/verdict)
                tone = beat.get("tone") or (raw if raw in _SAY_TONES else "teach")
                if tone not in _SAY_TONES:
                    return R.error("bad_args",
                                   f"bad say tone {tone!r} — use "
                                   f"{' | '.join(_SAY_TONES)}")
                entry["tone"] = tone
            if beat.get("concept_id"):
                entry["concept_id"] = beat["concept_id"]
            if kind == "ask" and beat.get("hints"):   # withheld ladder, vague->specific
                raw_hints = beat["hints"]
                hints = [h.strip() for h in raw_hints if isinstance(h, str) and h.strip()] \
                    if isinstance(raw_hints, list) else []
                if hints:
                    entry["hints"] = hints[:3]         # cap the ladder at three rungs
            shaped.append(entry)
        # Repaint the board FIRST (only on an explicit `board`), then append — so
        # each beat is stamped (in append_beats) with the board seq it describes and
        # the panel stays anchored to what's on the board.
        for beat in beats:
            b = beat.get("board")
            if isinstance(b, dict) and b.get("fen"):
                self.store.write_board(
                    b["fen"], arrows=b.get("arrows"), highlights=b.get("highlights"),
                    caption=b.get("caption"))
        indices = self.store.append_beats(shaped)
        stop_pos = next((i for i, b in enumerate(shaped) if b["stops"]), None)
        if stop_pos is not None:   # an ask locks the flow until the player answers
            # Persist WHICH probe is pending, not just that one is (§6.1 gate.pending_question) — so a
            # session resumed mid-probe can re-render the question + its hint ladder from the document,
            # not just come back locked with no idea what was asked. Cleared when read_input unlocks.
            ask = shaped[stop_pos]
            pending = {"beat_i": indices[stop_pos],
                       "question": " ".join(s.get("text", "") for s in ask.get("segments", []))}
            if ask.get("hints"):
                pending["hints"] = ask["hints"]
            self.store.set_gate(True, pending=pending)
        self._spoke_since_read = True         # the coach has spoken → the next read_input is a real turn
        return {"ok": True, "indices": indices}

    # -- push_analysis ----------------------------------------------------
    @_guarded
    def push_analysis(self, fen, verdict, observations) -> dict:
        """Render the single-position **analysis object** — the coach's natural-language
        conversion of the grounded facts + engine eval for THIS position. `verdict` is
        the one-line standing translated for the player; `observations` are the grounded
        points (each a short sentence). It is the *output* twin of
        `analyze_and_show(focus="analysis")`: you READ that briefing, then hand the
        player-facing version back here (the app renders it in the Analysis panel, not a
        paced beat stream). It does NOT coach — the conversation the player reads is
        `push_beat`; use this only as an extra static read, never instead of coaching.

        Grounded, never from memory: you MUST have called `analyze_and_show` on this exact
        fen this turn — a position you have not analysed is refused (board authority). Keep
        numbers out; translate the eval to meaning (say "winning", never "+1.7")."""
        if (g := self._gate()) is not None:
            return g
        if self._last_fen != fen:
            return R.error(
                "ungrounded_analysis",
                "push_analysis needs analyze_and_show on this exact fen first — analyse "
                "the position, then convert its facts + eval to natural language (never "
                "from memory).")
        if not isinstance(verdict, str) or not verdict.strip():
            return R.error("bad_args",
                           "analysis needs a one-line `verdict` (the standing, in words)")
        obs = [o.strip() for o in observations if isinstance(o, str) and o.strip()] \
            if isinstance(observations, list) else []
        if not obs:
            return R.error("bad_args", "analysis needs at least one grounded `observation`")
        seq = self.store.write_analysis(fen, verdict=verdict.strip(), observations=obs[:6])
        self._spoke_since_read = True   # a panel landed → the next read_input is a real turn
        return {"ok": True, "seq": seq, "fen": fen,
                "note": "Analysis panel updated — this is NOT coaching. If the player asked to be "
                        "coached (e.g. 'coach me'), the conversation still goes through push_beat; "
                        "analysis alone leaves the Coach panel empty."}

    # -- read_input -------------------------------------------------------
    @_guarded
    def read_input(self) -> dict:
        """Read the player's input AND classify the turn. Returns the input plus
        `classification`, `must_ground`, and `allowed_tools`. Side effects (the
        turn lock): sets the current class, resets the per-turn grounding flag, and
        ENDS any probe wait — in the interactive model a turn is read only because
        the player has responded (with a move, or a text answer that shows as
        `kind:"none"`). See contract M-classification."""
        data = self.store.read_input()
        was_awaiting = self._awaiting_input
        # IDEMPOTENT within a turn: a re-read before the coach has spoken (no beat pushed since the
        # last read) that finds nothing new returns the SAME classification and — crucially — does
        # NOT reset the grounding flag. Observed in real transcripts: a mid-turn re-read reset
        # `_grounded`, which forced a duplicate analyze_and_show just to unlock push_beat.
        if (data.get("kind") in (None, "none") and self._classified
                and not self._spoke_since_read and not was_awaiting):
            return self._last_read_out
        self._awaiting_input = False
        cls = _classify(data, was_awaiting)
        # Solved-drill navigation. The single-slot mailbox still holds `drill_solved` after a solve
        # (terminal text is invisible to the server, so a typed question can't overwrite it), so without
        # this the coach explains "you solved it" over whatever board the player has since navigated to.
        # TRANSITION by where they're actually looking (never overriding a LIVE drill turn):
        #   - exploring the poisoned line they dodged (on it OR a variation within it) -> DRILL_POISONED_LINE
        #     (read_input hands over the whole trap + a live eval of the current board);
        #   - a completely different board -> OPEN (coach the live board);
        #   - still on the solved position -> DRILL_SOLVED (the normal close).
        if cls not in ("DRILL_EVENT", "DRILL_WRONG") and self._in_poisoned_context():
            cls = "DRILL_POISONED_LINE"
        elif cls == "DRILL_SOLVED":
            bv, ev = self.store.board_view, data.get("fen")
            if bv and ev and bv.split()[0] != ev.split()[0]:
                cls = "OPEN"
        self._class = cls
        self._classified = True
        self._grounded = False
        self._spoke_since_read = False
        spec = _CLASS_TABLE[cls]
        out = dict(data)
        out["classification"] = cls
        out["must_ground"] = spec["must_ground"]
        out["allowed_tools"] = sorted(_SCOPED - spec["refused"])
        # The lane skill to load before coaching this turn (or None — the core spine handles it).
        out["load_skill"] = _CLASS_SKILL.get(cls)
        # The position the player is LOOKING AT right now (their moves + navigation), so a free
        # question like "what do you think about this position?" grounds in the real board, not memory.
        if (bv := self.store.board_view) is not None:
            out["board_fen"] = bv
        # A drill that closed deterministically. Consume it EITHER WAY (so it never leaks to a later
        # turn), but surface it ONLY when this turn is not already the DRILL_SOLVED closing turn: on
        # DRILL_SOLVED the coach is handling the solve, so a second "concluded/banked" signal is
        # redundant — it made the coach parrot "solved and banked" over the player's actual question.
        if (dc := self.store.take_drill_close()) is not None and cls != "DRILL_SOLVED":
            out["drill_concluded"] = dc
        # DRILL_POISONED_LINE: hand over the whole trap, grounded, so the coach just narrates it.
        if cls == "DRILL_POISONED_LINE":
            out["poisoned_line"] = self._poisoned_line_payload()
        self._last_read_out = out
        return out

    def _in_poisoned_context(self) -> bool:
        """True if the player is exploring the just-solved drill's poisoned line — on it OR in a variation
        WITHIN it. Detected by the on-screen line (plus the current board) passing THROUGH the trap
        positions, so a deeper sub-variation still counts. Placement-only, so move counters don't matter.
        No overlap with the poisoned line -> a completely different board (a plain OPEN turn)."""
        moves = (self.store._last_tree or {}).get("poisoned_line_moves") or []
        if not moves:
            return False
        poisoned = {(m.get("fen") or "").split()[0] for m in moves if m.get("fen")}
        seen = {(m.get("fen") or "").split()[0]
                for m in ((self.store.view or {}).get("line") or []) if m.get("fen")}
        if self.store.board_view:
            seen.add(self.store.board_view.split()[0])
        return bool(seen & poisoned)

    def _poisoned_line_payload(self) -> str:
        """The grounded poisoned-line context for a DRILL_POISONED_LINE turn — as ONE natural-language
        string, not a JSON blob (the coach reads sequential SAN + prose well, and drowns in nested dicts).
        Leads with the trap as numbered SAN ("1.Nxe4 Qe3 2.Bxf7+"), then Maia's motif (why it tempts, why
        it fails — precomputed on the tree), then TWO separately-named engine verdicts (translated to
        words, never a number): the evaluation of the poisoned line (its END position — where the trap
        lands you) and the evaluation of the current position (the board the player is looking at). Naming
        them apart stops the coach from conflating "where the trap ends" with "where the player is now"."""
        tree = self.store._last_tree or {}
        meta = tree.get("poisoned_line_meta") or {}
        moves = tree.get("poisoned_line_moves") or []
        san = _pgn_line(moves)
        parts = [f"The poisoned line the player sidestepped: {san}." if san
                 else "The player is exploring the poisoned line they sidestepped."]
        if (idea := meta.get("idea")):
            parts.append(f"The catch: {idea}.")
        if (fatal := meta.get("fatal")) and fatal not in (idea or ""):
            parts.append(f"The motif is a {fatal}.")
        # The two evaluations, named separately (both engine-gated → skipped without an engine).
        end_fen = (moves[-1] or {}).get("fen") if moves else None
        if end_fen and (poisoned_verdict := self._live_verdict(end_fen)):
            parts.append(f"The evaluation of the poisoned line {san} is: {poisoned_verdict}." if san
                         else f"The evaluation of the poisoned line is: {poisoned_verdict}.")
        if (current_verdict := self._live_verdict(self.store.board_view)):
            parts.append(f"The evaluation of the current position is: {current_verdict}.")
        return " ".join(parts)

    def _live_verdict(self, fen) -> str | None:
        """A plain-language verdict for `fen` from a quick engine eval — 'White is winning', 'roughly
        equal', etc. (never a number). None if there's no engine or the eval fails."""
        if not fen or self.engine is None:
            return None
        try:
            wp = R.eval_block(self._cached_analyse(fen, multipv=1, **self._lim(250)).best.score)["win_pct"]
        except Exception:
            return None
        side = "White" if fen.split()[1] == "w" else "Black"
        if wp >= 85:   return f"{side} is winning"
        if wp >= 62:   return f"{side} is clearly better"
        if wp > 38:    return "the position is roughly equal"
        if wp > 15:    return f"{side} is worse"
        return f"{side} is losing"

    # -- get_view (reader of the app's display state) ---------------------
    @_guarded
    def get_view(self) -> dict:
        """What the player is looking at right now — grounded, token-bounded. The POSITION is the ONE
        session board (`board_view`, the single source of truth — updated by both the coach's paint and
        the app's navigation, never a rival copy). This is NOT the retired `view.fen or board_view`
        chain: `board_view` is the sole position source, so the coach's own fresh paint can never be
        outranked by a stale app view (the H1 staleness the design forbids). The line/cursor/variations
        come from the app's reported view when present — a coach paint to a NEW position drops that
        stale view at the writer (see StateStore.write_board), so it's never surfaced under a fresh
        fen. See contract M-session-view."""
        fen = self.store.board_view
        if not fen:
            if not getattr(self.store, "_current", ""):
                return R.error("no_session", "no active coaching session; start one first")
            return R.error("no_view", "no position is being shown yet; paint or analyze a board first")
        try:
            board = Board(fen)
        except Exception as e:
            return R.error("illegal_fen", str(e))
        side = "white" if fen.split()[1] == "w" else "black"
        view = self.store.view
        out: dict = {
            "fen": fen,
            "side_to_move": side,
            "in_variation": bool((view or {}).get("in_variation")),
            "material": R.material(board)["standing"],
            "viewing": None,
            "cursor_san": None,
            "variations": [],
        }
        if not view:
            return out
        line = view.get("line") or []
        cursor = view.get("cursor") if isinstance(view.get("cursor"), int) else -1
        out["viewing"] = _viewing_line(line, cursor) or None
        if 0 <= cursor < len(line):
            out["cursor_san"] = line[cursor].get("san")
        vs: list[dict] = []
        for branch in (view.get("tree") or []):
            for var in (branch.get("variations") or []):
                vs.append({"from": _branch_label(branch),
                           "line": _pgn_line(var.get("line") or [])})
                if len(vs) >= 12:                            # token budget — cap the branch list
                    break
            if len(vs) >= 12:
                break
        out["variations"] = vs
        return out

    # -- submit_input -----------------------------------------------------
    @_guarded
    def submit_input(self, kind, move=None, fen=None) -> dict:
        """The **app's** structured input channel (replaces `input.json`): a played move, a
        "done", or a drill event — whatever the turn classifier reads on the next `read_input`.
        `kind` is the input class ("move", "drill", "drill_wrong", "start", "none", …); `move`
        is a UCI/SAN string; `fen` the board it was played on. Typed *player text* does NOT come
        here — that reaches the coach through the terminal (LLD-app §2.2)."""
        payload = {"kind": kind}
        if move is not None:
            payload["uci"] = move
        if fen is not None:
            payload["fen"] = fen
        self.store.set_input(payload)
        return {"ok": True, "input": payload}

    # -- drill staleness guard --------------------------------------------
    def _refusal_bypassed(self, name: str, args, kwargs) -> bool:
        """A scoped tool may WAIVE its out-of-class refusal for a specific call. Currently only
        `build_and_arm_drill` on a genuinely NEW position: the class gate refuses it in most classes
        with "you're not entering a fresh tactic here", but that premise is false when there is no
        live drill, the drill is finished, or the target fen has diverged from the active drill — in
        all of which arming a drill IS a fresh tactic. The only spurious case (a live, unfinished
        drill re-armed on its OWN current position) stays refused. Fixes the out_of_class thrown when
        the player enters a new position mid-turn (e.g. in PROBE_ANSWER)."""
        if name == "build_and_arm_drill":
            fen = kwargs.get("fen") or (args[0] if args else None)
            d = self._drill
            return d is None or d.finished or self._drill_diverged(fen)
        return False

    def _drill_diverged(self, fen: str | None, uci: str | None = None) -> bool:
        """Has the board moved OFF the active drill? A live drill keeps the board pinned to
        `drill.current.fen` (every `play()` repaints to the new solve position), so a fresh position —
        or a move played on one — means the coach set up something new WITHOUT a new `build_and_arm_drill`,
        leaving a stale walker installed. Adjudicating against it rejects correct moves and echoes raw
        UCI (`Board(stale_fen).san(uci)` raises → the coordinate fallback); the navigator keeps
        rendering the old line. Detect the divergence by the reported `fen` and, defensively, by the
        move's legality on the drill's own board."""
        d = self._drill
        if d is None or d.finished:
            return False
        cur = (d.current or {}).get("fen")
        if not cur:
            return True
        if fen and fen != cur:
            return True
        if uci:
            try:
                Board(cur).san(uci)
            except Exception:
                return True
        return False

    def _reset_drill_line(self, fen: str | None) -> None:
        """Retire any active drill and reset the move navigator to a clean start at `fen`. Called
        whenever the coach establishes a position that isn't a continuation of the current drill —
        otherwise the stale walker keeps adjudicating and the navigator keeps showing the old game."""
        self._drill = None
        self.store.set_drill_state(None)   # clear the persisted walker (P2c) so a relaunch can't re-arm it
        self.store.clear_tree()   # drops tree.json + the DB copy, so a relaunch can't re-arm it
        self.store.clear_poisoned()   # a new position retires any freeform trap held for the old one
        self.store.write_history(
            [{"n": 0, "san": None, "uci": None, "fen": fen}] if fen else [])

    # -- play_move (app pushes a raw move; the MCP adjudicates) ------------
    @_guarded
    def play_move(self, uci, fen=None, *, push_feedback=True) -> dict:
        """The APP pushes a raw played move; the MCP adjudicates it against the active drill (or,
        with no drill, records it for the coach). Server-side it updates the board, pushes the
        deterministic feedback beat + the Maia-grounded move-meaning beat, and records the drill
        event `read_input` returns — so the app is a pure renderer and `explain`/`ask why` has
        real move context."""
        drill = self._drill
        if drill is not None and not drill.finished and self._drill_diverged(fen, uci):
            # The move was played on a position that isn't the drill's current one. TWO cases:
            if fen and any(_norm_fen(fen) == _norm_fen((p or {}).get("fen", ""))
                           for p in (getattr(drill, "line", None) or [])):
                # (b) the player navigated BACK within the drill's OWN line to try a "what if". That is a
                # REQUEST to explore, not a fact that the drill is over (§4 — a non-authority value is a
                # request). Do NOT destroy the walker/tree; record the move for the coach and report the
                # drill SUSPENDED. It resumes when the board returns to drill.current. (Destroying a drill
                # is now only ever an explicit act — a new position / reset — never a move side effect.)
                self.store.set_input({"kind": "move", "uci": uci, **({"fen": fen} if fen else {})})
                return {"ok": True, "drill": "suspended",
                        "detail": f"the player explored off the drill line at {fen}; the drill is at "
                                  f"{drill.current.get('fen')} — invite them back to continue it, or "
                                  f"retire it deliberately (set up a new position) if they've moved on."}
            # (a) an UNRELATED position — the coach set up something new, so the walker is genuinely
            # stale. Retire it and treat this move as freeform on the real board (never adjudicate a
            # correct move against a dead position).
            self._reset_drill_line(fen or (self.store._last_board or {}).get("fen"))
            drill = None
        if drill is None or drill.finished:
            self.store.set_input({"kind": "move", "uci": uci, **({"fen": fen} if fen else {})})
            # FREEFORM play is still real state: record it (board + move line, both persisted) so a
            # played-out line survives an app relaunch instead of evaporating with the process.
            pre = fen or (self.store._last_board or {}).get("fen")
            if pre:
                try:
                    board = Board(pre)
                    san = board.san(uci)
                    after = board.apply(uci).fen
                    plies = list(self.store._history) or [
                        {"n": 0, "san": None, "uci": None, "fen": pre}]
                    plies.append({"n": len(plies), "san": san, "uci": uci, "fen": after})
                    self.store.append_beats([_you_beat(
                        f"Played {san}" + (f" — takes the {c}" if (c := _captured_piece(pre, uci)) else ""),
                        move=san, fen=after)])
                    # History BEFORE board: the app anchors orientation on history.first, so writing the
                    # board first (a black-to-move position) would flip the board upside-down for a frame
                    # on the first move (empty history) before history corrects it.
                    self.store.write_history(plies)
                    self.store.write_board(after)
                except Exception:
                    pass   # an unreplayable move (stale fen) — input already recorded for the coach
            return {"ok": True, "drill": False}
        pre_fen = drill.current.get("fen") or fen
        # Name the move (SAN) before adjudicating, so the drill records it in the move line.
        try:
            played = Board(pre_fen).san(uci)
        except Exception:
            played = uci
        r = drill.play(uci, played)
        # The player's turn was a board move, not typed text — echo it as "Played <move>" so the
        # beats column stays a conversation. Ground the captured piece here (SAN doesn't name it, so
        # the coach otherwise guesses — usually "pawn"): "Played Qxf2+ — takes the bishop".
        captured = _captured_piece(pre_fen, uci) if pre_fen else None
        echo = f"Played {played}" + (f" — takes the {captured}" if captured else "")
        # The move bubble carries the verdict as a BADGE (green check / red cross) — no separate
        # feedback beat. The LLM (coach_move) voices the why + what's next; the badge is the instant
        # right/wrong signal that used to be a canned beat. `move`/`fen` make the move chip clickable
        # (snap the board to the position right after it).
        try:
            after_player_fen = Board(pre_fen).apply(uci).fen if pre_fen else None
        except Exception:
            after_player_fen = None
        self.store.append_beats([_you_beat(echo, correct=r["correct"], move=played, fen=after_player_fen)])
        # History BEFORE board (orientation anchors on history.first — board-first would flip the board
        # for a frame on the opening move).
        self.store.write_history(r["plies"])
        if r.get("board"):
            self.store.write_board(r["board"])
        # Legacy canned verdict beat — only when the caller isn't LLM-coaching this move (push_feedback).
        if push_feedback:
            self.store.append_beats([r["feedback"]])
            if r.get("extra"):
                self.store.append_beats([r["extra"]])
            # Legacy deterministic poisoned-line nudge. On the LLM path (push_feedback=False) the coach
            # surfaces the trap in its own voice on solve (coach_move), so no template beat here.
            if r["finished"] and (self.store._last_tree or {}).get("has_poisoned_line"):
                self.store.append_beats([{
                    "kind": "say", "stops": False, "tone": "teach",
                    "segments": [{"text": "There's a poisoned line in this position — a move that looks "
                                          "winning but loses. Ask me to show you the line you sidestepped."}],
                }])
        self.store.set_input(r["event"])
        # DETERMINISTIC loop-close: the drill is solved and we know exactly what happened — bank the
        # mastery observation + drop a one-shot 'concluded' note (read_input hands it to the coach once).
        if r["finished"]:
            self._close_drill(drill)
        # The Maia move-meaning ('a common blunder at your level' etc.) is now returned, NOT pushed as
        # its own local beat — the LLM (coach_move) folds it into its grounded voice.
        meaning = self._move_meaning(pre_fen, uci)
        if push_feedback and meaning:
            self.store.append_beats([{"kind": "say", "stops": False, "tone": "teach",
                                      "segments": [{"text": meaning}]}])
        self.store.set_drill_state(drill.to_state())   # persist walker progress (P2c) — survives restart
        return {"ok": True, "drill": True, "correct": r["correct"], "finished": r["finished"],
                "meaning": meaning}

    def _close_drill(self, drill) -> None:
        """Close the loop on a solved drill DETERMINISTICALLY — no LLM. Banks the mastery observation
        for the concept the drill was armed with (grade derived from the wrong-attempt count on the
        walker), and records a one-shot 'concluded' note that read_input hands the coach exactly once —
        so a player who immediately asks something else doesn't get the finished drill re-praised from
        memory. A missing/unknown concept still closes cleanly (the note lands; only mastery is skipped)."""
        tree = self.store._last_tree or {}
        concept_id = tree.get("concept_id")
        wrong = int((drill.counters if drill is not None else {}).get("wrong", 0))
        # Clean solve = 1.0; each wrong attempt costs 0.25, floored so a struggled-but-solved still counts.
        quality = 1.0 if wrong == 0 else max(0.25, 1.0 - 0.25 * wrong)
        banked = None
        if concept_id and self.mastery is not None:
            try:
                self.mastery.record({"type": "recall", "concept": concept_id,
                                     "quality": quality, "resolved": True, "note": "drill solved"})
                self.store.add_banked(concept_id)
                banked = concept_id
            except ValueError:
                pass   # a bad/unknown concept_id must never break the move — close without the bank
        # `concept` reflects what was ACTUALLY banked (None if the id was invalid/absent), so the coach
        # never claims "banked <x>" when the observation was silently skipped.
        self.store.set_drill_close({"result": "solved", "concept": banked, "quality": quality})

    def _move_meaning(self, fen, uci):
        """Player-facing, Maia-grounded meaning of the played move (or None) — reuses assess_move."""
        if fen is None:
            return None
        try:
            res = self.assess_move(fen, uci)
            return res.get("meaning") if isinstance(res, dict) else None
        except Exception:
            return None

    # -- activity stack (P5): push a rabbit-hole, pop back --------------------
    def _rehydrate_drill(self) -> None:
        """Rebuild the drill walker to match the CURRENT top frame's persisted drill_state — the walker
        follows the live workspace across a push/pop, so a rabbit-hole's drill and the parent's drill
        don't bleed into each other."""
        st = self.store.drill_state
        tree = self.store._last_tree
        if st and tree:
            from .drill import DrillState
            # The move line has ONE home — the document's history — handed to the walker here, not a
            # second copy inside drill_state (which would drift the moment history changed without it).
            self._drill = DrillState.restore(tree, st, line=list(self.store._history or []))
        else:
            self._drill = None

    @_guarded
    def push_activity(self, kind: str = "conversation", seed: dict | None = None) -> dict:
        """Open a rabbit-hole: push a fresh activity frame (design §7). The current workspace (board,
        line, drill) is frozen on the stack and a new empty one becomes live — the SAME conversation
        continues (beats + the Socratic gate are session-level). Use for a "let me show you" that jumps
        to an UNRELATED position; a RELATED line is a variation, not a push. Pop with pop_activity when
        done to restore exactly what the player was looking at."""
        self.store.push_activity(kind, seed=seed)
        self._rehydrate_drill()
        self._signal_board_change()
        return {"ok": True, "depth": self.store.frame_depth, "kind": self.store.top_kind,
                "note": "Pushed a new activity. The player's previous board is preserved and restored "
                        "on pop_activity — don't rebuild it by hand."}

    @_guarded
    def pop_activity(self) -> dict:
        """Close the current rabbit-hole: pop the top activity and RESTORE the parent frame's frozen
        workspace exactly (board, line, drill, cursor). Refused at the base (nothing to pop)."""
        if not self.store.pop_activity():
            return R.error("cannot_pop",
                           "already at the base activity — there's nothing to pop back to.")
        self._rehydrate_drill()
        self._signal_board_change()
        return {"ok": True, "depth": self.store.frame_depth, "kind": self.store.top_kind}

    @_guarded
    def reset_to_start(self) -> dict:
        """App-initiated 'back to the previous concept' when there's no rabbit-hole to pop (base frame):
        reset the current frame to the standard starting position — retire any drill, clear the line, and
        repaint the start board. Not a coach tool; the app calls it via /reset when the activity stack is
        empty. Signals a board change so the coach re-grounds instead of coaching the old position."""
        self._reset_drill_line(_START_FEN)   # retire drill + reset navigator/history to the start ply
        self.store.write_board(_START_FEN)   # repaint the visible board to the standard start
        self._signal_board_change()
        return {"ok": True, "fen": _START_FEN}

    def _signal_board_change(self) -> None:
        """Flag a position change in the input mailbox so the coach's NEXT read_input sees it. An
        activity push/pop (e.g. the app's "Back to where we were") moves the board WITHOUT a word to
        the coach; without this the coach keeps answering from its memory of the old board. read_input
        already reports the current `board_fen` — this just makes the change unmissable (the defensive
        backstop to the contract's "re-ground on board_fen" rule)."""
        self.store.set_input({"kind": "none", "board_changed": True})

    # -- mastery tools (M6) -----------------------------------------------
    @_guarded
    def record_observation(self, concept_id, type, quality, *,
                           misconception=None, resolved=False, note=None) -> dict:
        """One honest observation per resolved exchange. `type ∈ probe|recall|bug`.
        Returns `{ok, mastery:<derived state for concept_id>}`."""
        if self.mastery is None:
            return R.error("mastery_unavailable", "no mastery engine in this session")
        try:
            _ev, learner = self.mastery.record({
                "type": type, "concept": concept_id, "quality": quality,
                "misconception": misconception,
                "resolved": True if resolved else None, "note": note,
            })
        except ValueError as e:
            return R.error("invalid_observation", str(e))
        self.store.add_banked(concept_id)   # this session touched this concept → conclude_session lists it
        return {"ok": True, "mastery": learner["concepts"].get(concept_id)}

    @_guarded
    def conclude_session(self) -> dict:
        """Close the loop: mark the session **complete** and hand back what was BANKED this session —
        each mastery concept you recorded, with its current derived mastery. Call it when the teaching
        arc is done (the player has resolved it, or says they're done); then push ONE closing beat that
        names what they banked (e.g. "Banked: removing the defender ✓") so the loop visibly closes."""
        banked = []
        for cid in self.store.banked:
            m = self.mastery.mastery(cid) if self.mastery is not None else None
            banked.append({"concept": cid, "mastery": (m or {}).get("mastery") if m else None})
        self.store.set_status("complete")
        return {"ok": True, "status": "complete", "banked": banked,
                "note": "Session concluded. Push one closing beat naming what the player banked — do not "
                        "leave the loop open. (Nothing is refused after this; a new position starts fresh.)"}

    @_guarded
    def undo_move(self) -> dict:
        """Snap the board BACK one move — undo the player's last move in a freeform coaching position
        and re-pose. Use it when the player's move was wrong and you want them to try again: 'that's
        not it — let's undo and look again.' The board + navigator return to the position before the
        move; then push a beat re-asking. (In a live drill you don't need this — the drill snaps back
        wrong moves for you.)"""
        if self._drill is not None and not self._drill.finished:
            return R.error("in_drill",
                           "a live drill already snaps back wrong moves — no manual undo needed here.")
        # Resolve the move to undo from the ONE canonical position — the session board (`board_view`) —
        # never from a stale `view.line`+cursor copy (which the coach may have painted past, and whose
        # fixed cursor makes two undos in a row re-undo the SAME move). Because the target is derived
        # from the board each call, consecutive undos walk back correctly: each re-reads the new board.
        current = self.store.board_view
        if not current:
            return R.error("nothing_to_undo", "there's no position on the board to undo from.")
        cur_n = _norm_fen(current)
        # Mainline: the board sits at the tip of the MCP-authored history → pop it (board + history).
        hist = list(self.store._history)
        if len(hist) >= 2 and _norm_fen((hist[-1] or {}).get("fen", "")) == cur_n:
            undone = hist.pop()
            prev = hist[-1]
            self.store.write_board(prev.get("fen"))
            self.store.write_history(hist)
            return {"ok": True, "fen": prev.get("fen"), "undone": undone.get("san"),
                    "note": "Undone — the board snapped back to before that move. Now re-pose the question in a beat."}
        # Otherwise the board is on a SIDELINE (or navigated history): the app's reported line is the
        # only record of it (variations are UI-authored), so LOCATE the current board within that line
        # and step back one ply WITHIN it. write_board keeps the view because the target is on the
        # view's own line (see StateStore.write_board), so a second undo re-resolves and steps again.
        line = (self.store.view or {}).get("line") or []
        idx = next((i for i, m in enumerate(line)
                    if _norm_fen((m or {}).get("fen", "")) == cur_n), None)
        if idx is not None and idx > 0 and (line[idx - 1] or {}).get("fen"):
            prev = line[idx - 1]["fen"]
            self.store.write_board(prev)
            return {"ok": True, "fen": prev, "undone": line[idx].get("san"),
                    "note": "Undone — snapped back to before that move in the line the player is on. Re-pose the question in a beat."}
        return R.error("nothing_to_undo",
                       "nothing to undo — the board is at the start of the line, or not on a known move line.")

    @_guarded
    @_scoped("get_mastery")
    def get_mastery(self, what="overview") -> dict:
        """`what ∈ overview | due_reviews | concept:<id>`."""
        if self.mastery is None:
            return R.error("mastery_unavailable", "no mastery engine in this session")
        if what == "overview":
            return self.mastery.overview()
        if what == "due_reviews":
            return {"due_reviews": self.mastery.due_reviews()}
        if what.startswith("concept:"):
            cid = what.split(":", 1)[1]
            derived = self.mastery.mastery(cid)
            if derived is None:
                return {"concept": cid, "mastery": None, "status": "started",
                        "observations": 0}
            return {"concept": cid, **derived}
        return R.error("bad_args",
                       f"unknown selector {what!r} — use overview | due_reviews | concept:<id>")

    @_guarded
    def record_pedagogy(self, kind, value, *, desc=None) -> dict:
        """`kind ∈ lands | falls_flat | add_pattern` (add_pattern needs `desc`)."""
        if self.mastery is None:
            return R.error("mastery_unavailable", "no mastery engine in this session")
        internal = {"lands": "what_lands", "falls_flat": "what_falls_flat",
                    "add_pattern": "bug_pattern"}.get(kind)
        if internal is None:
            return R.error("bad_args",
                           f"unknown pedagogy kind {kind!r} — use lands | falls_flat | add_pattern")
        try:
            self.mastery.record_pedagogy(internal, value, desc=desc)
        except ValueError as e:
            return R.error("invalid_pedagogy", str(e))
        return {"ok": True}
