"""Session state — the single-writer bridge to the app (session-state-design.md).

`StateStore` owns ONE canonical Session Document per session (the `session_document` blob in
`lucena.db`, plus the `beat` sidecar) and streams it to the app over SSE — a snapshot on connect,
deltas after. The retired per-fact JSON files (`board.json`/`beats.json`/`tree.json`/`view.json`)
are gone; only `heartbeat.json` (liveness, §8b) and `session.json` (the durable current-session id)
remain as files. Debug the live document with the `/dump` endpoint, not by poking at files.

The monotonic document `version` is the durability/ordering key; the per-channel `*_seq`s persist
alongside it only as the wire projection the app still consumes (residual debt — collapse toward the
one `version`, never add to it). Not thread-safe beyond publish: the server serializes every tool
call and app mutation behind one lock; the heartbeat uses its own `seq`.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from lucena_core.board import Board
from .statefile import write_state

SCHEMA = 1

# An activity's outcome, shown on its card in the base conversation. "" while it is still live.
ATTEMPTED_STATUS = "attempted"
SOLVED_STATUS = "solved"

# The bound chat session for the current execution context. Per-context, NOT per-process: two
# concurrent turns cannot see each other's value, which is the whole point — a background turn parked
# in the LLM must still write to the chat it started in, whatever anyone else has opened meanwhile.
# Bind it at entry points via `StateStore.bound(sid)`; never assign it directly outside this module.
_current_sid: contextvars.ContextVar[str] = contextvars.ContextVar("lucena_current_sid", default="")

# The authenticated user for the current execution context (the LOGIN session), bound at the same
# entry points as the chat. Separate from `_current_sid` on purpose: a user has many chats, so
# "who is this" and "which chat" are independent facts. None = the pre-accounts / in-memory path,
# which has exactly one implicit user.
_current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lucena_current_user", default=None)
VERSION = "0.1.0"


def _norm_fen(fen: str) -> str:
    """A position's identity ignoring clocks — placement + side + castling + ep (first 4 FEN fields).
    Matches the app's `VariationForest.norm`, so a stored view's mainline prefix can be checked
    against the session's move history without the halfmove/fullmove counters fragmenting it."""
    return " ".join(str(fen).split()[:4])

# Placeholder name a session gets the moment it's created — shown in the rail until Claude's
# ai-title (from the transcript) replaces it.
DEFAULT_SESSION_NAME = "New session"
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# board arrow/highlight styles are semantic names the app resolves against the
# palette — the server never sends hex colors (§5.1).
ARROW_STYLES = ("analysis", "correction", "gold", "ghost")


@dataclass
class _Workspace:
    """The chess a single activity is looking at (design §6.2, state-machine P3) — its board, analysis,
    drill tree/walker, move line and display view (+ the legacy wire seqs). Suspended with its frame on
    a rabbit-hole push and restored on pop (P5); for now there is one, on the base frame."""
    last_board: dict | None = None
    last_analysis: dict | None = None
    last_tree: dict | None = None
    history: list = field(default_factory=list)   # move line (ply 0 = start), for the navigator
    view: dict | None = None                      # the app's full display state (fen+line+tree+cursor)
    drill_state: dict | None = None               # DrillState.to_state() (P2c)
    # The FREEFORM poisoned line (§6.4): a durable {for_fen, moves, meta} — the trap lives on the
    # document from the moment it's detected, NEVER on the transient board. The board's
    # has_poisoned_line/poisoned_line are a pure PROJECTION of this (re-derived on every repaint, so
    # it can't evaporate on a paint the way a board-stored flag did). (A DRILL's trap lives on its
    # tree — that artifact is its own durable home; this slot is the non-drill counterpart.)
    poisoned: dict | None = None
    # The coach's board DECORATIONS for a position (§6.2): the LAST coach paint's
    # {for_fen, arrows, highlights, caption, eval}. Decorations are a property OF a position, so the
    # published board's arrows/highlights/caption/eval are a PROJECTION of this — shown only while the
    # board sits at `for_fen`. That's what lets the app navigate away and BACK without the coach's
    # marks being blanked (the app's own /position report no longer overwrites them with empties).
    decorations: dict | None = None
    board_seq: int = 0
    tree_seq: int = 0
    analysis_seq: int = 0
    history_seq: int = 0
    view_seq: int = 0


@dataclass
class _Frame:
    """One activity on the stack (design §7): its kind + the workspace it looks at + its OWN beats. Each
    activity owns its conversation: the base frame's beats are 'the session's beats'; a pushed puzzle
    activity has its own, so a puzzle can be its own saved conversation (per-activity beats). `kind`:
    `conversation` (base) | `puzzle` | …. Push/pop forks/restores beats along with the workspace."""
    kind: str = "conversation"
    workspace: _Workspace = field(default_factory=_Workspace)
    beats: list = field(default_factory=list)
    beats_seq: int = 0
    # For a saved, re-openable activity (a puzzle): a human title for its card in the parent
    # conversation, and its outcome (`""` while live | `attempted` | `solved`). Base frame leaves both
    # empty. The card beat in the base references this frame by index and mirrors these.
    title: str = ""
    status: str = ""
    # The drill this activity IS — so reopening it can make the drill LIVE again (reactivate + present),
    # not just show a static review. Empty on the base conversation.
    lesson_id: str = ""


@dataclass
class _Live:
    """One session's live state (state-machine P3). Session-LEVEL fields (the one continuous
    conversation `beats`, the durable Socratic `gate`, the monotonic `version`) are held here directly;
    the per-frame *workspace* (board/analysis/drill/line/view) AND its beats live on the activity STACK
    (`activities`, `[0]` = base). The per-frame attributes (`beats`, `last_board`, `board_seq`, …) are
    exposed as delegating properties onto the top frame, so every StateStore accessor stays unchanged.
    One bundle per Claude session, so switching sessions never leaks beats/board across sessions."""
    # The durable Socratic gate (P2b): persisted so a session resumed mid-probe stays LOCKED (§11).
    gate_awaiting: bool = False
    gate_pending: dict | None = None
    # The single monotonic document version (P1): bumped every mutation, persisted, restored on load —
    # the durable ordering/durability key that replaces the per-file seqs (kept as the wire projection).
    version: int = 0
    # Session lifecycle (durable): `active` until the coach concludes it (bank + close the loop), then
    # `complete`. `banked` = the mastery concept ids observed THIS session (for the concluding summary).
    status: str = "active"
    banked: list = field(default_factory=list)
    # Puzzle ids served THIS session (set_puzzle), so "give me another" never repeats one. Durable →
    # survives a mid-session relaunch; reset when a new session starts (a fresh deck).
    served_puzzles: list = field(default_factory=list)
    # A just-solved drill, closed DETERMINISTICALLY on the solving move (mastery banked + verdict beat
    # already posted by play_move). Surfaced to the coach ONCE via read_input so it doesn't re-praise a
    # drill the player has already moved on from, then cleared. Durable → survives a mid-solve relaunch.
    drill_close: dict | None = None
    # The activity LIST; `activities[0]` is the base conversation. `active_idx` points at the frame
    # currently IN VIEW (the one the wire projects). A puzzle is a sibling activity, not a transient
    # stack pop: entering a puzzle appends a frame and moves `active_idx` onto it; going back moves
    # `active_idx` to the base but KEEPS the puzzle frame (its beats/board/variations), so a card in
    # the base conversation can reopen it. (The parked what-if excursion still uses pop_activity.)
    activities: list = field(default_factory=lambda: [_Frame()])
    active_idx: int = 0
    # Per-session RUNTIME state (never persisted — transient). Session-scoped by construction so it can
    # NEVER leak across a switch/new-session: the app→coach input mailbox. (The "current board fen" is NOT
    # a separate field — it derives from the one session board, `last_board`; see StateStore.board_view.)
    input: dict | None = None
    # Post-move fens the coach has ALREADY commented on while walking a variation — so stepping BACKWARD
    # onto or re-visiting a move it already read stays silent (one comment per move landed on). Transient:
    # a fresh session/reconnect starts empty, and re-walking then is a fine re-read.
    walked: set = field(default_factory=set)

    @property
    def top(self) -> _Frame:
        """The activity currently IN VIEW (`active_idx`) — the frame the wire projects. Named `top`
        for historical reasons (it was the stack top); with re-openable activities it's the active
        frame, which is the base while a puzzle sits saved as a sibling."""
        return self.activities[self.active_idx]

    @property
    def ws(self) -> _Workspace:
        return self.top.workspace

    # Beats + beats_seq live on the FRAME (per-activity), delegated here so StateStore accessors and
    # the wire projection stay unchanged — they always read/write the ACTIVE activity's conversation.
    @property
    def beats(self) -> list:
        return self.top.beats

    @beats.setter
    def beats(self, v: list) -> None:
        self.top.beats = v

    @property
    def beats_seq(self) -> int:
        return self.top.beats_seq

    @beats_seq.setter
    def beats_seq(self, v: int) -> None:
        self.top.beats_seq = v

    # -- per-frame attributes delegate to the top workspace (keeps StateStore accessors unchanged) --
    @property
    def last_board(self): return self.ws.last_board
    @last_board.setter
    def last_board(self, v): self.ws.last_board = v
    @property
    def last_analysis(self): return self.ws.last_analysis
    @last_analysis.setter
    def last_analysis(self, v): self.ws.last_analysis = v
    @property
    def last_tree(self): return self.ws.last_tree
    @last_tree.setter
    def last_tree(self, v): self.ws.last_tree = v
    @property
    def history(self): return self.ws.history
    @history.setter
    def history(self, v): self.ws.history = v
    @property
    def view(self): return self.ws.view
    @view.setter
    def view(self, v): self.ws.view = v
    @property
    def drill_state(self): return self.ws.drill_state
    @drill_state.setter
    def drill_state(self, v): self.ws.drill_state = v
    @property
    def poisoned(self): return self.ws.poisoned
    @poisoned.setter
    def poisoned(self, v): self.ws.poisoned = v
    @property
    def decorations(self): return self.ws.decorations
    @decorations.setter
    def decorations(self, v): self.ws.decorations = v
    @property
    def board_seq(self): return self.ws.board_seq
    @board_seq.setter
    def board_seq(self, v): self.ws.board_seq = v
    @property
    def tree_seq(self): return self.ws.tree_seq
    @tree_seq.setter
    def tree_seq(self, v): self.ws.tree_seq = v
    @property
    def analysis_seq(self): return self.ws.analysis_seq
    @analysis_seq.setter
    def analysis_seq(self, v): self.ws.analysis_seq = v
    @property
    def history_seq(self): return self.ws.history_seq
    @history_seq.setter
    def history_seq(self, v): self.ws.history_seq = v
    @property
    def view_seq(self): return self.ws.view_seq
    @view_seq.setter
    def view_seq(self, v): self.ws.view_seq = v


def _workspace_to_dict(w: _Workspace) -> dict:
    return {
        "last_board": w.last_board, "last_analysis": w.last_analysis, "last_tree": w.last_tree,
        "history": w.history, "view": w.view, "drill_state": w.drill_state, "poisoned": w.poisoned,
        "decorations": w.decorations,
        "board_seq": w.board_seq, "tree_seq": w.tree_seq, "analysis_seq": w.analysis_seq,
        "history_seq": w.history_seq, "view_seq": w.view_seq,
    }


def _workspace_from_dict(d: dict | None) -> _Workspace:
    d = d or {}
    return _Workspace(
        last_board=d.get("last_board"), last_analysis=d.get("last_analysis"),
        last_tree=d.get("last_tree"), history=d.get("history") or [], view=d.get("view"),
        drill_state=d.get("drill_state"), poisoned=d.get("poisoned"),
        decorations=d.get("decorations"),
        board_seq=d.get("board_seq", 0), tree_seq=d.get("tree_seq", 0),
        analysis_seq=d.get("analysis_seq", 0), history_seq=d.get("history_seq", 0),
        view_seq=d.get("view_seq", 0))


def _frame_to_dict(f: _Frame) -> dict:
    return {"kind": f.kind, "workspace": _workspace_to_dict(f.workspace),
            "beats": list(f.beats), "beats_seq": f.beats_seq,
            "title": f.title, "status": f.status, "lesson_id": f.lesson_id}


def _frame_from_dict(d: dict | None) -> _Frame:
    d = d or {}
    return _Frame(kind=d.get("kind", "conversation"),
                  workspace=_workspace_from_dict(d.get("workspace")),
                  beats=list(d.get("beats") or []), beats_seq=d.get("beats_seq", 0),
                  title=d.get("title") or "", status=d.get("status") or "",
                  lesson_id=d.get("lesson_id") or "")


@dataclass
class Subscription:
    """One reader watching one chat — a queue, the loop that owns it, and which chat it is on.

    `epoch` bumps on every retarget. Events are stamped with the (chat, epoch) they were enqueued
    for, and `get` drops anything that no longer matches, so a socket that switches chats cannot be
    handed deltas from the chat it just left. The map lock alone cannot do this: by the time a switch
    happens, old-chat events may already be sitting in the queue, or their `call_soon_threadsafe`
    callbacks may already be scheduled — neither can be unsent.
    """

    queue: asyncio.Queue
    sid: str
    loop: object
    epoch: int = 0
    # Held by the reader across "check the stamp, then send", and by a retarget before it mutates
    # sid/epoch. Without it there is a third stale window the epoch alone cannot close: the reader can
    # dequeue a still-matching event, yield at the send, have a retarget land in that gap, and then
    # deliver the old chat's event to a socket that has already switched. The invariant this buys:
    # once retarget returns, no event dequeued under the old (sid, epoch) can still be sent.
    gate: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def next_event(self) -> tuple:
        """The next event still live for this subscription, as `(sid, epoch, channel, payload)`.

        Returns the stamp too, so the reader can RE-CHECK it under `gate` immediately before sending
        — the check here is necessary but not sufficient on its own.
        """
        while True:
            ev = await self.queue.get()
            if ev[0] == self.sid and ev[1] == self.epoch:
                return ev
            # else: stale — queued for a chat this reader has since left. Drop it.

    def is_live(self, sid: str, epoch: int) -> bool:
        """Does this stamp still match? Re-checked under `gate` after dequeue, before send."""
        return sid == self.sid and epoch == self.epoch

    async def get(self) -> tuple:
        """Convenience for non-socket readers: the next live event as `(channel, payload)`."""
        _sid, _epoch, channel, payload = await self.next_event()
        return channel, payload


@dataclass
class StateStore:
    """Owns the Lucena directory's state files and the per-session live view."""

    home: str
    db: object | None = None                         # durable store (db.DB); None → in-memory only
    # PROCESS-level resources only (NOT session state): the heartbeat/session-file write counters, the
    # SSE subscriber set, and the event loop. Everything that is per-coaching-session state lives in the
    # `_Live` bundle below, partitioned by session id — there are no session-state store globals.
    _hb_seq: int = 0
    _session_seq: int = 0
    # chat_session_id -> {queue: loop}. Per-CHAT, so a publish reaches only the sockets watching that
    # chat. The loop is stored PER SUBSCRIBER, not once on the store: `_publish` schedules queue puts
    # with call_soon_threadsafe, and a single store-wide loop is silently wrong the moment more than
    # one loop is involved (the last subscriber would win and earlier sockets would never wake).
    _subscribers: dict = field(default_factory=dict)
    # The session state is partitioned by chat session id; the app's board / beats / analysis / drill /
    # input / gate are ALWAYS the bound chat's. Every per-session field (`_beats`, `_last_board`,
    # `_input`, seqs, …) is a proxy property onto `_cur`, so nothing leaks across chats.
    _live: dict = field(default_factory=dict)        # chat_session_id -> _Live
    # Guards first-touch of `_live[sid]`: the check-then-insert in `_cur` does DB I/O in `_load_live`,
    # so two threads racing a cold chat would both load it.
    _live_lock: object = field(default_factory=threading.RLock)
    # When true, resolving an unbound cursor raises instead of silently using "". Tests turn this on so
    # a publish/write that escapes its chat is loud rather than landing in `_live[""]`.
    _strict: bool = False
    # Serializes the version-bump + publish so concurrent writers (an inline move in a worker thread,
    # the coach's background beat on the event loop, a still-running arm task) never race the monotonic
    # `version` — a stale/duplicate version makes the app's ordered-delta guard DROP the event (a
    # never-cleared "thinking" status or a missed board update = a frozen screen). Re-entrant: nested
    # persist/publish calls under one mutator are fine.
    _wlock: object = field(default_factory=threading.RLock)

    # -- the bound chat (the cursor) ---------------------------------------------------------------
    # `_current` is NOT stored on the store. It lives in a ContextVar, so two concurrent turns cannot
    # see each other's value: the cursor is per execution context, not per process. It propagates
    # across asyncio.create_task (context is copied at creation) and asyncio.to_thread (which does
    # copy_context().run), which is every path the server actually uses. It does NOT cross a raw
    # threading.Thread — such callers must be passed a session_id explicitly.
    @property
    def _current(self) -> str:
        return self.current_sid

    @property
    def current_sid(self) -> str:
        sid = _current_sid.get()
        if not sid and self._strict:
            raise RuntimeError(
                "no chat session is bound in this context: a read/write/publish escaped its chat. "
                "Bind one with `with store.bound(sid):` at the entry point."
            )
        return sid

    @property
    def current_user(self) -> str | None:
        """The authenticated user for this context (the login session). None pre-accounts."""
        return _current_user.get()

    @contextmanager
    def bound(self, sid: str, *, user_id: str | None = None):
        """Bind `sid` as the current chat for this execution context (and anything it spawns).

        `user_id` binds the login session too; omit it to keep whatever user is already bound (the
        common case — the entry point binds the user once, then binds chats under it).
        """
        token = _current_sid.set(sid)
        utoken = _current_user.set(user_id) if user_id is not None else None
        try:
            yield self
        finally:
            _current_sid.reset(token)
            if utoken is not None:
                _current_user.reset(utoken)

    @contextmanager
    def as_user(self, user_id: str | None):
        """Bind the login session for this context, without touching the chat cursor."""
        token = _current_user.set(user_id)
        try:
            yield self
        finally:
            _current_user.reset(token)

    # -- Lessons (coach mode) — per-user PROGRESS + shared CONTENT, both file-backed (LLD §7) ------
    # Lazy-built so the store has no import cycle with `coaching` at module load, and so plain
    # in-memory/test stores that never touch lessons pay nothing. Keyed off `current_user` (None =
    # anonymous), NOT the chat — a lesson's solved/open state is per-PERSON and cross-session.
    @property
    def _lesson_store(self):
        cache = getattr(self, "_lesson_store_cache", None)
        if cache is None:
            from ..coaching.lesson_store import LessonStore
            cache = LessonStore(self.home)
            self._lesson_store_cache = cache
        return cache

    @property
    def library(self):
        cache = getattr(self, "_library_cache", None)
        if cache is None:
            from ..coaching.library import Library
            cache = Library(self.home)
            self._library_cache = cache
        return cache

    def _pair(self, progress):
        """A full Lesson = shared spec (library) + this user's progress. None if the spec is gone."""
        if progress is None:
            return None
        spec = self.library.get(progress.lesson_id)
        if spec is None:
            return None
        from ..coaching.lesson import Lesson
        return Lesson(spec=spec, progress=progress)

    def active_lesson(self):
        """The live coach lesson for THIS CHAT: state==active AND meta is None AND chat_id==current
        chat (resolve_mode). Per-chat, so a lesson active in another chat doesn't force coach mode
        here. None → freeform. Paired with its library spec; a dangling progress reads None."""
        return self._pair(self._lesson_store.active(self.current_user, self.current_sid))

    def suspended_lesson(self):
        """The SUSPENDED drill for this chat (parked by a what-if), paired to its spec — so a move can
        resume it. None → nothing to resume."""
        return self._pair(self._lesson_store.suspended(self.current_user, self.current_sid))

    def activate_lesson(self, lesson_id: str) -> None:
        """Bind a lesson active to the CURRENT chat (coach entry)."""
        self._lesson_store.activate(self.current_user, lesson_id, self.current_sid)
        self.publish_mode()

    def get_lesson(self, lesson_id: str):
        return self._pair(self._lesson_store.get(self.current_user, lesson_id))

    def open_lessons(self) -> list:
        """Resumable items for 'pick up an open item' (state==open, meta None), paired to specs."""
        return [l for p in self._lesson_store.open_items(self.current_user)
                if (l := self._pair(p)) is not None]

    def save_lesson_progress(self, progress) -> None:
        self._lesson_store.put(self.current_user, progress)
        self.publish_mode()     # a solve sets meta=solved here, which flips the chat back to freeform

    def set_lesson_state(self, lesson_id: str, state: str) -> None:
        self._lesson_store.set_state(self.current_user, lesson_id, state)
        self.publish_mode()

    def save_lesson_spec(self, spec) -> None:
        self.library.put(spec)

    def get_library_spec(self, lesson_id: str):
        """A shared content spec by id (cache hit for a paste-time-computed puzzle). No progress."""
        return self.library.get(lesson_id)

    def lesson_solved(self, lesson_id: str) -> bool:
        return self._lesson_store.is_solved(self.current_user, lesson_id)

    @property
    def _cur(self) -> _Live:
        return self._live_for(self.current_sid)

    def _live_for(self, sid: str) -> _Live:
        with self._live_lock:
            if sid not in self._live:
                self._live[sid] = self._load_live(sid)
            return self._live[sid]

    # -- the last per-session runtime fields: input mailbox, /position fen, poisoned-line latch --
    @property
    def _input(self): return self._cur.input
    @_input.setter
    def _input(self, v): self._cur.input = v

    def _load_live(self, sid: str) -> _Live:
        """A session's live bundle — restored from the DB if we have one, else fresh. Prefers the
        canonical document blob (P1); falls back to the decomposed legacy columns for a pre-P1 session
        that has no blob yet (compose-on-load — the next flush writes its blob)."""
        if self.db is None or not sid:
            return _Live()
        legacy = self.db.load_view(sid)                       # beats (the sidecar) + legacy columns
        blob = self.db.load_document(sid) if hasattr(self.db, "load_document") else None
        if blob is not None:                                  # blob is the source of truth
            d = blob["document"]
            if "activities" in d:                             # P3+ frame shape
                frames = [_frame_from_dict(f) for f in d["activities"]] or [_Frame()]
                # Beats now persist ON the frame. Back-compat: a pre-per-activity blob has frames with no
                # beats and the conversation still in the sidecar → seed the BASE frame from the sidecar.
                if not frames[0].beats and legacy.get("beats"):
                    frames[0].beats = legacy["beats"]
                    frames[0].beats_seq = d.get("beats_seq", 0)
                gate = d.get("gate") or {}
                active_idx = d.get("active_idx", 0)
                if not (0 <= active_idx < len(frames)):        # a corrupt/stale pointer → the base
                    active_idx = 0
                return _Live(gate_awaiting=bool(gate.get("awaiting")), gate_pending=gate.get("pending"),
                             status=d.get("status", "active"), banked=d.get("banked") or [],
                             served_puzzles=d.get("served_puzzles") or [],
                             drill_close=d.get("drill_close"),
                             version=blob["version"], activities=frames, active_idx=active_idx)
            # P1/P2 flat blob → wrap the flat workspace fields into one base frame (compose-on-load)
            ws = _Workspace(
                last_board=d.get("last_board"), last_analysis=d.get("last_analysis"),
                last_tree=d.get("last_tree"), history=d.get("history") or [], view=d.get("view"),
                drill_state=d.get("drill_state"),
                board_seq=d.get("board_seq", 0), tree_seq=d.get("tree_seq", 0),
                analysis_seq=d.get("analysis_seq", 0), history_seq=d.get("history_seq", 0),
                view_seq=d.get("view_seq", 0))
            return _Live(gate_awaiting=bool(d.get("gate_awaiting")), gate_pending=d.get("gate_pending"),
                         version=blob["version"],
                         activities=[_Frame(workspace=ws, beats=legacy["beats"],
                                            beats_seq=d.get("beats_seq", 0))])
        view = legacy.get("view")                             # compose from legacy columns (pre-P1)
        ws = _Workspace(last_board=legacy["last_board"], last_analysis=legacy["last_analysis"],
                        last_tree=legacy["last_tree"], history=legacy.get("history") or [], view=view,
                        board_seq=legacy["board_seq"], tree_seq=legacy["tree_seq"],
                        analysis_seq=legacy["analysis_seq"], view_seq=(view or {}).get("seq", 0))
        return _Live(activities=[_Frame(workspace=ws, beats=legacy["beats"],
                                        beats_seq=legacy["beats_seq"])])

    def _document_dict(self, c: _Live) -> dict:
        """The canonical Session Document (design §6 shape, P3): session-level fields + the activity
        STACK, each frame carrying its workspace. Everything except beats (the sidecar)."""
        return {
            "version": c.version,
            "beats_seq": c.beats_seq,
            "status": c.status, "banked": c.banked,
            "served_puzzles": c.served_puzzles,
            "drill_close": c.drill_close,
            "gate": {"awaiting": c.gate_awaiting, "pending": c.gate_pending},
            "activities": [_frame_to_dict(f) for f in c.activities],
            "active_idx": c.active_idx,
        }

    def dump(self) -> dict:
        """A full snapshot of the current session document (design §5: debuggability via a `dump`
        command, not the retired live JSON files) — the document blob shape + beats + the session id."""
        c = self._cur
        return {**self._document_dict(c), "beats": list(c.beats), "session": self._current or None}

    def _persist_view(self) -> None:
        """Flush the current session document (P6: the ONE canonical blob — no more JSON files, no more
        decomposed columns). Bumps the monotonic `version` (in memory always, so it advances even with
        no DB), then persists the whole document. Beats ride ON their activity frame (per-activity
        conversation), so `_save_activities` persists them in the SAME transaction as the document —
        `beats_seq` and the beat rows never disagree."""
        with self._wlock:   # atomic bump (int += is read-modify-write; concurrent writers must not tear it)
            c = self._cur
            c.version += 1
            if self.db is None or not self._current:
                return
            self.db.save_document(self._current, self._document_dict(c), c.version)

    @property
    def _beats(self): return self._cur.beats
    @_beats.setter
    def _beats(self, v): self._cur.beats = v

    @property
    def _last_board(self): return self._cur.last_board
    @_last_board.setter
    def _last_board(self, v): self._cur.last_board = v

    @property
    def _last_analysis(self): return self._cur.last_analysis
    @_last_analysis.setter
    def _last_analysis(self, v): self._cur.last_analysis = v

    @property
    def _last_tree(self): return self._cur.last_tree
    @_last_tree.setter
    def _last_tree(self, v): self._cur.last_tree = v

    @property
    def _board_seq(self): return self._cur.board_seq
    @_board_seq.setter
    def _board_seq(self, v): self._cur.board_seq = v

    @property
    def _beats_seq(self): return self._cur.beats_seq
    @_beats_seq.setter
    def _beats_seq(self, v): self._cur.beats_seq = v

    @property
    def _tree_seq(self): return self._cur.tree_seq
    @_tree_seq.setter
    def _tree_seq(self, v): self._cur.tree_seq = v

    @property
    def _analysis_seq(self): return self._cur.analysis_seq
    @_analysis_seq.setter
    def _analysis_seq(self, v): self._cur.analysis_seq = v

    @property
    def _history(self): return self._cur.history
    @_history.setter
    def _history(self, v): self._cur.history = v

    @property
    def _history_seq(self): return self._cur.history_seq
    @_history_seq.setter
    def _history_seq(self, v): self._cur.history_seq = v

    @property
    def _view(self): return self._cur.view
    @_view.setter
    def _view(self, v): self._cur.view = v

    @property
    def _version(self): return self._cur.version
    @_version.setter
    def _version(self, v): self._cur.version = v

    @property
    def _gate_awaiting(self) -> bool: return self._cur.gate_awaiting
    @property
    def _gate_pending(self): return self._cur.gate_pending

    @property
    def drill_state(self) -> dict | None:
        """The persisted drill-walker state (DrillState.to_state()), or None if no drill is armed."""
        return self._cur.drill_state

    def set_drill_state(self, state: dict | None) -> None:
        """Persist the drill walker's resumable state (P2c). `None` clears it (drill retired). Written
        after every adjudicated move so a restart resumes the walk exactly (incl. backtrack progress)."""
        self._cur.drill_state = state
        self._persist_view()

    def set_gate(self, awaiting: bool, pending: dict | None = None) -> None:
        """Set the durable Socratic gate and persist it (P2b). Idempotent — a no-op (no flush, no
        version bump) when nothing changes, so the common case (read_input clearing an already-clear
        gate every turn) doesn't churn the document. Persisting the gate is what lets a session resume
        mid-probe still locked."""
        c = self._cur
        if bool(awaiting) == c.gate_awaiting and pending == c.gate_pending:
            return
        c.gate_awaiting = bool(awaiting)
        c.gate_pending = pending
        self._persist_view()

    # -- session lifecycle: bank a concept + conclude (close the loop) --------------------------------
    @property
    def status(self) -> str:
        return self._cur.status

    def set_status(self, status: str) -> None:
        """Set + persist the session lifecycle status (`active` | `complete`); also denormalise it to
        the rail (session table) so a concluded session shows as done."""
        self._cur.status = status
        if self.db is not None and self._current:
            self.db.set_session_status(self._current, status)
        self._persist_view()

    @property
    def banked(self) -> list:
        return list(self._cur.banked)

    def add_banked(self, concept_id: str) -> None:
        """Record that a mastery concept was observed THIS session — the concluding summary lists them."""
        if concept_id and concept_id not in self._cur.banked:
            self._cur.banked.append(concept_id)
            self._persist_view()

    @property
    def served_puzzles(self) -> list:
        return list(self._cur.served_puzzles)

    def mark_puzzle_served(self, puzzle_id: str) -> None:
        """Record that set_puzzle handed out this puzzle THIS session, so "give me another" won't
        repeat it. Durable; idempotent (a re-served id doesn't churn the document version)."""
        if puzzle_id and puzzle_id not in self._cur.served_puzzles:
            self._cur.served_puzzles.append(puzzle_id)
            self._persist_view()

    def set_drill_close(self, info: dict | None) -> None:
        """Note that a drill just closed deterministically (mastery banked + verdict beat posted on the
        solving move) so read_input can tell the coach ONCE. Durable → survives a relaunch before the
        coach reads it. Idempotent when unchanged, to avoid churning the document version."""
        if info == self._cur.drill_close:
            return
        self._cur.drill_close = info
        self._persist_view()

    def take_drill_close(self) -> dict | None:
        """Consume the pending drill-close note (read_input surfaces it exactly once), clearing it so the
        coach is told a drill concluded only on the first turn after the solve."""
        info = self._cur.drill_close
        if info is not None:
            self._cur.drill_close = None
            self._persist_view()
        return info

    # -- activity STACK (P5): push a rabbit-hole, pop back, set the base -----------------------------
    @property
    def frame_depth(self) -> int:
        """How many activities are on the stack (1 = just the base)."""
        return len(self._cur.activities)

    @property
    def top_kind(self) -> str:
        """The kind of the active (in-view) activity — the session's current surface."""
        return self._cur.top.kind

    @property
    def active_idx(self) -> int:
        """Which activity is currently in view (0 = base conversation)."""
        return self._cur.active_idx

    def _republish(self) -> None:
        """Re-render the CURRENT frame's workspace after an activity switch (push/open/finish/set_base).

        We do NOT send a hard `reset` here: `reset` nulls the board on the client, which FLICKERS it
        (blank → repaint) on every activity transition — and worse, a freshly PUSHED frame has no board
        yet (the drill's `present` writes it a beat later), so the board would sit blank in between. The
        snapshot fully re-specifies board/beats/history/view/activity/mode, so re-sending it REPLACES the
        old frame's projection in place. The only thing a plain replace can't do is CLEAR a channel the
        new frame lacks, so we explicitly clear the drill tree when this frame has none (else the old
        frame's walker would linger and mis-adjudicate)."""
        if self._last_tree is None:
            self._publish("tree_cleared", {})
        for channel, payload in self.snapshot():
            self._publish(channel, payload)

    # Fields a coach-supplied push seed may set. The UI-authored `view` (cursor/line) is NOT here —
    # §8: view is UI-authored, period, so a push can never smuggle in a cursor write — and the wire
    # `*_seq`s start fresh on a new frame, so they're excluded too.
    _SEEDABLE = frozenset({"last_board", "last_analysis", "last_tree", "history", "drill_state",
                           "poisoned", "decorations"})

    def active_frame_lesson_id(self) -> str:
        """The drill id bound to the activity currently in view (empty for the base conversation)."""
        return self._cur.activities[self._cur.active_idx].lesson_id

    def reactivate_lesson(self, lesson_id: str) -> bool:
        """Reopening a puzzle makes its drill LIVE again WITHOUT losing the player's work: bind the lesson
        active to this chat (mode → coach) but keep its bit progress, and keep the frame's board/history/
        view — the line played and the variations explored are PRESERVED, not reset. Clearing `meta` is
        what lets `active_lesson` resolve it again (a solved drill has meta set). Returns True if it
        reactivated a real lesson. Publishes the mode via save_lesson_progress."""
        from ..coaching.lesson import ACTIVE
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            return False
        prog = lesson.progress
        prog.meta = None                                   # so resolve_mode sees it as live again
        prog.state = ACTIVE
        prog.chat_id = self.current_sid                    # bound to THIS chat → resolve_mode = coach
        self.save_lesson_progress(prog)                    # keep bits: the played line stays on the board
        # Re-push the forcing tree so the reopened drill adjudicates on the CLIENT (like a fresh puzzle) —
        # without it the client has no walker and moves fall back to the server. The tree lives on the
        # lesson spec, so it survived the reload.
        tree = next((b.params.get("tree") for b in lesson.spec.bits if (b.params or {}).get("tree")), None)
        if tree:
            self.write_tree(tree)
        return True

    def push_activity(self, kind: str, *, seed: dict | None = None, title: str = "",
                      lesson_id: str = "") -> int:
        """Open a fresh activity (design §7): append a new frame and move `active_idx` onto it — with
        its OWN empty beat stream, so a puzzle activity is its own saved conversation. The frame the
        player was on is frozen in place (NOT popped): it stays a sibling so a card can bring the player
        back here later. The durable Socratic gate + version stay session-level. A coach-supplied `seed`
        may set the starting board/tree/history, but NEVER the UI-authored `view` — that field is
        stripped so a push can't author the cursor (§8). Persists + re-renders. Returns the new idx."""
        safe_seed = {k: v for k, v in (seed or {}).items() if k in self._SEEDABLE}
        self._cur.activities.append(
            _Frame(kind=kind, workspace=_workspace_from_dict(safe_seed), title=title, lesson_id=lesson_id))
        self._cur.active_idx = len(self._cur.activities) - 1
        self._persist_view()
        self._republish()
        return self._cur.active_idx

    def open_activity(self, idx: int) -> bool:
        """Switch the in-view activity to `idx` (a card click reopening a saved puzzle, or `0` to go
        back to the base conversation). No frame is removed — activities are re-openable siblings.
        Out-of-range → False. Persists + re-renders the now-active frame's workspace + beats."""
        if not (0 <= idx < len(self._cur.activities)):
            return False
        self._cur.active_idx = idx
        self._persist_view()
        self._republish()
        return True

    def finish_activity(self, *, status: str = ATTEMPTED_STATUS) -> bool:
        """The player leaves the current activity (the puzzle's back button): stamp the frame's outcome,
        switch the view back to the BASE conversation, and drop a `card` beat into the base referencing
        the saved activity so the player can reopen it. No-op (False) when already on the base or the
        active frame isn't a re-openable activity. The frame itself — its beats, board, variations — is
        preserved for reopening."""
        c = self._cur
        if c.active_idx == 0:
            return False
        idx = c.active_idx
        frame = c.activities[idx]
        if frame.kind == "conversation":
            return False
        frame.status = frame.status or status  # a solved drill already stamped itself; don't downgrade it
        c.active_idx = 0                       # back to the base conversation…
        base = c.activities[0]
        existing = next((b for b in base.beats
                         if b.get("kind") == "card" and b.get("activity_idx") == idx), None)
        if existing is None:
            self.append_beats([{"kind": "card", "activity_idx": idx, "title": frame.title or "Puzzle",
                                "activity_kind": frame.kind, "status": frame.status,
                                "segments": [{"text": frame.title or "Puzzle"}]}])
        else:
            existing["status"] = frame.status  # a re-visit may have upgraded attempted → solved
            self._persist_view()
        self._republish()
        return True

    def mark_activity_solved(self) -> None:
        """Stamp the active activity's frame `status=solved` (the deterministic solve moment), so its
        card reads 'solved' even though the player hasn't left yet. No-op on the base conversation."""
        c = self._cur
        if c.active_idx == 0:
            return
        c.activities[c.active_idx].status = SOLVED_STATUS
        self._persist_view()

    def pop_activity(self) -> bool:
        """Legacy destructive pop (the parked what-if excursion): drop the top frame and restore the one
        beneath. Distinct from the Activities feature, which keeps frames (finish_activity/open_activity).
        Cannot pop past the base (False). Only valid when the active frame IS the last one."""
        if len(self._cur.activities) <= 1:
            return False
        self._cur.activities.pop()
        self._cur.active_idx = min(self._cur.active_idx, len(self._cur.activities) - 1)
        self._persist_view()
        self._republish()
        return True

    def set_base_activity(self, kind: str) -> None:
        """Replace the whole stack with a single base frame of `kind` (a deliberate home change)."""
        self._cur.activities = [_Frame(kind=kind)]
        self._cur.active_idx = 0
        self._persist_view()
        self._republish()

    @property
    def _view_seq(self): return self._cur.view_seq
    @_view_seq.setter
    def _view_seq(self, v): self._cur.view_seq = v

    def __post_init__(self):
        os.makedirs(self.home, exist_ok=True)

    # -- session view (writer: app via /view; readers: get_view, read_input) --
    def set_view(self, snapshot: dict) -> dict | None:
        """Record the app's full display state — the exact board it's showing, the resolved line on
        screen (mainline prefix flowing into the active sideline), the whole variation forest, and the
        cursor. Persisted per session (so a resumed session restores the exact tree + cursor) and
        surfaced to the coach via `get_view`, so it coaches the position the player is actually
        staring at, not memory. Single-writer (this process), atomic; NOT published live (the app is
        the source — the view is only replayed on (re)connect/switch via `snapshot`).

        Returns the stored view object (with derived `side_to_move`), or None if `fen` is absent."""
        fen = (snapshot or {}).get("fen")
        if not fen:
            return None
        self._view_seq += 1
        side = "white" if fen.split()[1] == "w" else "black"
        pov = snapshot.get("pov")
        obj = {
            "schema": SCHEMA, "seq": self._view_seq, "session": self._current or None,
            "fen": fen, "side_to_move": side,                 # side derived here — never trust the client
            # The colour the player has the board ORIENTED to (StudySessionScreen.povColor) — what
            # "I"/"me" means in chat, deliberately independent of whose turn it is or who moved
            # last. Validated, not trusted blind: anything else collapses to unknown (None), which
            # the coach prompt treats as "don't guess an identity" rather than a wrong one.
            "pov": pov if pov in ("white", "black") else None,
            "cursor": int(snapshot.get("cursor") or 0),
            "in_variation": bool(snapshot.get("in_variation")),
            "line": snapshot.get("line") or [],
            "tree": snapshot.get("tree") or [],
            "ts": time.time(),
        }
        self._view = obj                 # the line/tree/cursor for display + resume (NOT a rival board source)
        self._report_board(fen)          # the app's navigation updates the ONE session board (persists)
        return obj

    def _build_board(self, fen: str) -> dict:
        """The published board object for `fen`: identity (fen/side/terminal) PLUS the derived
        decoration + poisoned layers. Each layer is a projection of its durable slot — shown only when
        the slot is for THIS position — so a board painted by the coach and then navigated away from
        and BACK re-shows its marks, and a bare position report never carries another position's marks.
        Bumps `board_seq` (the app re-renders on the change)."""
        self._board_seq += 1
        side = "white" if fen.split()[1] == "w" else "black"
        terminal = None
        try:
            b = Board(fen)
            if not b.legal_moves():
                terminal = "checkmate" if b.in_check else "stalemate"
        except Exception:
            pass
        d = self._cur.decorations or {}
        on = bool(d) and _norm_fen(d.get("for_fen", "")) == _norm_fen(fen)   # decorations for THIS fen?
        p = self._cur.poisoned
        poisoned_here = bool(p and p.get("moves") and _norm_fen(p.get("for_fen", "")) == _norm_fen(fen))
        return {
            "schema": SCHEMA, "seq": self._board_seq, "fen": fen,
            "side_to_move": side,               # explicit, so nothing has to parse the FEN
            "terminal": terminal,               # "checkmate" | "stalemate" | None
            "arrows": (d.get("arrows") or []) if on else [],
            "highlights": (d.get("highlights") or []) if on else [],
            "caption": d.get("caption") if on else None,
            "eval": d.get("eval") if on else None,
            "has_poisoned_line": poisoned_here,                        # derived: "there's a trap here for you"
            "poisoned_line": (p["moves"] if poisoned_here else None),  # derived: the full trap [{uci,san,fen}]
        }

    def _report_board(self, fen: str | None) -> None:
        """The app tells us the current board (its navigation, or the thin /position report). Update the
        ONE session board — the single source of truth `board_view` derives from — WITHOUT publishing
        (the app already shows it) and WITHOUT touching the decoration slot: the coach's marks belong to
        the position they were painted for, so navigating onto that position re-projects them and off it
        hides them (they are NOT blanked). This is `terminal -> nav -> update the session board`."""
        if not fen:
            return
        self._last_board = self._build_board(fen)
        self._persist_view()

    def set_board_view(self, fen: str | None) -> None:
        """Thin channel (`POST /position`): the app reports the fen it's displaying → updates the one
        session board (the source of truth for `board_fen`)."""
        self._report_board(fen)

    def mark_walked(self, fen: str | None) -> bool:
        """Claim `fen` as 'the coach has now commented on this walked move'. Returns True the FIRST time
        (→ comment on it) and False on every re-visit (→ silent). Idempotent + transient, so walking a
        variation forward comments each new move but stepping back over seen ones says nothing."""
        if not fen or fen in self._cur.walked:
            return False
        self._cur.walked.add(fen)
        return True

    @property
    def view(self) -> dict | None:
        """The app's current display state (full view object), or None if none reported yet."""
        return self._view

    @property
    def board_view(self) -> str | None:
        """The current board fen, DERIVED from the one source of truth — the session board (`_last_board`).
        Everything that changes the board updates that single field: the coach's paint (`write_board`),
        the app's navigation (`set_view`), the thin /position report (`set_board_view`). No priority chain
        across rival stores — the old `_view.fen`-first ordering let a stale app view outrank the coach's
        own fresh paint, so the coach was ignored and the board snapped back to the previous position."""
        return (self._last_board or {}).get("fen")

    # -- coach session id (durable; the app resumes it across relaunches) --
    # The server owns the Claude Code session id for this home so the coaching thread survives app
    # relaunches: the app fetches it at launch and resumes (or, first time, creates) that session.
    # A home can grow to several sessions later — this is the current/default one.
    def _session_path(self) -> str:
        return os.path.join(self.home, "session.json")

    def read_session_id(self) -> str | None:
        """The current session id — the one `is_active` row in the DB (B3: no more session.json).
        Falls back to the legacy file only when there is no DB (in-memory mode)."""
        if self.db is not None:
            return self.db.get_active_session(self.current_user)
        try:
            with open(self._session_path(), encoding="utf-8") as f:
                return (json.load(f) or {}).get("session_id") or None
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def write_session_id(self, session_id: str) -> str:
        """Persist the coach session id (single-writer, atomic) and make it the current live view.
        Returns it."""
        self._session_seq += 1
        if self.db is None:                       # in-memory mode keeps the legacy file
            write_state(self._session_path(), {
                "schema": SCHEMA, "seq": self._session_seq, "session_id": session_id,
            })
        self._switch_current(session_id)          # sets the DB is_active flag when there is a DB
        return session_id

    def ensure_session_id(self, requested: str | None = None) -> str:
        """Resolve the chat this caller should be on, minting one on first use — so the app always
        gets a stable id and the coaching thread is resumed, not restarted.

        `requested` attaches to a specific existing chat (a connection naming its own). Without it,
        fall back to the durable active-chat pointer, then to a fresh chat.

        Attaches rather than opens: this is called on every REST request and on socket connect, and
        must not replay a snapshot at the chat's sockets each time.
        """
        sid = requested or self.read_session_id() or str(uuid.uuid4())
        self.attach_chat(sid)
        self._seed_start_board()
        return sid

    def _seed_start_board(self) -> None:
        """A fresh session (no board yet) gets a PUBLISHED start position, so the app receives a real
        `board` event and the board is interactive from move one. Without this the app only has the
        placeholder start (never a `board` event) and locks the board on launch. No-op if a board exists."""
        if self._last_board is None:
            self.write_board(_START_FEN)

    def bind_current(self, sid: str) -> str:
        """Bind `sid` as this context's chat and make sure its bundle is loaded. Pure: no DB write,
        no publish. Use this to attach a caller to a chat that already exists."""
        if not sid:
            return self.current_sid
        _current_sid.set(sid)
        self._live_for(sid)          # restore this chat's beats/board from the DB if it is cold
        return sid

    def _authorize_chat(self, sid: str) -> None:
        """Refuse a chat this caller may not touch — BEFORE anything reads or writes it.

        Must run before `bind_current`, not just before the upsert. `bind_current` calls `_live_for`,
        which loads that chat's document/view out of the DB and caches it in `_live` — so checking
        later still lets a guessed chat id pull another user's coaching into memory. Authorization has
        to precede the load, not merely the mutation.

        Existence is distinguished from ownership on purpose:
          - no such chat  -> allowed; the caller is about to create it under themselves.
          - owned by me   -> allowed.
          - owned by someone else -> refused.
          - EXISTS with no owner (a pre-accounts row) -> refused for an authenticated user. Treating
            "unowned" as "free to take" would let a real user read and mutate legacy chats; adopting
            them silently would be worse. They are only reachable by the pre-accounts path itself.
        """
        if self.db is None:
            return                      # in-memory: one implicit user, nothing to guard
        exists, owner = self.db.session_owner(sid)
        if not exists:
            return
        me = self.current_user
        if me is None:
            # Pre-accounts caller: may only touch unowned rows, never a real user's chat.
            if owner is not None:
                raise PermissionError(f"chat {sid!r} belongs to a user; this caller is anonymous")
            return
        if owner != me:
            raise PermissionError(f"chat {sid!r} is not owned by user {me!r}")

    def _activate(self, sid: str) -> None:
        """Make `sid` the durable active chat FOR THE BOUND USER. No publish — see open_chat."""
        if self.db is None:
            return
        # The active chat has ONE canonical home: the DB's is_active row. We do NOT also stamp a
        # `current_session` DB-meta row — that was a rival copy with zero readers, and a second home
        # for one fact is exactly the drift this design forbids.
        # Record the chat so it shows in the rail immediately (named), before anything has titled it —
        # its beats are already tied to it in the DB.
        me = self.current_user
        self.db.upsert_session(sid, DEFAULT_SESSION_NAME, time.time(), user_id=me)
        self.db.set_active_session(sid, user_id=me)   # this user's active-chat pointer
        from .sessions import refresh_session_names
        refresh_session_names(self.home, self.db)

    def attach_chat(self, sid: str) -> str:
        """Bind `sid` and make it the active chat, WITHOUT replaying a snapshot.

        For callers that are merely resolving which chat they are on — every REST request, a socket
        on connect (which sends the snapshot itself). Publishing here would be wrong: a REST request
        starts with an unbound cursor, so a "did it change?" test against the caller's own cursor is
        always true, and every request would blast a reset + full snapshot at that chat's sockets
        (which is exactly what made the app see a stale seed board after a /move).
        """
        if not sid:
            return self.current_sid
        self._authorize_chat(sid)     # BEFORE bind_current — binding LOADS the document
        self.bind_current(sid)
        self._activate(sid)
        return sid

    def open_chat(self, sid: str) -> str:
        """Bind `sid`, make it active, AND replay its snapshot to the sockets watching it — so
        opening a chat swaps the board/beats/drill and never shows another chat's coaching.

        For the EXPLICIT act of opening a chat (the `open_chat` WS message, POST /session). Publishing
        is unconditional: 'already current' is a property of ONE context under a per-context cursor,
        so a fresh connection whose cursor is unbound must still get the snapshot even when another
        connection already has this chat open. Suppressing it on a cursor comparison would leave that
        socket bound to nothing, reading `_live[""]`, showing an empty board.
        """
        if not sid:
            return self.current_sid
        self._authorize_chat(sid)     # BEFORE bind_current — binding LOADS the document
        self.bind_current(sid)
        # No clearing needed: input / board_view / the poisoned-line latch are per-session fields on
        # `_Live`, so binding another chat's bundle IS the isolation — nothing to reset.
        self._activate(sid)
        self._publish_to(sid, "reset", {})
        for channel, payload in self.snapshot():
            self._publish_to(sid, channel, payload)
        return sid

    # Back-compat alias: the existing suite drives session switching through this name, and keeping it
    # green across the cursor rewrite is the regression net for the proxy rewiring.
    _switch_current = open_chat

    def open_chat_for(self, sub: Subscription, sid: str) -> str:
        """Open `sid` FOR ONE SUBSCRIPTION — authorize, activate, retarget, and hand that subscription
        its new chat's baseline, as one ordered operation. This is the socket's `open_chat` message.

        Why this exists instead of `retarget(); open_chat()` at the call site: those are two steps, and
        the window between them is a bug. The moment the retarget returns, this socket is a subscriber
        of `sid` — so any concurrent publisher for `sid` (a background coach beat on the chat being
        opened) can enqueue a DELTA ahead of the reset+snapshot, and the socket sees a delta for a
        chat whose baseline it has not been given: applied against the old chat's replica, or dropped
        as a version gap. Ordering the two calls cannot fix it; only making them one step can.

        The sequence is chosen so each failure lands where it costs least:
          - `_authorize_chat` FIRST, before the subscription moves — a refused chat leaves this socket
            exactly where it was, rather than stranded on a chat it may not read.
          - `_wlock` around snapshot-build + retarget + enqueue, so no writer can mutate the document
            between the snapshot we build and the baseline we send. Without it the socket could be
            handed snapshot(N) while delta(N+1) went out to the chat's other watchers just before we
            joined them — a gap that only a reconnect would heal.
          - the baseline is addressed to THIS subscription, not published to the chat: the other
            sockets already on `sid` have their baseline and must not be reset because someone else
            walked in.

        Lock order is `_wlock` -> `_live_lock`, the same order every writer takes them (a publish under
        `_wlock` takes `_live_lock` to enqueue). Both are re-entrant.
        """
        if not sid:
            return self.current_sid
        self._authorize_chat(sid)     # BEFORE bind_current — binding LOADS the document
        self.bind_current(sid)
        self._activate(sid)
        # Seed BEFORE the snapshot is built, so a fresh chat's start board arrives IN the baseline
        # rather than as a delta chasing it. (Seeding after would also be correct — it publishes to
        # `sid`, which by then includes us — but it makes the board a separate round trip.)
        self._seed_start_board()
        with self._wlock:
            events = [("reset", {})] + list(self.snapshot())
            with self._live_lock:
                self._retarget_locked(sub, sid)
                for channel, payload in events:
                    self._enqueue_locked(sid, (sub,), channel, payload)
        return sid

    # -- pub/sub (writer: the coach/app paths; readers: the app's WS connections) ------
    def subscribe(self, sid: str) -> Subscription:
        """Register a subscriber watching chat `sid`. Captures the running loop ALONGSIDE the queue,
        so sync writers publish thread-safely to the loop that actually owns it."""
        sub = Subscription(queue=asyncio.Queue(), sid=sid, loop=asyncio.get_running_loop())
        with self._live_lock:
            self._subscribers.setdefault(sid, {})[sub.queue] = sub
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._live_lock:
            watchers = self._subscribers.get(sub.sid)
            if watchers is not None:
                watchers.pop(sub.queue, None)
                if not watchers:
                    self._subscribers.pop(sub.sid, None)

    def retarget(self, sub: Subscription, new_sid: str) -> None:
        """Move a subscriber from one chat to another (a socket switching chats).

        Bumping `epoch` is what makes the switch clean. Taking `_live_lock` stops a publish from
        enqueuing into the chat this socket just left, but it CANNOT unsend what is already in the
        queue, nor cancel a `call_soon_threadsafe` callback already scheduled. Those carry the old
        (sid, epoch) and `Subscription.get` drops them.
        """
        if sub.sid == new_sid:
            return
        with self._live_lock:
            self._retarget_locked(sub, new_sid)

    def _retarget_locked(self, sub: Subscription, new_sid: str) -> None:
        """The move itself. CALLER MUST HOLD `_live_lock` — see `open_chat_for`, which needs the
        retarget and the baseline that follows it to be ONE locked step."""
        if sub.sid == new_sid:
            return
        (self._subscribers.get(sub.sid) or {}).pop(sub.queue, None)
        if not self._subscribers.get(sub.sid):
            self._subscribers.pop(sub.sid, None)
        sub.sid = new_sid
        sub.epoch += 1
        self._subscribers.setdefault(new_sid, {})[sub.queue] = sub

    def _publish(self, channel: str, payload: dict) -> None:
        """Fan a `(channel, payload)` event out to the subscribers of the CURRENTLY BOUND chat.

        Resolving the chat from the context here is what keeps every existing call site unchanged
        while making the fan-out session-scoped.
        """
        self._publish_to(self.current_sid, channel, payload)

    def _publish_to(self, sid: str, channel: str, payload: dict) -> None:
        """Fan an event out to the subscribers of chat `sid` ONLY — never to other chats' sockets.

        Callers that run off any loop-aware path (a raw thread, e.g. the live analyzer) must use this
        directly and pass their session id: contextvars do not cross `threading.Thread`.
        """
        # Enqueue under `_live_lock`, the same lock `retarget`/`unsubscribe` mutate the map with, so a
        # socket switching chats can never be handed a delta from the chat it just left: without it,
        # this could copy the queue out of the old chat's bucket and enqueue AFTER retarget moved it.
        # Safe to hold across the loop — call_soon_threadsafe is non-blocking — and it is an RLock, so
        # the nested `_live_for` below re-enters fine.
        with self._live_lock:
            watchers = self._subscribers.get(sid)
            if not watchers:
                return
            self._enqueue_locked(sid, list(watchers.values()), channel, payload)

    def _enqueue_locked(self, sid: str, subs, channel: str, payload: dict) -> None:
        """Stamp one event for chat `sid` and enqueue it to exactly `subs`. CALLER MUST HOLD
        `_live_lock`.

        Split out of `_publish_to` so `open_chat_for` can address ONE subscription instead of a
        chat's whole watcher set, inside the same locked step as its retarget.
        """
        # P4a: stamp the monotonic document `version` on every delta (non-mutating shallow copy,
        # so the stored payload stays clean). The app applies deltas in version order and
        # reconnects on a gap, so the version MUST come from the publishing chat's own bundle —
        # read `_live_for(sid)` directly rather than `_version` (which would resolve the caller's
        # cursor and could stamp a different chat's number, silently making the app DROP it).
        if isinstance(payload, dict) and "version" not in payload:
            payload = {**payload, "version": self._live_for(sid).version}
        # Every event carries the chat + subscription epoch it was enqueued for, so a reader that
        # has since switched chats can drop it (see Subscription.get).
        for sub in subs:
            stamped = (sid, sub.epoch, channel, payload)
            try:
                sub.loop.call_soon_threadsafe(sub.queue.put_nowait, stamped)
            except RuntimeError:
                pass

    def snapshot(self) -> list[tuple[str, dict]]:
        """The current UI state as `(channel, payload)` events, for replay on (re)connect —
        so a fresh app is fully synced before it starts consuming deltas."""
        evs: list[tuple[str, dict]] = []
        # History goes out BEFORE the board: the app anchors board orientation on `history.first`, so if
        # the board arrived first it would briefly flip to the board's side-to-move (a black-to-move
        # position flashes upside-down) before history corrects it. History first = stable orientation.
        # COPY the mutable lists into the payload. `_publish` enqueues the payload and the WS sender
        # serializes it LATER (async), so a live reference would capture mutations made between now and
        # serialization. `append_beats` mutates `self._beats` IN PLACE, so an "empty" push snapshot could
        # serialize with beats that were appended a beat later (present's challenge) — delivering it both
        # here AND in its own delta = a duplicate. A shallow copy freezes the length at snapshot time.
        evs.append(("history", {"schema": SCHEMA, "seq": self._history_seq, "plies": list(self._history)}))
        if self._last_board is not None:
            evs.append(("board", self._last_board))
        evs.append(("beats", {
            "schema": SCHEMA, "seq": self._beats_seq, "beats": list(self._beats),
            "cursor": self._beats[-1]["i"] if self._beats else 0,
        }))
        if self._last_analysis is not None:
            evs.append(("analysis", self._last_analysis))
        if self._last_tree is not None:
            evs.append(("tree", self._last_tree))
        # The app's persisted display state — replayed once so a resumed/switched session rebuilds the
        # exact board + variations + cursor it left. Absent when the session never explored, OR when the
        # mainline has diverged under it (a re-import): a view whose main prefix no longer matches the
        # session's history references dead positions, so we DROP it rather than replay a broken tree.
        if self._view is not None and self._view_matches_history(self._view):
            evs.append(("view", self._view))
        evs.append(("sessions", self._sessions_payload()))
        # P5: the activity stack's shape for the app's breadcrumb (depth 1 = just the base, no
        # breadcrumb). Re-sent by _republish after every push/pop, so the breadcrumb updates live.
        evs.append(("activity", {"depth": self.frame_depth, "kind": self.top_kind,
                                  "idx": self.active_idx}))
        # The conversation mode (coach/freeform), so a (re)connecting app shows the drill header
        # without waiting for the next lesson-state transition.
        evs.append(("mode", self._mode_payload()))
        # P4a: a snapshot is a point-in-time full sync — every event carries the same current version,
        # so the app knows the version its replica is at before it starts consuming deltas.
        v = self._version
        return [(ch, {**p, "version": v}) for ch, p in evs]

    def _view_matches_history(self, view: dict) -> bool:
        """Is a stored view still consistent with the session's mainline? Each `main` move in the
        view's resolved line must sit at its history ply (same position, ignoring clocks). Empty
        history → nothing to diverge from, keep it. A mismatch or a main prefix longer than history
        means the mainline moved (re-import) — the view's tree references dead positions, so drop it."""
        hist = self._history or []
        if not hist:
            return True
        for m in (view.get("line") or []):
            if m.get("kind") != "main":
                continue
            i = m.get("i")
            if not isinstance(i, int) or not (0 <= i < len(hist)):
                return False
            if _norm_fen(m.get("fen", "")) != _norm_fen((hist[i] or {}).get("fen", "")):
                return False
        return True

    def _sessions_payload(self) -> dict:
        """The rail's session list + which is current — pushed over /state (on connect and on every
        switch), so the rail updates live instead of the app having to re-poll."""
        from .sessions import list_sessions
        return {"sessions": list_sessions(self.home, self.db, user_id=self.current_user),
                "current": self._current}

    # -- live engine lines (writer: LiveAnalyzer; transient, not persisted) ----
    def publish_engine_lines(self, payload: dict, *, session_id: str) -> None:
        """Push a live multi-PV analysis snapshot for chat `session_id`. Transient — regenerated
        continuously by the analyzer, so it's not stored or replayed in the snapshot.

        `session_id` is REQUIRED and explicit: the analyzer deepens on its OWN raw thread
        (LiveAnalyzer._run), and contextvars do not cross `threading.Thread` — the bound cursor would
        resolve empty there and the event would be stamped with the wrong chat's version, which makes
        the app's ordered-delta guard silently DROP it.
        """
        self._publish_to(session_id, "engine_lines", payload)

    # -- margin pre-roll progress (writer: margin deep worker; transient) ------
    def publish_margin_progress(self, payload: dict, *, session_id: str) -> None:
        """Push one pre-roll analysis stage (label + squares to highlight) for
        the interactive loading (owner 2026-07-25: "as we analyze pawn
        structure, all the pawns must get highlighted..."). Transient — the
        margin's finished sheet supersedes it; never stored or replayed.
        `session_id` explicit for the same reason as publish_engine_lines:
        the margin deep worker is a raw thread, contextvars don't cross."""
        self._publish_to(session_id, "margin_progress", payload)

    # -- live agent status (writer: tool wrappers; transient, not persisted) ---
    def publish_status(self, text: str | None) -> None:
        """Push a one-line "what the coach is doing" phase (grounded in the tool it just called),
        or `None` to clear it. Transient — a UI hint, not persisted or replayed."""
        with self._wlock:   # ordered w.r.t. other publishes, so a `None` clear is never dropped as stale
            self._publish("status", {"schema": SCHEMA, "text": text})

    # -- conversation mode (writer: the lesson bridge methods above) ----------
    def _mode_payload(self) -> dict:
        """The chat's conversation mode as a wire fact (resolve_mode made visible): a live lesson →
        coach, else freeform — with `suspended` flagging a drill parked by a what-if excursion, and
        the governing lesson's type so the app can title the drill ("Puzzle")."""
        live = self.active_lesson()
        lesson = live or self.suspended_lesson()
        return {"schema": SCHEMA,
                "mode": "coach" if live is not None else "freeform",
                "suspended": live is None and lesson is not None,
                "lesson_type": lesson.spec.type if lesson is not None else None}

    def publish_mode(self) -> None:
        """Push the current mode to the bound chat's sockets. Called after every lesson-state
        mutation (activate / set_state / progress save) — the only places the mode can change —
        so the app never has to infer the mode from side effects."""
        with self._wlock:   # ordered w.r.t. other publishes, same as publish_status
            self._publish("mode", self._mode_payload())

    # -- app->coach input (writer: app via submit_input; reader: read_input) --
    def set_input(self, data: dict) -> None:
        """Record the app's structured input (a played move / "done" / drill event). In-memory
        replacement for the old app-written `input.json`; `read_input` prefers this."""
        self._input = dict(data)

    def _path(self, name: str) -> str:
        return os.path.join(self.home, name)

    def reset_session(self) -> None:
        """Clear the session view (board/beats/input) so a fresh server — i.e. a
        new `claude` session — begins with a clean slate: no stale beats from a
        prior session, and no seq collision (each process restarts seq at 0, so a
        leftover beats.json stamped seq 1 would otherwise reappear under the new
        session's seq-1 board). The MCP owns these files (single-writer safe);
        mastery memory under memory/ is untouched."""
        self._cur.activities = [_Frame()]     # reset the stack to a single clean base frame (P3)
        self._cur.active_idx = 0              # …and re-point the view before any beats/board write below
        self._version = 0
        self._beats_seq = 0
        self._beats = []
        self._cur.gate_awaiting = False
        self._cur.gate_pending = None
        self._cur.status = "active"
        self._cur.banked = []
        self._cur.served_puzzles = []          # a fresh session gets a fresh puzzle deck
        self._input = None
        # (the freeform poisoned slot lives on the workspace, already reset by the fresh _Frame() above)
        # P6: the live state JSON files are gone (board/beats/tree/analysis/view are the document now).
        # Only clear the app<->MCP assess channel (app-written) + any legacy leftovers on an old home.
        for name in ("input.json", "move_query.json", "move_meaning.json",
                     "board.json", "beats.json", "tree.json", "analysis_view.json", "view.json"):
            try:
                os.remove(self._path(name))
            except FileNotFoundError:
                pass

    def new_session(self) -> str:
        """Mint a brand-new session id, make it the active/current one, and return it. The BACKEND owns
        id creation (the app just adopts what it gets back). write_session_id persists it, sets the DB
        is_active flag, and switches the live view — which publishes reset + the fresh (empty) snapshot
        to the app, so 'New session' swaps to a clean slate."""
        sid = self.write_session_id(str(uuid.uuid4()))
        self._seed_start_board()   # interactive start board from move one (not a locked placeholder)
        return sid

    @property
    def session_name(self) -> str | None:
        """The current session's display name (from the DB), or None. `DEFAULT_SESSION_NAME` means it
        hasn't been titled yet — the coach names it on the first substantive turn."""
        if self.db is None or not self._current:
            return None
        return self.db.get_session_name(self._current)

    @property
    def session_unnamed(self) -> bool:
        """True when the current session still carries the placeholder title — the cue to have the coach
        propose a real one on this turn."""
        return (self.session_name or DEFAULT_SESSION_NAME) == DEFAULT_SESSION_NAME

    def set_session_name(self, name: str) -> None:
        """Title the current session and push the updated rail live, so the sidebar renames in place.
        Trimmed + length-capped; a blank name is ignored (keeps the placeholder)."""
        name = (name or "").strip()[:60]
        if self.db is None or not self._current or not name:
            return
        self.db.set_session_name(self._current, name)
        with self._wlock:
            self._publish("sessions", self._sessions_payload())

    # -- the durable freeform poisoned line (§6.4) ------------------------
    @property
    def poisoned(self) -> dict | None:
        """The current session's freeform poisoned line `{for_fen, moves, meta}`, or None."""
        return self._cur.poisoned

    def set_poisoned(self, for_fen: str, moves: list, meta: dict | None = None) -> None:
        """Record the detected freeform trap DURABLY (§6.4) — on the document, keyed by the position it
        lives at, never on the transient board. The board's poisoned fields are derived from this by
        `write_board`, so the trap survives any repaint. Repaint the board afterwards to re-project it."""
        if not (for_fen and moves):
            return
        self._cur.poisoned = {"for_fen": for_fen, "moves": list(moves), "meta": meta or {}}
        self._persist_view()

    def clear_poisoned(self, for_fen: str | None = None) -> None:
        """Drop the freeform trap. With `for_fen`, only clears when the slot is for THAT position (so a
        no-trap check on one position never clobbers a live trap held for another). No-op if empty."""
        p = self._cur.poisoned
        if p is None:
            return
        if for_fen is not None and _norm_fen(p.get("for_fen", "")) != _norm_fen(for_fen):
            return
        self._cur.poisoned = None
        self._persist_view()

    # -- board.json (writer: mcp) -----------------------------------------
    def write_board(
        self,
        fen: str,
        *,
        arrows: list[dict] | None = None,
        highlights: list[dict] | None = None,
        caption: str | None = None,
        eval: dict | None = None,
    ) -> int:
        """Repaint the board — a COACH paint. Returns the new `seq`. The passed decorations
        (arrows/highlights/caption/eval) are recorded in the durable `decorations` slot keyed to `fen`
        (§6.2), and the published board projects them (and the poisoned slot) from there. So decorations
        survive the app navigating away and back (the /position report no longer blanks them), and the
        poisoned line can never live only on the transient board (the old evaporate-on-repaint bug)."""
        # Record the coach's decorations for THIS position (the durable home); the board projects them.
        self._cur.decorations = {"for_fen": fen, "arrows": arrows or [], "highlights": highlights or [],
                                 "caption": caption, "eval": eval}
        obj = self._build_board(fen)
        self._last_board = obj
        # A coach paint to a position OFF the stored app view's whole line SUPERSEDES that view: the
        # coach moved the board somewhere the view doesn't describe, so the old view is stale and must
        # not linger as a rival the reader (get_view) could surface under the fresh fen. Drop it here,
        # at the writer — same single-source discipline as snapshot()'s drop-a-diverged-view. Painting
        # a position that IS on the view's line (adding arrows to what's on screen, or an undo stepping
        # back WITHIN the line) keeps the view, so navigation within a line doesn't destroy it. (A thin
        # app /position report goes through _report_board, NOT here, so it never supersedes a rich view
        # — that subordination of /position to /view is deliberate.)
        if self._view is not None:
            nf = _norm_fen(fen)
            on_view_line = nf == _norm_fen((self._view or {}).get("fen", "")) or any(
                nf == _norm_fen((m or {}).get("fen", "")) for m in (self._view.get("line") or []))
            if not on_view_line:
                self._view = None
        with self._wlock:
            self._persist_view()
            self._publish("board", obj)
        return self._board_seq

    # -- beats (writer: mcp) ----------------------------------------------
    def append_beats(self, beats: list[dict]) -> list[int]:
        """Append 1–N beats (each already shaped: segments/kind/stops/board?).

        Assigns each a global index `i`, returns the assigned indices. `cursor`
        is set to the newest beat; the app paces the reveal from there."""
        with self._wlock:   # atomic version-bump + publish (see _wlock) — concurrent writers stay ordered
            start = len(self._beats)
            now = time.time()
            indices = []
            for offset, beat in enumerate(beats):
                b = dict(beat)
                b["i"] = start + offset
                b["board_seq"] = self._board_seq   # the board this beat describes
                b["ts"] = now                      # wall-clock, so the app can interleave with user messages
                self._beats.append(b)
                indices.append(b["i"])
            self._beats_seq += 1
            # Beats now live ON THE FRAME → they persist inside the document blob (per activity). We no
            # longer write the `(session_id, i)` beat sidecar: per-activity beats restart at i=0, so a
            # pushed activity's beat 0 would COLLIDE with the base's beat 0 there. The blob is atomic.
            self._persist_view()
            self._publish("beats", {"schema": SCHEMA, "seq": self._beats_seq,
                                    "appended": self._beats[start:]})
        return indices

    # -- tree.json (writer: mcp) ------------------------------------------
    def clear_tree(self) -> None:
        """Drop any drill from the document + tell the app. Used when build_and_arm_drill finds the
        position isn't a single-only-move forcing win — there is nothing for the app to walk, and a
        leftover tree would make it reject every move as wrong."""
        self._last_tree = None
        with self._wlock:
            self._persist_view()
            self._publish("tree_cleared", {})

    def write_tree(self, tree: dict) -> int:
        """Write the forcing-line drill tree for the app to walk. Single-writer
        (mcp), atomic; `seq` bumps each build so the app can detect a new drill.
        The tree carries its own `schema`/`fen`/`root`; we stamp seq + wrap it."""
        self._tree_seq = getattr(self, "_tree_seq", 0) + 1
        obj = {"schema": SCHEMA, "seq": self._tree_seq, **tree}
        self._last_tree = obj
        with self._wlock:
            self._persist_view()
            self._publish("tree", obj)
        return self._tree_seq

    # -- move history (writer: mcp) ---------------------------------------
    def write_history(self, plies: list) -> int:
        """Set the session's move line (ply 0 = start; each ply {n, san, uci, fen}) — the move
        navigator. Single-writer (mcp); `seq` bumps each write so the app re-syncs."""
        self._history_seq += 1
        self._history = list(plies)
        with self._wlock:
            self._persist_view()
            self._publish("history", {"schema": SCHEMA, "seq": self._history_seq, "plies": self._history})
        return self._history_seq

    # -- analysis_view.json (writer: mcp) ---------------------------------
    def write_analysis(self, fen: str, *, verdict: str, observations: list[str],
                       board_seq: int | None = None) -> int:
        """Render a single-position **analysis object**: the coach's natural-language
        conversion of the grounded facts + engine eval into a player-facing read
        (`verdict` = the one-line standing, `observations` = the grounded points). This
        is the *output* twin of `analyze_and_show(focus="analysis")`'s briefing (its
        *input*) — same facts, now translated for the player. Single-writer (mcp),
        atomic, `seq` bumps each push. `board_seq` anchors it to the board the analysis
        describes (defaults to the current board)."""
        self._analysis_seq += 1
        side = "white" if fen.split()[1] == "w" else "black"
        obj = {
            "schema": SCHEMA, "seq": self._analysis_seq, "fen": fen,
            "side_to_move": side,
            "verdict": verdict, "observations": list(observations),
            "board_seq": board_seq if board_seq is not None else self._board_seq,
        }
        self._last_analysis = obj
        self._persist_view()
        self._publish("analysis", obj)
        return self._analysis_seq

    # -- input.json (writer: app; we only read) ---------------------------
    def read_input(self) -> dict:
        """Read AND CONSUME the app's structured input — a played move / drill event is a one-shot
        thing, so it's cleared once read. Otherwise a stale move lingers and hijacks the next turn
        (e.g. the coach re-coaches an old board drag instead of the player's fresh typed request).
        Prefers the in-memory value from `submit_input`; falls back to `input.json`. Returns
        `{"kind": "none"}` if there's nothing new (never raises into the tool layer)."""
        if self._input is not None:
            data, self._input = self._input, None      # consume
            return data
        try:
            with open(self._path("input.json"), encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"kind": "none"}
        try:
            os.remove(self._path("input.json"))         # consume the file too
        except OSError:
            pass
        return data

    # -- analysis.json (writer: import pass; we only read) ----------------
    def analysis_path(self, game_id: str) -> str:
        return os.path.join(self.home, "analysis", f"{game_id}.json")

    def read_analysis(self, game_id: str) -> dict | None:
        """Read a precomputed `analysis.json` by game id, or None if absent."""
        try:
            with open(self.analysis_path(game_id), encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # -- heartbeat.json (writer: mcp) -------------------------------------
    def write_heartbeat(self, *, engine_ok: bool, pid: int | None = None,
                        started_at: float | None = None, now: float | None = None) -> int:
        self._hb_seq += 1
        ts = now if now is not None else time.time()
        obj = {
            "schema": SCHEMA, "seq": self._hb_seq,
            "pid": pid if pid is not None else os.getpid(),
            "started_at": started_at if started_at is not None else ts,
            "refreshed_at": ts,
            "engine": "ok" if engine_ok else "down",
            "version": VERSION,
        }
        write_state(self._path("heartbeat.json"), obj)
        return self._hb_seq
