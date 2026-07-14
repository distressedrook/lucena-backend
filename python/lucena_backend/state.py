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
import json
import os
import time
import uuid
from dataclasses import dataclass, field

from lucena_engine.board import Board
from .statefile import write_state

SCHEMA = 1
VERSION = "0.1.0"


def _norm_fen(fen: str) -> str:
    """A position's identity ignoring clocks — placement + side + castling + ep (first 4 FEN fields).
    Matches the app's `VariationForest.norm`, so a stored view's mainline prefix can be checked
    against the session's move history without the halfmove/fullmove counters fragmenting it."""
    return " ".join(str(fen).split()[:4])

# Placeholder name a session gets the moment it's created — shown in the rail until Claude's
# ai-title (from the transcript) replaces it.
DEFAULT_SESSION_NAME = "New session"

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
    """One activity on the stack (design §7): its kind + the workspace it looks at. `conversation` is
    the base default; `drill` etc. add fields via the workspace's drill_state. Push/pop is P5."""
    kind: str = "conversation"
    workspace: _Workspace = field(default_factory=_Workspace)


@dataclass
class _Live:
    """One session's live state (state-machine P3). Session-LEVEL fields (the one continuous
    conversation `beats`, the durable Socratic `gate`, the monotonic `version`) are held here directly;
    the per-frame *workspace* (board/analysis/drill/line/view) lives on the activity STACK
    (`activities`, `[0]` = base). The per-frame attributes (`last_board`, `board_seq`, …) are exposed as
    delegating properties onto the top frame's workspace, so every StateStore accessor stays unchanged.
    One bundle per Claude session, so switching sessions never leaks beats/board across sessions."""
    beats: list = field(default_factory=list)
    beats_seq: int = 0
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
    # The activity STACK; activities[-1] is the live (top) frame. One base frame until P5 adds push/pop.
    activities: list = field(default_factory=lambda: [_Frame()])
    # Per-session RUNTIME state (never persisted — transient). Session-scoped by construction so it can
    # NEVER leak across a switch/new-session: the app→coach input mailbox. (The "current board fen" is NOT
    # a separate field — it derives from the one session board, `last_board`; see StateStore.board_view.)
    input: dict | None = None

    @property
    def top(self) -> _Frame:
        return self.activities[-1]

    @property
    def ws(self) -> _Workspace:
        return self.activities[-1].workspace

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
    return {"kind": f.kind, "workspace": _workspace_to_dict(f.workspace)}


def _frame_from_dict(d: dict | None) -> _Frame:
    d = d or {}
    return _Frame(kind=d.get("kind", "conversation"),
                  workspace=_workspace_from_dict(d.get("workspace")))


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
    _subscribers: set = field(default_factory=set)   # set[asyncio.Queue]
    _loop: object | None = None                      # the server's event loop (for threadsafe publish)
    _on_switch: object | None = None                 # optional callback fired after a session switch
                                                     # (serve_http uses it to reset the live analyzer,
                                                     #  whose target is otherwise the OLD session's fen)
    # The session state is partitioned by Claude Code session id (`_current`); the app's board / beats /
    # analysis / drill / input / gate are ALWAYS the current session's. Every per-session field
    # (`_beats`, `_last_board`, `_input`, seqs, …) is a proxy property onto `_cur`, so
    # nothing leaks across a switch.
    _live: dict = field(default_factory=dict)        # session_id -> _Live
    _current: str = ""                               # current session id; "" until one is set

    @property
    def _cur(self) -> _Live:
        if self._current not in self._live:
            self._live[self._current] = self._load_live(self._current)
        return self._live[self._current]

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
                gate = d.get("gate") or {}
                return _Live(beats=legacy["beats"], beats_seq=d.get("beats_seq", 0),
                             gate_awaiting=bool(gate.get("awaiting")), gate_pending=gate.get("pending"),
                             status=d.get("status", "active"), banked=d.get("banked") or [],
                             served_puzzles=d.get("served_puzzles") or [],
                             drill_close=d.get("drill_close"),
                             version=blob["version"], activities=frames)
            # P1/P2 flat blob → wrap the flat workspace fields into one base frame (compose-on-load)
            ws = _Workspace(
                last_board=d.get("last_board"), last_analysis=d.get("last_analysis"),
                last_tree=d.get("last_tree"), history=d.get("history") or [], view=d.get("view"),
                drill_state=d.get("drill_state"),
                board_seq=d.get("board_seq", 0), tree_seq=d.get("tree_seq", 0),
                analysis_seq=d.get("analysis_seq", 0), history_seq=d.get("history_seq", 0),
                view_seq=d.get("view_seq", 0))
            return _Live(beats=legacy["beats"], beats_seq=d.get("beats_seq", 0),
                         gate_awaiting=bool(d.get("gate_awaiting")), gate_pending=d.get("gate_pending"),
                         version=blob["version"], activities=[_Frame(workspace=ws)])
        view = legacy.get("view")                             # compose from legacy columns (pre-P1)
        ws = _Workspace(last_board=legacy["last_board"], last_analysis=legacy["last_analysis"],
                        last_tree=legacy["last_tree"], history=legacy.get("history") or [], view=view,
                        board_seq=legacy["board_seq"], tree_seq=legacy["tree_seq"],
                        analysis_seq=legacy["analysis_seq"], view_seq=(view or {}).get("seq", 0))
        return _Live(beats=legacy["beats"], beats_seq=legacy["beats_seq"],
                     activities=[_Frame(workspace=ws)])

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
        }

    def dump(self) -> dict:
        """A full snapshot of the current session document (design §5: debuggability via a `dump`
        command, not the retired live JSON files) — the document blob shape + beats + the session id."""
        c = self._cur
        return {**self._document_dict(c), "beats": list(c.beats), "session": self._current or None}

    def _persist_view(self, new_beats: list | None = None) -> None:
        """Flush the current session document (P6: the ONE canonical blob — no more JSON files, no more
        decomposed columns). Bumps the monotonic `version` (in memory always, so it advances even with
        no DB), then persists the whole document blob. `new_beats` (from append_beats) are written in
        the SAME transaction as the document, so `beats_seq` and the beat rows never disagree."""
        c = self._cur
        c.version += 1
        if self.db is None or not self._current:
            return
        self.db.save_document(self._current, self._document_dict(c), c.version, beats=new_beats)

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
        """The kind of the live (top) activity — the session's current mode."""
        return self._cur.top.kind

    def _republish(self) -> None:
        """Tell the app to reset and re-render the CURRENT top frame's workspace — used after a
        push/pop/set_base swaps which workspace is live (the wire is a projection of the top frame)."""
        self._publish("reset", {})
        for channel, payload in self.snapshot():
            self._publish(channel, payload)

    # Fields a coach-supplied push seed may set. The UI-authored `view` (cursor/line) is NOT here —
    # §8: view is UI-authored, period, so a push can never smuggle in a cursor write — and the wire
    # `*_seq`s start fresh on a new frame, so they're excluded too.
    _SEEDABLE = frozenset({"last_board", "last_analysis", "last_tree", "history", "drill_state",
                           "poisoned", "decorations"})

    def push_activity(self, kind: str, *, seed: dict | None = None) -> None:
        """Push a fresh activity frame (a rabbit-hole, design §7): the current top's workspace is
        frozen in place on the stack, a new empty workspace becomes live. Session-level state (beats,
        the Socratic gate, version) is CONTINUOUS across the push. A coach-supplied `seed` may set the
        starting board/tree/history, but NEVER the UI-authored `view` — that field is stripped so a
        push can't author the cursor (§8). Persists + re-renders the new frame."""
        safe_seed = {k: v for k, v in (seed or {}).items() if k in self._SEEDABLE}
        self._cur.activities.append(_Frame(kind=kind, workspace=_workspace_from_dict(safe_seed)))
        self._persist_view()
        self._republish()

    def pop_activity(self) -> bool:
        """Resolve the top rabbit-hole: drop the top frame and restore the parent's frozen workspace.
        Cannot pop past the base (returns False). Persists + re-renders the restored frame."""
        if len(self._cur.activities) <= 1:
            return False
        self._cur.activities.pop()
        self._persist_view()
        self._republish()
        return True

    def set_base_activity(self, kind: str) -> None:
        """Replace the whole stack with a single base frame of `kind` (a deliberate home change)."""
        self._cur.activities = [_Frame(kind=kind)]
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
        obj = {
            "schema": SCHEMA, "seq": self._view_seq, "session": self._current or None,
            "fen": fen, "side_to_move": side,                 # side derived here — never trust the client
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
            return self.db.get_active_session()
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

    def ensure_session_id(self) -> str:
        """The durable session id, minting + persisting a fresh UUID on first use — so the app
        always fetches a stable id and the coaching thread is resumed, not restarted."""
        sid = self.read_session_id() or self.write_session_id(str(uuid.uuid4()))
        self._switch_current(sid)
        return sid

    def _switch_current(self, sid: str) -> None:
        """Point the live view at `sid`'s bundle. On a real change, tell the app to reset and
        replay that session's snapshot — so switching sessions swaps the board/beats/drill and
        never shows another session's coaching."""
        if not sid or sid == self._current:
            return
        self._current = sid
        # No clearing needed: input / board_view / the poisoned-line latch are now per-session fields on
        # `_Live`, so pointing `_current` at another session's bundle IS the isolation — nothing to reset.
        if sid not in self._live:
            self._live[sid] = self._load_live(sid)     # restore this session's beats/board from DB
        if self.db is not None:
            # The current session has ONE canonical home: `_current` (memory, the authority) mirrored
            # to session.json for restart. We do NOT also stamp a `current_session` DB-meta row — that
            # was a rival copy with zero readers (`get_meta` is never called for it), and a second home
            # for one fact is exactly the drift this design forbids.
            # Record the session so it shows in the rail immediately (named), before Claude has
            # written any transcript — its beats are already tied to it in the DB.
            self.db.upsert_session(sid, DEFAULT_SESSION_NAME, time.time())
            self.db.set_active_session(sid)       # the current-session pointer (replaces session.json)
            # Switching sessions is an explicit event (off the event loop, under the tool lock) — a fine
            # place to cache any freshly-available transcript titles into the DB, so the rail we're about
            # to publish shows nice names. The snapshot's own read (_sessions_payload) stays pure.
            from .sessions import refresh_session_names
            refresh_session_names(self.home, self.db)
        # The live analyzer's target is a flat, process-level fen — reset it on a switch so it stops
        # deepening (and publishing) the OLD session's position under the new session's version. The
        # app re-arms it via /analyze for the new session when it wants live analysis.
        if callable(self._on_switch):
            try:
                self._on_switch()
            except Exception:
                pass
        self._publish("reset", {})
        for channel, payload in self.snapshot():
            self._publish(channel, payload)

    # -- SSE pub/sub (writer: mcp; readers: app `/state` connections) ------
    def subscribe(self) -> asyncio.Queue:
        """Register an SSE subscriber (called from the `/state` async handler). Captures the
        running loop so sync tool writers can publish thread-safely."""
        self._loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _publish(self, channel: str, payload: dict) -> None:
        """Fan a `(channel, payload)` event out to every SSE subscriber. Safe to call from a
        worker thread (FastMCP runs sync tools off-loop) — hops onto the loop via
        `call_soon_threadsafe`. No-op when nobody's listening."""
        if self._loop is None or not self._subscribers:
            return
        # P4a: stamp the monotonic document `version` on every delta (non-mutating shallow copy, so
        # the stored payload stays clean). The app ignores it until P4b, when it applies deltas in
        # version order and reconnects on a gap. Transient channels (engine_lines/status) get it too.
        if isinstance(payload, dict) and "version" not in payload:
            payload = {**payload, "version": self._version}
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, (channel, payload))
            except RuntimeError:
                pass

    def snapshot(self) -> list[tuple[str, dict]]:
        """The current UI state as `(channel, payload)` events, for replay on (re)connect —
        so a fresh app is fully synced before it starts consuming deltas."""
        evs: list[tuple[str, dict]] = []
        if self._last_board is not None:
            evs.append(("board", self._last_board))
        evs.append(("beats", {
            "schema": SCHEMA, "seq": self._beats_seq, "beats": self._beats,
            "cursor": self._beats[-1]["i"] if self._beats else 0,
        }))
        if self._last_analysis is not None:
            evs.append(("analysis", self._last_analysis))
        if self._last_tree is not None:
            evs.append(("tree", self._last_tree))
        evs.append(("history", {"schema": SCHEMA, "seq": self._history_seq, "plies": self._history}))
        # The app's persisted display state — replayed once so a resumed/switched session rebuilds the
        # exact board + variations + cursor it left. Absent when the session never explored, OR when the
        # mainline has diverged under it (a re-import): a view whose main prefix no longer matches the
        # session's history references dead positions, so we DROP it rather than replay a broken tree.
        if self._view is not None and self._view_matches_history(self._view):
            evs.append(("view", self._view))
        evs.append(("sessions", self._sessions_payload()))
        # P5: the activity stack's shape for the app's breadcrumb (depth 1 = just the base, no
        # breadcrumb). Re-sent by _republish after every push/pop, so the breadcrumb updates live.
        evs.append(("activity", {"depth": self.frame_depth, "kind": self.top_kind}))
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
        return {"sessions": list_sessions(self.home, self.db), "current": self._current}

    # -- live engine lines (writer: LiveAnalyzer; transient, not persisted) ----
    def publish_engine_lines(self, payload: dict) -> None:
        """Push a live multi-PV analysis snapshot for the current position. Transient — regenerated
        continuously by the analyzer, so it's not stored or replayed in the snapshot."""
        self._publish("engine_lines", payload)

    # -- live agent status (writer: tool wrappers; transient, not persisted) ---
    def publish_status(self, text: str | None) -> None:
        """Push a one-line "what the coach is doing" phase (grounded in the tool it just called),
        or `None` to clear it. Transient — a UI hint, not persisted or replayed."""
        self._publish("status", {"schema": SCHEMA, "text": text})

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
        self._persist_view()
        self._publish("board", obj)
        return self._board_seq

    # -- beats (writer: mcp) ----------------------------------------------
    def append_beats(self, beats: list[dict]) -> list[int]:
        """Append 1–N beats (each already shaped: segments/kind/stops/board?).

        Assigns each a global index `i`, returns the assigned indices. `cursor`
        is set to the newest beat; the app paces the reveal from there."""
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
        # Persist the new beats + the document (with the bumped beats_seq) in ONE transaction, so a
        # crash can't leave beats_seq ahead of the beat rows (or vice-versa).
        self._persist_view(new_beats=self._beats[start:])
        self._publish("beats", {"schema": SCHEMA, "seq": self._beats_seq,
                                "appended": self._beats[start:]})
        return indices

    # -- tree.json (writer: mcp) ------------------------------------------
    def clear_tree(self) -> None:
        """Drop any drill from the document + tell the app. Used when build_and_arm_drill finds the
        position isn't a single-only-move forcing win — there is nothing for the app to walk, and a
        leftover tree would make it reject every move as wrong."""
        self._last_tree = None
        self._persist_view()
        self._publish("tree_cleared", {})

    def write_tree(self, tree: dict) -> int:
        """Write the forcing-line drill tree for the app to walk. Single-writer
        (mcp), atomic; `seq` bumps each build so the app can detect a new drill.
        The tree carries its own `schema`/`fen`/`root`; we stamp seq + wrap it."""
        self._tree_seq = getattr(self, "_tree_seq", 0) + 1
        obj = {"schema": SCHEMA, "seq": self._tree_seq, **tree}
        self._last_tree = obj
        self._persist_view()
        self._publish("tree", obj)
        return self._tree_seq

    # -- move history (writer: mcp) ---------------------------------------
    def write_history(self, plies: list) -> int:
        """Set the session's move line (ply 0 = start; each ply {n, san, uci, fen}) — the move
        navigator. Single-writer (mcp); `seq` bumps each write so the app re-syncs."""
        self._history_seq += 1
        self._history = list(plies)
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
