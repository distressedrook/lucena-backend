"""SQLite durable store for the Lucena home — the single durable store the server owns.

One file, `lucena.db`, in the home directory; the MCP server is the sole writer (the single-writer
invariant, now held in a DB rather than a set of per-file JSON writes). WAL mode so the app could
read concurrently; one connection guarded by a lock, because some tool calls run off the event loop
(e.g. play_move in a worker thread) and SQLite connections aren't safe under concurrent use. The
schema is versioned via `PRAGMA user_version` so later phases (mastery, analysis) migrate forward.

Phase 1 owns the per-session live coaching view: each Claude Code session's beats + last board /
analysis / drill, so switching or resuming a session restores its panel instead of losing it on a
server restart. Values that are structured (board, analysis, tree, a beat) are stored as JSON text.
"""

from __future__ import annotations

import json
import sqlite3
import threading

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS session_view (
    session_id   TEXT PRIMARY KEY,
    last_board   TEXT,                      -- JSON | NULL
    last_analysis TEXT,                     -- JSON | NULL
    last_tree    TEXT,                      -- JSON | NULL
    board_seq    INTEGER NOT NULL DEFAULT 0,
    beats_seq    INTEGER NOT NULL DEFAULT 0,
    tree_seq     INTEGER NOT NULL DEFAULT 0,
    analysis_seq INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS beat (
    session_id TEXT NOT NULL,
    i          INTEGER NOT NULL,
    payload    TEXT NOT NULL,               -- JSON of the shaped beat
    ts         REAL NOT NULL,
    PRIMARY KEY (session_id, i)
);
"""

# v2: coaching sessions we own — inserted the moment one is created, so the current session shows
# in the rail (with its name + beats) before Claude has written any transcript.
_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS session (
    id         TEXT PRIMARY KEY,
    name       TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

# v3: the session's move history (JSON list of plies {n, san, uci, fen}) for the move navigator.
_SCHEMA_V3 = """
ALTER TABLE session_view ADD COLUMN history TEXT;
"""

# v4: the session view — the app's full display state (board fen + variation tree + cursor), so a
# resumed session restores the exact board + variations the user left. JSON of the view.json object.
_SCHEMA_V4 = """
ALTER TABLE session_view ADD COLUMN view TEXT;
"""

# v5 (state-machine P1): the canonical Session Document — ONE JSON blob per session holding the whole
# session state (board/analysis/tree/history/view + the monotonic `version`), the read-back source of
# truth. Beats stay in the `beat` sidecar (design §5). The decomposed `session_view` columns are NO
# longer written (`save_view` has no live caller); `load_view` still reads the beat rows, and the
# columns survive only as a compose-on-load fallback for pre-P1 sessions that have no blob yet.
_SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS session_document (
    session_id TEXT PRIMARY KEY,
    document   TEXT NOT NULL,               -- JSON of the canonical session document (beats excluded)
    version    INTEGER NOT NULL DEFAULT 0
);
"""

# v6: session lifecycle — a session can be concluded (bank + close the loop), so the rail can show it
# as done rather than active. Denormalised from the document's session-level status, written on conclude.
_SCHEMA_V6 = """
ALTER TABLE session ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
"""

# Applied in order; PRAGMA user_version is the count applied. Append new migrations, never edit.
_MIGRATIONS = [_SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6]
SCHEMA_VERSION = len(_MIGRATIONS)


def _loads(v):
    return json.loads(v) if v else None


class DB:
    """The home's SQLite store. All access is serialized behind one lock."""

    def __init__(self, path: str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for v in range(version, len(_MIGRATIONS)):
                self._conn.executescript(_MIGRATIONS[v])
                self._conn.execute(f"PRAGMA user_version={v + 1}")
            self._conn.commit()

    # -- meta (key/value; e.g. the current session pointer) ----------------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self._conn.commit()

    # -- sessions we own (id + name; the rail list) ------------------------
    def upsert_session(self, session_id: str, name: str, now: float) -> None:
        """Record a session on first sight (with `name`); thereafter just bump its `updated_at`
        (the name is kept — a later ai-title/rename replaces it explicitly)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO session(id, name, created_at, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at",
                (session_id, name, now, now))
            self._conn.commit()

    def set_session_name(self, session_id: str, name: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE session SET name=? WHERE id=?", (name, session_id))
            self._conn.commit()

    def set_session_status(self, session_id: str, status: str) -> None:
        """Set a session's lifecycle status (`active` | `complete`) — the rail shows concluded ones."""
        with self._lock:
            self._conn.execute("UPDATE session SET status=? WHERE id=?", (status, session_id))
            self._conn.commit()

    def list_sessions(self) -> list[dict]:
        """Every session we own, newest first: `{session_id, name, updated_at, status}`."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, updated_at, status FROM session ORDER BY updated_at DESC").fetchall()
        return [{"session_id": r["id"], "name": r["name"], "updated_at": r["updated_at"],
                 "status": r["status"] if "status" in r.keys() else "active"}
                for r in rows]

    # -- per-session live coaching view ------------------------------------
    def load_view(self, session_id: str) -> dict:
        """The session's stored view as a plain dict (empty defaults if never persisted)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_view WHERE session_id=?", (session_id,)).fetchone()
            beats = [json.loads(r["payload"]) for r in self._conn.execute(
                "SELECT payload FROM beat WHERE session_id=? ORDER BY i", (session_id,))]
        return {
            "beats": beats,
            "last_board": _loads(row["last_board"]) if row else None,
            "last_analysis": _loads(row["last_analysis"]) if row else None,
            "last_tree": _loads(row["last_tree"]) if row else None,
            "history": _loads(row["history"]) if row else None,
            "view": _loads(row["view"]) if row and "view" in row.keys() else None,
            "board_seq": row["board_seq"] if row else 0,
            "beats_seq": row["beats_seq"] if row else 0,
            "tree_seq": row["tree_seq"] if row else 0,
            "analysis_seq": row["analysis_seq"] if row else 0,
        }

    def save_view(self, session_id: str, *, last_board, last_analysis, last_tree, history,
                  board_seq, beats_seq, tree_seq, analysis_seq, view=None) -> None:
        """Upsert the session's non-beat view (board/analysis/tree/history/display-view + seqs).
        Beats append separately."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_view"
                "(session_id, last_board, last_analysis, last_tree, history, view,"
                " board_seq, beats_seq, tree_seq, analysis_seq)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                " last_board=excluded.last_board, last_analysis=excluded.last_analysis,"
                " last_tree=excluded.last_tree, history=excluded.history, view=excluded.view,"
                " board_seq=excluded.board_seq, beats_seq=excluded.beats_seq,"
                " tree_seq=excluded.tree_seq, analysis_seq=excluded.analysis_seq",
                (session_id,
                 json.dumps(last_board) if last_board is not None else None,
                 json.dumps(last_analysis) if last_analysis is not None else None,
                 json.dumps(last_tree) if last_tree is not None else None,
                 json.dumps(history) if history else None,
                 json.dumps(view) if view is not None else None,
                 board_seq, beats_seq, tree_seq, analysis_seq))
            self._conn.commit()

    # -- the canonical Session Document (P1: one blob per session) ----------
    def save_document(self, session_id: str, document: dict, version: int,
                      beats: list[dict] | None = None) -> None:
        """Upsert the whole session document blob + its monotonic `version`. Beats live in the `beat`
        sidecar, but when a write appends beats they are inserted in the SAME transaction as the
        document — so the document's `beats_seq` and the beat rows can never disagree after an
        ungraceful crash between two separate commits (the sidecar-drift the design warns about)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_document(session_id, document, version) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET document=excluded.document,"
                " version=excluded.version",
                (session_id, json.dumps(document), version))
            if beats:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO beat(session_id, i, payload, ts) VALUES(?,?,?,?)",
                    [(session_id, b["i"], json.dumps(b), b.get("ts", 0.0)) for b in beats])
            self._conn.commit()

    def load_document(self, session_id: str) -> dict | None:
        """The session's document blob + version as `{document: dict, version: int}`, or None if the
        session has no blob yet (a pre-P1 session — the caller composes from the legacy columns)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT document, version FROM session_document WHERE session_id=?",
                (session_id,)).fetchone()
        if row is None:
            return None
        return {"document": json.loads(row["document"]), "version": row["version"]}

    def add_beats(self, session_id: str, beats: list[dict]) -> None:
        """Persist newly-appended beats (each already carries its index `i` and `ts`)."""
        if not beats:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO beat(session_id, i, payload, ts) VALUES(?,?,?,?)",
                [(session_id, b["i"], json.dumps(b), b.get("ts", 0.0)) for b in beats])
            self._conn.commit()

    def clear_view(self, session_id: str) -> None:
        """Drop a session's stored view + document + beats (used when its live state is reset)."""
        with self._lock:
            self._conn.execute("DELETE FROM beat WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM session_view WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM session_document WHERE session_id=?", (session_id,))
            self._conn.commit()
