"""Postgres durable store — the single durable store the backend owns.

Relational, per B3: the session document is DECOMPOSED into real tables on save and
REASSEMBLED on load, so `state.py`'s in-memory model is untouched (it still calls
`save_document(document)` / `load_document() -> {document, version}`). jsonb is used
only for the four genuinely-variable/tree-shaped fields: `poisoned.meta`, `view.tree`,
`gate_pending`, `drill_close`. `last_analysis` / `last_tree` / `drill_state` are NOT
persisted (re-derived on resume); `last_board` is REBUILT from `fen` + decorations +
poisoned (the same projection `state.py._build_board` computes).

One connection, serialized behind a lock (some tool calls run off the event loop).
Test isolation: each `DB(path)` maps its path to a unique Postgres SCHEMA in the
configured database (default `postgresql:///lucena_dev`, override `LUCENA_PG_DSN`),
so per-test stores never collide.
"""

from __future__ import annotations

import hashlib
import os
import threading

import psycopg
from psycopg.types.json import Jsonb

from lucena_engine.board import Board
from lucena_engine._fen import norm_fen

SCHEMA_VERSION = 7               # bumped from the SQLite lineage (6) — the Postgres relational cut
_BOARD_SCHEMA = 1                # matches state.SCHEMA (the board object's "schema" field)

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text);
CREATE TABLE IF NOT EXISTS session (
    id text PRIMARY KEY, name text, created_at double precision, updated_at double precision,
    status text NOT NULL DEFAULT 'active', is_active boolean NOT NULL DEFAULT false);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_session ON session (is_active) WHERE is_active;

-- legacy per-session view (pre-P1 fallback + the beats sidecar's companion columns)
CREATE TABLE IF NOT EXISTS session_view (
    session_id text PRIMARY KEY,
    last_board jsonb, last_analysis jsonb, last_tree jsonb, history jsonb, view jsonb,
    board_seq int NOT NULL DEFAULT 0, beats_seq int NOT NULL DEFAULT 0,
    tree_seq int NOT NULL DEFAULT 0, analysis_seq int NOT NULL DEFAULT 0);

-- the canonical document, decomposed --------------------------------------
CREATE TABLE IF NOT EXISTS session_doc (
    session_id text PRIMARY KEY, version bigint NOT NULL DEFAULT 0, beats_seq int NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'active', gate_awaiting boolean NOT NULL DEFAULT false,
    gate_pending jsonb, drill_close jsonb);
CREATE TABLE IF NOT EXISTS banked (
    session_id text, ord int, concept_id text, PRIMARY KEY (session_id, ord));
CREATE TABLE IF NOT EXISTS served_puzzle (
    session_id text, ord int, puzzle_id text, PRIMARY KEY (session_id, ord));
CREATE TABLE IF NOT EXISTS activity (
    session_id text, idx int, kind text NOT NULL DEFAULT 'conversation',
    board_seq int NOT NULL DEFAULT 0, tree_seq int NOT NULL DEFAULT 0, analysis_seq int NOT NULL DEFAULT 0,
    history_seq int NOT NULL DEFAULT 0, view_seq int NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, idx));
CREATE TABLE IF NOT EXISTS board (           -- last_board's fen (last_board is rebuilt via projection)
    session_id text, activity_idx int, fen text, PRIMARY KEY (session_id, activity_idx));
CREATE TABLE IF NOT EXISTS decoration (
    session_id text, activity_idx int, for_fen text, caption text,
    eval_cp int, eval_win_pct double precision, has_eval boolean NOT NULL DEFAULT false,
    PRIMARY KEY (session_id, activity_idx));
CREATE TABLE IF NOT EXISTS decoration_arrow (
    session_id text, activity_idx int, ord int, from_sq text, to_sq text, style text, fact_id text,
    PRIMARY KEY (session_id, activity_idx, ord));
CREATE TABLE IF NOT EXISTS decoration_highlight (
    session_id text, activity_idx int, ord int, square text, style text, fact_id text,
    PRIMARY KEY (session_id, activity_idx, ord));
CREATE TABLE IF NOT EXISTS poisoned (
    session_id text, activity_idx int, for_fen text, meta jsonb, PRIMARY KEY (session_id, activity_idx));
CREATE TABLE IF NOT EXISTS poisoned_move (
    session_id text, activity_idx int, ord int, uci text, san text, fen text,
    PRIMARY KEY (session_id, activity_idx, ord));
CREATE TABLE IF NOT EXISTS ply (              -- history
    session_id text, activity_idx int, n int, san text, uci text, fen text,
    PRIMARY KEY (session_id, activity_idx, n));
CREATE TABLE IF NOT EXISTS view (
    session_id text, activity_idx int, fen text, cursor int, side_to_move text,
    line jsonb, tree jsonb, extra jsonb, PRIMARY KEY (session_id, activity_idx));
CREATE TABLE IF NOT EXISTS beat (
    session_id text, i int, payload jsonb NOT NULL, ts double precision NOT NULL,
    PRIMARY KEY (session_id, i));
"""

_DSN = os.environ.get("LUCENA_PG_DSN", "postgresql:///lucena_dev")

# view keys we lift into columns; everything else on the view rides in `extra` (still not a document blob).
_VIEW_COLS = {"fen", "cursor", "side_to_move", "line", "tree"}


def _schema_for(path: str) -> str:
    return "s_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:20]


class DB:
    """The backend's Postgres store. All access serialized behind one lock."""

    def __init__(self, path: str, *, dsn: str | None = None):
        self._lock = threading.RLock()
        self._schema = _schema_for(path)
        self._conn = psycopg.connect(dsn or _DSN, autocommit=False)
        with self._lock:
            self._conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"')
            self._conn.execute(f'SET search_path TO "{self._schema}"')
            self._conn.execute(_DDL)
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',%s) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(SCHEMA_VERSION),))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ex(self, sql, params=()):
        return self._conn.execute(sql, params)

    # -- meta --------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._ex("SELECT value FROM meta WHERE key=%s", (key,)).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._ex("INSERT INTO meta(key,value) VALUES(%s,%s) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self._conn.commit()

    # -- sessions rail -----------------------------------------------------
    def upsert_session(self, session_id: str, name: str, now: float) -> None:
        with self._lock:
            self._ex("INSERT INTO session(id,name,created_at,updated_at) VALUES(%s,%s,%s,%s) "
                     "ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at",
                     (session_id, name, now, now))
            self._conn.commit()

    def set_session_name(self, session_id: str, name: str) -> None:
        with self._lock:
            self._ex("UPDATE session SET name=%s WHERE id=%s", (name, session_id))
            self._conn.commit()

    def set_session_status(self, session_id: str, status: str) -> None:
        with self._lock:
            self._ex("UPDATE session SET status=%s WHERE id=%s", (status, session_id))
            self._conn.commit()

    def set_active_session(self, session_id: str) -> None:
        """At most one session is_active (replaces the session.json pointer)."""
        with self._lock:
            self._ex("UPDATE session SET is_active=false WHERE is_active")
            self._ex("UPDATE session SET is_active=true WHERE id=%s", (session_id,))
            self._conn.commit()

    def get_active_session(self) -> str | None:
        with self._lock:
            row = self._ex("SELECT id FROM session WHERE is_active").fetchone()
            return row[0] if row else None

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._ex("SELECT id,name,updated_at,status FROM session "
                            "ORDER BY updated_at DESC").fetchall()
        return [{"session_id": r[0], "name": r[1], "updated_at": r[2], "status": r[3] or "active"}
                for r in rows]

    # -- legacy per-session view (fallback + beats) ------------------------
    def load_view(self, session_id: str) -> dict:
        with self._lock:
            row = self._ex("SELECT last_board,last_analysis,last_tree,history,view,"
                           "board_seq,beats_seq,tree_seq,analysis_seq FROM session_view "
                           "WHERE session_id=%s", (session_id,)).fetchone()
            beats = [r[0] for r in self._ex(
                "SELECT payload FROM beat WHERE session_id=%s ORDER BY i", (session_id,)).fetchall()]
        return {
            "beats": beats,
            "last_board": row[0] if row else None, "last_analysis": row[1] if row else None,
            "last_tree": row[2] if row else None, "history": row[3] if row else None,
            "view": row[4] if row else None,
            "board_seq": row[5] if row else 0, "beats_seq": row[6] if row else 0,
            "tree_seq": row[7] if row else 0, "analysis_seq": row[8] if row else 0,
        }

    def save_view(self, session_id: str, *, last_board, last_analysis, last_tree, history,
                  board_seq, beats_seq, tree_seq, analysis_seq, view=None) -> None:
        with self._lock:
            self._ex(
                "INSERT INTO session_view(session_id,last_board,last_analysis,last_tree,history,view,"
                "board_seq,beats_seq,tree_seq,analysis_seq) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(session_id) DO UPDATE SET last_board=excluded.last_board,"
                "last_analysis=excluded.last_analysis,last_tree=excluded.last_tree,"
                "history=excluded.history,view=excluded.view,board_seq=excluded.board_seq,"
                "beats_seq=excluded.beats_seq,tree_seq=excluded.tree_seq,"
                "analysis_seq=excluded.analysis_seq",
                (session_id, _j(last_board), _j(last_analysis), _j(last_tree), _j(history), _j(view),
                 board_seq, beats_seq, tree_seq, analysis_seq))
            self._conn.commit()

    # -- the canonical document (decomposed) -------------------------------
    def save_document(self, session_id: str, document: dict, version: int,
                      beats: list[dict] | None = None) -> None:
        with self._lock:
            self._ex(
                "INSERT INTO session_doc(session_id,version,beats_seq,status,gate_awaiting,"
                "gate_pending,drill_close) VALUES(%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(session_id) DO UPDATE SET version=excluded.version,"
                "beats_seq=excluded.beats_seq,status=excluded.status,"
                "gate_awaiting=excluded.gate_awaiting,gate_pending=excluded.gate_pending,"
                "drill_close=excluded.drill_close",
                (session_id, version, document.get("beats_seq", 0), document.get("status", "active"),
                 bool((document.get("gate") or {}).get("awaiting")),
                 _j((document.get("gate") or {}).get("pending")), _j(document.get("drill_close"))))
            self._replace_list("banked", session_id, document.get("banked") or [])
            self._replace_list("served_puzzle", session_id, document.get("served_puzzles") or [])
            self._save_activities(session_id, document.get("activities") or [])
            if beats:
                self._insert_beats(session_id, beats)
            self._conn.commit()

    def load_document(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._ex("SELECT version,beats_seq,status,gate_awaiting,gate_pending,drill_close "
                           "FROM session_doc WHERE session_id=%s", (session_id,)).fetchone()
            if row is None:
                return None
            banked = [r[0] for r in self._ex(
                "SELECT concept_id FROM banked WHERE session_id=%s ORDER BY ord", (session_id,)).fetchall()]
            served = [r[0] for r in self._ex(
                "SELECT puzzle_id FROM served_puzzle WHERE session_id=%s ORDER BY ord",
                (session_id,)).fetchall()]
            activities = self._load_activities(session_id)
        document = {
            "version": row[0], "beats_seq": row[1], "status": row[2],
            "banked": banked, "served_puzzles": served, "drill_close": row[5],
            "gate": {"awaiting": bool(row[3]), "pending": row[4]},
            "activities": activities,
        }
        return {"document": document, "version": row[0]}

    def add_beats(self, session_id: str, beats: list[dict]) -> None:
        if not beats:
            return
        with self._lock:
            self._insert_beats(session_id, beats)
            self._conn.commit()

    def clear_view(self, session_id: str) -> None:
        with self._lock:
            for t in ("beat", "session_view", "session_doc", "banked", "served_puzzle", "activity",
                      "board", "decoration", "decoration_arrow", "decoration_highlight", "poisoned",
                      "poisoned_move", "ply", "view"):
                self._ex(f"DELETE FROM {t} WHERE session_id=%s", (session_id,))
            self._conn.commit()

    # -- decompose helpers -------------------------------------------------
    def _replace_list(self, table: str, session_id: str, items: list) -> None:
        self._ex(f"DELETE FROM {table} WHERE session_id=%s", (session_id,))
        col = "concept_id" if table == "banked" else "puzzle_id"
        for i, v in enumerate(items):
            self._ex(f"INSERT INTO {table}(session_id,ord,{col}) VALUES(%s,%s,%s)", (session_id, i, v))

    def _insert_beats(self, session_id: str, beats: list[dict]) -> None:
        for b in beats:
            self._ex("INSERT INTO beat(session_id,i,payload,ts) VALUES(%s,%s,%s,%s) "
                     "ON CONFLICT(session_id,i) DO UPDATE SET payload=excluded.payload,ts=excluded.ts",
                     (session_id, b["i"], Jsonb(b), b.get("ts", 0.0)))

    def _save_activities(self, session_id: str, activities: list[dict]) -> None:
        for t in ("activity", "board", "decoration", "decoration_arrow", "decoration_highlight",
                  "poisoned", "poisoned_move", "ply", "view"):
            self._ex(f"DELETE FROM {t} WHERE session_id=%s", (session_id,))
        for idx, frame in enumerate(activities):
            ws = frame.get("workspace") or {}
            self._ex("INSERT INTO activity(session_id,idx,kind,board_seq,tree_seq,analysis_seq,"
                     "history_seq,view_seq) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                     (session_id, idx, frame.get("kind", "conversation"),
                      ws.get("board_seq", 0), ws.get("tree_seq", 0), ws.get("analysis_seq", 0),
                      ws.get("history_seq", 0), ws.get("view_seq", 0)))
            lb = ws.get("last_board")
            if lb and lb.get("fen"):
                self._ex("INSERT INTO board(session_id,activity_idx,fen) VALUES(%s,%s,%s)",
                         (session_id, idx, lb["fen"]))
            self._save_decorations(session_id, idx, ws.get("decorations"))
            self._save_poisoned(session_id, idx, ws.get("poisoned"))
            for p in (ws.get("history") or []):
                self._ex("INSERT INTO ply(session_id,activity_idx,n,san,uci,fen) VALUES(%s,%s,%s,%s,%s,%s)",
                         (session_id, idx, p.get("n"), p.get("san"), p.get("uci"), p.get("fen")))
            self._save_view(session_id, idx, ws.get("view"))

    def _save_decorations(self, session_id, idx, d) -> None:
        if not d:
            return
        ev = d.get("eval")
        self._ex("INSERT INTO decoration(session_id,activity_idx,for_fen,caption,eval_cp,eval_win_pct,"
                 "has_eval) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                 (session_id, idx, d.get("for_fen"), d.get("caption"),
                  (ev or {}).get("cp"), (ev or {}).get("win_pct"), ev is not None))
        for i, a in enumerate(d.get("arrows") or []):
            self._ex("INSERT INTO decoration_arrow(session_id,activity_idx,ord,from_sq,to_sq,style,"
                     "fact_id) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                     (session_id, idx, i, a.get("from"), a.get("to"), a.get("style"), a.get("fact_id")))
        for i, h in enumerate(d.get("highlights") or []):
            self._ex("INSERT INTO decoration_highlight(session_id,activity_idx,ord,square,style,fact_id)"
                     " VALUES(%s,%s,%s,%s,%s,%s)",
                     (session_id, idx, i, h.get("square"), h.get("style"), h.get("fact_id")))

    def _save_poisoned(self, session_id, idx, p) -> None:
        if not p:
            return
        self._ex("INSERT INTO poisoned(session_id,activity_idx,for_fen,meta) VALUES(%s,%s,%s,%s)",
                 (session_id, idx, p.get("for_fen"), _j(p.get("meta"))))
        for i, m in enumerate(p.get("moves") or []):
            self._ex("INSERT INTO poisoned_move(session_id,activity_idx,ord,uci,san,fen) "
                     "VALUES(%s,%s,%s,%s,%s,%s)",
                     (session_id, idx, i, m.get("uci"), m.get("san"), m.get("fen")))

    def _save_view(self, session_id, idx, v) -> None:
        if not v:
            return
        extra = {k: val for k, val in v.items() if k not in _VIEW_COLS}
        self._ex("INSERT INTO view(session_id,activity_idx,fen,cursor,side_to_move,line,tree,extra) "
                 "VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                 (session_id, idx, v.get("fen"), v.get("cursor"), v.get("side_to_move"),
                  _j(v.get("line")), _j(v.get("tree")), _j(extra or None)))

    # -- reassemble helpers ------------------------------------------------
    def _load_activities(self, session_id: str) -> list[dict]:
        acts = self._ex("SELECT idx,kind,board_seq,tree_seq,analysis_seq,history_seq,view_seq "
                        "FROM activity WHERE session_id=%s ORDER BY idx", (session_id,)).fetchall()
        frames = []
        for a in acts:
            idx = a[0]
            decorations = self._load_decorations(session_id, idx)
            poisoned = self._load_poisoned(session_id, idx)
            brow = self._ex("SELECT fen FROM board WHERE session_id=%s AND activity_idx=%s",
                            (session_id, idx)).fetchone()
            last_board = _project_board(brow[0] if brow else None, a[2], decorations, poisoned)
            history = [{"n": r[0], "san": r[1], "uci": r[2], "fen": r[3]} for r in self._ex(
                "SELECT n,san,uci,fen FROM ply WHERE session_id=%s AND activity_idx=%s ORDER BY n",
                (session_id, idx)).fetchall()]
            ws = {
                "last_board": last_board, "last_analysis": None, "last_tree": None,
                "history": history, "view": self._load_view_row(session_id, idx),
                "drill_state": None, "poisoned": poisoned, "decorations": decorations,
                "board_seq": a[2], "tree_seq": a[3], "analysis_seq": a[4],
                "history_seq": a[5], "view_seq": a[6],
            }
            frames.append({"kind": a[1], "workspace": ws})
        return frames

    def _load_decorations(self, session_id, idx):
        d = self._ex("SELECT for_fen,caption,eval_cp,eval_win_pct,has_eval FROM decoration "
                     "WHERE session_id=%s AND activity_idx=%s", (session_id, idx)).fetchone()
        if d is None:
            return None
        arrows = [{"from": r[0], "to": r[1], "style": r[2], "fact_id": r[3]} for r in self._ex(
            "SELECT from_sq,to_sq,style,fact_id FROM decoration_arrow WHERE session_id=%s AND "
            "activity_idx=%s ORDER BY ord", (session_id, idx)).fetchall()]
        highlights = [{"square": r[0], "style": r[1], "fact_id": r[2]} for r in self._ex(
            "SELECT square,style,fact_id FROM decoration_highlight WHERE session_id=%s AND "
            "activity_idx=%s ORDER BY ord", (session_id, idx)).fetchall()]
        ev = {"cp": d[2], "win_pct": d[3]} if d[4] else None
        return {"for_fen": d[0], "arrows": arrows, "highlights": highlights,
                "caption": d[1], "eval": ev}

    def _load_poisoned(self, session_id, idx):
        p = self._ex("SELECT for_fen,meta FROM poisoned WHERE session_id=%s AND activity_idx=%s",
                     (session_id, idx)).fetchone()
        if p is None:
            return None
        moves = [{"uci": r[0], "san": r[1], "fen": r[2]} for r in self._ex(
            "SELECT uci,san,fen FROM poisoned_move WHERE session_id=%s AND activity_idx=%s ORDER BY ord",
            (session_id, idx)).fetchall()]
        return {"for_fen": p[0], "moves": moves, "meta": p[1] or {}}

    def _load_view_row(self, session_id, idx):
        v = self._ex("SELECT fen,cursor,side_to_move,line,tree,extra FROM view "
                     "WHERE session_id=%s AND activity_idx=%s", (session_id, idx)).fetchone()
        if v is None:
            return None
        out = {"fen": v[0], "cursor": v[1], "side_to_move": v[2], "line": v[3], "tree": v[4]}
        if v[5]:
            out.update(v[5])
        return {k: val for k, val in out.items() if val is not None}


def _j(obj):
    return Jsonb(obj) if obj is not None else None


def _project_board(fen, board_seq, decorations, poisoned):
    """Rebuild last_board — the same projection state.py._build_board computes from
    fen + the durable decoration/poisoned slots (shown only when keyed to THIS fen)."""
    if not fen:
        return None
    side = "white" if fen.split()[1] == "w" else "black"
    terminal = None
    try:
        b = Board(fen)
        if not b.legal_moves():
            terminal = "checkmate" if b.in_check else "stalemate"
    except Exception:
        pass
    d = decorations or {}
    on = bool(d) and norm_fen(d.get("for_fen", "")) == norm_fen(fen)
    p = poisoned
    poisoned_here = bool(p and p.get("moves") and norm_fen(p.get("for_fen", "")) == norm_fen(fen))
    return {
        "schema": _BOARD_SCHEMA, "seq": board_seq, "fen": fen, "side_to_move": side, "terminal": terminal,
        "arrows": (d.get("arrows") or []) if on else [],
        "highlights": (d.get("highlights") or []) if on else [],
        "caption": d.get("caption") if on else None,
        "eval": d.get("eval") if on else None,
        "has_poisoned_line": poisoned_here,
        "poisoned_line": (p["moves"] if poisoned_here else None),
    }
