"""The SQLite durable store + StateStore persistence (Phase 1: the per-session live view)."""

import os
import tempfile

import pytest

from lucena_backend.db import DB, SCHEMA_VERSION
from lucena_backend.state import StateStore


@pytest.fixture
def dbpath(tmp_path):
    return str(tmp_path / "lucena_backend.db")


def _say(text):
    return {"kind": "say", "segments": [{"text": text}]}


# -- DB unit ---------------------------------------------------------------

def test_schema_version_set(dbpath):
    db = DB(dbpath)
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_meta_round_trip(dbpath):
    db = DB(dbpath)
    assert db.get_meta("current_session") is None
    db.set_meta("current_session", "sess-A")
    assert db.get_meta("current_session") == "sess-A"
    db.set_meta("current_session", "sess-B")             # upsert
    assert db.get_meta("current_session") == "sess-B"


def test_load_view_empty_defaults(dbpath):
    view = DB(dbpath).load_view("never-seen")
    assert view["beats"] == [] and view["last_board"] is None and view["board_seq"] == 0


def test_save_and_load_view(dbpath):
    db = DB(dbpath)
    db.save_view("A", last_board={"fen": "x"}, last_analysis=None, last_tree=None, history=None,
                 board_seq=3, beats_seq=2, tree_seq=0, analysis_seq=1)
    db.add_beats("A", [{"i": 0, "ts": 1.0, **_say("one")},
                       {"i": 1, "ts": 2.0, **_say("two")}])
    view = db.load_view("A")
    assert view["last_board"] == {"fen": "x"} and view["board_seq"] == 3
    assert [b["segments"][0]["text"] for b in view["beats"]] == ["one", "two"]


def test_clear_view(dbpath):
    db = DB(dbpath)
    db.save_view("A", last_board={"fen": "x"}, last_analysis=None, last_tree=None, history=None,
                 board_seq=1, beats_seq=1, tree_seq=0, analysis_seq=0)
    db.add_beats("A", [{"i": 0, "ts": 1.0, **_say("one")}])
    db.clear_view("A")
    view = db.load_view("A")
    assert view["beats"] == [] and view["last_board"] is None


# -- StateStore persistence ------------------------------------------------

def test_beats_persist_across_restart(tmp_path):
    home, dbpath = str(tmp_path), str(tmp_path / "lucena_backend.db")
    s1 = StateStore(home, db=DB(dbpath))
    s1._switch_current("A")
    s1.write_board("r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1")
    s1.append_beats([_say("first")])
    s1.append_beats([{"kind": "you", "segments": [{"text": "Played bxc4"}]}])

    # A brand-new store + connection on the same file = a server restart.
    s2 = StateStore(home, db=DB(dbpath))
    s2._switch_current("A")
    assert [b["segments"][0]["text"] for b in s2._beats] == ["first", "Played bxc4"]
    assert s2._last_board["fen"].startswith("r1bq1rk1")
    assert s2._board_seq == 1


def test_append_beats_persists_in_one_transaction(tmp_path, monkeypatch):
    """Item 9 (regression): append_beats writes the new beats AND the document (with its bumped
    beats_seq) in a SINGLE save_document call — one transaction — not a separate add_beats then
    save_document. That closes the crash window where beats_seq could outrun the beat rows."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    calls = {"save_with_beats": 0, "add_beats": 0}
    real_save = s.db.save_document
    def spy_save(sid, doc, ver, beats=None):
        if beats:
            calls["save_with_beats"] += 1
        return real_save(sid, doc, ver, beats=beats)
    monkeypatch.setattr(s.db, "save_document", spy_save)
    monkeypatch.setattr(s.db, "add_beats",
                        lambda *a, **k: calls.__setitem__("add_beats", calls["add_beats"] + 1))
    s.append_beats([_say("one")])
    assert calls["save_with_beats"] == 1     # beats + document written together
    assert calls["add_beats"] == 0           # NOT a separate beat-only transaction


def test_sessions_are_isolated(tmp_path):
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    s.append_beats([_say("A-only")])
    s._switch_current("B")
    assert s._beats == []                     # B starts clean
    s.append_beats([_say("B-only")])
    s._switch_current("A")
    assert [b["segments"][0]["text"] for b in s._beats] == ["A-only"]   # A intact


def test_switch_publishes_reset_then_snapshot(tmp_path):
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    s.append_beats([_say("hi")])
    # Switching to B emits a reset followed by B's snapshot (which always replays beats).
    published = []
    s._publish = lambda ch, payload: published.append(ch)   # type: ignore[method-assign]
    s._switch_current("B")
    assert published[0] == "reset"
    assert "beats" in published                # snapshot always replays a beats event


def test_upsert_session_keeps_name_bumps_time(dbpath):
    db = DB(dbpath)
    db.upsert_session("A", "New session", now=100.0)
    db.upsert_session("A", "ignored-on-conflict", now=200.0)   # only updated_at moves
    rows = db.list_sessions()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "A" and rows[0]["name"] == "New session"
    assert rows[0]["updated_at"] == 200.0


def test_list_sessions_newest_first(dbpath):
    db = DB(dbpath)
    db.upsert_session("old", "o", now=1.0)
    db.upsert_session("new", "n", now=9.0)
    assert [r["session_id"] for r in db.list_sessions()] == ["new", "old"]


def test_set_session_name(dbpath):
    db = DB(dbpath)
    db.upsert_session("A", "New session", now=1.0)
    db.set_session_name("A", "Rook endgames")
    assert db.list_sessions()[0]["name"] == "Rook endgames"


def test_switch_current_records_named_session(tmp_path):
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    rows = s.db.list_sessions()
    assert rows[0]["session_id"] == "A" and rows[0]["name"]   # inserted with a name


def test_current_session_listed_before_any_transcript(tmp_path):
    from lucena_backend.sessions import list_sessions
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("current-123")
    listed = list_sessions(str(tmp_path), s.db)     # no transcripts on disk → DB is the source
    assert any(r["session_id"] == "current-123" and r["name"] for r in listed)


def test_list_sessions_is_pure_and_refresh_is_noop(tmp_path):
    """Item 12 (regression): list_sessions (used inside snapshot()) is a PURE DB read — no writes.
    In the ADK path there is no Claude transcript, so refresh_session_names is a no-op: the DB `name`
    is authoritative and untouched (transcript-title caching was removed in the pivot)."""
    from lucena_backend import sessions as S
    db = DB(str(tmp_path / "lucena_backend.db"))
    db.upsert_session("A", "New session", now=1.0)
    assert S.list_sessions(str(tmp_path), db)[0]["name"] == "New session"   # pure read: DB untouched
    S.refresh_session_names(str(tmp_path), db)                              # no-op in the ADK path
    assert db.list_sessions()[0]["name"] == "New session"                  # unchanged — no caching


def test_snapshot_does_not_write_the_db(tmp_path, monkeypatch):
    """Item 12 (regression): building the SSE snapshot must not MUTATE the DB — the old sessions-rail
    read cached titles via set_session_name mid-snapshot, on the event loop."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s.ensure_session_id()
    writes = []
    monkeypatch.setattr(s.db, "set_session_name", lambda *a, **k: writes.append(a))   # spy
    s.snapshot()                                     # builds every SSE payload incl. the sessions rail
    assert writes == []                              # …without a single DB name write


def test_no_db_is_pure_in_memory(tmp_path):
    # Without a DB the store still works (files + in-memory), and nothing tries to persist.
    s = StateStore(str(tmp_path))
    s._switch_current("A")
    s.append_beats([_say("x")])
    assert [b["segments"][0]["text"] for b in s._beats] == ["x"]
    assert not os.path.exists(str(tmp_path / "lucena_backend.db"))
