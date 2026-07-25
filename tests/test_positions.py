"""The fen → positionJSON cache (2026-07-26, owner: "the entire position is being recalculated.
Upon every move even when I go back ... There needs to be a fen: positionJSON cache").

Analysis is a pure function of the position, so it is cached by the position and shared across
chats and restarts. These pin the contract that matters: a cached position is ANSWERED, never
re-rolled; the key ignores the move clocks; and a cache that fails is a miss, never an error.
"""

import pytest

from lucena_backend import margin, positions
from lucena_backend.margin import build
from lucena_backend.positions import LINES, SHEET, PositionCache

OUT_OF_BOOK = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP2PPP/R2Q1RK1 w - - 4 9"
SAME_POSITION_LATER = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP2PPP/R2Q1RK1 w - - 31 40"


class _FakeDB:
    """The durable half, in a dict — the real one is Postgres."""

    def __init__(self, *, broken=False):
        self.rows, self.broken, self.writes = {}, broken, 0

    def get_position(self, fen, kind):
        if self.broken:
            raise RuntimeError("no database")
        return self.rows.get((fen, kind))

    def put_position(self, fen, kind, payload, now):
        if self.broken:
            raise RuntimeError("no database")
        self.writes += 1
        self.rows[(fen, kind)] = payload

    def prune_positions(self, keep):
        return 0


def test_the_clocks_are_not_part_of_the_position():
    """Halfmove/fullmove counters differ between two visits to the same position; an engine reads
    them as the same board, so the cache must too — including them fragmented every transposition."""
    db = _FakeDB()
    cache = PositionCache(db)
    cache.put(OUT_OF_BOOK, SHEET, {"schema": "x"})
    assert cache.get(SAME_POSITION_LATER, SHEET) == {"schema": "x"}
    assert positions.norm(OUT_OF_BOOK) == positions.norm(SAME_POSITION_LATER)


def test_memory_answers_without_touching_the_database():
    db = _FakeDB()
    cache = PositionCache(db)
    cache.put(OUT_OF_BOOK, SHEET, {"a": 1})
    db.rows.clear()                                    # only the memory layer can answer now
    assert cache.get(OUT_OF_BOOK, SHEET) == {"a": 1}


def test_a_restart_still_finds_it():
    db = _FakeDB()
    PositionCache(db).put(OUT_OF_BOOK, SHEET, {"a": 1})
    assert PositionCache(db).get(OUT_OF_BOOK, SHEET) == {"a": 1}     # fresh process, same database


def test_memory_is_bounded_by_recency():
    cache = PositionCache(None, mem_max=2)
    for i in range(3):
        cache.put(f"{i}/8/8/8/8/8/8/8 w - -", SHEET, {"i": i})
    assert cache.get("0/8/8/8/8/8/8/8 w - -", SHEET) is None         # oldest use evicted
    assert cache.get("2/8/8/8/8/8/8/8 w - -", SHEET) == {"i": 2}


def test_a_broken_database_is_a_miss_not_a_failure():
    """A cache is a convenience: every caller can recompute. It must never take a read down."""
    cache = PositionCache(_FakeDB(broken=True))
    cache.put(OUT_OF_BOOK, SHEET, {"a": 1})            # write fails, silently
    assert cache.get(OUT_OF_BOOK, SHEET) == {"a": 1}   # ...memory still has it
    other = PositionCache(_FakeDB(broken=True))
    assert other.get(OUT_OF_BOOK, SHEET) is None       # a cold process just misses


def test_kinds_do_not_collide():
    cache = PositionCache(_FakeDB())
    cache.put(OUT_OF_BOOK, SHEET, {"which": "sheet"})
    cache.put(OUT_OF_BOOK, LINES, {"which": "lines"})
    assert cache.get(OUT_OF_BOOK, SHEET) == {"which": "sheet"}
    assert cache.get(OUT_OF_BOOK, LINES) == {"which": "lines"}


# -- the margin reads through it ---------------------------------------------

@pytest.fixture
def margin_cache(monkeypatch):
    cache = PositionCache(_FakeDB())
    monkeypatch.setattr(margin, "_positions", cache)
    monkeypatch.setattr(margin, "_deep_cache", {})
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin, "_latest_by_session", {})
    monkeypatch.setattr(margin.theory, "theory_for", lambda fen: None)
    yield cache
    margin._deep_cache.clear()


def test_a_stored_sheet_is_served_instead_of_rolled(margin_cache, monkeypatch):
    """The whole point: a position analysed before is answered, not re-rolled."""
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin._worker, "submit", lambda *a, **k: submitted.append(a))
    margin_cache.put(OUT_OF_BOOK, SHEET, {"schema": "lucena-plans/sheet@1"})

    m = build(OUT_OF_BOOK)
    assert m["plansPending"] is False                  # answered outright
    assert m["sheet"] == {"schema": "lucena-plans/sheet@1"}
    assert '"schema"' in m["raw"]
    assert submitted == []                             # ...and nothing was rolled


def test_a_finished_roll_is_stored_but_a_pending_one_is_not(margin_cache):
    key = " ".join(OUT_OF_BOOK.split()[:4])
    margin._cache(key, {"pre": True}, None, True)      # mid-roll snapshot
    assert margin_cache.get(OUT_OF_BOOK, SHEET) is None
    margin._cache(key, {"post": True}, None, False)    # the verified sheet
    assert margin_cache.get(OUT_OF_BOOK, SHEET) == {"post": True}


def test_a_failed_roll_is_never_stored(margin_cache):
    """Tomorrow's visit should retry, not inherit today's failure."""
    key = " ".join(OUT_OF_BOOK.split()[:4])
    margin._cache(key, {"error": "sheet failed"}, "ERROR", False)
    assert margin_cache.get(OUT_OF_BOOK, SHEET) is None


# -- a broken cache must not poison the connection ---------------------------

def test_a_failed_cache_write_leaves_the_database_usable(tmp_path):
    """PositionCache SWALLOWS database errors to degrade to a miss — which is only survivable if the
    connection survives too. A failed statement leaves psycopg's transaction aborted, so without a
    rollback the next unrelated query dies with InFailedSqlTransaction and a broken cache takes the
    whole backend down (Codex)."""
    from lucena_backend.persistence.db import DB
    db = DB(str(tmp_path / "lucena"))
    db.set_meta("canary", "before")

    with pytest.raises(Exception):                     # a payload psycopg cannot adapt
        db.put_position("8/8/8/8/8/8/8/8 w - -", SHEET, {"bad": {1, 2, 3}}, 0.0)

    assert db.get_meta("canary") == "before"           # the connection still works...
    db.set_meta("canary", "after")                     # ...for writes as well
    assert db.get_meta("canary") == "after"
    db.put_position("8/8/8/8/8/8/8/8 w - -", SHEET, {"ok": True}, 1.0)
    assert db.get_position("8/8/8/8/8/8/8/8 w - -", SHEET) == {"ok": True}


def test_the_cache_swallows_that_same_failure(tmp_path):
    """...and the caller above it sees a miss, not an exception."""
    from lucena_backend.persistence.db import DB
    db = DB(str(tmp_path / "lucena"))
    cache = PositionCache(db)
    cache.put("8/8/8/8/8/8/8/8 w - -", SHEET, {"bad": {1, 2, 3}})   # no raise
    db.set_meta("canary", "still here")
    assert db.get_meta("canary") == "still here"
