"""The /margin content builder — JSON-inspection mode (owner, 2026-07-23):
the margin shows the plans layer's raw pre/post-verify JSON. The former
card builder (epigraph/theory/position cards) lives at backend fb4b2d7."""
import pytest

from lucena_backend import margin
from lucena_backend.margin import build

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUT_OF_BOOK = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP2PPP/R2Q1RK1 w - - 4 9"


def test_unconfigured_margin_is_bare_and_never_pends():
    m = build(OUT_OF_BOOK)                     # no pool configured, not live
    assert m["raw"] is None and m["plansPending"] is False


def test_cached_result_is_served_raw():
    key = " ".join(OUT_OF_BOOK.split()[:4])
    # _cache takes the SHEET DICT now (it does its own pretty-printing)
    margin._cache(key, {"schema": "lucena-plans/sheet@1"}, "POST-VERIFY", False)
    try:
        m = build(OUT_OF_BOOK)
        assert m["statusLine"] == "POST-VERIFY"
        assert '"schema"' in m["raw"]
        assert m["plansPending"] is False
    finally:
        margin._deep_cache.clear()


def test_pending_pre_keeps_polling_alive():
    # OUT_OF_BOOK, not START: the start position is stage 1 (the epigraph)
    # and never reaches the deep cache — only the out-of-book stage polls.
    key = " ".join(OUT_OF_BOOK.split()[:4])
    margin._cache(key, {}, "PRE-VERIFY · VERIFYING…", True)
    try:
        m = build(OUT_OF_BOOK)
        assert m["plansPending"] is True and m["raw"] == "{}" and m["sheet"] == {}
    finally:
        margin._deep_cache.clear()


def test_bad_fen_raises():
    with pytest.raises(ValueError):
        build("not a fen")


# -- the authored-content stages (/content wired 2026-07-24) ------------------

SICILIAN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"


def test_move_one_serves_a_sourced_epigraph():
    m = build(START, seed="session-1")
    e = m["epigraph"]
    assert e and e["quote"] and e["author"]
    assert m["theory"] is None and m["plansPending"] is False


def test_epigraph_is_seeded_and_stable():
    """A book keeps its epigraph: same seed -> same quote, across BOTH plies
    of move 1 (a fen-derived seed changed it between ply 0 and ply 1)."""
    ply1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    a = build(START, seed="session-1")["epigraph"]
    b = build(ply1, seed="session-1")["epigraph"]
    assert a == b
    assert build(START, seed="session-2")["epigraph"] != a


def test_in_book_serves_the_authored_theory_card():
    m = build(SICILIAN, seed="s")
    assert m["masthead"] == "Sicilian Defense"
    assert m["statusLine"] == "OPENING · MOVE 2"
    idea = m["theory"]["idea"]
    assert idea and "Sicilian" in idea
    # the card takes the LEAD sentences, not the whole essay. (Don't count
    # "." — move notation like "1.e4" contains one; _lead_sentences only
    # breaks on a period followed by whitespace, which is why it survives.)
    from lucena_core import content as authored
    full = authored.annotation_for("Sicilian Defense")
    assert len(idea) < len(full)
    assert full.startswith(idea[:40])
    doors = m["theory"]["doors"]
    assert doors and all(d["san"] and d["variation"] for d in doors)
    assert len(doors) <= 4
    assert m["epigraph"] is None and m["plansPending"] is False
