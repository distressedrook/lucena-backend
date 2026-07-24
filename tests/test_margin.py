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


def test_epigraph_is_the_move_zero_cover_only():
    """The epigraph is the MOVE-0 cover only: deterministic per session seed,
    and GONE the instant a move is played (owner: 'when I made a move it
    showed the quote again' — the cover must end at ply 0)."""
    a = build(START, seed="session-1")["epigraph"]
    assert a and build(START, seed="session-1")["epigraph"] == a   # stable per seed
    assert build(START, seed="session-2")["epigraph"] != a          # varies by seed
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert build(after_e4, seed="session-1")["epigraph"] is None     # cover ended


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


# -- the Wikibooks theory gate (2026-07-24) -----------------------------------

_WB_ENTRY = {
    "name": "Some Opening", "eco": "A00",
    "description": "A verbatim Wikibooks description of this position, long "
                   "enough to be a real lead paragraph and not a stub remnant.",
    "responses": ["2. Nf3 - Main line"],
    "source_url": "https://en.wikibooks.org/wiki/Chess_Opening_Theory/x",
}


def test_wikibooks_entry_gates_the_sheet(monkeypatch):
    """A position with a Wikibooks entry shows its THEORY (verbatim + CC BY-SA
    attribution) and MUST NOT roll the positional sheet — no deep job, no
    pending. (owner: 'don't show all the positions if it has a wikibooks
    entry')."""
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: _WB_ENTRY if fen == OUT_OF_BOOK else None)
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())          # "configured"
    monkeypatch.setattr(margin._worker, "submit",
                        lambda *a, **k: submitted.append(a))
    m = build(OUT_OF_BOOK, live=True)
    assert submitted == []                                  # never rolled
    assert m["plansPending"] is False and m["sheet"] is None
    assert m["masthead"] == "Some Opening"
    assert m["theory"]["idea"] and _WB_ENTRY["description"].startswith(
        m["theory"]["idea"][:40])
    attr = m["theory"]["attribution"]
    assert attr and attr["url"] == _WB_ENTRY["source_url"]
    assert "CC BY-SA" in attr["text"]


def test_no_wikibooks_entry_still_rolls(monkeypatch):
    """The gate is scoped to Wikibooks positions: an out-of-book position
    WITHOUT an entry must still roll the sheet exactly as before."""
    monkeypatch.setattr(margin.theory, "theory_for", lambda fen: None)
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin._worker, "submit",
                        lambda *a, **k: submitted.append(a))
    m = build(OUT_OF_BOOK, live=True)
    assert m["plansPending"] is True                        # rolled as before
    assert len(submitted) == 1


def test_named_without_annotation_falls_back_to_wikibooks(monkeypatch):
    """A position with an opening NAME but no authored annotation must still
    use the Wikibooks description + attribution — authored-first, Wikibooks-
    otherwise, independent of whether a name exists."""
    monkeypatch.setattr(margin.openings, "name_for", lambda fen: "Some Named Line")
    monkeypatch.setattr(margin.authored, "annotation_for", lambda name: None)
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: _WB_ENTRY if fen == OUT_OF_BOOK else None)
    m = build(OUT_OF_BOOK, seed="s")
    assert m["masthead"] == "Some Named Line"          # name wins the masthead
    assert m["theory"]["idea"] == _WB_ENTRY["description"]   # VERBATIM, full
    attr = m["theory"]["attribution"]
    assert attr and attr["url"] == _WB_ENTRY["source_url"]


def test_wikibooks_idea_is_verbatim_not_truncated(monkeypatch):
    """The Wikibooks lead is shown as-is (CC BY-SA 'as-is' contract) — never
    passed through _lead_sentences."""
    long_desc = ("First sentence of theory. Second sentence adds nuance. "
                 "Third sentence closes it out.")
    entry = {**_WB_ENTRY, "description": long_desc}
    monkeypatch.setattr(margin.openings, "name_for", lambda fen: None)
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: entry if fen == OUT_OF_BOOK else None)
    m = build(OUT_OF_BOOK, seed="s")
    assert m["theory"]["idea"] == long_desc            # full, untruncated


def test_unattributable_wikibooks_is_not_theory_and_still_rolls(monkeypatch):
    """A Wikibooks entry WITHOUT source_url can't be shown (CC BY-SA needs the
    credit), so it is NOT treated as theory: no empty card, and the positional
    sheet still rolls."""
    no_src = {"name": "X", "description": "text", "responses": []}   # no source_url
    monkeypatch.setattr(margin.openings, "name_for", lambda fen: None)
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: no_src if fen == OUT_OF_BOOK else None)
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin._worker, "submit",
                        lambda *a, **k: submitted.append(a))
    m = build(OUT_OF_BOOK, live=True)
    assert m["theory"] is None                     # not gated behind an empty card
    assert m["plansPending"] is True and len(submitted) == 1   # rolls as normal
