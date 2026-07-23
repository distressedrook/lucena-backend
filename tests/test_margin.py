"""The /margin content builder — the owner's staged wiring (2026-07-24):
move 1 → epigraph · in book → theory with labeled doors · out of book →
honest minimal status (plans zone pending)."""
import pytest

from lucena_backend.margin import build, _plies_played, _lead_sentences

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
SCANDI = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"   # 1.e4 d5
OUT_OF_BOOK = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP2PPP/R2Q1RK1 w - - 4 9"


def test_move_one_is_the_epigraph_and_it_is_seeded():
    a = build(START, seed="session-1")
    assert a["epigraph"] and a["epigraph"]["quote"] and a["epigraph"]["author"]
    assert a["theory"] is None and a["cards"] == []
    assert build(START, seed="session-1")["epigraph"] == a["epigraph"]   # stable
    b = build(AFTER_E4, seed="session-1")
    assert b["epigraph"] is not None                                      # ply 1 still move-1


def test_in_book_is_theory_with_labeled_doors():
    m = build(SCANDI)
    assert m["masthead"] == "Scandinavian Defense"
    assert m["statusLine"].startswith("OPENING")
    doors = m["theory"]["doors"]
    assert doors and all(d["variation"] for d in doors)
    assert doors[0]["variation"].startswith("Scandinavian")   # theory-first ordering
    assert any(d["san"] == "exd5" for d in doors)             # the mainline door
    assert m["theory"]["idea"]                             # authored annotation leads


def test_authored_idea_is_budgeted():
    from lucena_core.content import annotation_for
    idea = build(SCANDI)["theory"]["idea"]
    full = annotation_for("Scandinavian Defense")
    assert idea and len(idea) < len(full)          # trimmed, not the whole essay
    assert full.startswith(idea.split(".")[0])     # and it IS the lead


def test_out_of_book_is_minimal_but_never_empty():
    m = build(OUT_OF_BOOK)
    assert m["epigraph"] is None and m["theory"] is None
    assert m["cards"] and m["cards"][0]["id"] == "position"
    assert m["statusLine"].startswith("MOVE")


def test_bad_fen_raises():
    with pytest.raises(ValueError):
        build("not a fen")


def test_ply_arithmetic():
    assert _plies_played(START) == 0
    assert _plies_played(AFTER_E4) == 1
    assert _plies_played(SCANDI) == 2


def test_lead_sentences():
    assert _lead_sentences("One. Two. Three.", 2) == "One. Two."
