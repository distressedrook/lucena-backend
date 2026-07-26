"""The story pipeline's judgement calls, tested without an engine.

Everything here is a pure function over the numbers gamepass produces, so the
whole file runs in milliseconds. The engine-backed path is exercised by running
`analyse_pgn` on a real game, which is a slow offline job, not a unit test.
"""

import pytest

from lucena_backend.pipelines.moveclass import (MoveClass, classify_move,
                                                accuracy, MISS_GIFT, MISS_RETURN)
from lucena_backend.pipelines import gamestory as gs


def _ply(**kw):
    d = {"ply": 1, "move_no": 1, "side": "w", "san": "e4", "uci": "e2e4",
         "fen_before": "", "fen_after": "", "eval_cp": 0, "win_pct": 50.0,
         "delta_win_pct": 0.0, "class": "ok", "label": "best", "symbol": "",
         "best": {"san": "e4", "pv_san": [], "eval_cp": 0},
         "refutation_pv": [], "motifs": [], "gift_wp": 0.0}
    d.update(kw)
    return d


# ---------------------------------------------------------------- the ladder

def test_book_outranks_every_judgement():
    """A move still in theory is not graded — not even a bad-looking one."""
    assert classify_move(drop=40.0, played_is_best=False, in_book=True,
                         engine_class="blunder") is MoveClass.BOOK


def test_the_ladder_agrees_with_the_engine_thresholds():
    from lucena_engine.evalmodel import DUBIOUS, MISTAKE, BLUNDER
    mk = lambda d: classify_move(drop=d, played_is_best=False, in_book=False,
                                 engine_class="ok")
    assert mk(BLUNDER) is MoveClass.BLUNDER
    assert mk(BLUNDER - 0.1) is MoveClass.MISTAKE
    assert mk(MISTAKE) is MoveClass.MISTAKE
    assert mk(MISTAKE - 0.1) is MoveClass.INACCURACY
    assert mk(DUBIOUS) is MoveClass.INACCURACY
    assert mk(1.0) is MoveClass.EXCELLENT
    assert mk(4.0) is MoveClass.GOOD


def test_best_and_brilliant_and_great_come_from_the_audited_detectors():
    assert classify_move(drop=0.0, played_is_best=True, in_book=False,
                         engine_class="ok") is MoveClass.BEST
    assert classify_move(drop=1.0, played_is_best=False, in_book=False,
                         engine_class="brilliant") is MoveClass.BRILLIANT
    assert classify_move(drop=1.0, played_is_best=False, in_book=False,
                         engine_class="only_move") is MoveClass.GREAT


def test_a_miss_needs_both_a_gift_and_it_being_given_back():
    """Being handed a win is not a miss; handing it back is."""
    assert classify_move(drop=MISS_RETURN, played_is_best=False, in_book=False,
                         engine_class="ok", gift_wp=MISS_GIFT) is MoveClass.MISS
    # gift, but the player kept it -> not a miss
    assert classify_move(drop=0.5, played_is_best=False, in_book=False,
                         engine_class="ok", gift_wp=MISS_GIFT) is not MoveClass.MISS
    # gave a lot back, but nothing had been handed over -> an ordinary blunder
    assert classify_move(drop=30.0, played_is_best=False, in_book=False,
                         engine_class="ok", gift_wp=0.0) is MoveClass.BLUNDER


def test_accuracy_is_bounded_and_monotone():
    assert accuracy([]) == 100.0
    assert accuracy([0.0]) == 100.0
    perfect, sloppy = accuracy([0.0] * 10), accuracy([12.0] * 10)
    assert 0.0 <= sloppy < perfect <= 100.0


def test_one_catastrophe_is_not_averaged_away_to_nothing():
    """The plain mean is the deliberate choice — a 40-point blunder must still
    move the number even in a long game."""
    clean = accuracy([0.0] * 40)
    with_blunder = accuracy([0.0] * 39 + [40.0])
    assert clean - with_blunder > 1.5


# ---------------------------------------------------------------- moments

def test_a_decided_position_yields_no_turning_point():
    """You cannot lose what you did not have."""
    # win_pct is measured AFTER the move, so the position before it stood at
    # win_pct - delta: here 8% - already lost when the move was made.
    dead = [_ply(ply=1, win_pct=-2.0, delta_win_pct=-10.0)]
    assert gs._turning_points(dead) == []
    # ...and here 70%, a game worth losing
    live = [_ply(ply=1, win_pct=40.0, delta_win_pct=-30.0)]
    assert len(gs._turning_points(live)) == 1


def test_a_miss_moment_records_what_was_handed_over():
    plies = [_ply(ply=1, side="w", delta_win_pct=-25.0),
             _ply(ply=2, side="b", delta_win_pct=-20.0, label="miss")]
    got = gs._missed(plies)
    assert len(got) == 1 and got[0].gift_wp == 25.0


def test_gifts_look_only_at_the_move_immediately_before():
    plies = [_ply(ply=1, delta_win_pct=-30.0), _ply(ply=2, delta_win_pct=0.0),
             _ply(ply=3, delta_win_pct=0.0)]
    gs._gifts(plies)
    assert [p["gift_wp"] for p in plies] == [0.0, 30.0, 0.0]


# ---------------------------------------------------------------- the arc

def test_the_arc_tiles_the_game_with_no_gaps_and_no_repeats():
    """Acts must read as continuous prose: no two adjacent acts of the same
    kind, and no move numbers falling between them."""
    plies = []
    for i in range(1, 41):
        cp = 10 if i < 20 else 400          # level, then winning for White
        plies.append(_ply(ply=i, move_no=(i + 1) // 2,
                          side="w" if i % 2 else "b",
                          eval_cp=cp if i % 2 else -cp))
    acts = gs._arc(plies)
    assert acts
    for a, b in zip(acts, acts[1:]):
        assert (a["kind"], a["who"]) != (b["kind"], b["who"]), "adjacent duplicates"
        assert b["from_move"] <= a["to_move"] + 1, "gap between acts"


def test_the_arc_normalises_evaluation_to_whites_point_of_view():
    """eval_cp is mover-POV; a curve that forgets to flip Black's plies zigzags."""
    plies = [_ply(ply=1, side="w", eval_cp=300),
             _ply(ply=2, side="b", eval_cp=-300)]
    gs._arc(plies)
    assert [p["cp_white"] for p in plies] == [300, 300]


# ---------------------------------------------------------------- alignment

def _plans(*families, tier="engine"):
    return {"white": [{"idea": "x", "families": list(families), "tier": tier}],
            "black": []}


def test_alignment_reads_only_the_movers_own_plans():
    """A continuation names BOTH players' plans; crediting the mover with the
    opponent's ideas would make almost everything look 'on plan'."""
    intent = ["B:rook_activation"]          # the OPPONENT's plan
    a = gs._alignment(intent, _plans("rook_activation"), "w", "best")
    assert a["state"] == "unnamed"


def test_alignment_separates_a_bad_move_from_a_bad_idea():
    intent = ["W:rook_activation"]
    plans = _plans("rook_activation")
    assert gs._alignment(intent, plans, "w", "best")["state"] == "on-plan"
    assert gs._alignment(intent, plans, "w", "blunder")["state"] == \
        "right-idea-wrong-move"


def test_universal_plans_cannot_earn_a_right_idea_claim():
    """Nearly every opening position offers "castle" — matching it says nothing
    about whether the player read THIS position."""
    a = gs._alignment(["W:castle_kingside"], _plans("castle_kingside"),
                      "w", "mistake")
    assert a["state"] == "different-plan"


def test_an_unnamed_continuation_is_never_reported_as_planless():
    a = gs._alignment([], _plans("rook_activation"), "w", "best")
    assert a["state"] == "unnamed"


def test_a_plan_we_never_proposed_is_different_not_wrong():
    a = gs._alignment(["W:pawn_storm"], _plans("rook_activation"), "w", "best")
    assert a["state"] == "different-plan" and a["played"] == ["pawn_storm"]


# ------------------------------------------------- the plan-chapter gate

def _fake_game(monkeypatch, n_plies=30):
    """Drive build_story with a stubbed engine pass.

    The gate is a selection rule, not chess: stubbing the engine keeps this a
    fast unit test while still exercising the real code path that regressed."""
    plies = []
    for i in range(1, n_plies + 1):
        plies.append(_ply(ply=i, move_no=(i + 1) // 2,
                          side="w" if i % 2 else "b", san=f"m{i}",
                          uci="e2e4", fen_before=f"FEN{i}", fen_after=f"FEN{i}",
                          eval_cp=10, win_pct=50.0, delta_win_pct=0.0))
    monkeypatch.setattr(gs, "run_pass", lambda *a, **k: {
        "plies": plies, "game": {}, "summary": {"counts": {}}})

    class _G:
        headers = {"White": "W", "Black": "B"}
        plies = [type("P", (), {"ply": i, "fen_before": f"FEN{i}"})()
                 for i in range(1, n_plies + 1)]
    monkeypatch.setattr(gs, "parse_pgn", lambda _t: _G())
    monkeypatch.setattr(gs, "_static_read", lambda fen: {
        "phase": "middlegame", "developed": {}, "structure": "", "weaknesses": {}})
    monkeypatch.setattr(gs, "move_intent", lambda *a, **k: [])
    monkeypatch.setattr(gs, "game_phase", lambda fen: {"phase": "middlegame"})
    return plies


def _read(tier=None, text="a read"):
    plans = {"white": [], "black": []}
    if tier:
        plans["white"] = [{"idea": "x", "families": ["rook_activation"],
                           "tier": tier}]
    return {"read": text, "plans": plans, "character": "", "character_why": "",
            "initiative": {}, "king_risk": {}, "only_move": False}


def test_a_quiet_position_without_an_engine_confirmed_plan_is_not_a_chapter(monkeypatch):
    """The promise is 'the plan on offer'. A structure-tier hunch is not one,
    and an empty read is nothing at all — printing either would promise the
    reader exactly what this product exists to deliver and deliver nothing."""
    _fake_game(monkeypatch)
    monkeypatch.setattr(gs, "_plans_read", lambda *a, **k: _read("structure"))
    story = gs.build_story("pgn", None, None)
    assert [m for m in story["moments"] if m["kind"] == "plan"] == []
    assert story["plan_chapters_shown"] == 0
    assert story["plan_chapters_read"] > 0, "candidates must actually be read"


def test_an_empty_read_is_also_rejected(monkeypatch):
    _fake_game(monkeypatch)
    monkeypatch.setattr(gs, "_plans_read", lambda *a, **k: _read("engine", ""))
    story = gs.build_story("pgn", None, None)
    assert story["plan_chapters_shown"] == 0


def test_an_engine_confirmed_plan_earns_a_chapter_and_the_counts_agree(monkeypatch):
    _fake_game(monkeypatch)
    monkeypatch.setattr(gs, "_plans_read", lambda *a, **k: _read("engine"))
    story = gs.build_story("pgn", None, None, plan_chapters=2)
    shown = [m for m in story["moments"] if m["kind"] == "plan"]
    assert len(shown) == 2 == story["plan_chapters_shown"]
    assert story["plan_chapters_read"] >= story["plan_chapters_shown"]
    assert story["plan_chapters_considered"] >= story["plan_chapters_read"]


def test_each_position_is_rolled_at_most_once(monkeypatch):
    """Selection and enrichment both need the read; rolling twice would double
    the cost of the most expensive step in the pipeline."""
    _fake_game(monkeypatch)
    calls = []

    def counting(fen, *a, **k):
        calls.append(fen)
        return _read("engine")
    monkeypatch.setattr(gs, "_plans_read", counting)
    gs.build_story("pgn", None, None, plan_chapters=2)
    assert len(calls) == len(set(calls)), f"duplicate rolls: {calls}"
