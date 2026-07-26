"""The story pipeline's judgement calls, tested without an engine.

Everything here is a pure function over the numbers gamepass produces, so the
whole file runs in milliseconds. The engine-backed path is exercised by running
`analyse_pgn` on a real game, which is a slow offline job, not a unit test.
"""

import math

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
    """A move that loses the game must COST something a reader can feel.

    This test previously asserted `> 1.5` and passed at 2.1 while the function
    used an arithmetic mean — a bar too weak to catch the defect it existed to
    prevent. The harmonic mean charges 12.5 for the same blunder. The bar is
    now set where the statistic stops being misleading, not where the
    implementation happens to land."""
    clean = accuracy([0.0] * 40)
    with_blunder = accuracy([0.0] * 39 + [40.0])
    assert clean - with_blunder > 8.0, (clean, with_blunder)


def test_a_move_that_scores_zero_cannot_break_the_average():
    """The harmonic mean divides by each term, so a move bad enough to map to
    0.0 would raise ZeroDivisionError without the floor. Mate-in-one exists;
    this is reachable input, not a hypothetical."""
    assert accuracy([100.0]) == pytest.approx(1.0, abs=0.05)   # the floor itself
    worst = accuracy([0.0] * 20 + [100.0])
    assert 0.0 < worst < 100.0 and math.isfinite(worst)
    # ...and it must still be the harshest thing that can happen to a game
    assert worst < accuracy([0.0] * 20 + [40.0])


def test_the_worst_move_dominates_a_clean_average():
    """Two players, same number of moves, same total error — one spread thin,
    one concentrated in a single disaster. They must not score the same."""
    spread = accuracy([4.0] * 10)
    concentrated = accuracy([0.0] * 9 + [40.0])
    assert concentrated < spread - 5.0, (concentrated, spread)


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


# ------------------------------------------------- book, ending, spreading

def test_book_survives_the_tables_own_gaps():
    """The opening table names NODES, not every ply. Treating the first
    unnamed ply as the end of book cut a real French Exchange off at move 4
    and reported 2 book plies for a line named through move 7."""
    import chess
    from lucena_core.openings import name_for
    b, plies = chess.Board(), []
    for i, san in enumerate("e4 e6 Nf3 d5 exd5 exd5 d4 Bd6 Nc3 c6".split(), 1):
        b.push_san(san)
        plies.append(_ply(ply=i, move_no=(i + 1) // 2,
                          side="w" if i % 2 else "b", fen_after=b.fen()))
    # the table really does have a hole here — that is the point of the test
    assert name_for(plies[3]["fen_after"]) is None
    assert name_for(plies[6]["fen_after"]) is not None
    gs._gifts(plies)
    gs._label(plies)
    # the HOLE (plies 4-6) is bridged because the line RETURNS at ply 7...
    assert all(p["in_book"] for p in plies[:7]), \
        "a gap inside theory must not end the book"
    # ...and book ends at the deepest named node, not three plies past it
    assert not any(p["in_book"] for p in plies[7:])
    assert gs._opening(plies)["name"] == "French Defense: Exchange Variation"
    # the reader-facing tally must agree with the gate, not with the narrower
    # "was this exact ply a named node" field (which would count only 4)
    pat = gs._patterns(plies, [], {"White": "W", "Black": "B"})
    assert pat["w"]["book_plies"] + pat["b"]["book_plies"] == 7


def test_a_gap_only_counts_as_theory_if_the_line_comes_back():
    """Tolerating unnamed plies FORWARD hands free book status to a player who
    simply left theory with a bad move — after 1.d4 Nh6 the next plies would
    go ungraded. The whole game is in hand, so the gap is only inside theory
    when the line actually returns to a named position."""
    import chess
    from lucena_core.openings import name_for
    b, plies = chess.Board(), []
    for i, san in enumerate("d4 Nh6 e4 g5 Bxg5 f6".split(), 1):
        b.push_san(san)
        plies.append(_ply(ply=i, move_no=(i + 1) // 2,
                          side="w" if i % 2 else "b", fen_after=b.fen()))
    assert name_for(plies[0]["fen_after"]) is not None      # 1.d4 is named
    assert name_for(plies[1]["fen_after"]) is None          # 1...Nh6 is not
    gs._gifts(plies)
    gs._label(plies)
    assert plies[0]["in_book"], "1.d4 is theory"
    assert not any(p["in_book"] for p in plies[1:]), \
        "leaving theory must not buy three ungraded plies"


def test_leaving_the_book_is_permanent():
    """A late accidental transposition back into a named position must not
    re-open the book twenty moves after the players left theory."""
    import chess
    b, plies = chess.Board(), []
    for i, san in enumerate("a3 a6 h3 h6 a4 a5 h4 h5 Ra3 Ra6 Rb3 Rb6".split(), 1):
        b.push_san(san)
        plies.append(_ply(ply=i, move_no=(i + 1) // 2,
                          side="w" if i % 2 else "b", fen_after=b.fen()))
    gs._gifts(plies)
    gs._label(plies)
    assert not plies[-1]["in_book"]
    # and the shielding must stop at the exit, not three plies past it
    assert sum(1 for p in plies if p["in_book"]) < len(plies)


def test_a_repetition_is_read_off_the_board_not_the_headers():
    """This game's PGN says "drawn by agreement" while the players were
    shuffling. A claim we can verify beats one we are told."""
    import chess
    b, plies = chess.Board(), []
    # a real shuffle: knights out and back, returning to the same position
    for i, san in enumerate("Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8".split(), 1):
        b.push_san(san)
        plies.append(_ply(ply=i, move_no=(i + 1) // 2,
                          side="w" if i % 2 else "b", fen_after=b.fen()))
    rep = gs._ending(plies, {"Termination": "Game drawn by agreement"},
                     "1/2-1/2")["repetition"]
    assert rep["times"] >= 2 and rep["moves"]


def test_the_starting_position_counts_as_an_occurrence():
    """Repetition history built only from fen_after cannot see the opening
    position, so a knights-out-and-back shuffle reports one repeat too few."""
    import chess
    b, plies = chess.Board(), []
    start = b.fen()
    for i, san in enumerate("Nf3 Nf6 Ng1 Ng8".split(), 1):
        pr = _ply(ply=i, move_no=(i + 1) // 2, side="w" if i % 2 else "b",
                  fen_before=b.fen())
        b.push_san(san)
        pr["fen_after"] = b.fen()
        plies.append(pr)
    assert gs._norm(plies[-1]["fen_after"]) == gs._norm(start)
    rep = gs._ending(plies, {}, "1/2-1/2")["repetition"]
    assert rep["times"] == 2, "the start position is the first occurrence"
    assert rep["from_move"] == 1


def test_a_resignation_after_a_repeat_is_not_a_repetition_ending():
    """You can resign in a position that has occurred before. Explaining that
    resignation as a repetition would give the reader the wrong cause."""
    import chess
    b, plies = chess.Board(), []
    for i, san in enumerate("Nf3 Nf6 Ng1 Ng8".split(), 1):
        pr = _ply(ply=i, move_no=(i + 1) // 2, side="w" if i % 2 else "b",
                  fen_before=b.fen())
        b.push_san(san)
        pr["fen_after"] = b.fen()
        plies.append(pr)
    lost = gs._ending(plies, {"Termination": "White resigned"}, "0-1")
    assert "repetition" not in lost
    assert lost["repeats"] == 2, "the raw signal is still available"
    assert "repetition" in gs._ending(plies, {}, "1/2-1/2")


def test_a_decisive_game_reports_no_repetition():
    import chess
    b, plies = chess.Board(), []
    for i, san in enumerate("e4 e5 Nf3 Nc6 Bc4 Bc5".split(), 1):
        b.push_san(san)
        plies.append(_ply(ply=i, move_no=(i + 1) // 2,
                          side="w" if i % 2 else "b", fen_after=b.fen()))
    assert "repetition" not in gs._ending(plies, {}, "1-0")


def test_plan_candidates_reach_the_middlegame_before_the_fourth_opening_move():
    """Walking front to back put every plan chapter in the opening — three of
    four chapters discussed castling while the middlegame went unread."""
    plies = [_ply(ply=i, move_no=(i + 1) // 2, side="w" if i % 2 else "b")
             for i in range(1, 65)]
    order = [p["move_no"] for p in gs._plan_candidates(plies, set(), 4)]
    assert order, "there should be candidates in a 32-move game"
    # the first four tried must span the game, not cluster in its first act
    first_four = order[:4]
    assert max(first_four) - min(first_four) > 12, first_four
    # and nothing is discarded — the rest remain as fallbacks
    assert len(order) == len(set(order))


def test_a_transposition_after_the_book_closed_cannot_rename_the_opening():
    """book_name is deliberately sticky, so handing it the whole game lets a
    late accidental transposition rename the opening long after the players
    left theory — contradicting the boundary _label just computed."""
    plies = [_ply(ply=i, move_no=(i + 1) // 2, side="w" if i % 2 else "b",
                  fen_after=f"F{i}") for i in range(1, 9)]
    for i, p in enumerate(plies):
        p["in_book"] = i < 2                       # book closed after 2 plies
    seen = []

    def fake_book_name(fens):
        seen.append(list(fens))
        return "Some Opening"
    import lucena_core.openings as op
    old = op.book_name
    op.book_name = fake_book_name
    try:
        gs._opening(plies)
    finally:
        op.book_name = old
    assert seen and seen[0] == ["F1", "F2"], \
        f"only the in-book prefix may name the opening, got {seen}"


def test_a_theory_position_is_never_spent_on_a_plan_chapter():
    """A book position has nothing to teach about planning, and reading one
    burns budget on a move we just declined to grade."""
    plies = []
    for i in range(1, 41):
        p = _ply(ply=i, move_no=(i + 1) // 2, side="w" if i % 2 else "b")
        p["in_book"] = i <= 24                     # a long theoretical line
        plies.append(p)
    got = gs._plan_candidates(plies, set(), 4)
    assert got, "the out-of-book tail should still supply candidates"
    assert all(not p["in_book"] for p in got)
    assert min(p["ply"] for p in got) > 24
