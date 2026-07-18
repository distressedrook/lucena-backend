"""DrillState move-line accumulation + backtrack truncation (the move navigator's data)."""

from lucena_backend.grounding_tools.drill import DrillState


def _done(fen):
    return {"kind": "done", "fen": fen}


def _solve(fen, uci, san, after):
    return {"kind": "solve", "fen": fen, "expect_uci": uci, "expect_san": san, "after": after}


def _reply(fen, defenses):
    return {"kind": "reply", "fen": fen, "defenses": defenses}


# A drill with two defenses to your first move, each opening a second solve — so refuting the
# first defense backtracks to the second (the classic tree-walk the linear line must track).
TREE = {
    "fen": "START", "side_to_solve": "white",
    "root": _solve("START", "e2e4", "e4", _reply("AFTER_E4", [
        {"uci": "a7a6", "san": "a6",
         "then": _solve("AFTER_A6", "d2d4", "d4", _reply("AFTER_D4", [
             {"uci": "b7b6", "san": "b6", "then": _done("DONE_1")}]))},
        {"uci": "h7h6", "san": "h6",
         "then": _solve("AFTER_H6", "g1f3", "Nf3", _reply("AFTER_NF3", [
             {"uci": "g7g6", "san": "g6", "then": _done("DONE_2")}]))},
    ])),
}


def _sans(plies):
    return [p["san"] for p in plies]


def test_seeds_ply_zero_with_start():
    d = DrillState(TREE)
    assert d.line == [{"n": 0, "san": None, "uci": None, "fen": "START"}]


def test_records_your_move_and_the_played_defense():
    d = DrillState(TREE)
    r = d.play("e2e4", "e4")
    # your move, then the first live defense (a6) auto-played
    assert _sans(r["plies"]) == [None, "e4", "a6"]
    assert r["plies"][1]["fen"] == "AFTER_E4" and r["plies"][2]["fen"] == "AFTER_A6"


def test_backtrack_is_HELD_until_continue():
    d = DrillState(TREE)
    d.play("e2e4", "e4")                 # line: [start, e4, a6], now solving d4
    r = d.play("d2d4", "d4")             # completes the a6-line → HOLDS (Continue pending), no backtrack yet
    assert r["await_continue"] is True and r["event"]["event"] == "branch_done"
    assert _sans(r["plies"]) == [None, "e4", "a6", "d4"], "the board STAYS on the solution"
    c = d.continue_branch()              # Continue → NOW backtrack to the h6 defense
    assert _sans(c["plies"]) == [None, "e4", "h6"]
    assert c["plies"][2]["fen"] == "AFTER_H6"
    assert c["reply"]["new_line"] is True


def test_full_walk_ends_on_the_current_line():
    d = DrillState(TREE)
    d.play("e2e4", "e4")
    d.play("d2d4", "d4")                 # → branch solved, HELD
    d.continue_branch()                  # Continue → h6 line
    r = d.play("g1f3", "Nf3")            # solves the h6 line → done
    assert _sans(r["plies"]) == [None, "e4", "h6", "Nf3"]
    assert r["finished"] is True
    # The whole drill is solved → the coach is told (a DRILL_SOLVED turn), not dropped to OPEN.
    assert r["event"]["kind"] == "drill_solved"
    assert r["event"]["fen"] == "AFTER_NF3"   # the position after the winning move


def test_wrong_move_leaves_the_line_unchanged():
    d = DrillState(TREE)
    d.play("e2e4", "e4")
    before = list(d.line)
    r = d.play("a1a1", "??")            # not the expected move
    assert r["correct"] is False
    assert _sans(r["plies"]) == _sans(before)
    assert d.line == before


def test_matches_on_san_the_canonical_form():
    # SAN is the internal canonical: a correct move matches on SAN even if the UCI representation
    # is off (the bug that made a correct bxc4/b5c4 read as wrong when the two notations were mixed).
    d = DrillState(TREE)
    r = d.play("zzzz", "e4")            # bogus uci, correct SAN -> matched on SAN
    assert r["correct"] is True


def test_uci_is_a_defensive_fallback_when_san_absent():
    d = DrillState(TREE)
    r = d.play("e2e4", None)           # no SAN given -> falls back to the UCI prefix match
    assert r["correct"] is True


def test_wrong_in_both_notations_is_wrong():
    d = DrillState(TREE)
    r = d.play("a1a1", "Ra1")          # neither SAN nor UCI is the expected e4
    assert r["correct"] is False
