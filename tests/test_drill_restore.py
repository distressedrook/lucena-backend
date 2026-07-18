"""Serialize→restore must keep the move LINE coherent across a backtrack.

The walker's line is not in `to_state()` (its home is the session history), so `restore` needs the
line passed in. Without it the line defaulted to just [root]; the NEXT move landed at ply 1 and a
sibling-defence backtrack truncated the wrong prefix — the recorded history came out scrambled and
the board "went to weird states". This pins the fix (strategies pass `history` → `restore(line=…)`).
"""

from __future__ import annotations

from lucena_backend.grounding_tools.drill import DrillState

# root solve A → opponent reply with TWO sibling defences, each a one-move solve to done.
TREE = {"root": {"kind": "solve", "expect_san": "A", "expect_uci": "a1a2", "fen": "ROOT",
    "after": {"kind": "reply", "fen": "AFTER_A", "defenses": [
        {"san": "d1", "uci": "b1b2", "then": {"kind": "solve", "expect_san": "B", "expect_uci": "c1c2",
            "fen": "AFTER_d1", "after": {"kind": "done", "fen": "AFTER_B", "reason": "mate"}}},
        {"san": "d2", "uci": "e1e2", "then": {"kind": "solve", "expect_san": "C", "expect_uci": "f1f2",
            "fen": "AFTER_d2", "after": {"kind": "done", "fen": "AFTER_C", "reason": "mate"}}},
    ]}}}


def _line(plies):
    return [(p["san"], p["uci"]) for p in plies]


def test_restore_with_line_survives_a_backtrack():
    d = DrillState(TREE)
    d.play("a1a2", "A")                       # correct → reply d1 → line = [root, A, d1]
    state, hist = d.to_state(), list(d.line)  # the session history is the line's home

    # A fresh walker per move (as the strategy does), restored WITH the line:
    d2 = DrillState.restore(TREE, state, line=hist)
    r = d2.play("c1c2", "B")                  # solve line 1 → backtrack to sibling d2
    assert _line(r["plies"]) == [(None, None), ("A", "a1a2"), ("d2", "e1e2")], _line(r["plies"])
    assert (r.get("reply") or {}).get("new_line") is True


def test_restore_without_line_is_the_bug_we_fixed():
    # Documents the failure mode: no line → default [root] → the first move is lost, ply 1 is wrong.
    d = DrillState(TREE)
    d.play("a1a2", "A")
    d2 = DrillState.restore(TREE, d.to_state())    # NO line
    r = d2.play("c1c2", "B")
    assert _line(r["plies"]) != [(None, None), ("A", "a1a2"), ("d2", "e1e2")], \
        "without the line the history is scrambled — that was the bug"
