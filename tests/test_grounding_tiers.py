"""§4 grounding visibility tiers + the move briefing — the spoil-control invariants.

All pure functions: no LLM, no Stockfish, no DB. These pin the one property that, if it breaks,
silently ruins every puzzle — the solution or the trap detail leaking into what the solver sees
WHILE solving. The guard is structural (a withheld fact is physically absent from `solve_text()`),
so the test is: build grounding with a known trap, and assert the secret bytes are not in the
solve-time text and ARE in the reveal-time text.
"""

from __future__ import annotations

from lucena_backend.coaching.grounding import (
    TieredFacts, tiered_bit_grounding, _brief_move,
)

# A move_line tree carrying a poisoned line, shaped exactly as preview_drill emits it.
POISONED_TREE = {
    "has_poisoned_line": True,
    "poisoned_line_moves": [{"san": "Nxe4"}, {"san": "Qe3"}, {"san": "Nxg3"},
                            {"san": "hxg3"}, {"san": "bxc4"}],
    "poisoned_line_meta": {"idea": "hxg3", "fatal": "Nxg3"},
}
# The bytes that must NEVER appear at solve time (the trap line + its catch/motif).
SECRET_FRAGMENTS = ["Nxe4 Qe3 Nxg3 hxg3 bxc4", "the catch is hxg3", "the motif is a Nxg3"]

READ = {"analysis": ["White to move; Black is winning.", "Black is up a bishop."]}


# -- TieredFacts: the tier mechanics ----------------------------------------------------------------

def test_solve_text_is_always_plus_warn_never_reveal():
    tf = TieredFacts(always=["a positional read"], warn=["a trap exists"], reveal=["THE SOLUTION"])
    st = tf.solve_text()
    assert "a positional read" in st and "a trap exists" in st
    assert "THE SOLUTION" not in st, "reveal_on_resolve content leaked into solve_text()"


def test_reveal_text_is_only_reveal():
    tf = TieredFacts(always=["read"], warn=["warn"], reveal=["the secret"])
    assert tf.reveal_text() == "the secret"


def test_empty_tiers_degrade_safely():
    tf = TieredFacts()
    assert tf.solve_text() == "(no grounded read available)"
    assert tf.reveal_text() == "" and tf.warn_text() == ""


# -- tiered_bit_grounding: the poisoned line folds in as warn + reveal -------------------------------

def test_poisoned_line_is_absent_from_solve_text():
    tf = tiered_bit_grounding(READ, POISONED_TREE)
    st = tf.solve_text()
    for secret in SECRET_FRAGMENTS:
        assert secret not in st, f"solve_text() leaked the trap detail: {secret!r}"
    # the WARN (existence, not detail) is allowed through — it names no move.
    assert "tempting move" in st.lower()
    for m in ("Nxe4", "Qe3", "Nxg3"):
        assert m not in st, f"solve_text() named a trap move: {m}"


def test_poisoned_line_detail_is_present_at_reveal():
    rt = tiered_bit_grounding(READ, POISONED_TREE).reveal_text()
    for secret in SECRET_FRAGMENTS:
        assert secret in rt, f"reveal_text() is missing the trap detail: {secret!r}"


def test_positional_read_is_always_tier():
    tf = tiered_bit_grounding(READ, POISONED_TREE)
    assert "Black is up a bishop." in tf.solve_text(), "the positional read must survive as always-tier"


def test_no_tree_means_no_trap():
    tf = tiered_bit_grounding(READ, {})
    assert tf.warn_text() == "" and tf.reveal_text() == ""
    assert "Black is winning" in tf.solve_text()


def test_none_resp_still_reveals_the_trap():
    # _conclude calls this with resp=None (no positional read needed at reveal time).
    rt = tiered_bit_grounding(None, POISONED_TREE).reveal_text()
    assert "hxg3" in rt


# -- _brief_move: the played-move briefing, and its hide_best spoil gate -----------------------------

RIGHT_EVAL = {
    "san": "bxc4", "captured": "bishop", "class": "only_move",
    "best": {"san": "bxc4"}, "side_to_move": "black", "fen": "8/8/8/8/8/8/8/8 b - - 0 1",
    "facts": [{"kind": "hanging", "text": "bxc4 wins the bishop on c4"},
              {"kind": "threat", "text": "after a pass, Bb3 is strong for the opponent"}],
}
WRONG_EVAL = {
    "san": "Nxe4", "captured": "pawn", "class": "blunder",
    "best": {"san": "bxc4"}, "refutation_pv": ["Qe3", "Nef6"],
    "side_to_move": "black", "fen": "8/8/8/8/8/8/8/8 b - - 0 1",
    "facts": [{"kind": "hanging", "text": "bxc4 wins the bishop on c4"}],   # names the SOLUTION
}


def test_brief_move_right_includes_engine_facts():
    s = _brief_move(RIGHT_EVAL, hide_best=False)
    assert "Move played: bxc4" in s
    assert "wins the bishop on c4" in s, "the engine's own move-level 'why' was dropped"
    assert "Bb3 is strong" in s


def test_brief_move_wrong_hides_the_solution():
    s = _brief_move(WRONG_EVAL, hide_best=True)
    # The facts array here NAMES the best move (bxc4) — hide_best MUST withhold it.
    assert "bxc4" not in s, "a wrong-move briefing leaked the solution via the facts array"
    assert "best move" not in s.lower(), "hide_best must not state the engine's best move"
    # the flaw is still explained through the refutation line.
    assert "Qe3" in s


def test_brief_move_never_capitalises_a_move_token():
    # 'bxc4' (pawn) must not become 'Bxc4' (bishop) — the first-letter slip we fixed in the trap voice.
    s = _brief_move(RIGHT_EVAL, hide_best=False)
    assert "Bxc4" not in s


def test_brief_move_handles_error_and_empty():
    assert _brief_move({"error": "x"}) == "(no move read available)"
    assert _brief_move({}).startswith("Move played:")
