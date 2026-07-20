"""Wrong-move hint enrichments: the graduated Socratic ladder (`_hint_line`) and the empty-ladder
TARGET fallback (`_fallback_hint`). Both are answer-preserving — they point at the idea, never the move."""
from lucena_backend.coaching.grounding import _fallback_hint, _hint_line

# A real puzzle where the win (Bxe4) is a capture the derive_hints ladder does NOT cover → fallback.
PUZZLE = "r2qkb1r/pp2pppp/3p1n2/4n3/4b1PN/2PB3P/PP3P2/RNBQK2R w KQkq - 0 12"


def test_fallback_hint_points_at_the_best_moves_target():
    out = _fallback_hint(PUZZLE, {"best": {"pv_san": ["Bxe4"]}})
    assert out and "Black's bishop on e4" in out and "go after" in out
    assert "never the move" in out                   # the reveal guard rides with the fact
    assert "Bxe4" not in out                          # the target, NOT the move


def test_fallback_hint_silent_for_a_quiet_best_move():
    # a non-capturing best move has no target to point at → no fallback (coach asks a question instead)
    assert _fallback_hint(PUZZLE, {"best": {"pv_san": ["O-O"]}}) is None


def test_fallback_hint_handles_missing_fields():
    assert _fallback_hint(PUZZLE, {"best": {}}) is None
    assert _fallback_hint(PUZZLE, {}) is None
    assert _fallback_hint(None, {"best": {"pv_san": ["Bxe4"]}}) is None


HINTS = [{"rung": 1, "text": "a loose piece"}, {"rung": 2, "text": "a fork"}, {"rung": 3, "text": "that piece moves first"}]


def test_hint_line_escalates_with_attempts():
    assert "a loose piece" in _hint_line(HINTS, 0)
    assert "a fork" in _hint_line(HINTS, 1)
    assert "that piece moves first" in _hint_line(HINTS, 2)


def test_hint_line_caps_at_the_most_specific_rung():
    # more wrong tries than rungs → stay on the last rung, never past it (and never the move)
    assert "that piece moves first" in _hint_line(HINTS, 5)


def test_hint_line_carries_the_no_answer_guard():
    out = _hint_line(HINTS, 0)
    assert "Socratic nudge" in out and "never naming a move" in out


def test_hint_line_none_when_ladder_empty():
    assert _hint_line([], 0) is None
    assert _hint_line(None, 0) is None
    assert _hint_line([{"rung": 1, "text": ""}], 0) is None    # empty rung text → nothing to say


# -- null-move-threat filtering ('if White ignores Nc5 …' — describes passing, not the move played) --
from lucena_backend.coaching.grounding import _deep_tactics, _brief_move


def test_deep_tactics_drops_null_move_threat_clause():
    line = ("Tactics: warning — if White ignores Nc5, White goes from winning to losing; "
            "the bishop on d5 is hanging")
    out = _deep_tactics([line], solution_moves=[])
    assert out and "ignores" not in out and "Nc5" not in out
    assert "the bishop on d5 is hanging" in out          # the real, non-confusing point survives


def test_deep_tactics_none_when_only_a_null_move_threat():
    line = "Tactics: warning — if White ignores Nc5, White goes from winning to losing"
    assert _deep_tactics([line], solution_moves=[]) is None


def test_brief_move_drops_null_move_threat_fact():
    v = {"san": "Qxe4", "class": "only_move",
         "facts": [{"text": "warning — if White ignores Nc5, White goes from winning to losing"},
                   {"text": "the bishop on d5 is hanging"}]}
    out = _brief_move(v, hide_best=False)
    assert "ignores" not in out and "Nc5" not in out
    assert "the bishop on d5 is hanging" in out


def test_brief_move_drops_stale_threat_facts():
    # After Rxh3 the rook has LEFT c3, so a pre-move 'Black threatens Rxc3+, winning the rook on c3'
    # fact is stale and CONTRADICTS the real verdict (White is mating). live_fen drops it; the mate
    # fact (whose move IS legal in the after-position) survives.
    v = {"san": "Rxh3", "captured": "pawn", "class": "brilliant",
         "facts": [{"text": "there's a forced mate in 4 — it starts with Rxh3"},
                   {"text": "Black is easily winning — the attack starts with Rxc3+"},
                   {"text": "Black threatens Rxc3+, winning White's rook on c3"}]}
    after = "k7/2K5/1P6/8/7p/1r5R/7P/8 b - - 0 7"    # c3 vacated; the black rook now sits on h3
    out = _brief_move(v, hide_best=False, live_fen=after)
    assert "forced mate in 4" in out
    assert "Rxc3" not in out and "easily winning" not in out


def test_created_threat_names_the_mating_move_not_a_square():
    # The mate is stated by its MOVE ('Ra4#') and nothing else — no king square (the model confabulated
    # 'checkmate on a8' from the king's square; the prompt, not the fact, keeps it saying the move).
    from lucena_backend.coaching.grounding import _created_threat
    out = _created_threat(["Tactics: White threatens mate: Ra4#"], [])
    assert out == "The move just played creates this threat: White threatens mate: Ra4#."
    assert "a8" not in out and "king" not in out


def test_invented_motif_flags_ungrounded_tactical_label():
    from lucena_backend.coaching.grounding import _invented_motif
    facts = "there's a forced mate in 4 — it starts with Rxh3"
    assert _invented_motif("you execute a fork that attacks the king and rook", facts) == "fork"
    assert _invented_motif("this sets up a forced mate", facts) is None          # no motif named
    # a GROUNDED motif (the facts use the word) passes through
    assert _invented_motif("the knight forks the king and queen", "the knight forks the king") is None
    # word-stem boundary: a substring like 'opinion' must not false-trigger 'pin'
    assert _invented_motif("in my opinion this wins", "wins material") is None


def test_deep_tactics_leads_with_the_mate_not_the_pin():
    # For a forced-mate move, the mate IS the point — a pre-existing pin / defender relationship is true
    # background but not why the move is the move. The model kept leading with 'your rook is pinned'.
    line = ("Tactics: there's a forced mate in 4 — it starts with c4+; the rook on c5 is pinned to the "
            "king by the queen on a5 — it can't legally move off that line; the knight on d2 is the only "
            "defender of the bishop on b3")
    out = _deep_tactics([line], solution_moves=[])
    assert out == "The point of this move: there's a forced mate in 4 — it starts with c4+."
    assert "pinned" not in out and "only defender" not in out


def test_deep_tactics_uses_structural_feature_when_nothing_is_decisive():
    # No mate → the structural feature IS the point (the fallback the single-ply reasoners don't cover).
    line = "Tactics: the knight on c3 is the only defender of the bishop on a2"
    out = _deep_tactics([line], solution_moves=[])
    assert out == "The point of this move turns on: the knight on c3 is the only defender of the bishop on a2."


def test_pin_is_never_the_point_or_surfaced_in_move_read():
    from lucena_backend.coaching.grounding import _deep_tactics, _brief_move
    pin = "Tactics: the rook on c5 is pinned to the king by the queen on a5 — it can't move off that line"
    assert _deep_tactics([pin], []) is None                       # a static pin is not a move's point
    v = {"san": "c4+", "class": "only_move",
         "facts": [{"text": "the rook on c5 is pinned to the king by the queen on a5"},
                   {"text": "there is a forced mate in 4"}]}
    out = _brief_move(v, hide_best=False)
    assert "pinned" not in out and "there is a forced mate in 4" in out
