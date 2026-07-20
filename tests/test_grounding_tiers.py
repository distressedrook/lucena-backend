"""§4 grounding visibility tiers + the move briefing — the spoil-control invariants.

All pure functions: no LLM, no Stockfish, no DB. These pin the one property that, if it breaks,
silently ruins every puzzle — the solution or the trap detail leaking into what the solver sees
WHILE solving. The guard is structural (a withheld fact is physically absent from `solve_text()`),
so the test is: build grounding with a known trap, and assert the secret bytes are not in the
solve-time text and ARE in the reveal-time text.
"""

from __future__ import annotations

from lucena_backend.coaching.grounding import (
    TieredFacts, tiered_bit_grounding, _brief_move, _brief_reply, _pv_capture_victims, you_move_beat,
    _swing_phrase, _why_loses, _deep_tactics, _created_threat, _solution_moves, _node_at,
    _node_solutions, _draws_by_stalemate, _numbered_line,
)
# The undermine-the-defender reasoner moved to its own package (lucena_backend.reasoning) — its tests
# live in test_reasoning_undermine.py.


def test_numbered_line_is_pgn_style():
    # White carries the number; Black carries 'N...' ONLY when it OPENS the line, else it is bare.
    # A refutation PV opening with Black — '1... e5 2. Nf3 Nc6' — the SECOND black move (Nc6) is bare.
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"   # only parts[5] (fullmove) is read
    assert _numbered_line(["e5", "Nf3", "Nc6"], fen1, played_by_white=True) == \
        ["1... e5", "2. Nf3", "Nc6"]
    # A PV opening with White — every Black reply is bare, none dotted ('… e5 … Nc6').
    out = _numbered_line(["e4", "e5", "Nf3", "Nc6"], fen1, played_by_white=False)
    assert out == ["2. e4", "e5", "3. Nf3", "Nc6"]
    assert not any("..." in m for m in out), "no Black move opens this line → none is dotted"

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
              {"kind": "threat", "text": "Bb3 hits the rook"},
              {"kind": "threat", "text": "after a pass, Nd6 is strong for the opponent"}],
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
    assert "Bb3 hits the rook" in s, "a genuine engine 'why' fact must flow through"
    # ...but a null-move threat ('after a pass, Nd6 …') is filtered — it describes passing, not the move.
    assert "after a pass" not in s and "Nd6" not in s


def test_brief_move_wrong_hides_the_solution():
    s = _brief_move(WRONG_EVAL, hide_best=True)
    # The facts array here NAMES the best move (bxc4) — hide_best MUST withhold it.
    assert "bxc4" not in s, "a wrong-move briefing leaked the solution via the facts array"
    assert "best move" not in s.lower(), "hide_best must not state the engine's best move"
    # the flaw is still explained through the refutation line.
    assert "Qe3" in s


def test_brief_move_labels_each_side_in_the_refutation():
    # The refutation opens with the replying side's punishing move then alternates; without explicit
    # side labels the model flipped who's who (a live wrong-verdict cast the reply as the player's).
    # Labels are ABSOLUTE COLOURS, never relative you/opponent (which invert with perspective).
    # side_to_move is black here, so the player is Black and the replier is White: 2. Qe3 (no dots),
    # Black replies 2... Nef6.
    s = _brief_move(WRONG_EVAL, hide_best=True)
    assert "(White) 2. Qe3" in s, "the refuting move must be attributed to White by colour"
    assert "(Black) Nef6" in s, "Black's reply follows White in the line → bare SAN, no '2...'"
    assert "2... Nef6" not in s, "a Black move following White must not carry the dotted number"
    assert "White refutes it with 2. Qe3" in s
    assert "opponent" not in s and "(you)" not in s, "facts must use colours, not relative words"
    assert "ENDS at Nef6" in s and "nothing exists past Nef6" in s, "the line must be hard-bounded"


def test_brief_move_never_capitalises_a_move_token():
    # 'bxc4' (pawn) must not become 'Bxc4' (bishop) — the first-letter slip we fixed in the trap voice.
    s = _brief_move(RIGHT_EVAL, hide_best=False)
    assert "Bxc4" not in s


def test_pv_capture_victims_names_the_real_piece_taken():
    # White plays the wrong Kg2; Black refutes with Rxe1, taking the ROOK on e1 (not a knight — the
    # live hallucination). The victim is resolved on a board, so it is exactly what stands there.
    fen = "8/1k2N3/1p6/3p1p2/6p1/P5P1/1P3P2/4RK1r w - - 1 4"
    victims = _pv_capture_victims(fen, "Kg2", ["Rxe1", "Nxf5", "Kc7"])
    assert victims[0] == "rook", "Rxe1 takes the rook on e1, never a knight"
    assert victims[1] == "pawn", "Nxf5 takes the pawn on f5"
    assert victims[2] is None, "Kc7 is not a capture"


def test_pv_capture_victims_degrades_on_bad_input():
    assert _pv_capture_victims(None, "Kg2", ["Rxe1"]) == [None]
    assert _pv_capture_victims("8/8/8/8/8/8/8/8 w - - 0 1", "Kg2", ["Rxe1", "Qd1"]) == [None, None]


def test_brief_move_wrong_names_the_captured_piece_not_a_guess():
    # End-to-end: the wrong-move briefing states what the refuting capture actually takes.
    v = {"san": "Kg2", "class": "blunder", "side_to_move": "white",
         "fen": "8/1k2N3/1p6/3p1p2/6p1/P5P1/1P3P2/4RK1r w - - 1 4",
         "refutation_pv": ["Rxe1", "Nxf5", "Kc7"], "facts": []}
    s = _brief_move(v, hide_best=True)
    assert "captures the rook on e1" in s, "the victim must be named, not left for the model to guess"
    assert "[takes the rook]" in s


def test_brief_reply_grounds_the_opponent_reply():
    # The recapture 1... cxd5 (Black) takes White's queen that stood on d5. The reply is attributed
    # by COLOUR (Black moved; the victim is White's), never by relative you/opponent.
    from_fen = "1k5r/4q3/1pp5/3QNp2/6p1/P5P1/1P3P2/4RK2 b - - 0 1"
    s = _brief_reply(from_fen, "cxd5")
    assert "1... cxd5" in s, "the reply must be numbered as Black's"
    assert "Black has just replied" in s, "the mover is named by colour"
    assert "captures White's queen on d5" in s, "the victim must be resolved and colour-attributed"
    assert "opponent" not in s.lower(), "facts must use colours, not the relative word 'opponent'"


def test_brief_reply_flags_check_and_degrades():
    assert "gives check" in _brief_reply("8/8/8/8/8/8/5k2/4R1K1 w - - 0 1", "Re2+")
    assert _brief_reply(None, "cxd5") == "(no reply read available)"
    assert _brief_reply("x", None) == "(no reply read available)"


def test_you_move_beat_carries_badge_and_chip():
    # v1: the bubble is JUST the move — no "— takes the X" narration. It still carries the verdict badge,
    # the SAN chip, and the after-move FEN to snap to.
    b = you_move_beat("1k5r/4q3/1pp5/3bNp2/6p1/P5P1/1P3P2/3QRK2 w - - 0 1", "d1d5", "Qxd5", correct=True)
    assert b["kind"] == "you"
    assert b["segments"][0]["text"] == "Played Qxd5"          # no capture narration
    assert b["correct"] is True and b["move"] == "Qxd5"
    assert b["fen"].split()[1] == "b", "the after-move FEN is Black to move"


def test_you_move_beat_no_capture_no_badge():
    b = you_move_beat("8/8/8/8/8/8/5k2/4RK2 w - - 0 1", "e1e2", "Re2", correct=None)
    assert b["segments"][0]["text"] == "Played Re2"
    assert "correct" not in b, "no badge when correctness is unknown (freeform)"


def test_you_move_beat_carries_client_id_for_reconciliation():
    fen = "1k5r/4q3/1pp5/3bNp2/6p1/P5P1/1P3P2/3QRK2 w - - 0 1"
    assert you_move_beat(fen, "d1d5", "Qxd5", correct=True, client_id="nonce-1")["client_id"] == "nonce-1"
    assert "client_id" not in you_move_beat(fen, "d1d5", "Qxd5", correct=True), "absent when no nonce"


def test_swing_phrase_bands_the_consequence():
    assert _swing_phrase(19.7, 88.5) == "this move turns a winning position into a losing one for you"
    assert _swing_phrase(86.0, 88.5) is None, "no band change → no swing to explain"
    assert _swing_phrase(None, 88.5) is None


def test_why_loses_names_the_abandoned_defender():
    # After Nxe7 Rh1+, the White king on f1 is the ONLY defender of the rook on e1. Kg2 walks it off,
    # so Rxe1 wins the rook — the instructive mechanism, derived from the board, not guessed.
    why = _why_loses("8/1k2N3/1p6/3p1p2/6p1/P5P1/1P3P2/4RK1r w - - 1 4", "f1g2", ["Rxe1", "Nxf5"])
    assert why is not None
    assert "king you moved from f1" in why
    assert "rook on e1" in why and "Rxe1" in why


FORK_FEN = "1k5r/4q3/1pp5/3bNp2/6p1/P5P1/1P3P2/3QRK2 w - - 0 1"   # Nxc6+ forks Kb8+Qe7; Bd5 guards c6


def test_why_loses_credits_the_fork_only_when_the_single_solution_earns_it():
    # Nxc6+ played FIRST (premature): the knight forks the king and queen, but the d5 bishop guards c6,
    # so Bxc6 recaptures. With a SINGLE solution the idea is only credited when that solution supports
    # it — here Qxd5 removes the very guard first, so the fork is a genuine mistimed-good-idea: credit
    # it, name the guard mechanism (NOT 'undefended'), and point at the guard as the thing to deal with.
    why = _why_loses(FORK_FEN, "e5c6", ["Bxc6"], ["d1d5"], single_solution=True)
    assert why is not None
    assert "fork the king and the queen on e7" in why and "the right idea" in why
    assert "bishop on d5 still guards c6" in why
    assert "deal with first" in why
    assert "undefended" not in why, "a move-into-a-capture is not an 'undefended' story"


def test_why_loses_drops_the_fork_credit_when_the_single_solution_ignores_it():
    # Same real fork, single solution, but the win does something UNRELATED (a3-a4 stand-in): it neither
    # reuses the fork nor deals with the guard. A real-but-irrelevant fork must NOT be sold as 'the
    # right idea' (P3: Qxd4+ forks, but the win just grabs a hanging queen) — plain mechanical read.
    why = _why_loses(FORK_FEN, "e5c6", ["Bxc6"], ["a3a4"], single_solution=True)
    assert why is not None
    assert "the right idea" not in why and "deal with first" not in why
    assert "bishop on d5 still guards c6" in why and "Bxc6" in why   # the MECHANISM always stands


def test_why_loses_surfaces_the_fork_when_not_a_single_solution():
    # Several best moves, or a non-puzzle flow (no solution): we don't second-guess the idea — surface
    # it as before. The mechanism is derived regardless (a blunder is a blunder).
    for kwargs in ({}, {"solution_ucis": ["a3a4"], "single_solution": False}):
        why = _why_loses(FORK_FEN, "e5c6", ["Bxc6"], **kwargs)
        assert "the right idea" in why and "deal with first" in why, "not-single → old permissive credit"
        assert "bishop on d5 still guards c6" in why


def test_why_loses_does_not_invent_a_fork_for_a_non_fork_move():
    # Rxe5+ (e1e5) is a simple losing capture into a guarded square — it hits ONLY the king, not a fork.
    # The single solution Nxd6 removes the guard first, so 'deal with X first' is still earned — but the
    # wording must NOT claim a fork ('before the fork wins anything') that doesn't exist.
    why = _why_loses("4k3/8/3b4/4p3/2N5/8/8/4RK2 w - - 0 1", "e1e5", ["Bxe5"], ["c4d6"],
                     single_solution=True)
    assert why is not None
    assert "deal with first" in why and "bishop on d6 still guards e5" in why
    assert "fork" not in why and "the right idea" not in why, "a non-fork move must not be sold as a fork"


def test_why_loses_is_none_without_a_capture_refutation():
    assert _why_loses("8/1k2N3/1p6/3p1p2/6p1/P5P1/1P3P2/4RK1r w - - 1 4", "f1g2", ["Kb7"]) is None
    assert _why_loses(None, "f1g2", ["Rxe1"]) is None


def test_deep_tactics_keeps_resources_but_strips_the_solution():
    # The engine's tactical read names the SOLUTION (a 'forcing sequence starting with Qxd5') alongside
    # safe deep facts (the Rh1+ resource, the c6/d5 linchpin). The verdict must keep the safe facts and
    # drop the solution-naming clause — so the coach explains the trap without spoiling the answer.
    always = ["White to move; White is winning (eval +4.7, White 85%).",
              "Tactics: a forcing sequence starting with Qxd5 sacrifices material but wins; "
              "warning — ignore Rh1+ and you go from winning to losing; the pawn on c6 is the only "
              "defender of the bishop on d5."]
    out = _deep_tactics(always, ["Qxd5"])
    assert out is not None
    assert "Qxd5" not in out, "the solution move must be stripped"
    assert "ignore Rh1+" in out and "only defender of the bishop on d5" in out


def test_deep_tactics_none_without_a_tactics_line():
    assert _deep_tactics(["White is winning."], ["Qxd5"]) is None


def test_deep_tactics_drops_a_threat_the_move_made_impossible():
    # The FIX beat is grounded on the PRE-move board, so a wrong move that MOVES the attacked piece
    # (Rc3->c4) leaves it citing 'deal with Rxc3+' — a threat now impossible. With live_fen (the
    # after-move board, Black to move), a clause whose every move is illegal there is dropped as stale.
    facts = ["Tactics: Black threatens Rxc3+, winning White's rook on c3."]
    after_rc4 = "k7/2K5/1P6/8/2R4p/1r5p/7P/8 b - - 1 7"   # c3 empty → Rxc3+ illegal
    assert _deep_tactics(facts, [], live_fen=after_rc4) is None, "stale threat must be dropped"
    # Same fact, but grounded WITHOUT a live board (the right-move framing 'what this move addresses')
    # — the clause is kept.
    assert "Rxc3+" in (_deep_tactics(facts, []) or ""), "pre-move framing keeps the clause"
    # A structural clause naming no move survives the live filter (it isn't a move-threat).
    struct = ["Tactics: the pawn on c6 is the only defender of the bishop on d5."]
    assert "only defender" in (_deep_tactics(struct, [], live_fen=after_rc4) or "")


def test_deep_tactics_keeps_mate_claims_when_correctly_grounded():
    # Mate claims are NOT dropped — they're genuinely useful; the fix for the mis-sided 'Ra4#' is to
    # ground on the PRE-move position (player's turn) so the perspective is right, not to censor mates.
    always = ["Tactics: Black threatens mate in 13 — it starts with Rxc3+; "
              "Black threatens Rxc3+, winning White's rook on c3."]
    out = _deep_tactics(always, [])
    assert out is not None and "mate in 13" in out and "Rxc3+" in out


def test_created_threat_surfaces_the_mate_the_move_makes():
    # Read from the AFTER-move position (colour-attributed, so no perspective flip): the decisive
    # threat the played move creates. Instructive on both verdicts — 'it threatens mate, but…' / the
    # reward. Only loud mate threats qualify.
    after = ["Tactics: White threatens mate: Ra4#; Black has doubled pawns on the h-file."]
    out = _created_threat(after, [])
    assert out == "The move just played creates this threat: White threatens mate: Ra4#."


def test_created_threat_strips_a_solution_move_and_ignores_quiet_lines():
    # A threat whose move is the un-played solution must NOT be surfaced (it would spoil the drill).
    assert _created_threat(["Tactics: White threatens mate: Ra4#."], ["Ra4#"]) is None
    # No mate threat → nothing to surface (a mundane recapture is not a teaching point).
    assert _created_threat(["Tactics: the rook forks the king and the rook on h3."], []) is None
    assert _created_threat([], []) is None


def test_solution_moves_from_the_tree_root():
    assert _solution_moves({"root": {"kind": "solve", "expect_san": "Qxd5"}}) == ["Qxd5"]
    assert _solution_moves({"root": {"kind": "mate", "options": [{"san": "Qh7#"}, {"san": "Qb8#"}]}}) \
        == ["Qh7#", "Qb8#"]
    assert _solution_moves(None) == []


def test_node_at_anchors_on_the_current_position_and_counts_solutions():
    # The fork/idea validation anchors on the position the wrong move was played FROM — deep in a
    # multi-ply drill that is NOT the root — and is gated on the node having a SINGLE best move.
    tree = {"root": {"kind": "mate", "fen": "ROOT w - - 0 1", "options": [
        {"uci": "a1a2", "then": {"kind": "reply", "fen": "R1 b - - 0 1", "defenses": [
            {"uci": "b8b7", "then": {"kind": "solve", "fen": "DEEP w - - 5 3", "expect_uci": "a2a3",
                                     "after": {"kind": "reply", "defenses": [
                                         {"uci": "b7b6", "then": {"kind": "done"}}]}}}]}}]}}
    # clocks are ignored when matching a position
    assert _node_solutions(_node_at(tree, "ROOT w - - 9 9")) == ["a1a2"]   # single mate option → one
    assert _node_solutions(_node_at(tree, "DEEP w - - 0 0")) == ["a2a3"]   # anchored at the deeper node
    assert _node_at(tree, "NOSUCH w - - 0 1") is None                      # off the line → None
    # A 'mate' node with several options → several best moves (gating must NOT apply).
    multi = {"root": {"kind": "mate", "fen": "M w - - 0 1",
                      "options": [{"uci": "a1a2"}, {"uci": "b1b2"}]}}
    assert _node_solutions(_node_at(multi, "M w - - 0 1")) == ["a1a2", "b1b2"]
    # Non-puzzle / no tree → no node, no solutions.
    assert _node_at(None, "x") is None and _node_solutions(None) == []
    # SIBLING branch: a position reachable only via the SECOND mate option / second defense must still
    # be found (a wrong move after a Continue plays a sibling defense) — not just the main line.
    sib = {"root": {"kind": "mate", "fen": "TOP w - - 0 1", "options": [
        {"uci": "a1a2", "then": {"kind": "done"}},
        {"uci": "b1b2", "then": {"kind": "reply", "fen": "MID b - - 0 1", "defenses": [
            {"uci": "h8h7", "then": {"kind": "solve", "fen": "SIB w - - 0 2", "expect_uci": "b2b3"}}]}}]}}
    assert _node_solutions(_node_at(sib, "SIB w - - 9 9")) == ["b2b3"]   # found on the 2nd option branch


def test_draws_by_stalemate_is_grounded_not_inferred():
    # k7/2K5/1P6/8/7p/1rR4p/7P/8 w — Rxh3 is the only win; Rc1 draws because Rc3+ Rxc3 leaves Black
    # with no legal move (king boxed, h-pawns frozen): STALEMATE. Detected on the board, never guessed.
    FEN = "k7/2K5/1P6/8/7p/1rR4p/7P/8 w - - 0 7"
    out = _draws_by_stalemate(FEN, "c3c1", ["Rc3+", "Rxc3"])
    assert out is not None
    assert "DRAWS by STALEMATE" in out
    assert "7. Rc1 Rc3+ 8. Rxc3" in out, "PGN style: Black's reply after White is bare, not '7... Rc3+'"
    assert "Black has NO legal move" in out and "spare tempo" in out
    assert "Rxh3" not in out, "the solution move must never appear"


def test_draws_by_stalemate_none_when_the_line_does_not_stalemate():
    # A move that loses (Black keeps moves / mates) must not be reported as a stalemate draw.
    FEN = "k7/2K5/1P6/8/7p/1rR4p/7P/8 w - - 0 7"
    assert _draws_by_stalemate(FEN, "c7d7", ["Rc3", "Kd6"]) is None
    assert _draws_by_stalemate(None, "c3c1", ["Rc3+"]) is None


def test_brief_move_handles_error_and_empty():
    assert _brief_move({"error": "x"}) == "(no move read available)"
    assert _brief_move({}).startswith("Move played:")


def test_invented_moves_flags_ungrounded_move_tokens():
    from lucena_backend.coaching.grounding import _invented_moves
    facts = ("Move played: Rc4. The opponent refutes it with 7... Rb4 8. Rc5 8... Rb5 9. Rc6 9... Rb4. "
             "Black threatens Rxc3+ winning the rook.")
    # Ra4#, Kc8, Rb3 are NOT in the facts → invented; Rxc3+ IS grounded → clean.
    assert _invented_moves("Rc4 allows Rb4, a threat of Ra4#; then 8. Rc6 Rb3 9. Kc8.", facts, "Rc4") \
        == {"Ra4#", "Kc8", "Rb3"}
    assert _invented_moves("Rc4 lets Rb4 threaten Rxc3+, winning your rook.", facts, "Rc4") == set()
    # a queen promotion narrated as a king move
    assert _invented_moves("Your move Kb2 sets up mate.", "Move played: b8=Q+.", "b8=Q+") == {"Kb2"}
    # bare pawn pushes / square mentions must NOT false-positive
    assert _invented_moves("the rook on c3 is loose", facts, "Rc4") == set()


def test_invented_moves_catches_a_false_mate_on_a_grounded_move():
    # The bug the strip-suffix guard missed: the MOVE is grounded (a rook shuffle Ra4 in the pv) but
    # the model dresses it up as mate ('Ra4#'). A '#' the facts never gave is itself invented.
    from lucena_backend.coaching.grounding import _invented_moves
    facts = "7... Rb4 8. Ra4 8... Rb5 9. Rc6. Black threatens Rxc3+."
    assert _invented_moves("creates a threat of Ra4# you must address", facts, "Rc4") == {"Ra4#"}
    assert _invented_moves("the opponent plays Ra4, shuffling", facts, "Rc4") == set()  # quiet Ra4 is fine
    assert _invented_moves("threatens Rxc3+", facts, "Rc4") == set()                    # grounded check
    assert _invented_moves("plays Rxc3 to win", facts, "Rc4") == set()                  # dropping + is ok
