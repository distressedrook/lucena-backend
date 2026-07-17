"""Black-box tests for the M4 import pass (contract: docs/contracts/M4-import.md).

Written from the M4 contract ONLY. The implementation under
`python/lucena/engine/pgn.py`, `python/lucena/engine/gamepass.py`, and
`python/lucena/statefile.py` was NEVER read — every expected value below is
derived from the contract's documented semantics and cross-checked against an
independent reference (python-chess + Stockfish, in scratch, never imported
here — GPL hygiene).

Three sections:
  * parse_pgn / Ply / Game / PgnError  — pure text parsing, no engine.
  * write_state                        — atomic JSON state writes, no engine.
  * run_pass                           — the two-pass engine analysis.

Engine-touching tests use `nodes=` + `threads=1` (NEVER movetime) per the
load-bearing determinism rule, are marked `engine`, and are skipped without a
Stockfish binary. The parser and statefile tests need no engine.

Positions/games used (verified independently in scratch):
  SCHOLAR — 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#  (UserSide=b). The black
  blunder is ply 6 (...Nf6), win% ~53->2.5 (drop ~51, class blunder); the
  mating move is ply 7 (Qxf7#): eval_cp 1000, win% 100.0, class ok.
"""

import json
import os
import shutil

import pytest

from lucena_engine import Board
from lucena_engine import Engine
from lucena_engine.pgn import parse_pgn, Game, Ply, PgnError
from lucena_engine.gamepass import run_pass
from lucena_backend.persistence.statefile import write_state


# Node limits kept modest so the pass is fast; the Scholar's-mate signals are
# huge (mate-in-one), so classification is robust at these counts.
FAST_LIMIT = {"nodes": 30_000}
DEEP_LIMIT = {"nodes": 80_000}


def _have_stockfish():
    return bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")


requires_engine = pytest.mark.skipif(not _have_stockfish(), reason="no stockfish")


@pytest.fixture
def engine():
    with Engine(threads=1) as e:
        yield e


# The blunder game, reused across run_pass tests.
SCHOLAR_PGN = """[Event "Test"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[UserSide "b"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


def plies_by_index(result):
    """Map each per-ply dict by its 1-based `ply` field."""
    return {p["ply"]: p for p in result["plies"]}


# =====================================================================
# parse_pgn / Ply / Game / PgnError
# =====================================================================


def test_parse_returns_game_with_frozen_types():
    g = parse_pgn(SCHOLAR_PGN)
    assert isinstance(g, Game)
    assert isinstance(g.headers, dict)
    assert isinstance(g.plies, list)
    assert all(isinstance(p, Ply) for p in g.plies)


def test_headers_parsed():
    g = parse_pgn(SCHOLAR_PGN)
    assert g.headers["White"] == "Alice"
    assert g.headers["Black"] == "Bob"
    assert g.headers["Result"] == "1-0"
    assert g.headers["UserSide"] == "b"


def test_headers_unescape_quote_and_backslash():
    # \" -> " and \\ -> \ per the contract.
    pgn = '[Event "Fancy \\"Quoted\\" Event"]\n[Site "C:\\\\games"]\n\n1. e4 *\n'
    g = parse_pgn(pgn)
    assert g.headers["Event"] == 'Fancy "Quoted" Event'
    assert g.headers["Site"] == "C:\\games"


def test_header_parsing_stops_at_movetext():
    # A "[...]"-looking token inside movetext must not become a header.
    g = parse_pgn(SCHOLAR_PGN)
    assert "1" not in g.headers  # no stray movetext leaked in
    assert set(g.headers) >= {"Event", "White", "Black", "Result", "UserSide"}


def test_mainline_moves_of_short_game():
    # SANs/uci/move_no/side verified with python-chess in scratch.
    g = parse_pgn(SCHOLAR_PGN)
    assert len(g.plies) == 7
    sans = [p.san for p in g.plies]
    # san is the board core's canonical rendering; check-mark decoration may
    # differ from input, so compare the move stem, not the exact glyph.
    assert [s.rstrip("+#") for s in sans] == [
        "e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7",
    ]
    assert [p.uci for p in g.plies] == [
        "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7",
    ]
    assert [p.ply for p in g.plies] == [1, 2, 3, 4, 5, 6, 7]
    assert [p.move_no for p in g.plies] == [1, 1, 2, 2, 3, 3, 4]
    assert [p.side for p in g.plies] == ["w", "b", "w", "b", "w", "b", "w"]


def test_ply_fen_chain_through_board_core():
    # fen_before/fen_after are the board core's FENs (its ep convention may
    # differ from other engines) — verify them through the Board core itself:
    # first fen_before is the start, each fen_after == next fen_before, and
    # applying the uci to fen_before yields fen_after.
    g = parse_pgn(SCHOLAR_PGN)
    assert g.plies[0].fen_before == g.start_fen
    for i, p in enumerate(g.plies):
        assert Board(p.fen_before).apply(p.uci).fen == p.fen_after
        if i + 1 < len(g.plies):
            assert p.fen_after == g.plies[i + 1].fen_before


def test_default_start_fen_is_standard():
    g = parse_pgn(SCHOLAR_PGN)
    assert g.start_fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_comments_variations_nags_linecomments_stripped():
    # {…} comment, $1 NAG, (…) variation (excluded from mainline), ;-line
    # comment all removed; 6 mainline plies remain (verified in scratch).
    pgn = (
        "1. e4 {King's pawn} e5 2. Nf3 $1 (2. Bc4 Bc5) Nc6 ; trailing comment\n"
        "3. Bb5 a6 *\n"
    )
    g = parse_pgn(pgn)
    assert [p.san.rstrip("+#") for p in g.plies] == [
        "e4", "e5", "Nf3", "Nc6", "Bb5", "a6",
    ]


def test_comments_do_not_nest():
    # Per the PGN standard the first '}' ends the comment; a '{' inside a comment
    # is literal. So "{ a } b { c }" is two comments around the move e5.
    pgn = "1. e4 { first } e5 { second } 2. Nf3 Nc6 *\n"
    g = parse_pgn(pgn)
    assert [p.san.rstrip("+#") for p in g.plies] == ["e4", "e5", "Nf3", "Nc6"]


def test_nested_variations_and_comment_with_paren():
    # Variations DO nest (depth-counted); a '(' inside a comment is inert.
    pgn = ("1. e4 e5 2. Nf3 (2. Bc4 {develops (fast)} (2. d4 exd4)) "
           "Nc6 { a ( paren } 3. Bb5 *\n")
    g = parse_pgn(pgn)
    assert [p.san.rstrip("+#") for p in g.plies] == ["e4", "e5", "Nf3", "Nc6", "Bb5"]


def test_move_number_dot_and_ellipsis_forms_stripped():
    # 12. and 12... move-number tokens are removed either way.
    pgn = "1. e4 e5 2. Nf3 2... Nc6 3. Bb5 *\n"
    g = parse_pgn(pgn)
    assert [p.san.rstrip("+#") for p in g.plies] == [
        "e4", "e5", "Nf3", "Nc6", "Bb5",
    ]


def test_result_from_movetext_wins_over_header():
    # movetext ends 1-0 but header says 0-1 -> movetext token wins.
    g = parse_pgn('[Result "0-1"]\n\n1. e4 e5 1-0\n')
    assert g.result == "1-0"


def test_result_falls_back_to_header():
    # No result token in movetext -> the Result header is used.
    g = parse_pgn('[Result "0-1"]\n\n1. e4 e5 2. Nf3\n')
    assert g.result == "0-1"


def test_result_defaults_to_star():
    # No movetext token and no Result header -> "*".
    g = parse_pgn("1. e4 e5 2. Nf3\n")
    assert g.result == "*"


def test_setup_fen_custom_start_white_to_move():
    fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    g = parse_pgn(f'[SetUp "1"]\n[FEN "{fen}"]\n\n2. Nf3 Nc6 *\n')
    assert g.start_fen == fen
    assert g.plies[0].fen_before == fen
    assert g.plies[0].move_no == 2
    assert g.plies[0].side == "w"


def test_setup_fen_black_to_move_start():
    # Black-to-move start FEN with fullmove != 1, so move_no must come from the
    # FEN counter (5), not a reset to 1 (verified in scratch).
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 5"
    g = parse_pgn(f'[SetUp "1"]\n[FEN "{fen}"]\n\n5... Nc6 6. Nf3 *\n')
    assert g.start_fen == fen
    assert g.plies[0].move_no == 5
    assert g.plies[0].side == "b"
    assert g.plies[0].san.rstrip("+#") == "Nc6"
    assert g.plies[0].uci == "b8c6"
    assert g.plies[1].move_no == 6
    assert g.plies[1].side == "w"


def test_san_is_board_core_canonical_and_uci_conventions():
    # H5: canonical SAN == what the board core renders; castling/promotion uci
    # are convention-free anchors.
    g = parse_pgn("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 *\n")
    assert all(p.san == Board(p.fen_before).san(p.uci) for p in g.plies)
    castle = next(p for p in g.plies if p.san.startswith("O-O"))
    assert castle.uci == "e1g1"
    promo = parse_pgn('[SetUp "1"]\n[FEN "4k2r/P7/8/8/8/8/8/4K3 w k - 0 1"]\n\n'
                      "1. a8=Q Ke7 *\n").plies[0]
    assert promo.uci == "a7a8q"
    assert promo.san == "a8=Q+"  # board core decorates the check


def test_empty_movetext_game_has_no_plies():
    g = parse_pgn('[White "x"]\n[Black "y"]\n\n*\n')
    assert g.plies == []
    assert g.result == "*"


def test_setup_ignored_without_flag():
    # A FEN header without SetUp="1" does NOT change the start (contract: only
    # when SetUp == "1" AND a FEN header is present).
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    g = parse_pgn(f'[FEN "{fen}"]\n\n1. e4 e5 *\n')
    assert g.start_fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- error paths ---


def test_pgnerror_is_valueerror_subclass():
    assert issubclass(PgnError, ValueError)


def test_empty_input_raises():
    with pytest.raises(PgnError):
        parse_pgn("")


def test_whitespace_only_input_raises():
    with pytest.raises(PgnError):
        parse_pgn("   \n\t  \n")


def test_illegal_move_raises_pgnerror():
    # e2-e5 is not a legal first move; the parser must reject it.
    with pytest.raises(PgnError):
        parse_pgn("1. e5 *\n")


def test_illegal_move_midgame_raises_pgnerror():
    with pytest.raises(PgnError):
        parse_pgn("1. e4 e5 2. Qxd8 *\n")


def test_unbalanced_open_brace_raises():
    with pytest.raises(PgnError):
        parse_pgn("1. e4 {unclosed comment *\n")


def test_unbalanced_open_paren_raises():
    with pytest.raises(PgnError):
        parse_pgn("1. e4 (2. d4 *\n")


def test_invalid_start_fen_raises():
    with pytest.raises(PgnError):
        parse_pgn('[SetUp "1"]\n[FEN "not a valid fen"]\n\n1. e4 *\n')


# =====================================================================
# write_state
# =====================================================================


def test_write_state_roundtrips(tmp_path):
    p = str(tmp_path / "s.json")
    obj = {"schema": 1, "seq": 2, "status": "complete", "xs": [1, 2, 3]}
    write_state(p, obj)
    with open(p, encoding="utf-8") as fh:
        assert json.load(fh) == obj


def test_write_state_is_compact_and_utf8(tmp_path):
    p = str(tmp_path / "s.json")
    write_state(p, {"schema": 1, "seq": 1, "name": "café"})
    text = open(p, encoding="utf-8").read()
    # Compact: no ", " or ": " separators.
    assert ", " not in text
    assert ": " not in text
    # Non-ASCII preserved (not \u-escaped).
    assert "café" in text


def test_write_state_missing_schema_raises_and_writes_nothing(tmp_path):
    p = str(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        write_state(p, {"seq": 1})
    assert not os.path.exists(p)


def test_write_state_missing_seq_raises_and_writes_nothing(tmp_path):
    p = str(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        write_state(p, {"schema": 1})
    assert not os.path.exists(p)


def test_write_state_invalid_leaves_existing_file_untouched(tmp_path):
    p = str(tmp_path / "s.json")
    good = {"schema": 1, "seq": 1, "v": "old"}
    write_state(p, good)
    with pytest.raises(ValueError):
        write_state(p, {"seq": 2, "v": "new"})  # no schema
    # nothing written -> old content survives intact.
    with open(p, encoding="utf-8") as fh:
        assert json.load(fh) == good


def test_write_state_overwrites_existing(tmp_path):
    p = str(tmp_path / "s.json")
    write_state(p, {"schema": 1, "seq": 1, "v": "first"})
    write_state(p, {"schema": 1, "seq": 2, "v": "second"})
    with open(p, encoding="utf-8") as fh:
        assert json.load(fh) == {"schema": 1, "seq": 2, "v": "second"}


def test_write_state_file_is_complete_after_write(tmp_path):
    # Atomicity is hard to test directly; assert the file parses fully after
    # write (the temp+rename never leaves a truncated file visible).
    p = str(tmp_path / "s.json")
    obj = {"schema": 1, "seq": 2, "plies": [{"ply": i} for i in range(50)]}
    write_state(p, obj)
    assert json.load(open(p, encoding="utf-8")) == obj


# =====================================================================
# run_pass  (engine)
# =====================================================================


@pytest.mark.engine
@requires_engine
def test_run_pass_structure_keys(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    assert set(r) >= {"schema", "seq", "status", "game", "plies", "summary"}
    assert r["schema"] == 1
    assert r["status"] == "complete"
    assert r["seq"] == 2
    assert isinstance(r["plies"], list)
    assert isinstance(r["summary"], dict)


@pytest.mark.engine
@requires_engine
def test_run_pass_game_block(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    g = r["game"]
    assert set(g) >= {"id", "white", "black", "result", "user_side", "source", "date"}
    assert g["white"] == "Alice"
    assert g["black"] == "Bob"
    assert g["result"] == "1-0"
    assert g["user_side"] == "b"  # from UserSide header
    # No Site/id/source/date headers in SCHOLAR_PGN.
    assert g["id"] == "game"
    assert g["source"] is None
    assert g["date"] is None


@pytest.mark.engine
@requires_engine
def test_run_pass_per_ply_fields_present(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    for p in r["plies"]:
        assert set(p) >= {
            "ply", "move_no", "side", "san", "uci", "fen_after",
            "eval_cp", "win_pct", "delta_win_pct", "class", "best",
            "refutation_pv", "motifs",
        }
        # eval_cp is ceiled into [-1000, 1000].
        assert isinstance(p["eval_cp"], int)
        assert -1000 <= p["eval_cp"] <= 1000
        assert 0.0 <= p["win_pct"] <= 100.0
        assert p["class"] in {"ok", "only_move", "dubious", "mistake", "blunder"}
        # best block shape.
        b = p["best"]
        assert isinstance(b["san"], str)
        assert isinstance(b["pv_san"], list) and len(b["pv_san"]) <= 6
        assert isinstance(b["eval_cp"], int) and -1000 <= b["eval_cp"] <= 1000


@pytest.mark.engine
@requires_engine
def test_run_pass_blunder_ply(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    by = plies_by_index(r)
    blunder = by[6]  # ...Nf6, black
    assert blunder["side"] == "b"
    assert blunder["san"].rstrip("+#") == "Nf6"
    assert blunder["class"] == "blunder"
    assert blunder["delta_win_pct"] < 0  # lost ground
    assert blunder["win_pct"] < 25.0  # black is now lost (mate looming)


@pytest.mark.engine
@requires_engine
def test_run_pass_mating_move_ply(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    by = plies_by_index(r)
    mate = by[7]  # Qxf7#, white
    assert mate["side"] == "w"
    assert mate["class"] == "ok"  # mate is never an error
    assert mate["win_pct"] == 100.0
    assert mate["eval_cp"] == 1000


@pytest.mark.engine
@requires_engine
def test_run_pass_good_move_small_delta(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    by = plies_by_index(r)
    e4 = by[1]  # 1.e4 — a fine opening move
    assert e4["class"] == "ok"
    assert abs(e4["delta_win_pct"]) < 5.0


@pytest.mark.engine
@requires_engine
def test_run_pass_decisive_ply_is_the_blunder(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    assert r["summary"]["decisive_ply"] == 6


@pytest.mark.engine
@requires_engine
def test_run_pass_key_moment_has_refutation_and_motifs(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    by = plies_by_index(r)
    blunder = by[6]  # a key moment (blunder + the inflection)
    # opponent's best line after ...Nf6 is Qxf7# -> non-empty, <= 6 SAN.
    assert 1 <= len(blunder["refutation_pv"]) <= 6
    assert all(isinstance(s, str) for s in blunder["refutation_pv"])
    # fen_before (after 3.Qh5, black to move) carries a mate threat -> motifs.
    assert len(blunder["motifs"]) >= 1
    m = blunder["motifs"][0]
    assert set(m) >= {"motif", "confidence", "squares", "concept_id"}


@pytest.mark.engine
@requires_engine
def test_run_pass_non_key_ply_has_empty_refutation_and_motifs(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    by = plies_by_index(r)
    # ply 1 (1.e4) is a clean OK move and not the inflection -> not a key moment.
    non_key = by[1]
    assert non_key["class"] == "ok"
    assert non_key["refutation_pv"] == []
    assert non_key["motifs"] == []


@pytest.mark.engine
@requires_engine
def test_run_pass_summary_counts(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    counts = r["summary"]["counts"]
    assert isinstance(counts, dict)
    # counts cover both sides -> total == number of plies.
    assert sum(counts.values()) == len(r["plies"]) == 7
    assert list(counts.keys()) == sorted(counts.keys())
    assert counts.get("blunder", 0) >= 1


@pytest.mark.engine
@requires_engine
def test_run_pass_phase_losses_counts_user_mistakes(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    pl = r["summary"]["phase_losses"]
    assert set(pl) == {"opening", "middlegame", "endgame"}
    # The only black (user_side) mistake is ...Nf6 at move 3 -> opening.
    assert pl["opening"] >= 1
    assert pl["middlegame"] == 0
    assert pl["endgame"] == 0


@pytest.mark.engine
@requires_engine
def test_run_pass_weakness_hypotheses_empty_without_repeat(engine):
    # SCHOLAR has a single key moment carrying motifs -> nothing recurs >= 2.
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    wh = r["summary"]["weakness_hypotheses"]
    assert isinstance(wh, list)
    assert all(h["occurrences"] >= 2 for h in wh)
    assert wh == []


@pytest.mark.engine
@requires_engine
def test_run_pass_fast_only(engine):
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, fast_only=True)
    assert r["status"] == "fast-pass"
    assert r["seq"] == 1
    # provisional classes only: no motifs, no refutations anywhere.
    assert all(p["refutation_pv"] == [] for p in r["plies"])
    assert all(p["motifs"] == [] for p in r["plies"])
    # the blunder is still classified in the fast pass.
    assert plies_by_index(r)[6]["class"] == "blunder"


@pytest.mark.engine
@requires_engine
@pytest.mark.skip(reason="run_pass no longer writes; persistence re-homed to backend (write_state)")
def test_run_pass_out_path_writes_complete_file(engine, tmp_path):
    out = str(tmp_path / "analysis.json")
    engine.new_game()
    r = run_pass(
        SCHOLAR_PGN, engine,
        out_path=out, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT,
    )
    # returned dict is always produced.
    assert r["status"] == "complete"
    # final on-disk content parses and reflects the complete pass.
    with open(out, encoding="utf-8") as fh:
        disk = json.load(fh)
    assert disk["status"] == "complete"
    assert disk["seq"] == 2
    assert disk["schema"] == 1


@pytest.mark.engine
@requires_engine
def test_run_pass_deterministic_across_fresh_engines():
    def run():
        with Engine(threads=1) as e:
            r = run_pass(
                SCHOLAR_PGN, e, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT
            )
            return [(p["ply"], p["class"]) for p in r["plies"]], r["summary"][
                "decisive_ply"
            ]

    assert run() == run()


@pytest.mark.engine
@requires_engine
def test_run_pass_does_not_close_engine(engine):
    engine.new_game()
    run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    # engine still usable afterwards (run_pass never closes it).
    a = engine.analyse(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", nodes=30_000
    )
    assert a.best.pv


@pytest.mark.engine
@requires_engine
def test_run_pass_propagates_pgnerror(engine):
    engine.new_game()
    with pytest.raises(PgnError):
        run_pass("1. e5 *\n", engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)


# --- reconciliation additions (from the Agent B critique) -------------------

# White is up a queen and stalemates with Qg6 (verified: 0 legal black replies,
# not check). Scoring the terminal child without analysing it is required.
STALEMATE_PGN = (
    '[SetUp "1"]\n[FEN "7k/5K2/8/8/8/8/8/1Q6 w - - 0 1"]\n\n1. Qg6 *\n'
)


@pytest.mark.engine
@requires_engine
def test_run_pass_stalemate_scores_as_blunder(engine):
    # The dual of the mate case: stalemating a winning position is a blunder,
    # scored against a drawn result — and run_pass must not crash on the
    # terminal resulting position.
    engine.new_game()
    r = run_pass(STALEMATE_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    ply = r["plies"][0]
    assert ply["class"] == "blunder"
    assert ply["win_pct"] == 50.0
    assert ply["eval_cp"] == 0
    assert ply["delta_win_pct"] < -25  # threw away a ~winning position


@pytest.mark.engine
@requires_engine
def test_run_pass_eval_cp_is_mover_pov_signed(engine):
    # After 6...Nf6 it is White-to-move mate-in-one, so from the mover's (Black's)
    # POV the position is lost: eval_cp -1000, win_pct 2.5. Pins the POV negation
    # on a *black* ply (a sign bug reporting White's +1000 would pass every other
    # blunder assertion).
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    blunder = plies_by_index(r)[6]
    assert blunder["side"] == "b"
    assert blunder["eval_cp"] == -1000
    assert blunder["win_pct"] == 2.5


@pytest.mark.engine
@requires_engine
def test_decisive_ply_matches_the_gated_definition(engine):
    # Re-derive the contract's decisive rule from the ply data and confirm the
    # summary agrees: largest drop among plies whose win% BEFORE the move was
    # > 25 (not already lost), ties to the earliest; else None. This exercises
    # the gate (SCHOLAR's final plies happen from already-lost positions).
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    plies = r["plies"]
    qualifying = [
        p for p in plies
        if (p["win_pct"] - p["delta_win_pct"]) > 25 and -p["delta_win_pct"] > 0
    ]
    expected = (
        min(qualifying, key=lambda p: (p["delta_win_pct"], p["ply"]))["ply"]
        if qualifying else None
    )
    assert r["summary"]["decisive_ply"] == expected
    if expected is not None:
        chosen = next(p for p in plies if p["ply"] == expected)
        assert (chosen["win_pct"] - chosen["delta_win_pct"]) > 25  # not already lost


ALL_OK_PGN = "1. e4 e5 2. Nf3 Nc6 3. Bb5 *\n"


@pytest.mark.engine
@requires_engine
def test_inflection_ply_is_deep_analysed_even_when_not_a_mistake(engine):
    # A game with no mistakes still has one key moment: the inflection (largest
    # |delta|). Exactly one ply gets a refutation, and it is not a mistake.
    engine.new_game()
    r = run_pass(ALL_OK_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    with_refutation = [p for p in r["plies"] if p["refutation_pv"]]
    assert len(with_refutation) == 1
    assert with_refutation[0]["class"] not in {"dubious", "mistake", "blunder"}


@pytest.mark.engine
@requires_engine
@pytest.mark.skip(reason="run_pass no longer writes; persistence re-homed to backend (write_state)")
def test_fast_only_with_out_path_writes_one_fast_file(engine, tmp_path):
    # fast_only + out_path writes exactly the fast-pass file (seq 1), no complete.
    out = str(tmp_path / "analysis.json")
    engine.new_game()
    r = run_pass(SCHOLAR_PGN, engine, out_path=out, fast_limit=FAST_LIMIT, fast_only=True)
    assert r["status"] == "fast-pass" and r["seq"] == 1
    with open(out, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["status"] == "fast-pass"
    assert on_disk["seq"] == 1


@pytest.mark.engine
@requires_engine
def test_phase_losses_all_zero_when_user_side_unknown(engine):
    # No UserSide header -> user_side null -> no phase is attributed a loss even
    # though the game contains a blunder.
    pgn = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#\n"  # SCHOLAR minus UserSide
    engine.new_game()
    r = run_pass(pgn, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    assert r["game"]["user_side"] is None
    assert r["summary"]["phase_losses"] == {"opening": 0, "middlegame": 0, "endgame": 0}


@pytest.mark.engine
@requires_engine
def test_user_side_accepts_spelled_out_case_insensitive(engine):
    # A capitalized, spelled-out UserSide ("White") is normalized to "w".
    engine.new_game()
    r = run_pass('[UserSide "White"]\n\n1. e4 e5 *\n', engine, fast_limit=FAST_LIMIT,
                 fast_only=True)
    assert r["game"]["user_side"] == "w"
