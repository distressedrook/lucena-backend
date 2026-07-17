"""Black-box tests for the M5 MCP surface (contract: docs/contracts/M5-mcp.md).

Written from the M5 contract ONLY — nothing under `python/lucena/mcp/` was read,
opened, grepped, or run to discover an expected value. Every expectation below is
derived from the contract's documented semantics and cross-checked against an
independent reference (python-chess, in scratch, NEVER imported here — GPL hygiene).

Sections:
  * response  — pure helpers (error/piece_list/eval_block/pv_san/fact_wire/
                facts_to_board/estimate_tokens/enforce_budget). No engine, no I/O.
  * state     — StateStore file/seq/heartbeat/input/analysis behaviour. No engine.
  * analyze_position / evaluate_move / compare_moves / get_game_analysis — the
                engine-touching tools. Marked `engine`, skipped without Stockfish;
                determinism split: constructed with `limit={"nodes": N}`, threads=1.
  * ui-tools  — set_board / push_beat / read_input. No engine.

Positions (verified independently in scratch):
  START = standard start position.
  HANG  = 6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1 — Black knight on e5 hangs; the
          white best move is Rxe5 (e1e5) winning a clean knight.
  SCHOLAR_BEFORE_NF6 = position after 1.e4 e5 2.Bc4 Nc6 3.Qh5, Black to move; ...Nf6
          is the Scholar's-mate blunder that allows Qxf7#.
  SCHOLAR_BEFORE_QXF7 = position after ...Nf6, White to move; Qxf7 is mate.
"""

import asyncio
import copy
import json
import os
import shutil

import pytest

from lucena_engine import Board
from lucena_engine import Engine, Score
from lucena_engine.facts import Fact
from lucena_engine.gamepass import run_pass
from lucena_backend.grounding_tools import response as R
from lucena_backend.persistence.state import StateStore
from lucena_backend.persistence.statefile import write_state
from lucena_backend.grounding_tools.tools import ToolContext


# --------------------------------------------------------------------------
# Fixtures / helpers / constants
# --------------------------------------------------------------------------

NODES = 40_000
FAST_LIMIT = {"nodes": 30_000}
DEEP_LIMIT = {"nodes": 80_000}

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
HANG = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"
SCHOLAR_BEFORE_NF6 = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"
SCHOLAR_BEFORE_QXF7 = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"

# Expected piece_list renderings (order K,Q,R,B,N,P then square; verified in scratch).
START_PIECES = (
    "White: Ke1,Qd1,Ra1,Rh1,Bc1,Bf1,Nb1,Ng1,Pa2,Pb2,Pc2,Pd2,Pe2,Pf2,Pg2,Ph2\n"
    "Black: Ke8,Qd8,Ra8,Rh8,Bc8,Bf8,Nb8,Ng8,Pa7,Pb7,Pc7,Pd7,Pe7,Pf7,Pg7,Ph7"
)
HANG_PIECES = (
    "White: Kg1,Re1,Pf2,Pg2,Ph2\n"
    "Black: Kg8,Ne5,Pf7,Pg7,Ph7"
)

SCHOLAR_PGN = """[Event "Test"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[UserSide "b"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


def _have_stockfish():
    return bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")


requires_engine = pytest.mark.skipif(not _have_stockfish(), reason="no stockfish")


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def engine():
    with Engine(threads=1) as e:
        e.new_game()
        yield e


@pytest.fixture
def store(tmp_path):
    return StateStore(str(tmp_path))


@pytest.fixture
def ctx(engine, store):
    # Deterministic tool calls: nodes-limit + threads=1.
    return ToolContext(engine, store, limit={"nodes": NODES})


@pytest.fixture
def ui_ctx(store):
    # set_board / push_beat / read_input never reach the engine, so engine=None.
    # (Judgment call flagged in the return notes: the contract types `engine` as a
    # live Engine, but these three tools do no engine work.)
    return ToolContext(None, store)


@pytest.fixture
def analysis_ctx(engine, store):
    """A ToolContext whose store already holds analysis.json for game 'g'."""
    path = store.analysis_path("g")
    os.makedirs(os.path.dirname(path), exist_ok=True)  # defensive; see notes
    # Engine computes the digest; the backend persists it (run_pass no longer writes —
    # persistence moved out of the engine, per the extraction).
    digest = run_pass(SCHOLAR_PGN, engine, fast_limit=FAST_LIMIT, deep_limit=DEEP_LIMIT)
    write_state(path, digest)
    engine.new_game()
    return ToolContext(engine, store, limit={"nodes": NODES})


# ==========================================================================
# response helpers  (pure; no engine, no I/O)
# ==========================================================================


def test_error_shape():
    assert R.error("illegal_fen", "not a fen") == {
        "error": "illegal_fen",
        "detail": "not a fen",
    }


def test_piece_list_startpos():
    assert R.piece_list(Board(START)) == START_PIECES


def test_piece_list_two_lines_and_ordering():
    out = R.piece_list(Board(HANG))
    assert out == HANG_PIECES
    lines = out.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("White: ")
    assert lines[1].startswith("Black: ")
    # King leads each color (type order K,Q,R,B,N,P).
    assert lines[0].split(": ", 1)[1].split(",")[0].startswith("K")
    assert lines[1].split(": ", 1)[1].split(",")[0].startswith("K")


def test_eval_block_cp_and_winpct():
    assert R.eval_block(Score(cp=0)) == {"cp": 0, "win_pct": 50.0}
    assert R.eval_block(Score(cp=100)) == {"cp": 100, "win_pct": 59.1}
    assert R.eval_block(Score(cp=300)) == {"cp": 300, "win_pct": 75.1}


def test_eval_block_cp_ceiled_to_1000():
    # cp beyond +/-1000 is clamped by to_ceiled_cp; win_pct saturates ~97.5/2.5.
    assert R.eval_block(Score(cp=5000)) == {"cp": 1000, "win_pct": 97.5}
    assert R.eval_block(Score(cp=-5000)) == {"cp": -1000, "win_pct": 2.5}


def test_eval_block_mate_is_plus_minus_1000():
    assert R.eval_block(Score(mate=1)) == {"cp": 1000, "win_pct": 97.5}
    assert R.eval_block(Score(mate=3)) == {"cp": 1000, "win_pct": 97.5}
    assert R.eval_block(Score(mate=-1)) == {"cp": -1000, "win_pct": 2.5}
    assert R.eval_block(Score(mate=0)) == {"cp": -1000, "win_pct": 2.5}


def test_eval_block_pov_symmetry():
    # A score and its negation report from opposite POVs; win% sums to 100.
    a = R.eval_block(Score(cp=200))
    b = R.eval_block(Score(cp=-200))
    assert a == {"cp": 200, "win_pct": 67.6}
    assert b == {"cp": -200, "win_pct": 32.4}
    assert a["win_pct"] + b["win_pct"] == 100.0


def test_pv_san_renders_and_truncates_to_6():
    pv = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "d2d4"]
    assert R.pv_san(START, pv) == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]


def test_pv_san_custom_max_plies():
    pv = ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert R.pv_san(START, pv, max_plies=2) == ["e4", "e5"]


def test_pv_san_stops_early_on_illegal_element():
    # After e2e4 the e2 square is empty, so a second "e2e4" is illegal: stop there.
    assert R.pv_san(START, ["e2e4", "e2e4", "g1f3"]) == ["e4"]


def test_pv_san_empty_pv():
    assert R.pv_san(START, []) == []


def _hang_fact_2sq(id="F1"):
    return Fact(
        kind="hanging",
        squares=["e5", "e1"],
        text="Rxe5 wins the knight on e5",
        provenance="see:e1e5",
        salience=0.75,
        concept_id="hanging-pieces",
        id=id,
    )


def test_fact_wire_keys_and_drops_internal_fields():
    wire = R.fact_wire(_hang_fact_2sq())
    assert wire == {
        "id": "F1",
        "kind": "hanging",
        "squares": ["e5", "e1"],
        "text": "Rxe5 wins the knight on e5",
        "concept_id": "hanging-pieces",
    }
    assert "salience" not in wire
    assert "provenance" not in wire


def test_facts_to_board_arrow_highlight_and_ignore():
    two_sq = _hang_fact_2sq("F1")
    one_sq = Fact(
        kind="hanging", squares=["e5"], text="t", provenance="x",
        salience=0.5, concept_id="c", id="F2",
    )
    zero_sq = Fact(
        kind="threat", squares=[], text="t", provenance="x",
        salience=0.5, concept_id="c", id="F3",
    )
    arrows, highlights = R.facts_to_board([two_sq, one_sq, zero_sq], style="analysis")
    assert arrows == [
        {"from": "e5", "to": "e1", "style": "analysis", "fact_id": "F1"}
    ]
    assert highlights == [
        {"square": "e5", "style": "analysis", "fact_id": "F2"}
    ]


def test_estimate_tokens_is_monotonic_ish():
    small = R.estimate_tokens({"a": 1})
    big = R.estimate_tokens({"a": 1, "b": "x" * 400})
    assert isinstance(small, int) and isinstance(big, int)
    assert small <= big
    assert big > small  # a much larger object estimates larger


def test_enforce_budget_trims_pvs_then_facts():
    resp = {
        "fen": START,
        "pieces": START_PIECES,
        "eval": {"cp": 10, "win_pct": 51.0},
        "lines": [
            {"rank": 1, "eval": {"cp": 10, "win_pct": 51.0},
             "pv_san": ["Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]},
            {"rank": 2, "eval": {"cp": 5, "win_pct": 50.5},
             "pv_san": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]},
        ],
        "facts": [
            {"id": f"F{i}", "kind": "hanging", "squares": ["e5", "e1"],
             "text": "a reasonably long fact sentence to spend some tokens",
             "concept_id": "hanging-pieces"}
            for i in range(6)
        ],
    }
    full = R.estimate_tokens(resp)
    # (a) an intermediate budget reachable by PV-trim alone must NOT drop facts,
    #     proving PVs are trimmed *first*. Size it just under the full estimate.
    intermediate = dict(resp)
    intermediate["lines"] = [dict(ln) for ln in resp["lines"]]
    intermediate["facts"] = list(resp["facts"])
    out = R.enforce_budget(intermediate, max_tokens=full - 4)
    assert all(len(ln["pv_san"]) <= 4 for ln in out["lines"])
    assert len(out["facts"]) == 6          # facts untouched: PVs came first
    assert out is intermediate              # mutates in place, returns same object
    # (b) a smaller budget drops trailing facts too.
    out2 = R.enforce_budget(resp, max_tokens=60)
    assert all(len(ln["pv_san"]) <= 4 for ln in out2["lines"])
    assert len(out2["facts"]) < 6
    # best-effort floor: an impossibly-tiny budget shrinks as far as it can, but
    # never below the irreducible pieces+eval+lines body (no exception).
    R.enforce_budget(dict(resp, lines=[], facts=[]), max_tokens=1)


def test_enforce_budget_leaves_small_response_unchanged():
    resp = {"eval": {"cp": 0, "win_pct": 50.0}, "lines": [], "facts": []}
    before = copy.deepcopy(resp)
    out = R.enforce_budget(resp, max_tokens=300)
    assert out == before


# ==========================================================================
# StateStore  (no engine)
# ==========================================================================


def test_store_creates_home(tmp_path):
    home = tmp_path / "does" / "not" / "exist"
    StateStore(str(home))
    assert home.is_dir()


def test_write_board_seq_starts_at_one_and_increments(store, tmp_path):
    assert store.write_board(HANG) == 1
    assert store.write_board(HANG) == 2
    assert store.write_board(HANG) == 3


def test_write_board_file_shape_and_none_lists(store, tmp_path):
    seq = store.write_board(HANG)
    data = store._last_board
    assert data["schema"] == 1
    assert data["seq"] == seq == 1
    assert data["fen"] == HANG
    # None lists become [].
    assert data["arrows"] == []
    assert data["highlights"] == []
    assert "caption" in data
    assert "eval" in data


def test_write_board_records_arrows_and_caption(store, tmp_path):
    arrows = [{"from": "e5", "to": "e1", "style": "analysis"}]
    highlights = [{"square": "e5", "style": "analysis"}]
    store.write_board(
        HANG, arrows=arrows, highlights=highlights,
        caption="knight hangs", eval={"cp": 300, "win_pct": 75.1},
    )
    data = store._last_board
    assert data["arrows"] == arrows
    assert data["highlights"] == highlights
    assert data["caption"] == "knight hangs"
    assert data["eval"] == {"cp": 300, "win_pct": 75.1}


def test_coach_decorations_survive_navigation_away_and_back(store):
    """Item 6 (regression): the coach's decorations are keyed to the position they were painted for,
    so an app /position report to another square (and back) no longer BLANKS them. The old code
    overwrote _last_board with empty arrows on every navigation, losing the coach's marks for good."""
    A = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"
    B = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    arrows = [{"from": "e1", "to": "e5", "style": "analysis"}]
    store.write_board(A, arrows=arrows, caption="knight hangs")
    assert store._last_board["arrows"] == arrows
    store.set_board_view(B)                          # app navigates AWAY (/position)
    assert store._last_board["arrows"] == []         # not on A → the marks don't show…
    assert store._last_board["caption"] is None
    store.set_board_view(A)                          # app navigates BACK to the painted position
    assert store._last_board["arrows"] == arrows     # …and they reappear (were NOT destroyed)
    assert store._last_board["caption"] == "knight hangs"


def _beat(text="hi", kind="teach"):
    return {"segments": [{"text": text}], "kind": kind}


def test_append_beats_monotonic_indices_and_cursor(store, tmp_path):
    first = store.append_beats([_beat("a"), _beat("b")])
    assert first == [0, 1]
    second = store.append_beats([_beat("c")])
    assert second == [2]  # global, monotonic across calls
    data = {"beats": store._beats, "cursor": (store._beats[-1]["i"] if store._beats else 0), "seq": store._beats_seq}
    assert len(data["beats"]) == 3  # accumulates all beats so far
    assert data["cursor"] == 2  # newest beat's i


# (test_append_beats_records_session removed — the store-global `session` label is gone; beats are
#  tied to their session via the per-session bundle + the DB `beat` table, not a store attribute.)


def test_read_input_absent_returns_kind_none(store):
    assert store.read_input() == {"kind": "none"}


def test_read_input_returns_parsed_content(store, tmp_path):
    payload = {"kind": "pgn", "text": "1. e4 e5 *"}
    with open(tmp_path / "input.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    assert store.read_input() == payload


def test_read_input_unparseable_returns_kind_none(store, tmp_path):
    with open(tmp_path / "input.json", "w", encoding="utf-8") as fh:
        fh.write("{ this is not json")
    assert store.read_input() == {"kind": "none"}


def test_read_analysis_absent_returns_none(store):
    assert store.read_analysis("nope") is None


def test_analysis_path_shape(store, tmp_path):
    assert store.analysis_path("g") == os.path.join(str(tmp_path), "analysis", "g.json")


def test_write_heartbeat_shape_ok(store, tmp_path):
    seq = store.write_heartbeat(
        engine_ok=True, pid=4242, started_at=100.0, now=200.0
    )
    data = read_json(tmp_path / "heartbeat.json")
    assert data["schema"] == 1
    assert data["seq"] == seq == 1  # own seq counter
    assert data["engine"] == "ok"
    assert data["pid"] == 4242
    assert data["started_at"] == 100.0
    assert data["refreshed_at"] == 200.0
    assert "version" in data


def test_write_heartbeat_engine_down(store, tmp_path):
    store.write_heartbeat(engine_ok=False, pid=1, started_at=1.0, now=2.0)
    data = read_json(tmp_path / "heartbeat.json")
    assert data["engine"] == "down"


# ==========================================================================
# analyze_position  (engine)
# ==========================================================================


@pytest.mark.engine
@requires_engine
def test_analyze_position_result_keys(ctx):
    r = ctx.analyze_and_show(HANG)
    assert set(r) >= {"fen", "side_to_move", "pieces", "eval", "lines", "facts"}
    assert r["fen"] == HANG
    assert r["side_to_move"] == "white"
    assert r["pieces"] == HANG_PIECES


@pytest.mark.engine
@requires_engine
def test_analyze_position_eval_shape(ctx):
    r = ctx.analyze_and_show(HANG)
    ev = r["eval"]
    assert set(ev) == {"cp", "win_pct"}
    assert isinstance(ev["cp"], int) and -1000 <= ev["cp"] <= 1000
    assert 0.0 <= ev["win_pct"] <= 100.0


@pytest.mark.engine
@requires_engine
def test_analyze_position_lines_count_and_pv_len(ctx):
    r = ctx.analyze_and_show(HANG, multipv=2)
    assert len(r["lines"]) == 2
    for ln in r["lines"]:
        assert set(ln) >= {"rank", "eval", "pv_san"}
        assert len(ln["pv_san"]) <= 6
        assert set(ln["eval"]) == {"cp", "win_pct"}
    assert [ln["rank"] for ln in r["lines"]] == [1, 2]


@pytest.mark.engine
@requires_engine
def test_analyze_position_facts_are_wire_shaped(ctx):
    r = ctx.analyze_and_show(HANG)
    assert len(r["facts"]) <= 5
    for f in r["facts"]:
        assert set(f) == {"id", "kind", "squares", "text", "concept_id"}
        assert "salience" not in f and "provenance" not in f


@pytest.mark.engine
@requires_engine
def test_analyze_position_token_ceiling(ctx):
    r = ctx.analyze_and_show(HANG)
    assert R.estimate_tokens(r) <= 300


@pytest.mark.engine
@requires_engine
def test_analyze_position_focus_eval_drops_facts(ctx):
    r = ctx.analyze_and_show(HANG, focus="eval")
    assert r["facts"] == []


@pytest.mark.engine
@requires_engine
def test_analyze_position_focus_threats_only_threat_hanging(ctx):
    r = ctx.analyze_and_show(HANG, focus="threats")
    assert all(f["kind"] in {"threat", "hanging"} for f in r["facts"])


@pytest.mark.engine
@requires_engine
def test_analyze_position_board_push_writes_tagged_arrows(ctx, tmp_path):
    ctx.analyze_and_show(HANG, board_push=True)
    data = ctx.store._last_board
    assert data["fen"] == HANG
    # the hanging fact (2 squares) becomes a fact-tagged arrow.
    assert data["arrows"]
    assert all("fact_id" in a for a in data["arrows"])
    assert data["eval"] is not None


@pytest.mark.engine
@requires_engine
def test_analyze_position_board_push_false_does_not_write(ctx, tmp_path):
    ctx.analyze_and_show(HANG, board_push=False)
    assert ctx.store._last_board is None


@pytest.mark.engine
@requires_engine
def test_analyze_position_illegal_fen(ctx):
    r = ctx.analyze_and_show("not a fen at all")
    assert r["error"] == "illegal_fen"


@pytest.mark.engine
@requires_engine
def test_get_hints_grounded_ladder_for_a_winning_capture(ctx):
    # HANG: Rxe5 wins the loose knight -> a grounded (non-empty) ladder, each
    # rung traced to the engine PV/geometry, none over the token ceiling.
    r = ctx.get_hints(HANG)
    assert r["best"] == "Rxe5"
    assert 1 <= len(r["hints"]) <= 3
    assert [h["rung"] for h in r["hints"]] == sorted(h["rung"] for h in r["hints"])
    assert all(h["provenance"] and h["text"] for h in r["hints"])


@pytest.mark.engine
@requires_engine
def test_get_hints_illegal_fen(ctx):
    assert ctx.get_hints("nope")["error"] == "illegal_fen"


@pytest.mark.engine
@requires_engine
def test_get_hints_terminal_position(ctx):
    # checkmate: no legal moves -> deterministic error, no fabricated hints.
    assert ctx.get_hints("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")["error"] == "terminal_position"


@pytest.mark.engine
@requires_engine
def test_analyze_position_terminal_position(ctx):
    r = ctx.analyze_and_show(CHECKMATE_FEN)
    assert r["error"] == "terminal_position"


# ==========================================================================
# evaluate_move  (engine)
# ==========================================================================


@pytest.mark.engine
@requires_engine
def test_evaluate_move_best_move_low_delta(ctx):
    r = ctx.evaluate_and_show(HANG, "Rxe5", board_push=False)
    assert set(r) >= {
        "fen", "san", "class", "glyph", "delta_win_pct", "eval", "best",
        "refutation_pv", "facts",
    }
    assert r["san"] == "Rxe5"
    assert r["class"] in {"ok", "only_move"}
    assert r["glyph"] in {"", "!", "?!", "?", "??"}
    assert abs(r["delta_win_pct"]) < 5.0
    assert set(r["eval"]) == {"cp", "win_pct"}
    b = r["best"]
    assert set(b) >= {"san", "pv_san", "eval"}
    assert b["san"] == "Rxe5"
    assert len(b["pv_san"]) <= 6


def test_evaluate_move_accepts_uci_and_move_kwarg(ctx):
    # The coach passes the move as SAN or UCI, under `san` or `move` — all resolve to
    # the same move instead of failing a call on the wrong form (Rxe5 == e1e5 here).
    a = ctx.evaluate_and_show(HANG, "Rxe5", board_push=False)
    b = ctx.evaluate_and_show(HANG, "e1e5", board_push=False)          # UCI under san
    c = ctx.evaluate_and_show(HANG, move="e1e5", board_push=False)     # UCI under move kwarg
    assert a["san"] == b["san"] == c["san"] == "Rxe5"
    assert "error" not in a and "error" not in b and "error" not in c


def test_evaluate_move_no_move_is_structured_error(ctx):
    r = ctx.evaluate_and_show(HANG, board_push=False)
    assert r["error"] == "bad_args"


@pytest.mark.engine
@requires_engine
def test_evaluate_move_bad_move_is_mistake_or_blunder(ctx):
    r = ctx.evaluate_and_show(SCHOLAR_BEFORE_NF6, "Nf6", board_push=False)
    assert r["san"].rstrip("+#") == "Nf6"
    assert r["class"] in {"mistake", "blunder"}
    assert r["glyph"] in {"?", "??"}
    assert r["delta_win_pct"] < 0
    assert len(r["refutation_pv"]) >= 1  # Qxf7# refutes it


@pytest.mark.engine
@requires_engine
def test_evaluate_move_mating_move(ctx):
    r = ctx.evaluate_and_show(SCHOLAR_BEFORE_QXF7, "Qxf7", board_push=False)
    assert r["class"] == "ok"  # mate is never an error
    assert r["eval"] == {"cp": 1000, "win_pct": 100.0}
    assert r["refutation_pv"] == []


@pytest.mark.engine
@requires_engine
def test_evaluate_move_illegal_fen(ctx):
    r = ctx.evaluate_and_show("garbage fen", "Rxe5")
    assert r["error"] == "illegal_fen"


@pytest.mark.engine
@requires_engine
def test_evaluate_move_illegal_move(ctx):
    # No white queen exists in HANG, so "Qh5" is not a legal move here.
    r = ctx.evaluate_and_show(HANG, "Qh5")
    assert r["error"] == "illegal_move"


@pytest.mark.engine
@requires_engine
def test_evaluate_move_board_push_paints_class_arrow(ctx, tmp_path):
    ctx.evaluate_and_show(HANG, "Rxe5", board_push=True)
    data = ctx.store._last_board
    # played move e1->e5, colored by class (ok -> "analysis").
    played = [a for a in data["arrows"] if a["from"] == "e1" and a["to"] == "e5"]
    assert played
    assert played[0]["style"] == "analysis"


# ==========================================================================
# compare_moves  (engine)
# ==========================================================================


@pytest.mark.engine
@requires_engine
def test_compare_moves_sorted_by_win_pct_desc(ctx):
    r = ctx.evaluate_and_show(HANG, ["Re2", "Rxe5", "Kf1"], board_push=False)
    moves = r["moves"]
    assert len(moves) == 3
    for m in moves:
        assert set(m) >= {"san", "uci", "eval", "delta_win_pct"}
        assert set(m["eval"]) == {"cp", "win_pct"}
    win_pcts = [m["eval"]["win_pct"] for m in moves]
    assert win_pcts == sorted(win_pcts, reverse=True)
    # Winning the free knight is best, so Rxe5 leads.
    assert moves[0]["san"] == "Rxe5"
    # verdict is the sans joined " > " in sorted order.
    assert r["verdict"] == " > ".join(m["san"] for m in moves)
    assert r["verdict"].startswith("Rxe5")


@pytest.mark.engine
@requires_engine
def test_compare_moves_bad_args_zero_moves(ctx):
    r = ctx.evaluate_and_show(HANG, [])
    assert r["error"] == "bad_args"


@pytest.mark.engine
@requires_engine
def test_compare_moves_bad_args_too_many(ctx):
    r = ctx.evaluate_and_show(HANG, ["Rxe5", "Re2", "Kf1", "Re3", "Re4"])
    assert r["error"] == "bad_args"


@pytest.mark.engine
@requires_engine
def test_compare_moves_illegal_move(ctx):
    r = ctx.evaluate_and_show(HANG, ["Rxe5", "Qh5"])
    assert r["error"] == "illegal_move"


@pytest.mark.engine
@requires_engine
def test_compare_moves_illegal_fen(ctx):
    r = ctx.evaluate_and_show("not a fen", ["Rxe5"])
    assert r["error"] == "illegal_fen"


# ==========================================================================
# get_game_analysis  (engine — via a real analysis.json fixture)
# ==========================================================================


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_summary_has_no_plies(analysis_ctx):
    r = analysis_ctx.get_game_analysis("g", "summary")
    assert set(r) >= {"game", "status", "summary"}
    assert "plies" not in r
    assert isinstance(r["summary"], dict)
    assert r["game"]["white"] == "Alice"
    assert r["status"] == "complete"


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_default_is_summary(analysis_ctx):
    r = analysis_ctx.get_game_analysis("g")
    assert "plies" not in r
    assert set(r) >= {"game", "status", "summary"}


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_mistakes_only_classified(analysis_ctx):
    r = analysis_ctx.get_game_analysis("g", "mistakes")
    assert r["game_id"] == "g"
    assert all(
        p["class"] in {"dubious", "mistake", "blunder"} for p in r["plies"]
    )
    # the Scholar's-mate blunder (ply 6, ...Nf6) qualifies.
    assert any(p["ply"] == 6 for p in r["plies"])


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_ply_selector(analysis_ctx):
    r = analysis_ctx.get_game_analysis("g", "ply:6")
    assert r["ply"] == 6
    assert r["class"] == "blunder"


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_ply_not_found(analysis_ctx):
    r = analysis_ctx.get_game_analysis("g", "ply:99")
    assert r["error"] == "ply_not_found"


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_bad_selector(analysis_ctx):
    assert analysis_ctx.get_game_analysis("g", "ply:abc")["error"] == "bad_args"
    assert analysis_ctx.get_game_analysis("g", "nonsense")["error"] == "bad_args"


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_game_not_found(analysis_ctx):
    r = analysis_ctx.get_game_analysis("no-such-game", "summary")
    assert r["error"] == "game_not_found"


# ==========================================================================
# ui-tools: set_board / push_beat / read_input  (no engine)
# ==========================================================================


def test_set_board_ok_and_writes(ui_ctx, tmp_path):
    r = ui_ctx.set_board(HANG)
    assert r["ok"] is True
    assert isinstance(r["seq"], int) and r["seq"] == 1
    data = ui_ctx.store._last_board
    assert data["fen"] == HANG


def test_set_board_illegal_fen(ui_ctx):
    r = ui_ctx.set_board("definitely not a fen")
    assert r["error"] == "illegal_fen"


def test_push_beat_ok_and_writes(ui_ctx, tmp_path):
    r = ui_ctx.push_beat([_beat("a")])
    assert r["ok"] is True
    assert r["indices"] == [0]
    data = {"beats": ui_ctx.store._beats, "cursor": (ui_ctx.store._beats[-1]["i"] if ui_ctx.store._beats else 0), "seq": ui_ctx.store._beats_seq}
    assert len(data["beats"]) == 1


def test_push_beat_indices_monotonic_across_calls(ui_ctx):
    first = ui_ctx.push_beat([_beat("a"), _beat("b")])["indices"]
    second = ui_ctx.push_beat([_beat("c")])["indices"]
    assert first == [0, 1]
    assert second == [2]


def test_push_beat_bad_args_empty_list(ui_ctx):
    assert ui_ctx.push_beat([])["error"] == "bad_args"


def test_push_beat_bad_args_too_many(ui_ctx):
    beats = [_beat(str(i)) for i in range(5)]
    assert ui_ctx.push_beat(beats)["error"] == "bad_args"


def test_push_beat_bad_args_bad_kind(ui_ctx):
    assert ui_ctx.push_beat([_beat("a", kind="nope")])["error"] == "bad_args"


def test_push_beat_bad_args_missing_segments(ui_ctx):
    assert ui_ctx.push_beat([{"kind": "teach"}])["error"] == "bad_args"
    assert ui_ctx.push_beat([{"kind": "teach", "segments": []}])["error"] == "bad_args"


def test_push_beat_board_repaints(ui_ctx, tmp_path):
    beat = {"segments": [{"text": "x"}], "kind": "teach", "board": {"fen": START}}
    r = ui_ctx.push_beat([beat])
    assert r["ok"] is True
    data = ui_ctx.store._last_board
    assert data["fen"] == START


# -- push_analysis: the single-position analysis object ---------------------

def test_push_analysis_ungrounded_without_analyze(ui_ctx):
    # No analyze_position has run, so no position is grounded -> refused (board authority).
    r = ui_ctx.push_analysis(START, "Black is winning", ["The knight is strong"])
    assert r["error"] == "ungrounded_analysis"


@pytest.mark.engine
@requires_engine
def test_push_analysis_ungrounded_for_unanalysed_fen(ctx):
    ctx.analyze_and_show(HANG)                         # grounds HANG, not START
    r = ctx.push_analysis(START, "roughly equal", ["nothing hanging"])
    assert r["error"] == "ungrounded_analysis"


@pytest.mark.engine
@requires_engine
def test_push_analysis_ok_and_writes(ctx, tmp_path):
    ctx.analyze_and_show(HANG)
    r = ctx.push_analysis(
        HANG, "White is winning — the knight on e5 is hanging.",
        ["The rook on e1 attacks the undefended knight.",
         "Black has no way to defend it in time."])
    assert r["ok"] is True and r["fen"] == HANG
    data = ctx.store._last_analysis
    assert data["fen"] == HANG
    assert data["side_to_move"] == "white"
    assert data["verdict"].startswith("White is winning")
    assert len(data["observations"]) == 2
    assert data["seq"] == 1


@pytest.mark.engine
@requires_engine
def test_push_analysis_bad_args_empty_verdict(ctx):
    ctx.analyze_and_show(HANG)
    assert ctx.push_analysis(HANG, "   ", ["x"])["error"] == "bad_args"


@pytest.mark.engine
@requires_engine
def test_push_analysis_bad_args_no_observations(ctx):
    ctx.analyze_and_show(HANG)
    assert ctx.push_analysis(HANG, "White is better", [])["error"] == "bad_args"
    assert ctx.push_analysis(HANG, "White is better", ["  ", ""])["error"] == "bad_args"
    assert ctx.push_analysis(HANG, "White is better", "not a list")["error"] == "bad_args"


@pytest.mark.engine
@requires_engine
def test_push_analysis_strips_blanks_and_caps_at_six(ctx, tmp_path):
    ctx.analyze_and_show(HANG)
    obs = ["  keep me  ", "", "  ", *[f"point {i}" for i in range(8)]]
    ctx.push_analysis(HANG, "White is winning", obs)
    data = ctx.store._last_analysis
    assert data["observations"][0] == "keep me"        # stripped, blanks dropped
    assert len(data["observations"]) == 6              # capped


@pytest.mark.engine
@requires_engine
def test_push_analysis_blocked_while_awaiting_probe(ctx):
    ctx.analyze_and_show(HANG)
    ctx.push_beat([{"kind": "ask", "text": "what's hanging?"}])   # locks the flow
    r = ctx.push_analysis(HANG, "White is winning", ["knight hangs"])
    assert r["error"] == "awaiting_input"


# -- submit_input + SSE state spine (LLD-app §2) ----------------------------

def test_submit_input_roundtrips_through_read_input(ui_ctx):
    r = ui_ctx.submit_input("move", move="e2e4", fen=START)
    assert r["ok"] is True
    assert r["input"] == {"kind": "move", "uci": "e2e4", "fen": START}
    got = ui_ctx.read_input()                          # prefers the in-memory input
    assert got["kind"] == "move" and got["uci"] == "e2e4"
    assert got["classification"] == "MOVE_EXPLORE"


def test_submit_input_none_classifies_open(ui_ctx):
    ui_ctx.submit_input("none")
    assert ui_ctx.read_input()["classification"] == "OPEN"


def test_snapshot_reflects_writes(store):
    store.write_board(START)
    store.append_beats([{"kind": "say", "stops": False, "segments": [{"text": "hi"}]}])
    store.write_analysis(START, verdict="equal", observations=["nothing hanging"])
    snap = dict(store.snapshot())                      # channels are unique -> a dict is fine
    assert snap["board"]["fen"] == START
    assert snap["beats"]["beats"][0]["segments"][0]["text"] == "hi"
    assert snap["analysis"]["verdict"] == "equal"


def test_snapshot_empty_still_has_beats_channel(store):
    snap = dict(store.snapshot())
    assert snap["beats"]["beats"] == []                # always present, even empty
    assert "board" not in snap                         # board only after a write


def test_state_publish_delivers_board_event_to_subscriber(store):
    # Mirrors production: subscribe on the loop, then a WORKER THREAD writes (sync tools run
    # off-loop), so the publish must hop back via call_soon_threadsafe.
    # asyncio.to_thread — NOT loop.run_in_executor — because to_thread copies the context, so the
    # bound chat crosses into the worker. run_in_executor does not, and the write would land in an
    # unbound chat. Production only ever uses to_thread.
    async def scenario():
        with store.bound("chat-1"):
            sub = store.subscribe("chat-1")
            await asyncio.to_thread(store.write_board, START)
            return await asyncio.wait_for(sub.get(), timeout=2.0)

    channel, payload = asyncio.run(scenario())
    assert channel == "board"
    assert payload["fen"] == START


def test_state_publish_beats_are_append_deltas(store):
    async def scenario():
        with store.bound("chat-1"):
            sub = store.subscribe("chat-1")
            await asyncio.to_thread(
                store.append_beats,
                [{"kind": "say", "stops": False, "segments": [{"text": "one"}]}])
            return await asyncio.wait_for(sub.get(), timeout=2.0)

    channel, payload = asyncio.run(scenario())
    assert channel == "beats"
    assert payload["appended"][0]["segments"][0]["text"] == "one"


def test_read_input_tool_delegates_absent(ui_ctx):
    out = ui_ctx.read_input()
    assert out["kind"] == "none"                       # no input.json -> none
    assert out["classification"] == "OPEN"             # no probe pending (M-classification, additive)
    assert out["must_ground"] == "analyze_and_show" and "build_and_arm_drill" in out["allowed_tools"]


def test_read_input_tool_delegates_content(ui_ctx, tmp_path):
    payload = {"kind": "pgn", "text": "1. e4 *"}
    with open(tmp_path / "input.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    out = ui_ctx.read_input()
    assert {k: out[k] for k in payload} == payload     # input fields preserved (additive)
    assert out["classification"] == "OPEN"             # unknown kind -> OPEN catch-all


# ==========================================================================
# reconciliation additions (from the Agent B critique)
# ==========================================================================

# Black to move, Black rook e2, White knight e5 hangs — a good BLACK capture.
BLACK_WINS = "6k1/5ppp/8/4N3/8/8/4rPPP/6K1 b - - 0 1"
# White up a queen; Qb6 stalemates the a8 king.
STALEMATE_FEN = "k7/8/8/8/8/1Q6/8/K7 w - - 0 1"


def _fresh_ctx(tmp_path, name, nodes=40_000):
    from lucena_backend.persistence.state import StateStore
    from lucena_backend.grounding_tools.tools import ToolContext
    e = Engine(threads=1)
    return e, ToolContext(e, StateStore(str(tmp_path / name)), limit={"nodes": nodes})


@pytest.mark.engine
@requires_engine
def test_analyze_position_deterministic_across_fresh_engines(tmp_path):
    # H1: nodes + threads=1 -> identical result and identical board writes.
    e1, c1 = _fresh_ctx(tmp_path, "a")
    e2, c2 = _fresh_ctx(tmp_path, "b")
    try:
        r1, r2 = c1.analyze_and_show(HANG), c2.analyze_and_show(HANG)
    finally:
        e1.close(); e2.close()
    assert r1 == r2
    assert c1.store._last_board["arrows"] == \
           c2.store._last_board["arrows"]


@pytest.mark.engine
@requires_engine
def test_evaluate_move_deterministic_across_fresh_engines(tmp_path):
    e1, c1 = _fresh_ctx(tmp_path, "c")
    e2, c2 = _fresh_ctx(tmp_path, "d")
    try:
        assert c1.evaluate_and_show(HANG, "Rxe5") == c2.evaluate_and_show(HANG, "Rxe5")
    finally:
        e1.close(); e2.close()


@pytest.mark.engine
@requires_engine
def test_token_ceiling_evaluate_and_compare(ctx):
    ev = ctx.evaluate_and_show(SCHOLAR_BEFORE_NF6, "Nf6")
    assert R.estimate_tokens(ev) <= 300
    cm = ctx.evaluate_and_show(HANG, ["Rxe5", "Kf1", "Re2", "Rd1"])
    assert R.estimate_tokens(cm) <= 300


@pytest.mark.engine
@requires_engine
def test_evaluate_move_stalemate_scores_as_blunder(ctx):
    # H3: stalemating a winning position -> blunder vs a drawn result.
    r = ctx.evaluate_and_show(STALEMATE_FEN, "Qb6")
    assert r["class"] == "blunder"
    assert r["eval"] == {"cp": 0, "win_pct": 50.0}
    assert r["refutation_pv"] == []
    assert r["delta_win_pct"] < -25


@pytest.mark.engine
@requires_engine
def test_evaluate_move_eval_is_mover_pov_for_black(ctx):
    # H4: Black wins a knight -> from Black's (mover) POV eval.cp is POSITIVE.
    r = ctx.evaluate_and_show(BLACK_WINS, "Rxe5")
    assert r["eval"]["cp"] > 0
    assert r["eval"]["win_pct"] > 50


@pytest.mark.engine
@requires_engine
def test_analyze_position_arrows_are_fact_tagged_to_returned_facts(ctx, store, tmp_path):
    # M5: every arrow's fact_id belongs to the returned facts.
    r = ctx.analyze_and_show(HANG)
    data = store._last_board
    ids = {f["id"] for f in r["facts"]}
    assert data["arrows"]
    assert all(a["fact_id"] in ids for a in data["arrows"])


@pytest.mark.engine
@requires_engine
def test_evaluate_move_blunder_paints_correction_arrow(ctx, store, tmp_path):
    # M2: a blunder -> played-move arrow style "correction".
    ctx.evaluate_and_show(SCHOLAR_BEFORE_NF6, "Nf6")
    data = store._last_board
    played = [a for a in data["arrows"] if a["from"] == "g8" and a["to"] == "f6"]
    assert played and played[0]["style"] == "correction"


@pytest.mark.engine
@requires_engine
def test_compare_moves_board_push_styles(ctx, store, tmp_path):
    # M1: best move "analysis", the rest "ghost".
    ctx.evaluate_and_show(HANG, ["Re2", "Rxe5", "Kf1"], board_push=True)
    arrows = store._last_board["arrows"]
    assert arrows[0]["style"] == "analysis"  # best (Rxe5) first
    assert all(a["style"] == "ghost" for a in arrows[1:])


@pytest.mark.engine
@requires_engine
def test_board_push_false_writes_nothing_for_eval_and_compare(tmp_path):
    e1, c1 = _fresh_ctx(tmp_path, "np")
    try:
        c1.evaluate_and_show(HANG, "Rxe5", board_push=False)
        c1.evaluate_and_show(HANG, ["Rxe5", "Kf1"], board_push=False)
    finally:
        e1.close()
    assert c1.store._last_board is None


def test_push_beat_preserves_stops_tone_and_bumps_board_seq(ui_ctx, store, tmp_path):
    # M7: beat content round-trips; a per-beat board repaint bumps board seq.
    store.write_board(START)  # board seq now 1
    ui_ctx.push_beat([
        {"kind": "ask",
         "segments": [{"text": "why?", "tone": "emphasis"}, {"text": " plain"}],
         "board": {"fen": HANG}},
    ])
    beats = store._beats
    assert beats[-1]["stops"] is True and beats[-1]["kind"] == "ask"  # ask implies stops
    assert beats[-1]["segments"][0]["tone"] == "emphasis"
    assert "tone" not in beats[-1]["segments"][1]
    board = store._last_board
    assert board["fen"] == HANG and board["seq"] == 2  # repaint bumped seq


@pytest.mark.engine
@requires_engine
def test_get_game_analysis_mistakes_carry_motifs_and_refutation(analysis_ctx):
    r = analysis_ctx.get_game_analysis("g", "mistakes")
    blunder = next(p for p in r["plies"] if p["class"] == "blunder")
    assert blunder["motifs"]
    assert blunder["refutation_pv"]


# --- push_beat ergonomics + say/ask schema --------------------------------

def test_push_beat_accepts_flat_text_and_type_alias(ui_ctx, tmp_path):
    # flat text, and `type` + the `probe` alias (-> ask) that a live Claude reached for.
    assert ui_ctx.push_beat([{"type": "probe", "text": "is it protected?"}])["ok"] is True
    beats = ui_ctx.store._beats
    assert beats[-1]["kind"] == "ask" and beats[-1]["stops"] is True  # probe -> ask -> stops
    assert beats[-1]["segments"] == [{"text": "is it protected?"}]


def test_push_beat_still_rejects_empty_beat(ui_ctx):
    assert ui_ctx.push_beat([{"kind": "ask"}])["error"] == "bad_args"


def test_push_beat_say_carries_tone_and_concept_id(ui_ctx, tmp_path):
    # `teach` aliases to a `say` toned "teach"; concept_id round-trips; no fact ids.
    ui_ctx.push_beat([{"kind": "teach", "text": "x", "concept_id": "hanging-pieces"}])
    beat = ui_ctx.store._beats[-1]
    assert beat["kind"] == "say" and beat["tone"] == "teach"
    assert beat["concept_id"] == "hanging-pieces"
    assert "fact_ids" not in beat


def test_push_beat_say_tone_praise_and_correct(ui_ctx, tmp_path):
    ui_ctx.push_beat([{"kind": "say", "tone": "praise", "text": "nice"},
                      {"kind": "say", "tone": "correct", "text": "but watch the knight"}])
    beats = ui_ctx.store._beats
    assert beats[-2]["tone"] == "praise" and beats[-1]["tone"] == "correct"


def test_push_beat_probe_carries_hint_ladder(ui_ctx, tmp_path):
    # A probe's hints round-trip into beats.json in order (the Socratic ladder).
    ui_ctx.push_beat([{"kind": "probe", "text": "what does the knight hit?",
                       "stops": True,
                       "hints": ["look at the e6 square", "count its targets",
                                 "it checks and attacks at once"]}])
    beat = ui_ctx.store._beats[-1]
    assert beat["hints"] == ["look at the e6 square", "count its targets",
                             "it checks and attacks at once"]


def test_push_beat_hints_capped_at_three_and_stripped(ui_ctx, tmp_path):
    ui_ctx.push_beat([{"kind": "probe", "text": "q", "stops": True,
                       "hints": [" a ", "", "b", "c", "d"]}])
    beat = ui_ctx.store._beats[-1]
    assert beat["hints"] == ["a", "b", "c"]  # empties dropped, capped at three


def test_push_beat_hints_ignored_on_non_probe(ui_ctx, tmp_path):
    # Hints belong to the question; a teach/verdict beat must not carry them.
    ui_ctx.push_beat([{"kind": "teach", "text": "x", "hints": ["nope"]}])
    beat = ui_ctx.store._beats[-1]
    assert "hints" not in beat


def test_push_beat_ignores_unknown_fields_and_paints_no_board(ui_ctx, tmp_path):
    # fact_ids are gone from the schema (arrows auto-derive from analyze_position);
    # an unknown field is ignored, the beat still lands, and nothing is painted.
    assert ui_ctx.push_beat([{"kind": "say", "text": "x", "fact_ids": ["F1"]}])["ok"]
    assert ui_ctx.store._last_board is None


def test_reset_session_clears_view_and_buffer(store, tmp_path):
    # A fresh server resets the session view so old beats don't reappear (and
    # don't collide with the new session's seq-1 board).
    store.write_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    store.append_beats([{"kind": "teach", "segments": [{"text": "old"}]}])
    assert store._last_board is not None and store._beats

    store.reset_session()
    assert store._last_board is None
    assert not store._beats

    # buffer cleared: the next beat starts at i=0, board_seq back at 1
    store.write_board("8/8/8/8/8/8/8/K6k w - - 0 1")
    store.append_beats([{"kind": "teach", "segments": [{"text": "fresh"}]}])
    data = {"beats": store._beats, "cursor": (store._beats[-1]["i"] if store._beats else 0), "seq": store._beats_seq}
    assert [b["i"] for b in data["beats"]] == [0]
    assert data["beats"][0]["board_seq"] == 1


# --- advance: walk a line forward to coach a combination move by move -------

def test_advance_applies_san_sequence_and_repaints(ui_ctx, tmp_path):
    r = ui_ctx.explore_and_show(START, ["e4", "e5", "Nf3"], analyze=False)
    assert r["line"] == ["e4", "e5", "Nf3"]
    assert r["side_to_move"] == "black"
    assert ui_ctx.store._last_board["fen"] == r["fen"]


def test_advance_accepts_uci(ui_ctx):
    r = ui_ctx.explore_and_show(START, ["e2e4"], analyze=False)
    assert r["line"] == ["e4"] and r["side_to_move"] == "black"


def test_advance_illegal_move_errors(ui_ctx):
    # e4 (white) is legal; then "e4" for black is illegal — the sequence errors.
    assert ui_ctx.explore_and_show(START, ["e4", "e4"])["error"] == "illegal_move"


def test_advance_illegal_fen_errors(ui_ctx):
    assert ui_ctx.explore_and_show("not a fen", ["e4"])["error"] == "illegal_fen"


def test_advance_bad_args_empty(ui_ctx):
    assert ui_ctx.explore_and_show(START, [])["error"] == "bad_args"


# -- P2: turn/drill context is partitioned per session (fragility #2) -----------

def test_turn_context_is_partitioned_per_session(ui_ctx, store):
    # The drill walker + the Socratic/classification gate are per SESSION, not per process — switching
    # sessions must not leak or desync them (M-turn-partition).
    store._switch_current("A")
    ui_ctx._drill = "DRILL_A"
    ui_ctx._awaiting_input = True
    ui_ctx._class = "DRILL_EVENT"

    store._switch_current("B")                       # a different session sees a clean context
    assert ui_ctx._drill is None
    assert ui_ctx._awaiting_input is False
    assert ui_ctx._class == "OPEN"

    store._switch_current("A")                       # switching back restores session A's context
    assert ui_ctx._drill == "DRILL_A"
    assert ui_ctx._awaiting_input is True
    assert ui_ctx._class == "DRILL_EVENT"


def test_session_switch_rehydrates_drill_from_the_document(store):
    """Item 3 (regression): switching to a session whose DOCUMENT holds an armed drill (e.g. one
    armed in a prior process) must rebuild the walker from that document — not leave a fresh
    _SessCtx with drill=None, which makes play_move treat every drill move as freeform and silently
    kills the drill while its tree is still on screen. The walker is a derivation of the document,
    rehydrated on the first access to a session's context."""
    from lucena_backend.grounding_tools.drill import DrillState
    store._switch_current("A")                     # session A arms a drill (persisted to its document)
    store.write_tree(_NUDGE_TREE)
    store.set_drill_state(DrillState(store._last_tree).to_state())
    ctx = ToolContext(None, store)                 # a context that has NEVER built A's session ctx
    store._switch_current("B")                     # land on a different session first
    assert ctx._drill is None                      # B has no drill
    store._switch_current("A")                     # switch (back) to A — the document has the drill
    assert isinstance(ctx._drill, DrillState), "drill not rehydrated on switch — play_move would go freeform"
    assert ctx._drill.tree["root"] == store._last_tree["root"]


def test_phrasing_counters_are_per_session(store):
    """Item 18 (regression): per-session caches/counters live on _SessCtx, so they don't bleed across
    a session switch (the flat-side-object scope P2 removed). A switch away and back preserves each
    session's own counter."""
    ctx = ToolContext(None, store)
    store._switch_current("A")
    ctx._sc.ptxt_i = 5
    ctx._sc.maia_cache_key = ("fenA", 1500, 5)
    store._switch_current("B")
    assert ctx._sc.ptxt_i == 0 and ctx._sc.maia_cache_key is None   # a different session starts fresh
    store._switch_current("A")
    assert ctx._sc.ptxt_i == 5 and ctx._sc.maia_cache_key == ("fenA", 1500, 5)   # A's are intact


def test_open_chat_binds_and_loads_each_chat(tmp_path):
    """Opening a chat binds it for this context and loads its bundle.

    Replaces test_session_switch_fires_on_switch_callback. `_on_switch` was a hook whose only real
    consumer (serve_http) no longer exists — it was never assigned in production, so the deleted test
    was keeping dead code alive by asserting on it. Its stated purpose (resetting the live analyzer
    off the OLD session's position) is moot twice over: the analyzer is entirely unwired, and
    publishes are now addressed to a chat rather than fanned out globally.
    """
    from lucena_backend.persistence.db import DB
    store = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    store.open_chat("A")
    assert store.current_sid == "A"
    store.open_chat("B")
    assert store.current_sid == "B"
    assert {"A", "B"} <= set(store._live)           # each chat kept its own bundle
    store.open_chat("B")                            # idempotent: no change, still bound
    assert store.current_sid == "B"


def test_push_activity_seed_cannot_author_the_view(store):
    """Item 14 (regression): a coach push seed may set the starting board/tree but NEVER the
    UI-authored view (cursor/line) — §8. The view field is stripped from the seed."""
    store._switch_current("A")
    store.push_activity("conversation",
                        seed={"last_board": {"fen": "x"}, "view": {"cursor": 5, "fen": "sneaky"}})
    assert store.view is None                       # the seeded view was stripped (coach can't write it)
    assert store._last_board == {"fen": "x"}        # legitimate fields still seed the frame


def test_ask_beat_persists_the_pending_probe(ui_ctx, store):
    """Item 15 (regression): an `ask` beat records WHICH probe is pending (question + hints) in the
    durable gate, so a session resumed mid-probe can re-render it — not just come back locked. Cleared
    when read_input unlocks the turn."""
    ui_ctx.push_beat([{"kind": "ask", "text": "Why is the knight strong here?",
                       "hints": ["look at the outpost"]}])
    assert store._gate_awaiting is True
    pending = store._gate_pending
    assert pending and "knight" in pending["question"].lower()
    assert pending["hints"] == ["look at the outpost"]
    ui_ctx.read_input()
    assert store._gate_awaiting is False and store._gate_pending is None


def test_undo_move_reverts_last_move(ui_ctx, store):
    # Coach-initiated snap-back: undo the player's last freeform move ("that's not it — try again").
    F0 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    F1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    store.write_history([{"n": 0, "fen": F0}, {"n": 1, "san": "e4", "uci": "e2e4", "fen": F1}])
    store.write_board(F1)
    r = ui_ctx.undo_move()
    assert r["ok"] is True and r["undone"] == "e4"
    assert store._last_board["fen"] == F0            # board snapped back
    assert len(store._history) == 1
    assert ui_ctx.undo_move()["error"] == "nothing_to_undo"   # nothing left to undo


def test_undo_move_reverts_within_a_reported_sideline(ui_ctx, store):
    # The wrong move was played in a SIDELINE the app reported via /view — undo snaps back within THAT
    # line (the server's authoritative current line), not the mainline history (which stays untouched).
    F0 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Fmain = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"     # 1.e4 (the mainline)
    Fside = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"     # 1.Nf3 (the sideline, "wrong")
    store.write_history([{"n": 0, "fen": F0}, {"n": 1, "san": "e4", "uci": "e2e4", "fen": Fmain}])
    store.set_view({"fen": Fside, "cursor": 1, "in_variation": True,
                    "line": [{"fen": F0}, {"san": "Nf3", "fen": Fside}]})   # on the sideline's move
    r = ui_ctx.undo_move()
    assert r["ok"] is True and r["undone"] == "Nf3"
    assert store._last_board["fen"] == F0                # snapped back WITHIN the sideline
    assert len(store._history) == 2                      # the mainline history is untouched


def test_undo_move_twice_walks_back_two_moves_in_a_sideline(ui_ctx, store):
    """Item 5 (regression): two undos in a row (a real coaching pattern — 'not it, undo… still not
    it, undo') must undo TWO distinct moves. The old code read a FIXED view.cursor, so the second
    undo re-undid the same move; deriving the target from board_view each call walks back correctly."""
    F0 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Fmain = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"     # 1.e4 (the mainline)
    Fs1 = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"       # 1.Nf3 (sideline ply 1)
    Fs2 = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"     # 1.Nf3 Nf6 (sideline ply 2)
    store.write_history([{"n": 0, "fen": F0}, {"n": 1, "san": "e4", "uci": "e2e4", "fen": Fmain}])
    store.set_view({"fen": Fs2, "cursor": 2, "in_variation": True,
                    "line": [{"fen": F0}, {"san": "Nf3", "fen": Fs1}, {"san": "Nf6", "fen": Fs2}]})
    r1 = ui_ctx.undo_move()
    assert r1["ok"] and r1["undone"] == "Nf6"
    assert store._last_board["fen"] == Fs1
    r2 = ui_ctx.undo_move()                    # the SECOND undo steps back ANOTHER move, not the same one
    assert r2["ok"] and r2["undone"] == "Nf3"
    assert store._last_board["fen"] == F0
    assert len(store._history) == 2            # the mainline history is untouched throughout


def test_undo_move_ignores_a_stale_view(ui_ctx, store):
    """Item 5 (regression): the undo target comes from the canonical board (board_view) + the
    MCP-authored history, never a stale view. The coach painted back to the mainline tip while an old
    sideline view lingered; undo pops the mainline move, not the dead sideline's."""
    F0 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    Fmain = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"     # 1.e4 mainline tip
    Fdead = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"     # a dead 1.Nf3 sideline
    store.write_history([{"n": 0, "fen": F0}, {"n": 1, "san": "e4", "uci": "e2e4", "fen": Fmain}])
    store.set_view({"fen": Fdead, "cursor": 1, "in_variation": True,
                    "line": [{"fen": F0}, {"san": "Nf3", "fen": Fdead}]})
    store.write_board(Fmain)                    # coach paints back to the mainline tip (off the dead view)
    r = ui_ctx.undo_move()
    assert r["ok"] and r["undone"] == "e4"      # the MAINLINE move, not the stale sideline's Nf3
    assert store._last_board["fen"] == F0
    assert len(store._history) == 1


def test_conclude_session_marks_complete_and_banks(tmp_path):
    # Close the loop: conclude marks the session complete + returns what was banked (and persists it to
    # the rail). record_observation feeds `banked`; here we bank directly (no mastery engine on ui_ctx).
    from lucena_backend.persistence.db import DB
    dbpath = str(tmp_path / "lucena_backend.db")
    store = StateStore(str(tmp_path), db=DB(dbpath))
    store.ensure_session_id()
    ctx = ToolContext(None, store)
    store.add_banked("removing-the-defender")
    store.add_banked("hanging-pieces")
    r = ctx.conclude_session()
    assert r["status"] == "complete"
    assert [b["concept"] for b in r["banked"]] == ["removing-the-defender", "hanging-pieces"]
    assert store.status == "complete"
    assert DB(dbpath).list_sessions()[0]["status"] == "complete"   # persisted to the rail


def test_board_view_does_not_leak_across_sessions(tmp_path):
    # The /position transient fen (`board_view`) is the one store-global position field. A NEW session
    # must NOT inherit the previous session's board through it — regression for "started a new session,
    # entered a position, and the coach taught the OLD session's position instead."
    from lucena_backend.persistence.db import DB
    store = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    store.write_session_id("sess-A")
    store.set_board_view("r2nrk2/ppp2pp1/3B3p/1Q6/8/2P2N2/PP3PPP/R3K2R w - - 0 1")
    assert store.board_view is not None
    store.write_session_id("sess-B")                 # start a fresh session
    assert store.board_view is None                  # clean — the old board doesn't leak in


def test_socratic_gate_survives_restart(tmp_path):
    # P2b: the durable gate — a session locked at a probe resumes LOCKED after a process restart
    # (the coach still can't advance), instead of silently reopening the turn.
    from lucena_backend.persistence.db import DB
    dbpath = str(tmp_path / "lucena_backend.db")
    store = StateStore(str(tmp_path), db=DB(dbpath))
    store.ensure_session_id()
    sid = store._current
    ctx = ToolContext(None, store)
    ctx._awaiting_input = True                        # a probe locked the flow
    assert ctx._gate() is not None                   # gated now

    store2 = StateStore(str(tmp_path), db=DB(dbpath))  # simulate a fresh server on the same home
    store2._switch_current(sid)
    ctx2 = ToolContext(None, store2)
    assert ctx2._awaiting_input is True              # resumed still locked
    assert ctx2._gate() is not None


def test_drill_walker_serializes_through_a_backtrack():
    # P2c: the walker's full state — current node + backtrack stack — survives a serialize/restore
    # (a restart) deterministically, incl. mid-line progress, so no move-replay is needed and no
    # progress is lost. A two-defense tree exercises the backtrack (the fragile part).
    import json as _json
    from lucena_backend.grounding_tools.drill import DrillState
    tree = {"fen": "F0", "side_to_solve": "white", "root": {
        "kind": "solve", "fen": "F0", "expect_uci": "a1a2", "expect_san": "Ra2",
        "after": {"kind": "reply", "fen": "F1", "defenses": [
            {"uci": "b1b2", "san": "Rb2", "then": {"kind": "solve", "fen": "F2a",
             "expect_uci": "c1c2", "expect_san": "Rc2", "after": {"kind": "done", "fen": "F3a"}}},
            {"uci": "b1b3", "san": "Rb3", "then": {"kind": "solve", "fen": "F2b",
             "expect_uci": "d1d2", "expect_san": "Rd2", "after": {"kind": "done", "fen": "F3b"}}},
        ]}}}
    d = DrillState(tree)
    d.play("a1a2", "Ra2")                      # solve root -> auto-play the first defense
    d.play("c1c2", "Rc2")                      # solve branch A -> backtrack onto branch B
    assert d.current.get("fen") == "F2b" and d.solved == 2 and not d.stack and not d.finished

    # simulate a restart: serialize, then restore against a FRESHLY-parsed tree (new object identities)
    restored = DrillState.restore(_json.loads(_json.dumps(tree)),
                                  _json.loads(_json.dumps(d.to_state())))
    assert restored.current.get("fen") == "F2b"
    assert restored.solved == 2 and not restored.stack and not restored.finished
    # and the restored walker adjudicates the next move correctly (proves current resolved to a real node)
    r = restored.play("d1d2", "Rd2")
    assert r["correct"] is True and r["finished"] is True


def test_drill_state_line_has_one_home_the_history(store):
    """Item 8 (regression): the walker's move line is NOT duplicated inside drill_state — it has ONE
    home, the document's history, handed back on restore. Two copies would drift the moment a writer
    touched history without the walker (and the old code preferred the walker's stale copy)."""
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)
    d = DrillState(store._last_tree)
    state = d.to_state()
    assert "line" not in state                         # no rival line copy is serialized
    hist = [{"n": 0, "san": None, "uci": None, "fen": START},
            {"n": 1, "san": "e4", "uci": "e2e4", "fen": _E4}]
    restored = DrillState.restore(store._last_tree, state, line=hist)
    assert restored.line == hist                        # the line comes from history, its one home
    # Back-compat: a legacy state that still embeds a line is honoured only when no line is passed.
    legacy = {**state, "line": [{"n": 0, "fen": "LEGACY"}]}
    assert DrillState.restore(store._last_tree, legacy).line == [{"n": 0, "fen": "LEGACY"}]


# -- P5: the activity stack (push a rabbit-hole, pop back) ----------------------

def test_activity_stack_push_freezes_and_pop_restores(ui_ctx, store):
    store.write_board("BASE w - - 0 1")
    assert store.frame_depth == 1 and store._last_board["fen"] == "BASE w - - 0 1"

    ui_ctx.push_activity("conversation")             # rabbit-hole: fresh empty workspace
    assert store.frame_depth == 2
    assert store._last_board is None                 # the base workspace is frozen, not visible
    store.write_board("RABBIT b - - 0 1")
    assert store._last_board["fen"] == "RABBIT b - - 0 1"

    assert ui_ctx.pop_activity()["ok"] is True        # pop restores the base EXACTLY
    assert store.frame_depth == 1
    assert store._last_board["fen"] == "BASE w - - 0 1"
    assert ui_ctx.pop_activity()["error"] == "cannot_pop"   # can't pop past the base


# -- explore_line: walk a player-named line, then hand back the engine's read ---------

def test_explore_line_illegal_move_errors(ui_ctx):
    r = ui_ctx.explore_and_show(START, ["e4", "e4"])   # e4 for black is illegal
    assert r["error"] == "illegal_move" and "played so far" in r["detail"]


def test_explore_line_illegal_fen_errors(ui_ctx):
    assert ui_ctx.explore_and_show("not a fen", ["e4"])["error"] == "illegal_fen"


def test_explore_line_bad_args_empty(ui_ctx):
    assert ui_ctx.explore_and_show(START, [])["error"] == "bad_args"


@requires_engine
def test_explore_line_surfaces_the_mate_the_player_walks_into(ctx, tmp_path):
    # The exact failure shape from the field: the player proposes a line (…Nf6) that walks into a
    # mate (Qxf7#). explore_line must SURFACE it — a grounded mate_in + the mating move — off the
    # real board it built, never from memory. (Analog of "…bxc4 runs into Qh6, mate in 4".)
    r = ctx.explore_and_show(SCHOLAR_BEFORE_NF6, ["Nf6"])
    assert r["line"] == ["Nf6"]
    assert r["fen"] == SCHOLAR_BEFORE_QXF7          # the position was BUILT, not hallucinated
    assert r["side_to_move"] == "white"
    assert r["mate_in"] is not None and r["mate_in"] > 0   # White (side to move) mates
    assert "f7" in r["best"]                        # Qxf7 is the point
    # painted to the line's end for the player to see.
    assert ctx.store._last_board["fen"] == SCHOLAR_BEFORE_QXF7


@requires_engine
def test_explore_line_reports_terminal_when_the_line_mates(ctx):
    # If the player's line ENDS on the mating move, there's nothing to analyse — report terminal.
    r = ctx.explore_and_show(SCHOLAR_BEFORE_QXF7, ["Qxf7#"])
    assert r["terminal"] == "checkmate" and r["line"] == ["Qxf7#"]


# -- on-solve poisoned-line nudge (server-pushed beat, deterministic) --------------------

_NUDGE_TREE = {   # real fens (write_board parses the side field); a one-move drill (e4 → done)
    "fen": START, "side_to_solve": "white",
    "root": {"kind": "solve", "fen": START, "expect_uci": "e2e4", "expect_san": "e4",
             "after": {"kind": "done", "fen": START}},
}


def test_solve_pushes_poisoned_line_nudge_when_a_trap_was_flagged(ui_ctx, store):
    # A drill whose TREE carries has_poisoned_line=True (set deterministically at build, durable on the
    # document) → solving it pushes a deterministic "there's a poisoned line — ask me" beat. The tree is
    # the ONE source; there is no separate store latch.
    from lucena_backend.grounding_tools.drill import DrillState
    tree = {**_NUDGE_TREE, "has_poisoned_line": True}   # the drill hid a trap
    store.write_tree(tree)
    ui_ctx._drill = DrillState(tree)
    r = ui_ctx.play_move("e2e4", START)
    assert r["finished"] is True
    texts = " ".join(b["segments"][0]["text"] for b in store._beats if b.get("segments"))
    assert "poisoned line" in texts.lower()


def test_solve_no_nudge_when_no_trap(ui_ctx, store):
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)                       # latch cleared, never set
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    r = ui_ctx.play_move("e2e4", START)
    assert r["finished"] is True
    texts = " ".join(b["segments"][0]["text"] for b in store._beats if b.get("segments"))
    assert "poisoned line" not in texts.lower()


def test_freeform_poisoned_line_is_durable_across_repaints(store):
    """Item 4 (regression): the freeform trap lives on the DOCUMENT (set_poisoned); the board's
    has_poisoned_line/poisoned_line are a PROJECTION of it, re-derived on every repaint. So a repaint
    of an unrelated position can't evaporate the trap (the old bug — it stored the trap on the
    transient board), and returning to the trap position re-shows it. Exactly one durable home."""
    TRAP = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"
    moves = [{"uci": "d7e5", "san": "Nxe5", "fen": TRAP}]
    store.set_poisoned(TRAP, moves, {"fatal": "fork", "idea": "looks free but Qg4 wins"})
    store.write_board(TRAP)                             # on the trap position → board projects it
    assert store._last_board["has_poisoned_line"] is True
    assert store._last_board["poisoned_line"] == moves
    store.write_board(START)                            # repaint elsewhere → no trap here…
    assert store._last_board["has_poisoned_line"] is False
    assert store._last_board["poisoned_line"] is None
    store.write_board(TRAP)                             # …but the durable fact survived — re-projects
    assert store._last_board["has_poisoned_line"] is True
    assert store._last_board["poisoned_line"] == moves
    store.clear_poisoned()                              # once dropped, the trap no longer projects
    store.write_board(TRAP)
    assert store._last_board["has_poisoned_line"] is False


# -- deterministic drill close (the loop closes at the source, not via the LLM) -----------------
def test_solve_closes_drill_deterministically_without_a_concept(ui_ctx, store):
    # A drill armed with no concept still CLOSES on solve: a one-shot "concluded" note is set (so the
    # coach can be told once) even though there's no mastery engine and no concept to bank.
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    assert store.take_drill_close() is None                 # nothing pending before the solve
    r = ui_ctx.play_move("e2e4", START)
    assert r["finished"] is True
    dc = store.take_drill_close()
    assert dc is not None and dc["result"] == "solved" and dc["concept"] is None
    assert dc["quality"] == 1.0                             # clean solve, no wrong tries
    assert store.take_drill_close() is None                 # consumed — surfaced exactly once
    assert store.banked == []                               # no concept → nothing banked


@pytest.mark.skip(reason="mastery parked this cycle")
def test_solve_banks_mastery_deterministically_from_the_drill_concept(store, tmp_path):
    # The whole point: a drill armed with a concept_id banks mastery on solve WITHOUT the LLM — the MCP
    # knows exactly what happened (concept + how cleanly). No record_observation call by the coach.
    from lucena_backend.grounding_tools.drill import DrillState
    from lucena.mastery import MasteryEngine
    domain_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "memory", "domain.json")
    domain = json.loads(open(domain_path).read())
    mastery = MasteryEngine(str(tmp_path / "mem"), domain)
    ctx = ToolContext(None, store, mastery=mastery)
    tree = {**_NUDGE_TREE, "concept_id": "hanging-pieces"}   # armed with a concept
    store.write_tree(tree)
    ctx._drill = DrillState(tree)
    r = ctx.play_move("e2e4", START)
    assert r["finished"] is True
    assert store.banked == ["hanging-pieces"]               # banked deterministically, at the source
    assert mastery.mastery("hanging-pieces") is not None    # an observation actually landed
    dc = store.take_drill_close()
    assert dc["concept"] == "hanging-pieces" and dc["quality"] == 1.0


def test_read_input_surfaces_drill_concluded_once_then_clears(ui_ctx, store):
    # The plea half: after a solve, the NEXT read_input tells the coach the drill concluded (so it won't
    # re-praise a finished drill), and only that once — a later turn is clean.
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    ui_ctx.play_move("e2e4", START)
    store.set_input({"kind": "none"})                       # the player asks something else
    out = ui_ctx.read_input()
    assert out.get("drill_concluded", {}).get("result") == "solved"
    store.set_input({"kind": "none"})
    ui_ctx._spoke_since_read = True                         # a real new turn (not an idempotent re-read)
    out2 = ui_ctx.read_input()
    assert "drill_concluded" not in out2                    # surfaced exactly once


def test_drill_concluded_not_surfaced_on_the_solved_turn(ui_ctx, store):
    # On the DRILL_SOLVED closing turn the coach already handles the solve — a second "concluded/banked"
    # signal is redundant and made it parrot "solved and banked" over the player's actual question. It's
    # consumed anyway (so it can't leak to a later turn), just not surfaced here.
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    ui_ctx.play_move("e2e4", START)                        # solves → mailbox holds the drill_solved event
    out = ui_ctx.read_input()                              # the DRILL_SOLVED closing turn
    assert out["classification"] == "DRILL_SOLVED"
    assert "drill_concluded" not in out                    # redundant on this turn — not surfaced
    store.set_input({"kind": "none"})
    ui_ctx._spoke_since_read = True
    assert "drill_concluded" not in ui_ctx.read_input()    # consumed on the solved turn → never leaks


@pytest.mark.skip(reason="mastery parked this cycle")
def test_solve_with_invalid_concept_does_not_claim_a_bank(store, tmp_path):
    # The coach armed a drill with a bad concept id ("hanging-piece" vs "hanging-pieces"); the bank is
    # silently skipped, so drill_close must NOT name the concept — else the coach claims "banked <x>"
    # when nothing was recorded.
    from lucena_backend.grounding_tools.drill import DrillState
    from lucena.mastery import MasteryEngine
    domain_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "memory", "domain.json")
    mastery = MasteryEngine(str(tmp_path / "mem"), json.loads(open(domain_path).read()))
    ctx = ToolContext(None, store, mastery=mastery)
    tree = {**_NUDGE_TREE, "concept_id": "hanging-piece"}   # invalid — real id is "hanging-pieces"
    store.write_tree(tree)
    ctx._drill = DrillState(tree)
    ctx.play_move("e2e4", START)
    assert store.banked == []                               # nothing banked (invalid concept)
    assert store.take_drill_close()["concept"] is None      # and drill_close doesn't claim it


def test_reset_to_start_resets_board_drill_and_line(ui_ctx, store):
    # The app's "back to the previous concept" with an empty stack → standard start position: retire the
    # drill, clear the tree, reset the navigator to the start ply, repaint the start board, and flag the
    # board change so the coach re-grounds.
    from lucena_backend.grounding_tools.drill import DrillState
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    store.write_tree(_NUDGE_TREE)
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    store.write_history(ui_ctx._drill.line)
    r = ui_ctx.reset_to_start()
    assert r["ok"] is True and r["fen"] == start_fen
    assert ui_ctx._drill is None                            # drill retired
    assert store._last_tree is None                         # tree cleared
    assert store._last_board["fen"] == start_fen            # board repainted to the start
    assert store._history == [{"n": 0, "san": None, "uci": None, "fen": start_fen}]
    assert (store._input or {}).get("board_changed") is True   # coach flagged to re-ground


def test_stale_drill_solved_downgrades_to_open_when_board_moved_on(ui_ctx, store):
    # Player solved, then navigated INTO the poisoned line they dodged and typed a question about it. The
    # single-slot mailbox still holds drill_solved (terminal text can't overwrite it), but board_view has
    # moved on → classify OPEN so the coach grounds on the live board, not "you solved it".
    solved = "r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    poisoned = "r1bqkbnr/pppp1ppp/8/4N3/2Bn4/8/PPPP1PPP/RNBQK2R b KQkq - 0 1"   # different placement
    store.set_input({"kind": "drill_solved", "fen": solved})
    store.set_board_view(poisoned)
    assert ui_ctx.read_input()["classification"] == "OPEN"      # board moved on → not a solve turn
    # But a genuine solve turn (board still on the solved position) stays DRILL_SOLVED:
    store.set_input({"kind": "drill_solved", "fen": solved})
    store.set_board_view(solved)
    assert ui_ctx.read_input()["classification"] == "DRILL_SOLVED"


def test_drill_poisoned_line_transition_and_payload(ui_ctx, store):
    # Solved, then exploring the poisoned line they dodged → DRILL_POISONED_LINE, with the whole trap
    # (moves + Maia motif) handed over. On it, a variation WITHIN it, or completely elsewhere are the
    # three outcomes. (ui_ctx has no engine, so the live-eval `current` block is skipped — the line +
    # motif are the core grounding.)
    solved = "r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    p1 = "r1bqkbnr/pppp1ppp/8/4N3/2Bn4/8/PPPP1PPP/RNBQK2R b KQkq - 0 1"       # a node on the poisoned line
    elsewhere = "8/8/8/8/8/8/8/4K2k w - - 0 9"                                 # unrelated board
    store.write_tree({"root": {}, "has_poisoned_line": True,
                      "poisoned_line_moves": [{"uci": "f3e5", "san": "Nxe5", "fen": p1}],
                      "poisoned_line_meta": {"fatal": "fork", "idea": "the knight looks free but Qg4 wins"}})
    # (a) on the poisoned line
    store.set_input({"kind": "drill_solved", "fen": solved})
    store.set_board_view(p1)
    out = ui_ctx.read_input()
    assert out["classification"] == "DRILL_POISONED_LINE" and out["load_skill"] == "drills"
    pl = out["poisoned_line"]                               # ONE natural-language string, not a JSON blob
    assert isinstance(pl, str)
    assert "1.Nxe5" in pl and "Qg4 wins" in pl and "fork" in pl   # numbered SAN + Maia's motif, in prose
    # (b) a variation WITHIN it: the current board isn't a poisoned node, but the on-screen line passes
    # through p1 → still the poisoned-line context.
    store.set_input({"kind": "none"}); ui_ctx._spoke_since_read = True
    store.set_view({"fen": elsewhere, "line": [{"fen": p1}, {"fen": elsewhere}]})   # line passes through p1
    store.set_board_view(elsewhere)
    assert ui_ctx.read_input()["classification"] == "DRILL_POISONED_LINE"
    # (c) completely different board (no overlap with the poisoned line) → OPEN
    store.set_input({"kind": "drill_solved", "fen": solved}); ui_ctx._spoke_since_read = True
    store.set_view({"fen": elsewhere, "line": []})
    store.set_board_view(elsewhere)
    assert ui_ctx.read_input()["classification"] == "OPEN"


@requires_engine
def test_poisoned_line_payload_names_both_evaluations(ctx, store):
    # The DRILL_POISONED_LINE prose names TWO evaluations separately: the poisoned line's own (its END
    # position — where the trap lands you) and the current position (the board in front of the player).
    end = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"      # end of the trap (White wins the knight)
    cur = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    store.write_tree({"root": {}, "has_poisoned_line": True,
                      "poisoned_line_moves": [{"uci": "f3e5", "san": "Nxe5", "fen": end}],
                      "poisoned_line_meta": {"fatal": "fork", "idea": "the knight looks free but Qg4 wins"}})
    store.set_board_view(cur)
    pl = ctx._poisoned_line_payload()
    assert "The evaluation of the poisoned line" in pl and "Nxe5" in pl
    assert "The evaluation of the current position is:" in pl
    # the two are named apart, not merged into one verdict clause
    assert pl.count("The evaluation of the") == 2


# -- stale-drill guard --------------------------------------------------------
# Field report: after the coach set up a fresh endgame, a LEFTOVER drill (from a previous game)
# kept adjudicating — rejecting the correct move against the wrong board and echoing raw UCI
# ("Played h4h3"), while the navigator still showed the old game's line. A live drill pins the
# board to its solve fen, so a move on a DIFFERENT board means the drill is stale: retire it.
_ENDGAME = "2b5/8/pp4p1/3kP1P1/5K1p/P7/1PB5/8 b - - 2 41"   # best move: h3 (h4h3)


def test_drill_diverged_detects_a_new_position(ui_ctx):
    from lucena_backend.grounding_tools.drill import DrillState
    ui_ctx._drill = DrillState(_NUDGE_TREE)                # a drill on START
    assert ui_ctx._drill_diverged(_ENDGAME) is True        # a different board
    assert ui_ctx._drill_diverged(START) is False          # the drill's own board
    assert ui_ctx._drill_diverged(None, "h4h3") is True    # illegal on START → diverged
    assert ui_ctx._drill_diverged(None, "e2e4") is False   # legal on START → still on the drill
    ui_ctx._drill = None
    assert ui_ctx._drill_diverged(_ENDGAME) is False       # no drill → never "diverged"


def test_reset_drill_line_clears_walker_tree_and_navigator(ui_ctx, store):
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    store.write_history(ui_ctx._drill.line)
    ui_ctx._reset_drill_line(_ENDGAME)
    assert ui_ctx._drill is None
    assert store._last_tree is None                         # tree dropped (DB copy too)
    assert store._history == [{"n": 0, "san": None, "uci": None, "fen": _ENDGAME}]


def test_play_move_retires_stale_drill_when_board_diverged(ui_ctx, store):
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)                          # a drill on START (expects e2e4)…
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    store.write_history(ui_ctx._drill.line)               # …navigator seeded with the START line
    # The coach has since set the board to an unrelated endgame; the user plays ITS correct move.
    r = ui_ctx.play_move("h4h3", _ENDGAME)
    assert r["drill"] is False                            # handled freeform, not adjudicated+rejected
    assert ui_ctx._drill is None                          # the stale walker is retired
    echoes = [b["segments"][0]["text"] for b in store._beats if b.get("segments")]
    assert "Played h3" in echoes                          # echoed as SAN…
    assert not any("h4h3" in t for t in echoes)           # …never the raw coordinate
    assert store._history[0]["fen"] == _ENDGAME           # navigator reset to the real board
    assert store._history[-1]["san"] == "h3"


def test_play_move_keeps_a_valid_drill(ui_ctx, store):
    # The guard must NOT false-fire when the move IS on the drill's own board.
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    r = ui_ctx.play_move("e2e4", START)
    assert r["drill"] is True and r["correct"] is True


# 1.e4 e5 2.Nf3 — a two-solve drill, so its `line` holds an EARLIER position to navigate back to.
_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
_E4E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
_E4E5NF3 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
_TWO_MOVE_TREE = {
    "fen": START, "side_to_solve": "white",
    "root": {"kind": "solve", "fen": START, "expect_uci": "e2e4", "expect_san": "e4",
             "after": {"kind": "reply", "fen": _E4,
                       "defenses": [{"san": "e5", "uci": "e7e5", "then":
                           {"kind": "solve", "fen": _E4E5, "expect_uci": "g1f3", "expect_san": "Nf3",
                            "after": {"kind": "done", "fen": _E4E5NF3}}}]}},
}


def test_play_move_off_drill_line_suspends_not_destroys(ui_ctx, store):
    """Item 7 (regression): a move played on a position WITHIN the drill's own line (the player
    navigated back to try a 'what if') must NOT retire the walker or clear the tree — that is a
    request to explore, not a fact the drill is over. The drill is reported SUSPENDED and resumes
    when the board returns to its position. The old code destroyed the whole drill on the report."""
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_TWO_MOVE_TREE)
    ui_ctx._drill = DrillState(store._last_tree)
    store.write_history(ui_ctx._drill.line)
    assert ui_ctx.play_move("e2e4", START)["correct"] is True   # solve move 1 → drill.current advances to _E4E5
    # The player navigates BACK to the start position and tries a different move.
    r = ui_ctx.play_move("d2d4", START)
    assert r["drill"] == "suspended"                            # not destroyed…
    assert ui_ctx._drill is not None                           # …the walker survives…
    assert store._last_tree is not None                        # …and so does the tree
    # Returning to the drill position and playing its move resumes the drill and solves it.
    r2 = ui_ctx.play_move("g1f3", _E4E5)
    assert r2["drill"] is True and r2["correct"] is True and r2["finished"] is True


def test_play_move_retires_drill_on_an_unrelated_board(ui_ctx, store):
    """Item 7 (companion): a move on a genuinely UNRELATED position (the coach set up something new,
    off the drill's line entirely) still retires the stale walker and goes freeform — divergence
    within the line suspends; divergence off the line entirely is the coach moving on."""
    from lucena_backend.grounding_tools.drill import DrillState
    store.write_tree(_NUDGE_TREE)                              # a drill on START…
    ui_ctx._drill = DrillState(_NUDGE_TREE)
    store.write_history(ui_ctx._drill.line)
    r = ui_ctx.play_move("h4h3", _ENDGAME)                     # …a move on an unrelated endgame
    assert r["drill"] is False                                # handled freeform
    assert ui_ctx._drill is None                              # the stale walker is retired


def test_material_balance_deterministic():
    # the position where the coach hallucinated "up a rook and knight" — BOTH sides
    # have a rook, so material is nearly even (Black up a pawn). Grounded, not counted.
    m = R.material(Board("6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2"))
    assert m["white"] == 10 and m["black"] == 11 and m["net"] == -1
    assert m["standing"] == "Black is up a pawn"


def test_material_even_at_start():
    m = R.material(Board(START))
    assert m["net"] == 0 and m["standing"] == "material is even"


def test_standing_names_the_exact_imbalance_not_a_bucket():
    # the same session's END position: net +3, but that is a ROOK FOR TWO PAWNS
    # (with an even N-for-B minor trade), NOT "roughly a minor piece". The coach
    # reads this verbatim, so it must be the precise imbalance.
    m = R.material(Board("1R6/p5pk/7p/7K/P6N/8/5b2/8 b - - 0 4"))
    assert m["net"] == 3
    assert m["standing"] == "White is up a rook for two pawns"


def test_standing_clean_surplus_and_exchange():
    assert R.material(Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1"))["standing"] \
        == "White is up a knight"
    assert R.material(Board("4k3/8/8/8/8/8/8/R2nK3 w - - 0 1"))["standing"] \
        == "White is up the exchange"          # rook for a knight
    # a bare knight-for-bishop swap is materially even — cancelled, not named
    assert R.material(Board("4k3/8/8/8/8/8/8/N2bK3 w - - 0 1"))["standing"] \
        == "material is even"


# --- deterministic Socratic gate: a stops-probe LOCKS the flow --------------

def _write_input(tmp_path, obj):
    with open(tmp_path / "input.json", "w", encoding="utf-8") as f:
        json.dump(obj, f)


def test_probe_locks_position_and_reveal_tools(ui_ctx):
    ui_ctx.push_beat([{"kind": "probe", "text": "your move?", "stops": True}])
    # every position / reveal / advance tool is now blocked (gate runs before the
    # engine, so this holds even with engine=None)
    assert ui_ctx.push_beat([{"kind": "teach", "text": "it's Rxe5"}])["error"] == "awaiting_input"
    assert ui_ctx.set_board(START)["error"] == "awaiting_input"
    assert ui_ctx.explore_and_show(START, ["e4"])["error"] == "awaiting_input"
    assert ui_ctx.analyze_and_show(START)["error"] == "awaiting_input"
    assert ui_ctx.get_hints(START)["error"] == "awaiting_input"


def test_real_input_unlocks_the_flow(ui_ctx, tmp_path):
    ui_ctx.push_beat([{"kind": "probe", "text": "your move?", "stops": True}])
    _write_input(tmp_path, {"kind": "move", "uci": "e2e4"})
    assert ui_ctx.read_input()["kind"] == "move"     # a real answer
    assert ui_ctx.set_board(START)["ok"] is True      # unlocked


def test_read_input_ends_the_probe_wait(ui_ctx):
    # CHANGED (M-classification): a probe locks, but the coach's next turn
    # (read_input) ends the wait — interactively a turn is read only because the
    # player responded; a text answer shows as kind:none -> PROBE_ANSWER.
    ui_ctx.push_beat([{"kind": "probe", "text": "your move?", "stops": True}])
    out = ui_ctx.read_input()
    assert out["kind"] == "none" and out["classification"] == "PROBE_ANSWER"
    assert ui_ctx.set_board(START)["ok"] is True      # unlocked


def test_non_stopping_beat_does_not_lock(ui_ctx):
    ui_ctx.push_beat([{"kind": "teach", "text": "look at the knight"}])
    assert ui_ctx.set_board(START)["ok"] is True      # no lock without a stops-probe


# --------------------------------------------------------------------------
# structured-error contract: a *runtime* failure never escapes as an
# exception — the tool returns the {error, detail} shape, and an engine
# failure is the recoverable `engine_unavailable` code (M5 contract §errors).
# --------------------------------------------------------------------------

class _BoomEngine:
    """A stand-in engine whose calls fail — models a crashed/timed-out Stockfish
    (EngineError) or an unexpected bug (generic). No real process."""

    def __init__(self, exc):
        self._exc = exc

    def new_game(self):
        return None

    def analyse(self, *a, **k):
        raise self._exc


def test_engine_failure_returns_engine_unavailable_not_raised(store):
    from lucena_engine import EngineError
    ctx = ToolContext(_BoomEngine(EngineError("no response")), store,
                      limit={"nodes": NODES})
    out = ctx.analyze_and_show(HANG)          # must not raise
    assert out["error"] == "engine_unavailable"
    assert "retry" in out["detail"].lower()   # actionable recovery


def test_unexpected_failure_returns_internal_not_raised(store):
    ctx = ToolContext(_BoomEngine(RuntimeError("boom")), store,
                      limit={"nodes": NODES})
    out = ctx.evaluate_and_show(HANG, "Rxe5")     # must not raise
    assert out["error"] == "internal"
    assert "retry" in out["detail"].lower()


def test_bad_args_details_enumerate_valid_options(ui_ctx, analysis_ctx):
    # a wrong argument must tell the coach what IS valid (actionable), not just
    # that it was wrong.
    bad_kind = ui_ctx.push_beat([{"kind": "bogus", "text": "x"}])
    assert bad_kind["error"] == "bad_args"
    assert "say" in bad_kind["detail"] and "ask" in bad_kind["detail"]
    bad_sel = analysis_ctx.get_game_analysis("g", "bogus")   # 'g' exists in analysis_ctx
    assert "summary" in bad_sel["detail"] and "mistakes" in bad_sel["detail"]


# --- focus="positional": the grounded five-term strategic read --------------

def test_analyze_position_focus_positional_returns_five_term_read(ctx):
    # a sharp position so at least one term leads and carries features
    KING_ATTACK = "r4rk1/ppp2p2/3p3p/4p3/4P1nq/2NP4/PPP2PP1/R3QRK1 w - - 0 15"
    out = ctx.analyze_and_show(KING_ATTACK, focus="positional")
    assert out["facts"] == []                       # tactical fact sheet skipped
    assert out["lines"] == []                       # no PVs on a strategic read
    p = out["positional"]
    assert set(p["terms"]) == {"material", "king_safety", "activity", "pawns", "center"}
    assert 0.0 <= p["phase"] <= 1.0
    for t in p["terms"].values():
        assert isinstance(t["cp"], int) and isinstance(t["standing"], str)
    assert p["leads"], "a sharp position should surface at least one lead"
    for name in p["leads"]:                          # lead terms carry citable features
        assert "features" in p["terms"][name]


# --- focus="analysis": the grounded natural-language briefing ---------------

def test_analyze_position_focus_analysis_returns_nl_briefing(ctx):
    KING_ATTACK = "r4rk1/ppp2p2/3p3p/4p3/4P1nq/2NP4/PPP2PP1/R3QRK1 w - - 0 15"
    out = ctx.analyze_and_show(KING_ATTACK, focus="analysis")
    assert "analysis" in out and isinstance(out["analysis"], list) and out["analysis"]
    assert all(isinstance(s, str) and s.strip() for s in out["analysis"])
    assert "facts" not in out and "positional" not in out   # briefing replaces them
    assert "White to move" in out["analysis"][0]            # grounded side-to-move
    assert out["pieces"]                                    # roster kept for grounding


def test_focus_analysis_paints_board_without_citation(ctx, store):
    # arrows auto-derive from the computed fact sheet; no fact_ids involved
    ctx.analyze_and_show("6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2", focus="analysis")
    board = store._last_board
    assert board["fen"] == "6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2"


# --- get_line_tree: build the drill the app walks --------------------------

def test_get_line_tree_writes_drill_and_summarizes(ctx, store):
    out = ctx.build_and_arm_drill("6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2")
    assert out["drillable"] is True
    assert out["load_skill"] == "drills"        # a drillable position → the app-driven drill lane
    assert out["first_move"] == "Rb8+"          # the forced only-move
    assert out["side_to_solve"] == "white" and out["lines"] >= 2
    tree = store._last_tree
    assert tree["seq"] == 1 and tree["root"]["kind"] == "solve"
    assert tree["root"]["expect_san"] == "Rb8+"  # full tree is on disk for the app


def test_get_line_tree_non_forcing_is_not_drillable(ctx, store):
    out = ctx.build_and_arm_drill(START)
    assert out["drillable"] is False and out["root_kind"] == "done"
    assert out["load_skill"] == "coaching-a-position"   # not forcing → coach it freeform
    assert out["first_move"] is None
    # A non-drillable position must NOT leave a tree.json — the app would present a
    # broken drill that rejects every move as wrong (root is "done", nothing to solve).
    assert out["seq"] is None
    assert store._last_tree is None


def test_get_line_tree_non_forcing_clears_a_prior_drill(ctx, store):
    ctx.build_and_arm_drill("6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2")   # drillable → writes tree.json
    assert store._last_tree is not None
    ctx.build_and_arm_drill(START)                                          # non-forcing → clears it
    assert store._last_tree is None


def test_get_line_tree_returns_no_bulk_authoring_payload(ctx):
    # The coach coaches each node LIVE (analyze_position per event); get_line_tree
    # returns a compact summary only — no `nodes` batch to author cold (the long
    # context that drove hallucination). The full tree stays on disk for the app.
    out = ctx.build_and_arm_drill("6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2")
    assert "nodes" not in out


# -- Maia prediction in grounding responses (currentPlayerRating) ----------

def _have_maia():
    return bool(os.environ.get("LUCENA_MAIA")) or shutil.which("maia3-5m")


requires_maia = pytest.mark.skipif(not _have_maia(), reason="no maia3 (set LUCENA_MAIA)")


def test_player_tendency_absent_without_predictor(ctx):
    # Never load-bearing: no predictor -> no `player_tendency` key, tool still works.
    r = ctx.analyze_and_show(HANG, focus="analysis", board_push=False)
    assert "player_tendency" not in r


@requires_engine
@requires_maia
def test_grounding_responses_carry_player_tendency_as_plain_text(engine, tmp_path):
    from lucena_engine.maia import MaiaEngine
    tac = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"
    with MaiaEngine() as m:
        ctx = ToolContext(engine, StateStore(str(tmp_path)), limit={"nodes": 60_000},
                          maia=m, player_rating=1300)
        r = ctx.analyze_and_show(tac, focus="analysis", board_push=False)
        txt = r["player_tendency"]
        assert isinstance(txt, str)                    # a sentence, not raw ranks/probs
        assert "1300" in txt and "bxc4" in txt         # the rating + the likely move, in words
        assert "maia" not in txt.lower()               # the coach never learns the source
        # Firewall by construction: only move names + the (grounded) best move — no eval
        # numbers to misread. The engine still condemns the tempting move independently.
        ev = ctx.evaluate_and_show(tac, san="Nxe4", board_push=False)
        assert "player_tendency" in ev and ev["class"] == "blunder"


@requires_engine
@requires_maia
def test_move_read_is_the_meaning_the_assess_channel_relays(engine, tmp_path):
    # evaluate_move computes `move_read` — the deterministic MEANING the app's assess
    # watcher writes to move_meaning.json. Level-typical blunder here -> a "common
    # mistake" read; a plain best move most players find -> nothing notable (None).
    from lucena_engine.maia import MaiaEngine
    tac = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"
    with MaiaEngine() as m:
        ctx = ToolContext(engine, StateStore(str(tmp_path)), limit={"nodes": 60_000},
                          maia=m, player_rating=1300)
        r = ctx.evaluate_and_show(tac, san="Nxe4", board_push=False)
        assert r["class"] == "blunder"
        # `move_read` = the coach's private notes (imperatives are fine there).
        assert "COMMON blunder" in r["move_read"] and "maia" not in r["move_read"].lower()
        # `move_meaning` = what the app shows the PLAYER — addressed to them, no coach
        # instructions like "teach the pattern".
        pm = r["move_meaning"].lower()
        assert "your level" in pm or "your rating" in pm      # addressed to the player, at their level
        assert "coach" in pm                                  # nudges a conversation with the coach
        assert "teach the pattern" not in pm and "be kind" not in pm   # not the coach's private notes
        # bxc4 is best AND a common human pick at 1300 -> not beyond their level -> nothing to show
        best = ctx.evaluate_and_show(tac, san="bxc4", board_push=False)
        assert best.get("move_read") is None and best.get("move_meaning") is None


@requires_engine
def test_assess_move_serializes_on_the_tool_lock(engine, tmp_path):
    """Item 2 (regression): assess_move is a direct tool the APP calls; it MUST run under the
    single-writer @_guarded lock for its own chat, or its engine calls interleave with a coach tool
    in that chat and one side reads the other's bestmove. We prove the lock is held for the duration
    of the engine call by probing it from ANOTHER thread while assess_move runs: guarded → the probe
    can't acquire; unguarded → it acquires freely.

    The lock is now PER CHAT (`_lock_for(sid)`), so the probe must ask for the same chat's lock —
    probing a different one would acquire freely and the test would pass while asserting nothing.
    test_two_chats_do_not_serialize_on_each_other covers the other half.
    """
    import threading
    store = StateStore(str(tmp_path))
    ctx = ToolContext(engine, store, limit={"nodes": 40_000})
    other_could_acquire = []
    real_analyse = engine.analyse

    def probing_analyse(*a, **k):
        def probe():
            lock = ctx._lock_for(store.current_sid)      # THIS chat's lock
            got = lock.acquire(blocking=False)
            other_could_acquire.append(got)
            if got:
                lock.release()
        t = threading.Thread(target=probe)
        t.start()
        t.join()
        return real_analyse(*a, **k)

    engine.analyse = probing_analyse
    try:
        r = ctx.assess_move(HANG, "Rxe5")
    finally:
        engine.analyse = real_analyse
    assert other_could_acquire, "engine.analyse was never called — test didn't exercise the lock"
    assert all(x is False for x in other_could_acquire), \
        "assess_move ran without holding its chat's tool lock — its engine calls can interleave"
    assert r["san"] == "Rxe5"   # sanity: the guarded call still returns the assessment
    assert store._last_board is None   # still read-only under the guard


def test_two_chats_do_not_serialize_on_each_other(engine, tmp_path):
    """The point of the per-chat lock: chat B must NOT be held out while chat A is mid-tool. One
    process-wide lock made every user's turn queue behind every other user's."""
    import threading
    store = StateStore(str(tmp_path))
    ctx = ToolContext(engine, store, limit={"nodes": 40_000})
    other_could_acquire = []
    real_analyse = engine.analyse

    def probing_analyse(*a, **k):
        def probe():
            lock = ctx._lock_for("some-other-chat")      # a DIFFERENT chat's lock
            got = lock.acquire(blocking=False)
            other_could_acquire.append(got)
            if got:
                lock.release()
        t = threading.Thread(target=probe)
        t.start()
        t.join()
        return real_analyse(*a, **k)

    engine.analyse = probing_analyse
    with store.bound("chat-a"):
        try:
            ctx.assess_move(HANG, "Rxe5")
        finally:
            engine.analyse = real_analyse
    assert other_could_acquire, "engine.analyse was never called — test didn't exercise the lock"
    assert all(x is True for x in other_could_acquire), \
        "another chat was blocked while chat-a ran a tool — the lock is not per-chat"


def test_assess_move_tool_is_read_only_and_player_facing(engine, tmp_path):
    # The app calls assess_move directly over MCP: {san, class, meaning}, player-facing,
    # no state writes, no fact sheet (lean/fast).
    from lucena_engine.maia import MaiaEngine
    tac = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"
    with MaiaEngine() as m:
        store = StateStore(str(tmp_path))
        ctx = ToolContext(engine, store, limit={"nodes": 60_000}, maia=m, player_rating=1300)
        r = ctx.assess_move(tac, "Nxe4")
        assert r["san"] == "Nxe4" and r["class"] == "blunder"
        pm = r["meaning"].lower()
        assert ("your level" in pm or "your rating" in pm) and "coach" in pm
        assert "teach the pattern" not in pm             # player-facing, not coach notes
        assert ctx.assess_move(tac, "bxc4")["meaning"] is None
        assert ctx.assess_move(tac, "Qz9")["error"] == "illegal_move"
        assert store._last_board is None   # read-only


@requires_engine
@requires_maia
def test_common_mistakes_maia_selects_stockfish_evaluates(engine, tmp_path):
    # "what do folks at my level get wrong here?" — Maia picks WHICH moves (human-likely
    # at this rating), Stockfish evaluates each; the coach narrates the observations.
    from lucena_engine.maia import MaiaEngine
    tac = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"
    with MaiaEngine() as m:
        ctx = ToolContext(engine, StateStore(str(tmp_path)), limit={"nodes": 60_000},
                          maia=m, player_rating=1500)
        r = ctx.get_common_mistakes(tac)
        assert r["best"] == "bxc4"
        sans = [mv["san"] for mv in r["moves"]]
        assert 1 <= len(sans) <= 5 and "Nxe4" in sans        # Maia's likely moves at 1500
        nxe4 = next(mv for mv in r["moves"] if mv["san"] == "Nxe4")
        assert nxe4["class"] == "blunder"                     # Stockfish's verdict
        assert nxe4["delta_win_pct"] < 0 and nxe4["refutation"]   # eval + the punishing line
        assert "Nxe4" in r["mistakes"]


# --------------------------------------------------------------------------
# set_puzzle — the "give me a puzzle" one-call flow (select + push + paint + arm)
# --------------------------------------------------------------------------

_PUZZLE_FORK_A = {
    "puzzle_id": "eiDZO",
    "position_fen": "5R2/2k3r1/4pN2/4Pp2/3n1n2/1P5P/6PK/8 w - - 10 40",
    "themes": "advantage endgame fork short", "rating": 1400,
}
_PUZZLE_FORK_B = {
    "puzzle_id": "Wa80T",
    "position_fen": "1Rb1r3/5pk1/6p1/8/3b4/3B2K1/1PP3P1/5R2 b - - 7 30",
    "themes": "advantage endgame fork short", "rating": 1600,
}


def _seed_puzzles(tmp_path, monkeypatch, rows):
    from lucena_backend.grounding_tools import puzzle_content as pc
    d = tmp_path / "puzzles"
    d.mkdir()
    (d / "deck.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("LUCENA_PUZZLES", str(d))
    pc._cache.clear()
    return d


@requires_engine
def test_set_puzzle_selects_pushes_paints_and_arms(tmp_path, monkeypatch):
    _seed_puzzles(tmp_path, monkeypatch, [_PUZZLE_FORK_A])
    store = StateStore(str(tmp_path / "home"))
    with Engine(threads=1) as e:
        ctx = ToolContext(e, store, limit={"nodes": 60_000})
        r = ctx.set_puzzle(theme="fork")

    assert r.get("ok") and "error" not in r
    assert r["puzzle_id"] == "eiDZO"
    assert r["concept_id"] == "forks-double-attack"     # theme -> concept mapping
    assert r["drillable"] is True and r["first_move"]   # a forcing win was armed
    assert r["load_skill"] == "drills"
    # a fresh activity frame was pushed (the puzzle is an unrelated position)
    assert store.frame_depth == 2
    # the board shows the puzzle position (solver to move), WITHOUT arms/spoilers
    assert store.board_view == _PUZZLE_FORK_A["position_fen"]
    # served this session
    assert store.served_puzzles == ["eiDZO"]
    # the intro is a LOCAL beat (server-authored) — not left to the LLM to narrate
    texts = " ".join(seg.get("text", "")
                     for b in store._beats for seg in b.get("segments", []))
    assert "board" in texts.lower()


@requires_engine
def test_set_puzzle_never_repeats_within_a_session(tmp_path, monkeypatch):
    _seed_puzzles(tmp_path, monkeypatch, [_PUZZLE_FORK_A, _PUZZLE_FORK_B])
    store = StateStore(str(tmp_path / "home"))
    with Engine(threads=1) as e:
        ctx = ToolContext(e, store, limit={"nodes": 60_000})
        first = ctx.set_puzzle(theme="fork")
        second = ctx.set_puzzle(theme="fork")
        exhausted = ctx.set_puzzle(theme="fork")

    assert first["puzzle_id"] != second["puzzle_id"]
    assert set(store.served_puzzles) == {"eiDZO", "Wa80T"}
    # both fork puzzles served -> the deck is exhausted, a structured error (no invented position)
    assert exhausted.get("error") == "no_puzzles"


@requires_engine
def test_set_puzzle_empty_deck_is_structured_error(tmp_path, monkeypatch):
    _seed_puzzles(tmp_path, monkeypatch, [])       # empty deck file
    store = StateStore(str(tmp_path / "home"))
    with Engine(threads=1) as e:
        ctx = ToolContext(e, store, limit={"nodes": 60_000})
        r = ctx.set_puzzle()
    assert r.get("error") == "no_puzzles" and r.get("detail")
    assert store.frame_depth == 1                  # nothing pushed on failure
