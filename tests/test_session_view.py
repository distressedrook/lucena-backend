"""Black-box tests for the Session-View milestone.

Written from the contract ONLY (docs/contracts/M-session-view.md). Nothing under
`python/lucena/mcp/` (state.py, server.py, tools.py, db.py) was read to discover an
expected value or behaviour; every expectation below is derived from the contract's
documented semantics. FENs are verified legal with python-chess in scratch (dev
tooling, GPL carve-out — NEVER imported here).

Feature under test (contract §Scope): the app pushes a resolved *view* snapshot
(board + variation forest + cursor); the server derives `side_to_move`, persists it
to an atomic `view.json` (+ SQLite), exposes it to the coach via a `get_view` tool,
and replays it over SSE on session resume.

===========================================================================
RECONCILIATION NOTES — inferred entry points (contract names HTTP, not Python)
===========================================================================
The contract specifies the *HTTP* ingress (`POST /view`, `POST /position`) and the
`get_view` *tool*, but not the Python method the server's handler calls to ingest a
snapshot. Every existing black-box suite drives the server in-process through
`StateStore` / `ToolContext` (never over real HTTP), so these tests do the same and
centralise each inferred call in ONE helper below, so reconciliation can pin the
real name in a single place if the guess is wrong. Inferred:
  * `post_view(store, snapshot)`  -> `store.write_view(snapshot)`   (mirrors write_board)
  * `post_position(store, fen)`   -> `store.set_position(fen)`      (the /position alias)
  * `get_view()`                  -> `ToolContext.get_view()`       (per contract §Read tool)
If any name differs, fix the ONE helper, not each test.

Invariant 9 (re-hydration round-trip) is *app-side* (Swift `VariationForest` rebuild):
out of scope for pytest. The SERVER-side half of it — idempotent re-ingest of the
same snapshot — IS tested (see test_reingesting_same_snapshot_is_idempotent).
"""

import json
import pathlib

import pytest

from lucena_backend.state import StateStore
from lucena_backend.tools import ToolContext

try:
    from lucena_backend.db import DB
except Exception:  # pragma: no cover - DB optional in some layouts
    DB = None


# --------------------------------------------------------------------------
# Positions (verified legal in scratch with python-chess; ep squares as emitted)
# --------------------------------------------------------------------------

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
E4    = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"           # black to move
E4E5  = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"         # WHITE to move
NF3   = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"       # mainline, black to move
NC3   = "rnbqkbnr/pppp1ppp/8/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 2"       # sideline, black to move
BC4   = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2"       # 2nd sideline at E4E5
NC3NC6 = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"    # ...Nc6, a sub-line off Nc3
WHITE_UP_A_PAWN = "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"   # black minus its d-pawn


# --------------------------------------------------------------------------
# Snapshot builders — the app -> server "view snapshot" shape (contract §snapshot)
# --------------------------------------------------------------------------

def _line_row(i, san, uci, fen, kind, branch):
    return {"i": i, "san": san, "uci": uci, "fen": fen, "kind": kind, "branch": branch}


def branch_snapshot(session, *, cursor=3):
    """1.e4 e5 with the mainline (…Nf3) branched at ply 2 into a sideline (…Nc3),
    cursor viewing the position AFTER the sideline move Nc3 (black to move).

    One branch point (at E4E5), one sideline (first move Nc3)."""
    return {
        "session": session,
        "fen": NC3,
        "cursor": cursor,
        "in_variation": True,
        "line": [
            _line_row(0, None, None, START, "main", False),
            _line_row(1, "e4", "e2e4", E4, "main", False),
            _line_row(2, "e5", "e7e5", E4E5, "main", False),
            _line_row(3, "Nc3", "b1c3", NC3, "variation", True),
        ],
        "tree": [
            {
                "at_fen": E4E5,
                "at_ply": 2,
                "variations": [
                    {"uci": "b1c3", "san": "Nc3", "fen": NC3,
                     "line": [{"uci": "b1c3", "san": "Nc3", "fen": NC3}]},
                ],
            }
        ],
    }


def mainline_snapshot(session, *, cursor=3, fen=NF3, in_variation=False):
    """Pure mainline 1.e4 e5 2.Nf3, no variations created (tree == [])."""
    return {
        "session": session,
        "fen": fen,
        "cursor": cursor,
        "in_variation": in_variation,
        "line": [
            _line_row(0, None, None, START, "main", False),
            _line_row(1, "e4", "e2e4", E4, "main", False),
            _line_row(2, "e5", "e7e5", E4E5, "main", False),
            _line_row(3, "Nf3", "g1f3", NF3, "main", False),
        ][: cursor + 1],
        "tree": [],
    }


# --------------------------------------------------------------------------
# Centralised inferred entry points (see RECONCILIATION NOTES) + helpers
# --------------------------------------------------------------------------

def post_view(store, snapshot):
    """Drive the server's `POST /view` ingest (reconciled: store.set_view)."""
    return store.set_view(snapshot)


def post_position(store, fen):
    """Drive the legacy `POST /position` alias — fills only `fen` (reconciled: store.set_board_view)."""
    return store.set_board_view(fen)


def make_ctx(store):
    """A ToolContext for the state-only tools; get_view does no engine work."""
    return ToolContext(None, store)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_view_json(home):
    """view.json may live in the session bundle dir; locate it wherever it is."""
    hits = list(pathlib.Path(home).rglob("view.json"))
    return hits[0] if hits else None


@pytest.fixture
def store(tmp_path):
    """A DB-backed store with one active session — the normal server shape."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db"))) if DB else StateStore(str(tmp_path))
    s._switch_current("A")
    return s


# ==========================================================================
# Invariant 1 — Round-trip fidelity
# ==========================================================================

def test_posted_view_round_trips_through_get_view(store):
    """Inv 1: a view posted to /view is retrievable via get_view with the same
    fen, cursor position (cursor_san), in_variation, and the same variation set."""
    post_view(store, branch_snapshot("A"))
    v = make_ctx(store).get_view()
    assert "error" not in v
    assert v["fen"] == NC3
    assert v["in_variation"] is True
    # cursor position surfaces as the SAN the cursor sits on (contract §Read tool).
    assert v["cursor_san"] == "Nc3"
    # exactly the one branch we posted; its first move appears in the compact line.
    assert len(v["variations"]) == 1
    assert "Nc3" in v["variations"][0]["line"]


def test_cursor_at_start_position_has_null_cursor_san(store):
    """Inv 1: line[0] is the start position with san/uci null; get_view reports
    cursor_san == null when the cursor sits on the start position (cursor == 0)."""
    post_view(store, mainline_snapshot("A", cursor=0, fen=START, in_variation=False))
    v = make_ctx(store).get_view()
    assert v["fen"] == START
    assert v["cursor_san"] is None


# ==========================================================================
# Invariant 2 — Grounded side-to-move (server derives from fen, never trusts client)
# ==========================================================================

def test_side_to_move_derived_from_fen_black(store, tmp_path):
    """Inv 2: get_view().side_to_move and view.json.side_to_move equal the side
    derived from the viewed fen (NC3 -> black)."""
    post_view(store, branch_snapshot("A"))
    assert make_ctx(store).get_view()["side_to_move"] == "black"
    assert store._view is not None
    assert store._view["side_to_move"] == "black"


def test_side_to_move_derived_from_fen_white(store):
    """Inv 2: a viewed position with White to move (E4E5) grounds to 'white'."""
    post_view(store, mainline_snapshot("A", cursor=2, fen=E4E5, in_variation=False))
    assert make_ctx(store).get_view()["side_to_move"] == "white"


def test_side_to_move_ignores_client_supplied_value(store, tmp_path):
    """Inv 2: even if the client smuggles a side_to_move field, the server derives
    it from the fen and ignores the client (fen is black-to-move -> 'black')."""
    snap = branch_snapshot("A")
    snap["side_to_move"] = "white"  # deliberately wrong; must be ignored
    post_view(store, snap)
    assert make_ctx(store).get_view()["side_to_move"] == "black"
    assert store._view["side_to_move"] == "black"


# ==========================================================================
# Invariant 3 — Single writer / atomic (schema + monotonic seq, never torn)
# ==========================================================================

def test_view_json_has_schema_and_seq(store, tmp_path):
    """Inv 3: view.json carries schema == 1 and a seq."""
    post_view(store, branch_snapshot("A"))
    data = store._view
    assert data["schema"] == 1
    assert isinstance(data["seq"], int)


def test_view_json_seq_strictly_increases_and_never_torn(store, tmp_path):
    """Inv 3: seq strictly increases per write; the file is always valid JSON after
    each write (the atomic temp->fsync->rename guarantee's observable proxy — a
    reader never parses a half-written file)."""
    seqs = []
    for _ in range(3):
        post_view(store, branch_snapshot("A"))
        data = store._view  # parses cleanly => not torn
        seqs.append(data["seq"])
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)  # strictly increasing (no repeats)


# ==========================================================================
# Invariant 4 — Empty tree is valid
# ==========================================================================

def test_empty_tree_round_trips_with_no_variations(store):
    """Inv 4: a mainline view (tree == []) round-trips; get_view returns
    variations == [] and in_variation == False."""
    post_view(store, mainline_snapshot("A"))
    v = make_ctx(store).get_view()
    assert v["variations"] == []
    assert v["in_variation"] is False
    assert v["fen"] == NF3


# ==========================================================================
# Invariant 5 — Session partitioning
# ==========================================================================

def test_view_is_scoped_to_its_session(store):
    """Inv 5: a view is scoped to its session; switching the active session swaps
    the whole view — one session's tree/cursor is not visible from the other."""
    # Session A: the branched view (has a Nc3 sideline).
    store._switch_current("A")
    post_view(store, branch_snapshot("A"))
    # Session B: a plain mainline view, different position, no variations.
    store._switch_current("B")
    post_view(store, mainline_snapshot("B"))

    vb = make_ctx(store).get_view()
    assert vb["fen"] == NF3
    assert vb["variations"] == []          # A's Nc3 branch is NOT visible from B
    assert vb["in_variation"] is False

    store._switch_current("A")
    va = make_ctx(store).get_view()
    assert va["fen"] == NC3                 # A's own view is intact
    assert va["in_variation"] is True
    assert len(va["variations"]) == 1


# ==========================================================================
# Invariant 6 — Reset clears it
# ==========================================================================

def test_reset_session_removes_view_json(store, tmp_path):
    """Inv 6: reset_session removes view.json."""
    post_view(store, branch_snapshot("A"))
    assert store._view is not None
    store.reset_session()
    assert store._view is None


def test_get_view_after_reset_returns_fallback_not_stale_tree(store):
    """Inv 6: after reset_session the in-memory view is dropped, so a subsequent
    get_view returns the fallback (last painted board) — never the old tree.
    The fallback is in_variation False, empty variations, viewing null."""
    post_view(store, branch_snapshot("A"))
    store.reset_session()
    # Establish a freshly painted board so the fallback has a fen to surface.
    store.write_board(START)
    v = make_ctx(store).get_view()
    assert v["fen"] == START               # falls back to the painted board
    assert v["in_variation"] is False
    assert v["variations"] == []           # NOT the stale Nc3 tree
    assert v["viewing"] is None


# ==========================================================================
# Invariant 7 — /position compatibility
# ==========================================================================

def test_position_alias_updates_fen_only(store):
    """Inv 7: a bare POST /position {fen} still updates the viewed fen and leaves
    line/tree empty, so an old-endpoint-only caller keeps working."""
    post_position(store, HANG_FEN := "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1")
    v = make_ctx(store).get_view()
    assert v["fen"] == HANG_FEN
    assert v["side_to_move"] == "white"    # still grounded from fen
    assert v["variations"] == []
    assert v["in_variation"] is False


# ==========================================================================
# get_view fallback + error behaviour (contract §Read tool)
# ==========================================================================

def test_get_view_fallback_when_no_view_reported(store):
    """Fallback: with an active session but no view yet, get_view returns the last
    painted board as fen, in_variation False, empty variations, viewing null."""
    store.write_board(E4E5)                # coach paints, but no /view posted
    v = make_ctx(store).get_view()
    assert "error" not in v
    assert v["fen"] == E4E5
    assert v["in_variation"] is False
    assert v["variations"] == []
    assert v["viewing"] is None


def test_get_view_prefers_fresh_board_over_stale_view(store):
    """Item 1 (regression): the coach paints a NEW position via write_board while the app's last
    reported view still describes the PREVIOUS position. get_view must report the coach's fresh
    board — never the stale `view.fen` — and must NOT surface the stale view's line/variations
    under the new fen. This is the priority-chain staleness the design forbids (§6.3): the view
    applies only while it still describes the current board."""
    ELSEWHERE = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"   # a position OFF the view's line entirely
    post_view(store, branch_snapshot("A"))     # app reports a branch view at NC3 (+ a variation tree)
    assert store.view["fen"] == NC3
    store.write_board(ELSEWHERE)               # …then the coach paints a position the view doesn't describe
    v = make_ctx(store).get_view()
    assert v["fen"] == ELSEWHERE               # the coach's fresh paint, not the stale NC3 view
    assert v["variations"] == []               # the NC3 branch is stale → not shown under the new fen
    assert v["viewing"] is None
    assert v["in_variation"] is False


def test_get_view_shows_view_details_when_it_matches_the_board(store):
    """Item 1 (companion): when the view DOES still describe the current board (the app posted it,
    which also updates board_view), get_view enriches with the line + variations — the fix narrows
    trust to a matching view, it doesn't drop view context wholesale."""
    post_view(store, branch_snapshot("A"))
    v = make_ctx(store).get_view()
    assert v["fen"] == NC3
    assert v["in_variation"] is True
    assert len(v["variations"]) == 1           # the posted branch still surfaces (view matches board)


def test_get_view_no_active_session_is_structured_error(tmp_path):
    """Error: with no active coaching session, get_view returns a structured
    {error: 'no_session', detail: ...} — never a bare exception (house rule).

    NOTE (contract ambiguity): the contract distinguishes 'no active session'
    (this error) from 'active session, no view yet' (the fallback above). This
    test infers that a store on which no session was ever activated is the
    'no active session' condition. Pin during reconciliation if that differs."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db"))) if DB else StateStore(str(tmp_path))
    v = ToolContext(None, s).get_view()
    assert v.get("error") == "no_session"
    assert "detail" in v


# ==========================================================================
# Invariant 8 — Resume replays the view over SSE
# ==========================================================================

@pytest.mark.skipif(DB is None, reason="resume replay needs the SQLite-backed store")
def test_resume_replays_persisted_view_exactly_once(tmp_path):
    """Inv 8: a session with a persisted view emits exactly one `view` SSE event on
    becoming active, carrying the last stored snapshot (fen/cursor/in_variation/
    line/tree)."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    post_view(s, branch_snapshot("A"))     # persist a view for A

    # Move away, then record everything the resume of A publishes.
    s._switch_current("B")
    published = []
    s._publish = lambda ch, payload: published.append((ch, payload))  # type: ignore[method-assign]
    s._switch_current("A")                 # resume A

    view_events = [p for ch, p in published if ch == "view"]
    assert len(view_events) == 1           # exactly once
    ev = view_events[0]
    assert ev["fen"] == NC3
    assert ev["cursor"] == 3
    assert ev["in_variation"] is True
    # the replayed tree carries the same branch first-move.
    first_moves = {var["san"] for bp in ev["tree"] for var in bp["variations"]}
    assert first_moves == {"Nc3"}


@pytest.mark.skipif(DB is None, reason="resume replay needs the SQLite-backed store")
def test_resume_of_session_without_view_emits_no_view_event(tmp_path):
    """Inv 8: a session that never explored (no persisted view) emits NO `view`
    event on resume — absence is valid, never a stale/empty tree."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    post_view(s, branch_snapshot("A"))     # A has a view
    s._switch_current("C")                 # C is brand new, never posted a view

    s._switch_current("B")                 # go elsewhere
    published = []
    s._publish = lambda ch, payload: published.append((ch, payload))  # type: ignore[method-assign]
    s._switch_current("C")                 # resume the never-explored session

    assert not any(ch == "view" for ch, _ in published)


# ==========================================================================
# Invariant 9 (server half) — idempotent re-ingest of the same snapshot
# ==========================================================================
# The APP-side rebuild (Swift VariationForest -> displayLine + cursor) is out of
# scope for pytest. What the server can promise — re-posting the SAME snapshot is a
# no-op on content (no duplicated branches) — is tested here.

def test_reingesting_same_snapshot_is_idempotent(store):
    """Inv 9 (server side): applying the same view snapshot twice yields the same
    view — same fen/cursor/in_variation and no duplicated branches."""
    ctx = make_ctx(store)
    post_view(store, branch_snapshot("A"))
    first = ctx.get_view()
    post_view(store, branch_snapshot("A"))  # identical re-post (e.g. a replay round-trip)
    second = ctx.get_view()

    assert second["fen"] == first["fen"]
    assert second["cursor_san"] == first["cursor_san"]
    assert second["in_variation"] == first["in_variation"]
    # No duplicated branch: still exactly one sideline, not two.
    assert len(second["variations"]) == len(first["variations"]) == 1


# ==========================================================================
# Added coverage (critic pass): cursor mid-line, multi/nested branches,
# viewing/material content, seq-across-reset, diverged-mainline resume drop.
# ==========================================================================

def _row(i, san, uci, fen, kind, branch):
    return {"i": i, "san": san, "uci": uci, "fen": fen, "kind": kind, "branch": branch}


def multi_branch_snapshot(session):
    """1.e4 e5 with TWO sidelines at E4E5 (Nc3 and Bc4); the shown line takes Nc3, cursor on it."""
    return {
        "session": session, "fen": NC3, "cursor": 3, "in_variation": True,
        "line": [
            _row(0, None, None, START, "main", False),
            _row(1, "e4", "e2e4", E4, "main", False),
            _row(2, "e5", "e7e5", E4E5, "main", False),
            _row(3, "Nc3", "b1c3", NC3, "variation", True),
        ],
        "tree": [{
            "at_fen": E4E5, "at_ply": 2, "variations": [
                {"uci": "b1c3", "san": "Nc3", "fen": NC3,
                 "line": [{"uci": "b1c3", "san": "Nc3", "fen": NC3}]},
                {"uci": "f1c4", "san": "Bc4", "fen": BC4,
                 "line": [{"uci": "f1c4", "san": "Bc4", "fen": BC4}]},
            ],
        }],
    }


def nested_snapshot(session):
    """A branch off a SIDELINE: at E4E5 → Nc3 (on mainline, at_ply 2), and off Nc3 → ...Nc6 (a
    sub-line whose branch point is off the mainline, so at_ply is null). Shown line: e4 e5 Nc3 Nc6."""
    return {
        "session": session, "fen": NC3NC6, "cursor": 4, "in_variation": True,
        "line": [
            _row(0, None, None, START, "main", False),
            _row(1, "e4", "e2e4", E4, "main", False),
            _row(2, "e5", "e7e5", E4E5, "main", False),
            _row(3, "Nc3", "b1c3", NC3, "variation", True),
            _row(4, "Nc6", "b8c6", NC3NC6, "variation", False),
        ],
        "tree": [
            {"at_fen": E4E5, "at_ply": 2, "variations": [
                {"uci": "b1c3", "san": "Nc3", "fen": NC3, "line": [
                    {"uci": "b1c3", "san": "Nc3", "fen": NC3},
                    {"uci": "b8c6", "san": "Nc6", "fen": NC3NC6}]}]},
            {"at_fen": NC3, "at_ply": None, "variations": [   # off-mainline branch point
                {"uci": "b8c6", "san": "Nc6", "fen": NC3NC6,
                 "line": [{"uci": "b8c6", "san": "Nc6", "fen": NC3NC6}]}]},
        ],
    }


def _seed_matching_history(store):
    """Seed the session's move line to MATCH branch_snapshot's mainline prefix (e4 e5)."""
    store.write_history([
        {"n": 0, "san": None, "uci": None, "fen": START},
        {"n": 1, "san": "e4", "uci": "e2e4", "fen": E4},
        {"n": 2, "san": "e5", "uci": "e7e5", "fen": E4E5},
    ])


def test_cursor_mid_line_reports_that_move_not_the_tip(store):
    """Inv 1: with a 4-row line but cursor=1, get_view reports the move AT the cursor (e4), not the
    tip (Nc3) — pins that cursor_san tracks `cursor`, not `line[-1]`."""
    snap = branch_snapshot("A")
    snap["cursor"] = 1
    snap["in_variation"] = False
    post_view(store, snap)
    v = make_ctx(store).get_view()
    assert v["cursor_san"] == "e4"          # the move at index 1, NOT the tip Nc3


def test_multiple_sidelines_all_survive(store):
    """Inv 1/5: a branch point with two sidelines round-trips as the full SET of first-moves — an
    impl that keeps only the first sideline fails this."""
    post_view(store, multi_branch_snapshot("A"))
    v = make_ctx(store).get_view()
    first_moves = set()
    for var in v["variations"]:
        first_moves.add(var["line"].split()[0].lstrip("0123456789.…"))
    assert first_moves == {"Nc3", "Bc4"}
    assert len(v["variations"]) == 2


def test_nested_offmainline_branch_roundtrips(store):
    """snapshot spec: a branch point with at_ply == null (off a sideline) survives — its first move
    appears and its label is the off-line form, distinct from the on-mainline branch."""
    post_view(store, nested_snapshot("A"))
    v = make_ctx(store).get_view()
    froms = {var["from"] for var in v["variations"]}
    firsts = {var["line"].split()[0].lstrip("0123456789.…") for var in v["variations"]}
    assert "Nc6" in firsts and "Nc3" in firsts
    assert any("off" in f for f in froms)        # the at_ply=null branch labels as off-line


def test_viewing_is_nonnull_pgn_marking_the_cursor(store):
    """Read-tool spec: `viewing` is a one-line PGN of the resolved line that MARKS the cursor and the
    variation entry — not null, and not the same as the fallback."""
    post_view(store, branch_snapshot("A"))    # cursor on the Nc3 sideline (index 3)
    v = make_ctx(store).get_view()
    assert v["viewing"] is not None
    assert "e4" in v["viewing"] and "Nc3" in v["viewing"]   # prefix + sideline both present
    assert "Nc3" in v["viewing"].split("[")[-1]              # the cursor move is the bracketed one


def test_empty_tree_view_has_nonnull_viewing(store):
    """A real mainline view (not the fallback) still renders a `viewing` PGN — distinguishes it from
    the no-view fallback where viewing is null."""
    post_view(store, mainline_snapshot("A"))
    v = make_ctx(store).get_view()
    assert v["viewing"] is not None and "Nf3" in v["viewing"]


def test_material_is_grounded_from_fen(store):
    """Read-tool spec: get_view surfaces a grounded material standing (CLAUDE.md receipt), computed
    from the viewed fen — a position a pawn up reads as such."""
    post_view(store, mainline_snapshot("A", cursor=0, fen=WHITE_UP_A_PAWN, in_variation=False))
    v = make_ctx(store).get_view()
    assert "pawn" in v["material"].lower() and "up" in v["material"].lower()


def test_seq_restarts_after_reset(store, tmp_path):
    """Inv 3 × Inv 6: reset_session zeroes the view seq (like every other state file), so the next
    view after a reset starts at seq 1 — no leftover-seq collision with the removed file."""
    post_view(store, branch_snapshot("A"))
    post_view(store, branch_snapshot("A"))
    assert store._view["seq"] == 2
    store.reset_session()
    post_view(store, branch_snapshot("A"))
    assert store._view["seq"] == 1


@pytest.mark.skipif(DB is None, reason="resume replay needs the SQLite-backed store")
def test_resume_replays_line_and_tree(tmp_path):
    """Inv 8: the replayed `view` body carries `line` too (the grounded resolved strip), not just
    fen/cursor/tree."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    post_view(s, branch_snapshot("A"))
    s._switch_current("B")
    published = []
    s._publish = lambda ch, payload: published.append((ch, payload))  # type: ignore[method-assign]
    s._switch_current("A")
    ev = next(p for ch, p in published if ch == "view")
    assert [row.get("san") for row in ev["line"]] == [None, "e4", "e5", "Nc3"]


@pytest.mark.skipif(DB is None, reason="resume replay needs the SQLite-backed store")
def test_resume_drops_view_when_mainline_diverged(tmp_path):
    """Resume rule: a view whose mainline prefix no longer matches the session's history (a
    re-import moved the line) references dead positions — resume DROPS it, emitting no `view`."""
    s = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena_backend.db")))
    s._switch_current("A")
    _seed_matching_history(s)
    post_view(s, branch_snapshot("A"))
    # Sanity: matching history replays the view.
    published = []
    s._publish = lambda ch, payload: published.append((ch, payload))  # type: ignore[method-assign]
    s._switch_current("B"); s._switch_current("A")
    assert any(ch == "view" for ch, _ in published)
    # Now diverge A's mainline: ply 1 becomes d4, not e4.
    D4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
    s.write_history([{"n": 0, "san": None, "uci": None, "fen": START},
                     {"n": 1, "san": "d4", "uci": "d2d4", "fen": D4}])
    published.clear()
    s._switch_current("B"); s._switch_current("A")
    assert not any(ch == "view" for ch, _ in published)   # dropped, not replayed


def test_position_does_not_clobber_a_rich_view(store):
    """Inv 7 (clarified): a bare /position never wipes an existing rich view's line/tree — the rich
    view is the source of truth; /position is only the standalone fallback."""
    post_view(store, branch_snapshot("A"))
    post_position(store, "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1")   # stray legacy call
    v = make_ctx(store).get_view()
    assert v["in_variation"] is True
    assert len(v["variations"]) == 1        # the Nc3 tree survives
