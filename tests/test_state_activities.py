"""Characterization tests for the StateStore beats + activity-stack behavior — the safety net for the
'beats become per-activity' refactor. These pin the CURRENT contract so the change can be made without
silently breaking the untested core (session document). db=None → in-memory (no Postgres needed) for the
behavioural tests; a persist round-trip variant uses a real DB.

NOTE: `test_push_keeps_beats_continuous_TODAY` documents the invariant we are about to INVERT (a pushed
activity currently SHARES the conversation). When beats go per-activity, that test flips to
'push forks beats' — it is the intended behaviour change, not a regression.
"""
from __future__ import annotations

from lucena_backend.persistence.state import StateStore


def _beat(text):
    return {"kind": "say", "tone": "teach", "stops": False, "segments": [{"text": text}]}


def _store(tmp_path):
    return StateStore(str(tmp_path))          # db=None → in-memory


# -- beats -----------------------------------------------------------------------------------------

def test_append_beats_assigns_sequential_indices(tmp_path):
    s = _store(tmp_path)
    assert s.append_beats([_beat("a"), _beat("b")]) == [0, 1]
    assert s.append_beats([_beat("c")]) == [2]
    texts = [b["segments"][0]["text"] for b in s.dump()["beats"]]
    assert texts == ["a", "b", "c"]


def test_append_beats_stamps_board_seq_and_index(tmp_path):
    s = _store(tmp_path)
    s.write_board("8/8/8/8/8/8/8/K6k w - - 0 1")
    s.append_beats([_beat("x")])
    b = s.dump()["beats"][0]
    assert b["i"] == 0 and b["board_seq"] >= 1 and "ts" in b


# -- board / workspace -----------------------------------------------------------------------------

def test_write_board_and_board_view(tmp_path):
    s = _store(tmp_path)
    fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    s.write_board(fen)
    assert s.board_view == fen


# -- activity stack --------------------------------------------------------------------------------

def test_base_stack_is_one_activity(tmp_path):
    s = _store(tmp_path)
    assert len(s.dump()["activities"]) == 1
    assert s.dump()["activities"][0]["kind"] == "conversation"


def test_push_and_pop_activity_forks_and_restores_the_workspace(tmp_path):
    s = _store(tmp_path)
    base_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    s.write_board(base_fen)
    s.push_activity("puzzle", seed={"last_board": {"fen": "8/8/8/8/8/8/8/K6k b - - 0 1"}})
    assert len(s.dump()["activities"]) == 2
    assert s.dump()["activities"][-1]["kind"] == "puzzle"
    # workspace forked: the pushed frame is its own board, the base is frozen underneath
    s.write_board("8/8/8/8/8/8/8/7k w - - 0 1")
    assert s.board_view == "8/8/8/8/8/8/8/7k w - - 0 1"
    assert s.pop_activity() is True
    assert len(s.dump()["activities"]) == 1
    assert s.board_view == base_fen            # base workspace restored intact

    assert s.pop_activity() is False           # cannot pop past the base


def test_push_forks_beats_and_pop_restores_the_parent_conversation(tmp_path):
    # NEW per-activity-beats contract: a pushed activity owns its OWN conversation — it starts EMPTY,
    # beats added inside it do not leak to the parent, and pop restores the parent's beats intact. (This
    # is what makes a puzzle activity its own saved conversation.)
    s = _store(tmp_path)
    s.append_beats([_beat("base-1")])
    s.push_activity("puzzle")
    assert s.dump()["beats"] == []                                   # the puzzle's own empty stream
    s.append_beats([_beat("in-puzzle")])
    assert [b["segments"][0]["text"] for b in s.dump()["beats"]] == ["in-puzzle"]   # only the puzzle's
    s.pop_activity()
    assert [b["segments"][0]["text"] for b in s.dump()["beats"]] == ["base-1"]      # parent restored, no leak


def test_finish_activity_cards_the_base_and_switches_home(tmp_path):
    # Leaving a puzzle drops a clickable card into the BASE conversation (referencing the saved frame)
    # and switches the view back to the base — the puzzle frame is preserved for reopening.
    s = _store(tmp_path)
    s.append_beats([_beat("base-1")])
    idx = s.push_activity("puzzle", title="Puzzle")
    s.append_beats([_beat("in-puzzle")])
    assert s.finish_activity() is True
    assert s.active_idx == 0                                          # view is home
    d = s.dump()
    assert [b.get("kind") for b in d["beats"]][-1] == "card"          # a card now sits in the base
    card = d["beats"][-1]
    assert card["activity_idx"] == idx and card["status"] == "attempted"
    # the puzzle frame (and its own beats) survived — it was not popped
    assert len(d["activities"]) == 2
    assert [b["segments"][0]["text"] for b in d["activities"][idx]["beats"]] == ["in-puzzle"]


def test_open_activity_reopens_a_saved_puzzle_and_zero_returns_home(tmp_path):
    s = _store(tmp_path)
    s.write_board("8/8/8/8/8/8/8/K6k w - - 0 1")
    idx = s.push_activity("puzzle", title="Puzzle")
    s.write_board("8/8/8/8/8/8/8/7k w - - 0 1")
    s.append_beats([_beat("puzzle-beat")])
    s.finish_activity()
    assert s.board_view == "8/8/8/8/8/8/8/K6k w - - 0 1"             # base board
    assert s.open_activity(idx) is True                              # reopen the puzzle
    assert s.active_idx == idx
    assert s.board_view == "8/8/8/8/8/8/8/7k w - - 0 1"             # puzzle board restored
    assert [b["segments"][0]["text"] for b in s.dump()["beats"]] == ["puzzle-beat"]
    assert s.open_activity(9) is False                              # out of range
    assert s.open_activity(0) is True and s.active_idx == 0         # back home


def test_solved_stamp_then_finish_card_reads_solved(tmp_path):
    s = _store(tmp_path)
    idx = s.push_activity("puzzle", title="Puzzle")
    s.mark_activity_solved()
    assert s.finish_activity() is True
    card = s.dump()["beats"][-1]
    assert card["kind"] == "card" and card["status"] == "solved"    # solve preserved, not downgraded


def test_finish_activity_is_idempotent_no_duplicate_card(tmp_path):
    s = _store(tmp_path)
    idx = s.push_activity("puzzle", title="Puzzle")
    s.finish_activity()
    s.open_activity(idx)
    s.finish_activity()                                             # leave again
    cards = [b for b in s.dump()["beats"] if b.get("kind") == "card"]
    assert len(cards) == 1                                          # still ONE card for this activity


def test_active_idx_and_card_survive_a_reload(tmp_path):
    from lucena_backend.persistence.db import DB
    sid = "sess-act"
    s1 = StateStore(str(tmp_path), db=DB(str(tmp_path / "db")))
    with s1.bound(sid):
        s1.append_beats([_beat("base")])
        idx = s1.push_activity("puzzle", title="Puzzle")
        s1.append_beats([_beat("inside")])
        s1.mark_activity_solved()
        s1.finish_activity()
        s1.open_activity(idx)                                       # leave the view ON the puzzle
    s2 = StateStore(str(tmp_path), db=DB(str(tmp_path / "db")))
    with s2.bound(sid):
        d = s2.dump()
        assert s2.active_idx == idx                                 # pointer restored
        assert [b["segments"][0]["text"] for b in d["beats"]] == ["inside"]   # puzzle's own stream
        base = d["activities"][0]
        card = [b for b in base["beats"] if b.get("kind") == "card"][0]
        assert card["activity_idx"] == idx and card["status"] == "solved"
        assert d["activities"][idx]["status"] == "solved"


def test_variations_are_recorded_per_activity_and_restore_on_reopen(tmp_path):
    # The app authors variations into the view (line/tree/cursor); the view lives on the ACTIVE frame's
    # workspace, so a puzzle's variations belong to the puzzle activity and replay when it's reopened —
    # the base conversation never inherits them.
    s = _store(tmp_path)
    fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    idx = s.push_activity("puzzle", title="Puzzle")
    s.set_view({"fen": fen, "cursor": 0, "line": [],
                "tree": [{"i": 0, "san": "Ka2", "kind": "variation"}]})
    assert (s.view or {}).get("tree")                       # variations recorded on the puzzle frame
    s.finish_activity()
    assert not (s.view or {}).get("tree")                   # the base conversation has none of them
    s.open_activity(idx)
    assert s.view["tree"][0]["san"] == "Ka2"                # the puzzle's variations came back


def test_set_base_activity_replaces_the_stack(tmp_path):
    s = _store(tmp_path)
    s.push_activity("puzzle")
    assert len(s.dump()["activities"]) == 2
    s.set_base_activity("conversation")
    assert len(s.dump()["activities"]) == 1
    assert s.dump()["activities"][0]["kind"] == "conversation"


# -- persist round-trip (needs a DB) ---------------------------------------------------------------

def test_beats_and_activities_survive_a_reload(tmp_path):
    from lucena_backend.persistence.db import DB
    sid = "sess-1"
    s1 = StateStore(str(tmp_path), db=DB(str(tmp_path / "db")))
    with s1.bound(sid):
        s1.write_board("8/8/8/8/8/8/8/K6k w - - 0 1")
        s1.append_beats([_beat("hello"), _beat("world")])   # each write persists via _persist_view
    # a fresh store over the SAME db → reloads the document + beats
    s2 = StateStore(str(tmp_path), db=DB(str(tmp_path / "db")))
    with s2.bound(sid):
        d = s2.dump()
        assert [b["segments"][0]["text"] for b in d["beats"]] == ["hello", "world"]
        assert s2.board_view == "8/8/8/8/8/8/8/K6k w - - 0 1"
