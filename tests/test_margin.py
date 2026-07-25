"""The /margin content builder — JSON-inspection mode (owner, 2026-07-23):
the margin shows the plans layer's raw pre/post-verify JSON. The former
card builder (epigraph/theory/position cards) lives at backend fb4b2d7."""
import time

import pytest

from lucena_backend import margin
from lucena_backend.margin import build

@pytest.fixture(autouse=True)
def _isolated_margin(monkeypatch):
    """`margin` keeps its plumbing in MODULE GLOBALS that `build_app` writes (pool, publish, the
    position cache). Any test that builds an app therefore configures this module for every test
    that follows, which made these order-dependent: a leaked position cache answered a position one
    of these tests expected to see ROLLED. Each test starts from an unconfigured module."""
    monkeypatch.setattr(margin, "_pool", None)
    monkeypatch.setattr(margin, "_maia", None)
    monkeypatch.setattr(margin, "_publish", None)
    monkeypatch.setattr(margin, "_positions", None)
    monkeypatch.setattr(margin, "_deep_cache", {})
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin, "_latest_by_session", {})


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUT_OF_BOOK = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP2PPP/R2Q1RK1 w - - 4 9"


def test_unconfigured_margin_is_bare_and_never_pends(monkeypatch):
    # `_pool` is set EXPLICITLY here (2026-07-26): with the live/scrub gate
    # gone, a configured pool is the only thing standing between a request
    # and a roll, so this test has to state the unconfigured premise itself
    # rather than lean on another test not having configured the module.
    monkeypatch.setattr(margin, "_pool", None)
    m = build(OUT_OF_BOOK)
    assert m["raw"] is None and m["plansPending"] is False


def test_cached_result_is_served_raw():
    key = " ".join(OUT_OF_BOOK.split()[:4])
    # _cache takes the SHEET DICT now (it does its own pretty-printing)
    margin._cache(key, {"schema": "lucena-plans/sheet@1"}, "POST-VERIFY", False)
    try:
        m = build(OUT_OF_BOOK)
        assert m["statusLine"] == "POST-VERIFY"
        assert '"schema"' in m["raw"]
        assert m["plansPending"] is False
    finally:
        margin._deep_cache.clear()


def test_pending_pre_keeps_polling_alive():
    # OUT_OF_BOOK, not START: the start position is stage 1 (the epigraph)
    # and never reaches the deep cache — only the out-of-book stage polls.
    key = " ".join(OUT_OF_BOOK.split()[:4])
    margin._cache(key, {}, "PRE-VERIFY · VERIFYING…", True)
    try:
        m = build(OUT_OF_BOOK)
        assert m["plansPending"] is True and m["raw"] == "{}" and m["sheet"] == {}
    finally:
        margin._deep_cache.clear()


def test_bad_fen_raises():
    with pytest.raises(ValueError):
        build("not a fen")


# -- the authored-content stages (/content wired 2026-07-24) ------------------

SICILIAN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"


def test_move_one_serves_a_sourced_epigraph():
    m = build(START, seed="session-1")
    e = m["epigraph"]
    assert e and e["quote"] and e["author"]
    assert m["theory"] is None and m["plansPending"] is False


def test_epigraph_is_the_move_zero_cover_only():
    """The epigraph is the MOVE-0 cover only: deterministic per session seed,
    and GONE the instant a move is played (owner: 'when I made a move it
    showed the quote again' — the cover must end at ply 0)."""
    a = build(START, seed="session-1")["epigraph"]
    assert a and build(START, seed="session-1")["epigraph"] == a   # stable per seed
    assert build(START, seed="session-2")["epigraph"] != a          # varies by seed
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert build(after_e4, seed="session-1")["epigraph"] is None     # cover ended


def test_in_book_prefers_wikibooks_over_authored():
    """WIKIBOOKS FIRST (owner): even where we HAVE an authored annotation, an
    attributed Wikibooks entry is shown instead — with its CC BY-SA credit —
    and never the authored prose. The opening name still leads the masthead
    and the authored doors still show continuations."""
    m = build(SICILIAN, seed="s")
    assert m["masthead"] == "Sicilian Defense"          # name wins the masthead
    assert m["statusLine"] == "OPENING · MOVE 2"
    idea = m["theory"]["idea"]
    assert idea and "Sicilian" in idea
    attr = m["theory"]["attribution"]                    # Wikibooks, attributed
    assert attr and "Wikibooks" in attr["text"]
    assert attr["url"].startswith("https://en.wikibooks.org/")
    from lucena_core import content as authored
    assert idea != authored.annotation_for("Sicilian Defense")   # NOT our prose
    assert m["theory"]["doors"] == []                            # continuations dropped
    assert m["epigraph"] is None and m["plansPending"] is False


def test_authored_used_only_when_no_wikibooks(monkeypatch):
    """Authored annotation is the FALLBACK — used only where Wikibooks has no
    entry — and carries no attribution (it is our own prose)."""
    monkeypatch.setattr(margin.theory, "theory_for", lambda fen: None)
    m = build(SICILIAN, seed="s")
    assert m["masthead"] == "Sicilian Defense"
    idea = m["theory"]["idea"]
    from lucena_core import content as authored
    full = authored.annotation_for("Sicilian Defense")
    assert idea and full.startswith(idea[:40])           # the authored lead
    assert m["theory"]["attribution"] is None            # our prose, no credit


# -- the Wikibooks theory gate (2026-07-24) -----------------------------------

_WB_ENTRY = {
    "name": "Some Opening", "eco": "A00",
    "description": "A verbatim Wikibooks description of this position, long "
                   "enough to be a real lead paragraph and not a stub remnant.",
    "responses": ["2. Nf3 - Main line"],
    "source_url": "https://en.wikibooks.org/wiki/Chess_Opening_Theory/x",
}


def test_wikibooks_entry_gates_the_sheet(monkeypatch):
    """A position with a Wikibooks entry shows its THEORY (verbatim + CC BY-SA
    attribution) and MUST NOT roll the positional sheet — no deep job, no
    pending. (owner: 'don't show all the positions if it has a wikibooks
    entry')."""
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: _WB_ENTRY if fen == OUT_OF_BOOK else None)
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())          # "configured"
    monkeypatch.setattr(margin._worker, "submit",
                        lambda *a, **k: submitted.append(a))
    m = build(OUT_OF_BOOK)
    assert submitted == []                                  # never rolled
    assert m["plansPending"] is False and m["sheet"] is None
    assert m["masthead"] == "Some Opening"
    assert m["theory"]["idea"] and _WB_ENTRY["description"].startswith(
        m["theory"]["idea"][:40])
    attr = m["theory"]["attribution"]
    assert attr and attr["url"] == _WB_ENTRY["source_url"]
    assert "CC BY-SA" in attr["text"]


def test_no_wikibooks_entry_still_rolls(monkeypatch):
    """The gate is scoped to Wikibooks positions: an out-of-book position
    WITHOUT an entry must still roll the sheet exactly as before."""
    monkeypatch.setattr(margin.theory, "theory_for", lambda fen: None)
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin._worker, "submit",
                        lambda *a, **k: submitted.append(a))
    m = build(OUT_OF_BOOK)
    assert m["plansPending"] is True                        # rolled as before
    assert len(submitted) == 1


def test_named_without_annotation_falls_back_to_wikibooks(monkeypatch):
    """A position with an opening NAME but no authored annotation must still
    use the Wikibooks description + attribution — authored-first, Wikibooks-
    otherwise, independent of whether a name exists."""
    monkeypatch.setattr(margin.openings, "name_for", lambda fen: "Some Named Line")
    monkeypatch.setattr(margin.authored, "annotation_for", lambda name: None)
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: _WB_ENTRY if fen == OUT_OF_BOOK else None)
    m = build(OUT_OF_BOOK, seed="s")
    assert m["masthead"] == "Some Named Line"          # name wins the masthead
    assert m["theory"]["idea"] == _WB_ENTRY["description"]   # VERBATIM, full
    attr = m["theory"]["attribution"]
    assert attr and attr["url"] == _WB_ENTRY["source_url"]


def test_wikibooks_idea_is_verbatim_not_truncated(monkeypatch):
    """The Wikibooks lead is shown as-is (CC BY-SA 'as-is' contract) — never
    passed through _lead_sentences."""
    long_desc = ("First sentence of theory. Second sentence adds nuance. "
                 "Third sentence closes it out.")
    entry = {**_WB_ENTRY, "description": long_desc}
    monkeypatch.setattr(margin.openings, "name_for", lambda fen: None)
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: entry if fen == OUT_OF_BOOK else None)
    m = build(OUT_OF_BOOK, seed="s")
    assert m["theory"]["idea"] == long_desc            # full, untruncated


def test_unattributable_wikibooks_is_not_theory_and_still_rolls(monkeypatch):
    """A Wikibooks entry WITHOUT source_url can't be shown (CC BY-SA needs the
    credit), so it is NOT treated as theory: no empty card, and the positional
    sheet still rolls."""
    no_src = {"name": "X", "description": "text", "responses": []}   # no source_url
    monkeypatch.setattr(margin.openings, "name_for", lambda fen: None)
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: no_src if fen == OUT_OF_BOOK else None)
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin._worker, "submit",
                        lambda *a, **k: submitted.append(a))
    m = build(OUT_OF_BOOK)
    assert m["theory"] is None                     # not gated behind an empty card
    assert m["plansPending"] is True and len(submitted) == 1   # rolls as normal


# -- the loading stream's terminal contract (Codex P2, 2026-07-25) ------------

def _capture_publish(events):
    return lambda payload, *, session_id: events.append((session_id, payload))


def test_deep_job_publishes_done_on_success(monkeypatch):
    import lucena_backend.plans.service as svc
    events = []
    margin.configure(pool=None, maia=None, publish=_capture_publish(events))
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (None, {"ok": True}))
    key = " ".join(OUT_OF_BOOK.split()[:4])
    # the tested position must BE this session's latest (2026-07-26):
    # _deep_job drops a job the user has navigated away from, so a direct
    # call has to state that premise or it silently exercises nothing.
    monkeypatch.setattr(margin, "_latest_by_session",
                        {"sess-1": (key, __import__("time").monotonic())})
    try:
        margin._deep_job(OUT_OF_BOOK, "sess-1")
        assert events[-1][1]["stage"] == "done" and events[-1][0] == "sess-1"
        assert margin._deep_cache[key]["pending"] is False
        assert margin._deep_cache[key]["sheet"] == {"ok": True}
    finally:
        margin._deep_cache.clear()
        margin.configure(pool=None, maia=None)


def test_deep_job_publishes_done_on_failure_too(monkeypatch):
    # the app's loading cycle must END against a terminal error, not spin
    import lucena_backend.plans.service as svc
    events = []
    margin.configure(pool=None, maia=None, publish=_capture_publish(events))
    def boom(fen, pool, maia, on_pre):
        raise RuntimeError("roll failed")
    monkeypatch.setattr(svc, "sheet_json_staged", boom)
    key = " ".join(OUT_OF_BOOK.split()[:4])
    # the tested position must BE this session's latest (2026-07-26):
    # _deep_job drops a job the user has navigated away from, so a direct
    # call has to state that premise or it silently exercises nothing.
    monkeypatch.setattr(margin, "_latest_by_session",
                        {"sess-1": (key, __import__("time").monotonic())})
    try:
        margin._deep_job(OUT_OF_BOOK, "sess-1")
        assert events[-1][1]["stage"] == "done"
        assert margin._deep_cache[key]["pending"] is False
        assert "error" in margin._deep_cache[key]["sheet"]
    finally:
        margin._deep_cache.clear()
        margin.configure(pool=None, maia=None)


def test_raising_publisher_never_corrupts_a_successful_sheet(monkeypatch):
    import lucena_backend.plans.service as svc
    def bad_publish(payload, *, session_id):
        raise RuntimeError("socket gone")
    margin.configure(pool=None, maia=None, publish=bad_publish)
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (None, {"ok": True}))
    key = " ".join(OUT_OF_BOOK.split()[:4])
    # the tested position must BE this session's latest (2026-07-26):
    # _deep_job drops a job the user has navigated away from, so a direct
    # call has to state that premise or it silently exercises nothing.
    monkeypatch.setattr(margin, "_latest_by_session",
                        {"sess-1": (key, __import__("time").monotonic())})
    try:
        margin._deep_job(OUT_OF_BOOK, "sess-1")
        assert margin._deep_cache[key]["pending"] is False
        assert margin._deep_cache[key]["sheet"] == {"ok": True}   # not ERROR
    finally:
        margin._deep_cache.clear()
        margin.configure(pool=None, maia=None)


def test_preroll_streams_from_the_request_thread_not_the_roll_worker(monkeypatch):
    # the fast PRE phase must fire from build(), decoupled from the serialized
    # roll worker (owner 2026-07-25: stream stopped when rolls backed up)
    calls = []
    margin.configure(pool=object(), maia=None, publish=lambda p, *, session_id: None)
    monkeypatch.setattr(margin._worker, "submit", lambda *a, **k: None)
    monkeypatch.setattr(margin, "_stream_preroll",
                        lambda fen, sid: calls.append((fen, sid)))
    try:
        m = margin.build(OUT_OF_BOOK, seed="sess-1")
        assert m["plansPending"] is True
        assert calls == [(OUT_OF_BOOK, "sess-1")]      # streamed once, upstream
        margin.build(OUT_OF_BOOK, seed="sess-1")  # a poll retry
        assert len(calls) == 1                          # inflight -> no re-stream
    finally:
        margin._deep_cache.clear(); margin._inflight.clear()
        margin.configure(pool=None, maia=None)


def test_all_pre_events_precede_done(monkeypatch):
    # Codex P1: a fast/immediate worker must not publish `done` ahead of the
    # PRE stream — pre-roll is emitted BEFORE the roll is submitted.
    events = []
    margin.configure(pool=object(), maia=None,
                     publish=lambda p, *, session_id: events.append(p))

    def two_pre(fen, sid):
        for i in range(2):
            margin._publish({"fen": fen, "i": i, "stage": "pawns"}, session_id=sid)
    monkeypatch.setattr(margin, "_stream_preroll", two_pre)
    monkeypatch.setattr(margin._worker, "submit", lambda fn, *a: fn(*a))  # inline
    import lucena_backend.plans.service as svc
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (None, {"ok": True}))
    try:
        margin.build(OUT_OF_BOOK, seed="s1")
        assert [e.get("stage") for e in events] == ["pawns", "pawns", "done"]
    finally:
        margin._deep_cache.clear(); margin._inflight.clear()
        margin.configure(pool=None, maia=None)


def test_roll_submit_failure_clears_inflight_for_retry(monkeypatch):
    margin.configure(pool=object(), maia=None, publish=lambda p, *, session_id: None)
    monkeypatch.setattr(margin, "_stream_preroll", lambda fen, sid: None)
    def boom(*a, **k):
        raise RuntimeError("executor down")
    monkeypatch.setattr(margin._worker, "submit", boom)
    key = " ".join(OUT_OF_BOOK.split()[:4])
    try:
        margin.build(OUT_OF_BOOK, seed="s1")
        assert key not in margin._inflight        # cleared -> a later poll retries
    finally:
        margin._deep_cache.clear(); margin._inflight.clear()
        margin.configure(pool=None, maia=None)


IN_BOOK = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"  # 1.e4 e5


def test_stream_ignores_game_phase_streams_whenever_called():
    # owner 2026-07-25: "it should stream when not in book, not on the phase
    # of the game." _stream_preroll no longer gates on game_phase — an early
    # opening-phase position streams the same as a middlegame one.
    events = []
    margin.configure(pool=None, maia=None,
                     publish=lambda p, *, session_id: events.append(p))
    try:
        opening = "rnbqkb1r/pppp1ppp/5n2/4p3/2P5/6P1/PP1PPP1P/RNBQKBNR w KQkq - 0 3"
        margin._stream_preroll(opening, "s1")            # opening phase
        assert events and events[0]["stage"] == "pawns"  # ...still streams
    finally:
        margin.configure(pool=None, maia=None)


def test_in_book_position_does_not_stream():
    # the IN-BOOK gate lives in build(): a named opening returns the theory
    # card BEFORE reaching the deep-job submit, so no pre-roll fires.
    calls = []
    margin.configure(pool=object(), maia=None, publish=lambda p, *, session_id: None)
    monkeypatch_free = margin._stream_preroll
    margin._stream_preroll = lambda fen, sid: calls.append(fen)  # type: ignore
    try:
        m = build(IN_BOOK, seed="s1")
        assert m["theory"] is not None and calls == []   # theory card, no stream
    finally:
        margin._stream_preroll = monkeypatch_free
        margin._deep_cache.clear(); margin._inflight.clear()
        margin.configure(pool=None, maia=None)


# -- every position gets the cycle; latest-wins protects the worker ----------
# (owner 2026-07-26: "Variations must go through the same cycle as well.
# Nothing is happening when there is a variation.")

def test_a_variation_rolls_just_like_the_live_position(monkeypatch):
    """The deep pass used to be gated on a `live` flag, so a variation or a
    scrub sat on the instant layer forever. Every out-of-book position the
    margin is asked about now submits a roll."""
    variation = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP1QPPP/R4RK1 b - - 5 9"
    monkeypatch.setattr(margin.theory, "theory_for", lambda fen: None)
    submitted = []
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin, "_stream_preroll", lambda fen, sid: None)
    monkeypatch.setattr(margin._worker, "submit",
                        lambda fn, fen, sid: submitted.append(fen))
    try:
        assert build(OUT_OF_BOOK)["plansPending"] is True
        assert build(variation)["plansPending"] is True
        assert submitted == [OUT_OF_BOOK, variation]
    finally:
        margin._deep_cache.clear()


def test_a_stale_job_drops_instead_of_rolling(monkeypatch):
    """LATEST-WINS: scrubbing past a position must not spend 5s of engine on
    it. The dropped job clears its inflight mark, so coming back re-submits
    (the app polls every 1.5-2.5s)."""
    rolled = []
    import lucena_backend.plans.service as svc
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (rolled.append(fen),
                                                         {"ok": True})[1])
    stale_key = " ".join(OUT_OF_BOOK.split()[:4])
    monkeypatch.setattr(margin, "_inflight", {stale_key})
    monkeypatch.setattr(margin, "_latest_by_session",
                        {"s1": ("somewhere else entirely", time.monotonic())})
    try:
        margin._deep_job(OUT_OF_BOOK, "s1")
        assert rolled == []                          # never rolled
        assert stale_key not in margin._inflight     # ...and retryable
        assert margin._deep_cache == {}              # nothing cached either
        # the same job IS rolled once it is that session's latest again
        margin._latest_by_session["s1"] = (stale_key, time.monotonic())
        margin._deep_job(OUT_OF_BOOK, "s1")
        assert rolled == [OUT_OF_BOOK]
    finally:
        margin._deep_cache.clear()


def test_one_session_never_cancels_another_session_s_roll(monkeypatch):
    """Latest-wins is per SESSION (Codex): two windows navigating at once must
    not starve each other. Session A's queued job still rolls after session B
    has asked for a different position."""
    other = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP1QPPP/R4RK1 b - - 5 9"
    rolled = []
    import lucena_backend.plans.service as svc
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (rolled.append(fen),
                                                         {"ok": True})[1])
    monkeypatch.setattr(margin.theory, "theory_for", lambda fen: None)
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin, "_latest_by_session", {})
    monkeypatch.setattr(margin, "_stream_preroll", lambda fen, sid: None)
    monkeypatch.setattr(margin._worker, "submit", lambda *a, **k: None)  # queue it
    try:
        margin.build(OUT_OF_BOOK, seed="A")      # session A asks for A's fen
        margin.build(other, seed="B")            # session B moves on elsewhere
        margin._deep_job(OUT_OF_BOOK, "A")       # A's queued job finally runs
        assert rolled == [OUT_OF_BOOK]           # ...and still rolls
    finally:
        margin._deep_cache.clear()


def test_a_second_window_on_the_same_position_keeps_the_roll_alive(monkeypatch):
    """The roll belongs to the POSITION, not to whoever submitted it (Codex):
    A and B both sit on X, A navigates away, and X must still roll for B —
    without waiting for B's next poll."""
    other = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP1QPPP/R4RK1 b - - 5 9"
    key = " ".join(OUT_OF_BOOK.split()[:4])
    rolled = []
    import lucena_backend.plans.service as svc
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (rolled.append(fen),
                                                         {"ok": True})[1])
    # A and B are both on X; then A moves on to `other`. B's latest is still X.
    monkeypatch.setattr(margin, "_inflight", {key})
    monkeypatch.setattr(margin, "_latest_by_session",
                        {"A": (" ".join(other.split()[:4]), time.monotonic()),
                         "B": (key, time.monotonic())})
    try:
        margin._deep_job(OUT_OF_BOOK, "A")       # submitted by A, wanted by B
        assert rolled == [OUT_OF_BOOK]
    finally:
        margin._deep_cache.clear()


def test_navigating_to_an_in_book_position_also_drops_the_queued_job(monkeypatch):
    """Latest-wins is recorded on EVERY request (Codex): walking from an
    out-of-book position into theory (or into an already-cached position) is
    still walking away, so the queued roll for the old position must drop."""
    key = " ".join(OUT_OF_BOOK.split()[:4])
    rolled = []
    import lucena_backend.plans.service as svc
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (rolled.append(fen),
                                                         {"ok": True})[1])
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin, "_latest_by_session", {})
    monkeypatch.setattr(margin, "_stream_preroll", lambda fen, sid: None)
    monkeypatch.setattr(margin._worker, "submit", lambda *a, **k: None)   # queue it
    monkeypatch.setattr(margin.theory, "theory_for",
                        lambda fen: None if fen == OUT_OF_BOOK else _WB_ENTRY)
    try:
        assert margin.build(OUT_OF_BOOK, seed="A")["plansPending"] is True
        margin.build(SICILIAN, seed="A")             # ...into theory
        margin._deep_job(OUT_OF_BOOK, "A")           # the job finally runs
        assert rolled == []                          # nobody is there any more
        assert key not in margin._inflight           # and it stays retryable
    finally:
        margin._deep_cache.clear()


def test_many_live_sessions_all_keep_their_own_rolls(monkeypatch):
    """Interest is pruned by AGE, never by count (Codex): a count cap evicts
    by arrival order, so a session still sitting on a position could have its
    own roll dropped as unwanted. 40 live sessions, and the oldest one's job
    still rolls; a session that stopped asking an hour ago does not keep one
    alive."""
    key = " ".join(OUT_OF_BOOK.split()[:4])
    rolled = []
    import lucena_backend.plans.service as svc
    monkeypatch.setattr(svc, "sheet_json_staged",
                        lambda fen, pool, maia, on_pre: (rolled.append(fen),
                                                         {"ok": True})[1])
    monkeypatch.setattr(margin, "_pool", object())
    monkeypatch.setattr(margin, "_inflight", set())
    monkeypatch.setattr(margin, "_latest_by_session", {})
    monkeypatch.setattr(margin, "_stream_preroll", lambda fen, sid: None)
    monkeypatch.setattr(margin._worker, "submit", lambda *a, **k: None)
    monkeypatch.setattr(margin.theory, "theory_for", lambda fen: None)
    try:
        margin.build(OUT_OF_BOOK, seed="first")          # the oldest session
        for i in range(40):                              # 40 more arrive after
            margin.build(OUT_OF_BOOK, seed=f"s{i}")
        margin._deep_job(OUT_OF_BOOK, "first")
        assert rolled == [OUT_OF_BOOK]                   # still wanted

        # ...but an entry older than the TTL is gone, and keeps nothing alive
        rolled.clear()
        margin._deep_cache.clear()
        # ...and the WORKER prunes too, so no later request is needed: an
        # expired interest cannot keep a roll alive through a quiet period.
        stale = time.monotonic() - margin._INTEREST_TTL - 1
        monkeypatch.setattr(margin, "_latest_by_session", {"ghost": (key, stale)})
        margin._deep_job(OUT_OF_BOOK, "ghost")
        assert rolled == []
        assert "ghost" not in margin._latest_by_session
    finally:
        margin._deep_cache.clear()
