"""The /analyze route and the live analyzer behind it (wired 2026-07-26).

`LiveAnalyzer` existed but nothing ever constructed it, so /analyze was a stub
that returned ok and did nothing — the Analysis panel had no engine (owner:
"hook it up with the engine as well ... it should show 4 pv"). These pin the
wiring: the route targets the analyzer at whatever position the app is showing
(a variation included), the panel gets FOUR lines, the engine starts on first
use rather than at boot, and a chat is resolved rather than trusted.
"""

import os
import shutil

import pytest

from fastapi.testclient import TestClient

from lucena_backend import httpserver
from lucena_backend.grounding_tools import live_analysis

_have_sf = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_sf, reason="no stockfish")

# a position off the mainline — the case that motivated the wiring
VARIATION = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


class _FakeAnalyzer:
    """Records what the route asks of it."""
    made: list = []

    def __init__(self, engine, store, *, multipv=3, **kw):
        self.engine, self.multipv, self.targets, self.started = engine, multipv, [], False
        _FakeAnalyzer.made.append(self)

    def start(self):
        self.started = True

    def set_target(self, fen, on, *, session_id):
        self.targets.append((fen, on, session_id))

    def stop(self):
        self.started = False


@pytest.fixture
def client(tmp_path, monkeypatch):
    _FakeAnalyzer.made = []
    monkeypatch.setattr(live_analysis, "LiveAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(httpserver, "Engine", lambda *a, **k: object())
    app = httpserver.build_app(home=str(tmp_path), llm=None, model="stub",
                               engine=object())
    with TestClient(app) as c:
        yield c


def test_the_engine_starts_on_first_use_not_at_boot(client):
    assert _FakeAnalyzer.made == []                  # building the app spawned nothing
    client.post("/analyze", json={"on": True, "fen": VARIATION})
    assert len(_FakeAnalyzer.made) == 1 and _FakeAnalyzer.made[0].started
    client.post("/analyze", json={"on": True, "fen": VARIATION})
    assert len(_FakeAnalyzer.made) == 1              # ...and only ever one


def test_the_panel_gets_four_lines(client):
    client.post("/analyze", json={"on": True, "fen": VARIATION})
    assert _FakeAnalyzer.made[0].multipv == 4


def test_it_analyzes_whatever_position_is_shown_including_a_variation(client):
    r = client.post("/analyze", json={"on": True, "fen": VARIATION})
    assert r.json()["on"] is True
    fen, on, sid = _FakeAnalyzer.made[0].targets[-1]
    assert (fen, on) == (VARIATION, True)
    assert sid                                       # addressed to a real chat


def test_on_without_a_board_is_not_analysis(client):
    """`on` with no fen would leave the loop waiting on nothing; it is an off."""
    assert client.post("/analyze", json={"on": True}).json()["on"] is False


def test_a_dead_engine_degrades_the_route_not_the_app(tmp_path, monkeypatch):
    """No Stockfish here → the panel shows no lines. It must not 500, and it
    must not retry the spawn on every keystroke."""
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("no engine")

    monkeypatch.setattr(httpserver, "Engine", boom)
    app = httpserver.build_app(home=str(tmp_path), llm=None, model="stub",
                               engine=object())
    with TestClient(app) as c:
        for _ in range(3):
            r = c.post("/analyze", json={"on": True, "fen": VARIATION})
            assert r.status_code == 200 and r.json()["ok"] is False
    assert len(calls) == 1                           # tried once, then stayed off


def test_a_bad_fen_is_refused_not_handed_to_the_engine(client):
    """The deepen loop retries its target the instant a search raises, so a FEN
    the engine rejects would spin the dedicated thread. It never gets there."""
    r = client.post("/analyze", json={"on": True, "fen": "not a fen"})
    assert r.status_code == 400 and r.json()["error"] == "bad_fen"
    assert _FakeAnalyzer.made == []                  # no engine spawned either


def test_switching_off_never_starts_an_engine(client):
    """An off switch — or a panel that opens before the board exists — must not
    spawn Stockfish just to say 'stop'."""
    assert client.post("/analyze", json={"on": False, "fen": VARIATION}).json()["on"] is False
    assert client.post("/analyze", json={"on": True}).json()["on"] is False
    assert _FakeAnalyzer.made == []
    # ...and once one IS running, off still reaches it
    client.post("/analyze", json={"on": True, "fen": VARIATION})
    client.post("/analyze", json={"on": False, "fen": VARIATION})
    assert _FakeAnalyzer.made[0].targets[-1][:2] == (None, False)


# -- the fen -> lines cache (2026-07-26) --------------------------------------

class _RecordingStore:
    def __init__(self):
        self.published = []

    def publish_engine_lines(self, payload, *, session_id):
        self.published.append(payload)


class _FakeEngine:
    name = "Stockfish test"

    def __init__(self):
        self.searched = []

    def analyse(self, fen, *, depth, multipv):
        self.searched.append(depth)

        class _Score:
            @staticmethod
            def to_ceiled_cp():
                return 12

        class _Line:
            rank, pv, score = 1, [], _Score()

        class _Analysis:
            lines = [_Line()]

        return _Analysis()

    def close(self):
        pass


def _analyzer(store, engine, cache, max_depth=12):
    from lucena_backend.grounding_tools.live_analysis import LiveAnalyzer
    return LiveAnalyzer(engine, store, max_depth=max_depth, multipv=4, positions=cache)


def _deepen_once(a, fen, sid="s1"):
    """Run one deepen pass to completion. `_deepen` parks on its condition once it reaches max
    depth (that is how it holds a finished position), so the test releases it with stop()."""
    import threading
    t = threading.Thread(target=a._deepen, args=(fen, 0, sid), daemon=True)
    t.start()
    t.join(timeout=2)          # the fake engine is instant; this is the parked wait
    a.stop()
    t.join(timeout=2)
    assert not t.is_alive(), "the deepen loop never parked"


def test_a_known_position_is_republished_and_not_re_searched():
    """Walking BACK through a game re-deepened every position from depth 1. A position already
    taken to the maximum depth now answers instantly and searches not at all."""
    from lucena_backend.positions import LINES, PositionCache
    store, engine = _RecordingStore(), _FakeEngine()
    cache = PositionCache(None)
    cached = {"fen": VARIATION, "depth": 12, "engine": "Stockfish 18",
              "opening": None, "lines": [{"rank": 1, "eval_white_cp": 25,
                                          "win_pct": 52.0, "pv_san": ["d3"]}]}
    cache.put(VARIATION, LINES, cached)
    _deepen_once(_analyzer(store, engine, cache), VARIATION)
    assert store.published == [cached]           # answered from the cache...
    assert engine.searched == []                 # ...and the engine never ran


def test_a_partly_known_position_resumes_past_it():
    from lucena_backend.positions import LINES, PositionCache
    store, engine = _RecordingStore(), _FakeEngine()
    cache = PositionCache(None)
    cache.put(VARIATION, LINES, {"fen": VARIATION, "depth": 9, "engine": "x",
                                 "lines": [{"rank": 1, "eval_white_cp": 1,
                                            "win_pct": 50.0, "pv_san": []}]})
    _deepen_once(_analyzer(store, engine, cache), VARIATION)
    assert engine.searched == [10, 11, 12]       # picks up where it left off
    assert store.published[0]["depth"] == 9      # the known line-up went out first


def test_the_line_up_is_stored_at_a_few_depths_not_all_of_them(monkeypatch):
    """Every depth would be ~28 writes per position for a readout that only improves."""
    from lucena_backend.positions import PositionCache
    store, engine = _RecordingStore(), _FakeEngine()
    cache = PositionCache(None)
    stored = []
    monkeypatch.setattr(cache, "put", lambda fen, kind, payload: stored.append(payload["depth"]))
    _deepen_once(_analyzer(store, engine, cache), VARIATION)
    assert stored == [8, 12]                     # the settling depth, then every fourth / the last
