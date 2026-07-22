"""The plans layer (lucena_backend.plans): rolls, gates, and the freeform route.

The (fen, pvs, rolls) contract: lucena-plans only checks lines; the backend
produces them. These tests cover the producers' shapes (real Stockfish at
tiny nodes — same availability assumption as the engine tests), the
deterministic gates, and _plans_read's fallback discipline. No Postgres, no
LLM: the handler is built bare and _gen_json is stubbed.
"""

import asyncio

import pytest

from lucena_backend.coaching.freeform import FreeformHandler
from lucena_backend.engine_io.enginepool import EnginePool
from lucena_backend.plans import is_endgame, sheet_for
from lucena_backend.plans.rolls import roll_engine, roll_maia

MID = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2P1PN2/PP1N1PPP/R2QKB1R w KQ - 0 8"
END = "4k3/8/8/8/8/8/R7/4K3 w - - 0 1"


@pytest.fixture(scope="module")
def pool():
    return EnginePool(size=1, threads=1, hash_mb=16)


# ---------------------------------------------------------------- gates

def test_is_endgame():
    assert is_endgame(END)
    assert not is_endgame(MID)
    # QR each side (npm 14) is still a middlegame under the 13 threshold.
    assert not is_endgame("q3k2r/8/8/8/8/8/8/Q3K2R w Kk - 0 1")


# ---------------------------------------------------------------- rolls

def test_roll_engine_shape(pool):
    with pool.lease() as eng:
        pvs = roll_engine(eng, MID, horizon=8, multipv=2,
                          main_nodes=20_000, ext_nodes=5_000)
    assert pvs and len(pvs) <= 2
    for p in pvs:
        assert isinstance(p["cp"], int)
        assert 1 <= len(p["ucis"]) <= 8
    # best line first; cp is White-POV (equalish opening stays in a sane band)
    assert abs(pvs[0]["cp"]) < 300


def test_roll_maia_absent_leg():
    assert roll_maia(None, MID) is None


def test_roll_maia_gate_and_determinism():
    class FakeMaia:
        """Board-agnostic policy: always prefer the first legal move, with one
        contested rival — exercises the 40-60 gate without a real model."""
        def top_human_moves(self, fen, rating, *, n=5):
            from lucena_engine.board import Board
            legal = Board(fen).legal_moves()
            rows = [{"uci": legal[0], "rank": 1, "policy": 0.5}]
            if len(legal) > 1:
                rows.append({"uci": legal[1], "rank": 2, "policy": 0.4})  # contested
            if len(legal) > 2:
                rows.append({"uci": legal[2], "rank": 3, "policy": 0.1})  # gated out
            return rows

    a = roll_maia(FakeMaia(), MID, horizon=6, k=3)
    b = roll_maia(FakeMaia(), MID, horizon=6, k=3)
    assert a == b, "fen-seeded rollouts must be reproducible"
    assert len(a) == 3 and all(1 <= len(r) <= 6 for r in a)
    # policy-less wrapper -> no rollouts at all (leg absent), not garbage
    class NoPolicy:
        def top_human_moves(self, fen, rating, *, n=5):
            return [{"uci": "g1f3", "rank": 1}]
    assert roll_maia(NoPolicy(), MID, horizon=4, k=2) == [[], []]


# ---------------------------------------------------------------- sheet

def test_sheet_for_end_to_end(pool, monkeypatch):
    # Production defaults are calibration-grade (1M/250k, ~1 min); tests dial
    # the module constants down (late-bound in roll_engine, so this works).
    from lucena_backend.plans import rolls as _rolls
    monkeypatch.setattr(_rolls, "MAIN_NODES", 20_000)
    monkeypatch.setattr(_rolls, "EXT_NODES", 5_000)
    sheet, pid = sheet_for(MID, pool, None, horizon=8)
    assert pid.startswith("POSITION-")
    assert "PLAN FOR WHITE" in sheet and "ASSESSMENT" in sheet
    assert MID.split()[0] not in sheet, "FEN must stay redacted"


# ---------------------------------------------------------------- route

class _Ground:
    def __init__(self, cp, pool=None):
        self.cp, self._pool, self.calls = cp, pool, 0

    def analyze_and_show(self, fen, **kw):
        self.calls += 1
        return {"eval": {"cp": self.cp}}


class _Ctx:
    maia = None


def _handler(ground):
    h = FreeformHandler(ctx=_Ctx(), store=None, llm=None, model="m", ground=ground)

    async def fake_gen(system, prompt, **kw):
        assert "FACT SHEET" in system
        return {"text": "narrated"}
    h._gen_json = fake_gen
    return h


def test_plans_read_gates(pool, monkeypatch):
    from lucena_backend.plans import rolls as _rolls
    monkeypatch.setattr(_rolls, "MAIN_NODES", 20_000)
    monkeypatch.setattr(_rolls, "EXT_NODES", 5_000)

    async def main():
        # endgame: refused before the eval probe is even paid
        g = _Ground(0)
        assert await _handler(g)._plans_read(END) is None and g.calls == 0
        # outside the equalish band: refused after the probe
        g = _Ground(320)
        assert await _handler(g)._plans_read(MID) is None and g.calls == 1
        # inside the band: rolled, sheeted, narrated
        out = await _handler(_Ground(40, pool))._plans_read(MID)
        assert out == "narrated"
        # layer failure (dead pool) degrades to None -> plain read fallback
        class Dead:
            def lease(self):
                raise RuntimeError("down")
        assert await _handler(_Ground(40, Dead()))._plans_read(MID) is None
    asyncio.run(main())
