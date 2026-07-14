"""End-to-end poisoned-line detection over the REAL chain: real Stockfish +
the real policy-emitting Maia wrapper. This is the test the fake-based suite
(test_poisoned_line_detector.py) structurally cannot be — it injects fakes that
always supply `policy`, so it never notices when the production Maia wiring drops
it (the "green tests, dead in prod" bug this file guards against).

Slow + needs both binaries. The oracle is the contract's worked example
(docs/contracts/M-poisoned-line-detector.md "Worked example").
"""

import os
import shutil

import pytest

from lucena_engine import Engine
from lucena_engine.maia import MaiaEngine
from lucena_engine.poisoned_line_detector import find_poisoned_lines

pytestmark = [pytest.mark.slow, pytest.mark.engine]

# The contract's worked example: Black to move, solution bxc4. The poisoned line:
# seed Nxe4, the mover walks ...hxg3, and Nxg3 is the fatal spike.
WORKED = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"
NODES = 60_000  # the contract's faster mining-pass count; qualitative result holds at the 200k default

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_POLICY_MAIA = f"{_REPO}/.venv-maia/bin/python {_REPO}/tools/maia_policy_uci.py"


def _have_stockfish():
    return bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")


def _have_policy_maia():
    return all(os.path.exists(p) for p in _POLICY_MAIA.split())


pytestmark += [
    pytest.mark.skipif(not _have_stockfish(), reason="no stockfish"),
    pytest.mark.skipif(not _have_policy_maia(), reason="no .venv-maia policy wrapper"),
]


@pytest.fixture(scope="module")
def engine():
    with Engine(threads=1) as e:                          # single-threaded: determinism
        yield e


@pytest.fixture(scope="module")
def maia():
    with MaiaEngine(_POLICY_MAIA) as m:                  # the REAL policy-emitting wrapper
        yield m


def test_worked_example_detects_the_poisoned_line(engine, maia):
    res = find_poisoned_lines(WORKED, engine, maia, rating=1500, nodes=NODES)

    assert res["has_poisoned_line"] is True
    assert res["temptations"], "the worked example must surface at least one temptation"

    # The documented trap: idea hxg3, fatal Nxg3, seeded by Nxe4.
    trap = next((t for t in res["temptations"] if t["idea"] == "hxg3"), None)
    assert trap is not None, f"expected an hxg3 temptation, got {[t['idea'] for t in res['temptations']]}"
    assert trap["fatal"] == "Nxg3"
    assert "Nxe4" in trap["seeds"]
    assert trap["deep"] is True and trap["avoids"] is True
    assert trap["drop"] >= 20

    # The crux — real policy actually flowed through the multiplication. A dropped
    # `policy` (the bug) would zero this; the contract pins it to the "common" tier
    # at a stable ~5-6%.
    assert trap["poisoned"] > 0
    assert trap["tier"] == "common"


def test_solution_is_never_a_temptation(engine, maia):
    # bxc4 is the right move; it must never be reported as a poisoned line.
    res = find_poisoned_lines(WORKED, engine, maia, rating=1500, nodes=NODES, solution_uci="b5c4")
    for t in res["temptations"]:
        assert "bxc4" not in t["seeds"]
