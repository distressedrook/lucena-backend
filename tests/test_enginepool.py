"""Phase 2 — the engine pool. Concurrency without oversubscribing the box.

These use a fake engine: the properties under test (exclusivity, bounding, respawn, explicit
threads) are about the pool, not about Stockfish, and a real one would make them slow and
CPU-bound for no extra signal. test_mcp covers the pool with a real engine end to end.
"""

import threading
import time

import pytest

from lucena_backend.enginepool import EnginePool, SingleEnginePool, default_size
from lucena_engine.uci import EngineError


class _FakeEngine:
    def __init__(self, **kw):
        self.kwargs = kw
        self.new_games = 0
        self.closed = False
        self.in_use = False

    def new_game(self):
        self.new_games += 1

    def close(self):
        self.closed = True


def test_lease_is_exclusive():
    """Two callers must never hold the same engine: options are sticky (MultiPV) and callers mutate
    them mid-call, so a shared engine mid-conversation corrupts both sides."""
    pool = EnginePool(size=2, factory=_FakeEngine)
    seen_overlap = []

    def worker():
        with pool.lease() as eng:
            if eng.in_use:
                seen_overlap.append(True)
            eng.in_use = True
            time.sleep(0.02)
            eng.in_use = False

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not seen_overlap
    pool.close()


def test_pool_bounds_concurrency():
    """`size` is the cap on simultaneous analyses. Without it, N chats = N Stockfish processes."""
    pool = EnginePool(size=2, factory=_FakeEngine)
    live, peak = 0, 0
    lock = threading.Lock()

    def worker():
        nonlocal live, peak
        with pool.lease():
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            with lock:
                live -= 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak <= 2, f"{peak} engines in use at once with size=2"
    pool.close()


def test_engines_are_spawned_lazily_and_reused():
    """Constructing an engine is Popen + handshake + version assert; a cold pool would make startup
    pay for capacity nobody asked for."""
    made = []
    pool = EnginePool(size=4, factory=lambda: made.append(_FakeEngine()) or made[-1])
    assert made == []                       # nothing spawned until first use
    with pool.lease():
        pass
    assert len(made) == 1
    for _ in range(3):
        with pool.lease():
            pass
    assert len(made) == 1, "a warm slot should be reused, not respawned"
    pool.close()


def test_new_game_on_acquire():
    """Engines are fungible across chats ONLY because each lease resets the engine."""
    pool = EnginePool(size=1, factory=_FakeEngine)
    with pool.lease() as e1:
        pass
    with pool.lease() as e2:
        pass
    assert e1 is e2 and e2.new_games == 2
    pool.close()


def test_a_dead_engine_is_discarded_and_the_slot_respawns():
    """A failed engine must not go back in the pool — the next lease would inherit the failure — and
    capacity must survive the failure."""
    made = []

    def factory():
        e = _FakeEngine()
        made.append(e)
        return e

    pool = EnginePool(size=1, factory=factory)
    with pytest.raises(EngineError):
        with pool.lease() as eng:
            first = eng
            raise EngineError("stockfish died")
    assert first.closed, "the dead engine was not closed"
    with pool.lease() as eng:               # the slot still works
        assert eng is not first, "a dead engine was handed back out"
    assert len(made) == 2
    pool.close()


def test_pool_passes_explicit_threads_and_hash():
    """The trap: Engine's defaults are PER INSTANCE (threads = max(2, cpu-2), hash = 256MB), so a
    pool at the defaults oversubscribes the CPU ~size x and reserves size x 256MB of RAM."""
    made = []

    def factory(**kw):
        e = _FakeEngine(**kw)
        made.append(e)
        return e

    pool = EnginePool(size=2, threads=1, hash_mb=32,
                      factory=lambda: factory(threads=1, hash_mb=32))
    with pool.lease():
        pass
    assert made[0].kwargs == {"threads": 1, "hash_mb": 32}
    pool.close()


def test_default_size_leaves_room_for_the_rest_of_the_box():
    assert 2 <= default_size() <= max(2, (__import__("os").cpu_count() or 4))


def test_close_closes_live_engines():
    pool = EnginePool(size=2, factory=_FakeEngine)
    with pool.lease() as e:
        pass
    pool.close()
    assert e.closed


def test_single_engine_pool_yields_the_caller_engine():
    """ToolContext(engine, store) keeps working: a caller-supplied engine becomes a pool of one."""
    eng = _FakeEngine()
    pool = SingleEnginePool(eng)
    with pool.lease() as leased:
        assert leased is eng
    assert eng.new_games == 0, "the engine is the caller's; the pool must not reset it"


def test_single_engine_pool_is_reentrant():
    """A nested lease must not deadlock a pool of one (nested @_guarded calls are normal:
    play_move -> assess_move)."""
    pool = SingleEnginePool(_FakeEngine())
    with pool.lease() as a:
        with pool.lease() as b:
            assert a is b
