"""A bounded pool of Stockfish processes, leased per tool call.

Why a pool: one shared `Engine` serializes every user's analysis behind its own internal lock, so
concurrent chats queue no matter how well the state layer is partitioned. Engines are FUNGIBLE —
`new_game()` on acquire resets any cross-position state — which is what makes pooling sound.

Sizing is the whole risk. Each lease is a real OS process with its own hash table:
  * `threads` is MANDATORY. `Engine`'s default is `max(2, cpu_count-2)` PER INSTANCE, so a pool of N
    at the default oversubscribes the box by roughly N×.
  * `hash_mb` is per engine, so the pool's floor is `size × hash_mb` of RAM. Stockfish's own default
    is 256MB — a pool of 8 at that default silently reserves 2GB.

Modelled on `lucena_engine.server.serve.EngineHolder.acquire` (lease + `new_game()` on acquire),
generalised from one engine to a bounded set.
"""

from __future__ import annotations

import os
import queue
import threading
from contextlib import contextmanager

from lucena_engine.uci import Engine, EngineError


def default_size() -> int:
    """Concurrent analyses to allow. Half the box, so Stockfish threads + Maia + the backend itself
    still have somewhere to run."""
    return max(2, (os.cpu_count() or 4) // 2)


class EnginePool:
    """`size` Stockfish processes, handed out one at a time.

    Engines are spawned LAZILY: constructing one is a Popen + UCI handshake + version assert + option
    set, so a cold pool would make startup pay for capacity nobody has asked for yet.
    """

    def __init__(self, *, size: int | None = None, threads: int = 1, hash_mb: int = 64,
                 factory=None):
        self._size = size or default_size()
        self._threads = threads
        self._hash_mb = hash_mb
        self._factory = factory or (lambda: Engine(threads=self._threads, hash_mb=self._hash_mb))
        # Slots, not engines: a token to take is what bounds concurrency. `None` = "a slot you may
        # use, engine not spawned yet"; the lease materialises it.
        self._slots: queue.LifoQueue = queue.LifoQueue()
        for _ in range(self._size):
            self._slots.put(None)
        self._live: list[Engine] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return self._size

    @contextmanager
    def lease(self, timeout: float | None = None):
        """Check out an engine for the duration of the block.

        Exclusive for the whole lease, deliberately: callers mutate per-engine UCI options mid-call
        (`explore_and_show` sets Threads=1 and restores it), and MultiPV is sticky, so an engine must
        not be shared mid-conversation.
        """
        eng = self._slots.get(timeout=timeout)
        try:
            if eng is None:
                eng = self._factory()
                with self._lock:
                    self._live.append(eng)
            eng.new_game()          # engines are fungible ONLY because of this
            yield eng
        except EngineError:
            # A dead engine must not go back in the pool — the next lease would inherit the failure.
            # Drop it and return an empty slot, so capacity is preserved and the next lease respawns.
            self._discard(eng)
            eng = None
            raise
        finally:
            self._slots.put(eng)

    def _discard(self, eng) -> None:
        if eng is None:
            return
        with self._lock:
            if eng in self._live:
                self._live.remove(eng)
        try:
            eng.close()
        except Exception:           # noqa: BLE001 — already failing; closing is best-effort
            pass

    def close(self) -> None:
        with self._lock:
            live, self._live = self._live, []
        for eng in live:
            try:
                eng.close()
            except Exception:       # noqa: BLE001
                pass


class SingleEnginePool:
    """One caller-supplied engine behind the pool interface.

    Keeps `ToolContext(engine, store)` working for the ~25 call sites (mostly tests) that hand over
    one engine they own, without them needing to know about pooling. No `new_game()` here: the engine
    is not ours, and the callers that pass one drive it themselves.
    """

    size = 1

    def __init__(self, engine):
        self._engine = engine
        self._lock = threading.RLock()

    @contextmanager
    def lease(self, timeout: float | None = None):
        with self._lock:
            yield self._engine

    def close(self) -> None:
        pass
