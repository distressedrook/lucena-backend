"""Gamepass surfaces brilliant plies — moved from engine test_brilliant.py
(core-migration Phase 5, 2026-07-23) when gamepass landed in the backend."""
import os
import shutil

import pytest

_have_engine = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_engine, reason="needs stockfish")

LEGAL_SAC_SAN = "Nxe5"

LEGAL_PGN = """[Event "Legal"]
[White "A"]
[Black "B"]
[Result "*"]
[UserSide "w"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 d6 4. Nc3 Bg4 5. h3 Bh5 6. Nxe5 *
"""



@requires_engine
def test_gamepass_surfaces_brilliant_ply(tmp_path):
    # A full-game pass over Legal's opening: the played 6.Nxe5 (ply 11) is a sound
    # sacrifice over an otherwise-ok move -> its ply class is "brilliant".
    from lucena_backend.pipelines.gamepass import run_pass
    from lucena_engine.uci import Engine

    # run_pass returns the digest directly (persistence is the backend's job, not the engine's).
    with Engine(threads=1) as engine:
        engine.new_game()
        data = run_pass(
            LEGAL_PGN, engine,
            fast_limit={"nodes": 400_000},
            deep_limit={"nodes": 1_200_000},
        )
    nxe5 = next(p for p in data["plies"] if p.get("san") == LEGAL_SAC_SAN)
    assert nxe5["class"] == "brilliant"




# B#3 — "worthwhile" must gate on played_win, not best_win (the contract fix).
