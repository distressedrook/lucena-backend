"""CoachHandler._move_facts — how the verdict is grounded (the fix for the 'isolated pawn' invention).

The invariant: a MOVE verdict is grounded in the engine's read of the move JUST PLAYED, not the
spoil-stripped solve_text(). Right → the full read (facts included, since the move is on the board);
wrong → the refutation with the best move hidden (explain the flaw, never name the solution); a typed
answer (no move) → the solve-time facts. Engine is faked, so this is deterministic and fast.
"""

from __future__ import annotations

import asyncio

from lucena_backend.coaching.coach import CoachHandler
from lucena_backend.coaching.grounding import TieredFacts
from lucena_backend.coaching.loop import Input


class FakeGround:
    """Stands in for the read-only ground ctx: `evaluate` + `analyze_and_show` are exercised."""
    def __init__(self, evaluations, analysis=None):
        self._evals = evaluations           # {uci: eval-dict}
        self._analysis = analysis or []     # pre-move deep read (drives _deep_tactics on both paths)
        self.calls = []

    def evaluate(self, fen, ucis):
        self.calls.append((fen, tuple(ucis)))
        return self._evals[ucis[0]]

    def analyze_and_show(self, fen, **kw):
        return {"analysis": self._analysis}


def _handler(ground):
    # store/llm/ctx aren't touched by _move_facts; pass inert stand-ins.
    return CoachHandler(ctx=object(), store=object(), llm=object(), model="stub", ground=ground)


RIGHT = {   # b5c4 / bxc4 — the solution
    "san": "bxc4", "captured": "bishop", "class": "only_move", "best": {"san": "bxc4"},
    "side_to_move": "black", "fen": "F",
    "refutation_pv": ["Bh4", "Bb7"],   # a best move still has an opponent reply — must be dropped
    "facts": [{"text": "bxc4 wins the bishop on c4"}],
}
WRONG = {   # f6e4 / Nxe4 — a blunder that walks into Qe3
    "san": "Nxe4", "captured": "pawn", "class": "blunder", "best": {"san": "bxc4"},
    "side_to_move": "black", "fen": "F", "refutation_pv": ["Qe3", "Nef6"],
    "facts": [{"text": "bxc4 wins the bishop on c4"}],   # NAMES the solution
}
GROUNDING = TieredFacts(always=["Black is up a bishop.", "White's pieces are more active."], warn=[])


def _facts(handler, uci, correct):
    inp = Input(kind="move", uci=uci, fen="F", san=None)
    return asyncio.run(handler._move_facts(inp, correct, GROUNDING))


def test_right_verdict_includes_engine_facts_and_positional_read():
    s = _facts(_handler(FakeGround({"b5c4": RIGHT})), "b5c4", correct=True)
    assert "bxc4 wins the bishop on c4" in s, "the move's own 'why' must be grounded"
    assert "Black is up a bishop." in s, "the safe positional (always) tier must be included"


def test_right_verdict_carries_the_deep_point():
    # A correct verdict is a mini-lesson: it now gets the pre-move position's tactical POINT (the
    # linchpin / resource the simple tries fail against) — the SAME deep read the wrong path gets, so
    # the praise is instructive instead of a bare "nice, that wins the piece". (Solution-stripping of
    # the deep read is covered on the wrong path in test_grounding_tiers.)
    analysis = ["Tactics: Rh1+ is a saving check for White; the c6/d5 pawns mutually defend."]
    g = FakeGround({"b5c4": RIGHT}, analysis=analysis)
    s = _facts(_handler(g), "b5c4", correct=True)
    assert "Rh1+" in s, "the deep tactical point (the defensive resource) must reach the verdict"
    assert "The point of this move" in s, "deep-tactics framing must be present on the right path"


def test_right_verdict_drops_the_refutation_framing():
    # A best move has no 'refutation' — that line reads as if the right move were dubious.
    s = _facts(_handler(FakeGround({"b5c4": RIGHT})), "b5c4", correct=True)
    assert "refutes" not in s and "Bh4" not in s, "the correct move must not carry a refutation line"


def test_wrong_verdict_hides_the_solution():
    s = _facts(_handler(FakeGround({"f6e4": WRONG})), "f6e4", correct=False)
    assert "bxc4" not in s, "the wrong verdict leaked the solution move"
    assert "best move" not in s.lower()
    assert "Qe3" in s, "the flaw is explained through the refutation line"


def test_typed_answer_falls_back_to_solve_text():
    # No uci → nothing to evaluate → the free_text grader keeps the solve-time facts.
    g = FakeGround({})
    inp = Input(kind="text", text="a pin", uci=None, fen=None)
    out = asyncio.run(_handler(g)._move_facts(inp, correct=True, grounding=GROUNDING))
    assert out == GROUNDING.solve_text() and g.calls == [], "a typed answer must not call evaluate"
