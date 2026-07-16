"""Freeform opening narration: the cadence, the swing fork, and the voice.

Three layers, all deterministic and none of them touching an LLM or a Stockfish binary:

1. `_book_route` is pure, so the cadence is tested by REPLAYING real openings through it and
   asserting the exact fire sequence. This is where the coarsening regressions are pinned.
2. The prompt SELECTION and the prompt TEXT are asserted through a fake adapter. We test the input we
   hand the model, never its prose — "did it write a good paragraph" is ungroundable, and a test that
   asserts on generated text tests the weather.
3. The voice: assert no "you" reaches the model on any freeform path, and that every drill path still
   gets one. That is the drills-are-unaffected proof.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lucena_backend.llm import Completion, Usage
from lucena_backend.orchestrator import (
    Orchestrator, _book_route, _is_swing, _mover, _move_system, _move_system_tail,
    _MOVE_SYSTEM_HEAD, _COACH_SYSTEM_HEAD, _COACH_SYSTEM_TAIL,
    _NARRATE_SYSTEM_HEAD, _NARRATE_SYSTEM_TAIL, _ENDBOOK_SYSTEM, _perspective,
    _NARRATE, _ENDBOOK, _COACH, _SILENT, _OFF_BOOK_AT,
)
from lucena_engine.board import Board

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _fens(*ucis: str) -> list:
    """Replay a line from the start position, returning [start, ...after each ply].

    Through the REAL board core, never FEN literals: the opening table is keyed on our own core's
    en-passant convention, and a literal-keyed test would keep passing if the core's convention
    changed underneath it. Replaying is what pins the two together.
    """
    b = Board(START)
    out = [START]
    for u in ucis:
        b = b.apply(u)
        out.append(b.fen)
    return out


def _routes(ucis: list, swing: bool = False) -> list:
    """The route for each ply of a line, as (san-ish uci, route, name)."""
    fens = _fens(*ucis)
    return [(ucis[i - 1],) + _book_route(fens[: i + 1], swing)[:2] for i in range(1, len(fens))]


# -- 1. the cadence ------------------------------------------------------------------------------

RUY = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]
NAJDORF = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]
QGD = ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]
FRENCH = ["e2e4", "e7e6", "d2d4", "d7d5"]


def test_a_coarser_name_never_fires():
    """The regression that killed `deepest()`: the table re-attaches COARSER names deeper in a line.

    Najdorf `d6` names "Sicilian Defense: Modern Variations", then `d4` re-names the position plain
    "Sicilian Defense". A last-wins walk narrates the LESS specific name it just moved past — the
    coach audibly forgetting what it said one ply ago. These three plies are the exact ones that
    exposed it, so they are asserted by name rather than left to a general property.
    """
    for line, ply in ((NAJDORF, "d2d4"), (QGD, "f8e7"), (FRENCH, "d7d5")):
        routes = {u: r for u, r, _ in _routes(line)}
        assert routes[ply] != _NARRATE, (
            f"{ply} fired a narration — it re-attaches a COARSER name than the one already narrated, "
            f"so the coach would announce a less specific opening than it just used"
        )


def test_a_genuine_branch_fires():
    """The other half: a real branch into a named variation MUST fire, or the fold is just a mute."""
    ruy = {u: r for u, r, _ in _routes(RUY)}
    assert ruy["f1b5"] == _NARRATE, "Bb5 enters the Ruy Lopez and must be narrated"
    assert ruy["a7a6"] == _NARRATE, "a6 enters the Morphy Defense — a genuine refinement, not a coarsening"


def test_an_unnamed_ply_inside_the_book_is_sticky_and_silent():
    """A line that dips out of the table for a ply stays in its opening: silent, not end-of-book."""
    routes = _routes(RUY)
    assert all(r != _ENDBOOK for _, r, _ in routes), \
        f"the Ruy Lopez mainline announced end-of-book: {[(u, r) for u, r, _ in routes]}"
    quiet = [(u, r) for u, r, _ in routes if r == _SILENT]
    assert quiet, "no ply was silent — every book ply narrating is the volume problem, not the fix"


def test_the_first_move_narrates_and_the_reply_does_not():
    """The design goal, at its smallest: 1.e4 gets a paragraph; 1...e5 adds no new name, so silence."""
    r = _routes(["e2e4", "e7e5"])
    assert r[0][1] == _NARRATE and r[0][2] == "King's Pawn Game"
    assert r[1][1] == _SILENT


# -- 2. off-book ---------------------------------------------------------------------------------

# Enters the book at ply 1 (Anderssen's Opening) and then wanders straight out of it — measured, not
# assumed: a line that was NEVER named exercises a different branch (see the None case below).
WANDER = ["a2a3", "a7a6", "h2h3", "h7h6", "a3a4", "h6h5", "a1a3", "a8a7"]


def test_off_book_announces_exactly_once():
    """Fires at the threshold and never again — the property that means no latch has to be stored."""
    routes = [r for _, r, _ in _routes(WANDER)]
    assert routes.count(_ENDBOOK) == 1, f"expected exactly one end-of-book, got {routes}"
    assert routes.index(_ENDBOOK) == _OFF_BOOK_AT, \
        f"end-of-book fired at ply {routes.index(_ENDBOOK)}, not at the threshold: {routes}"
    assert all(r == _COACH for r in routes[_OFF_BOOK_AT + 1:]), \
        f"past the threshold every ply must be normal coaching: {routes}"


def test_a_line_that_was_never_in_the_book_never_announces():
    """A drill or a pasted midgame FEN has no book to leave. `plies_since_named` returns None and the
    router must fall through to normal coaching — announcing "you have left opening theory" to someone
    who was handed a rook endgame is nonsense."""
    midgame = ["8/5k2/8/8/8/8/5K2/4R3 w - - 0 1"] * 8
    for i in range(1, len(midgame)):
        route, _, _ = _book_route(midgame[: i + 1], swing=False)
        assert route == _COACH, f"a never-named line routed {route!r} at ply {i}"


def test_off_book_fires_at_the_threshold_not_before():
    fens = _fens(*RUY)
    named_through = [i for i in range(1, len(fens))
                     if _book_route(fens[: i + 1], False)[0] != _ENDBOOK]
    assert len(named_through) == len(fens) - 1, "the Ruy went off-book; the threshold is too tight"
    assert _OFF_BOOK_AT == 4, "the threshold is a measured heuristic — changing it needs the measurement"


# -- 3. the swing fork ---------------------------------------------------------------------------

def test_swing_fires_without_a_name_change():
    """The whole point of the gambit case: the swing trigger is INDEPENDENT of the name cadence.

    1...e5 is in the book and adds no new name (both plies are "King's Pawn Game"), so it is silent
    when quiet — and must still speak when the eval swings. A cadence keyed only on the name would be
    mute on exactly the move worth explaining.
    """
    line = _fens("e2e4", "e7e5")
    assert _book_route(line, swing=False)[0] == _SILENT, "fixture drifted: this ply must be quiet-silent"
    loud, name, prev = _book_route(line, swing=True)
    assert loud == _NARRATE, "a swing inside the book must narrate even when the name did not change"
    assert prev is None, "there is no delta to write when the name did not change — prev must not be passed"
    assert name, "the name is still passed as CONTEXT on a swing"


def test_a_swing_off_book_is_normal_coaching_not_narration():
    assert _book_route(_fens(*WANDER), swing=True)[0] == _COACH, \
        "a blunder outside the book is a blunder, not theory to explain"


def test_is_swing_reads_the_engines_own_classes():
    assert _is_swing({"class": "dubious"}) and _is_swing({"class": "blunder"})
    assert not _is_swing({"class": "ok"}) and not _is_swing({"class": "brilliant"})
    assert not _is_swing({}) and not _is_swing(None)


def test_mover_is_the_side_to_move_of_the_position_played_from():
    assert _mover(START) == "White"
    assert _mover(_fens("e2e4")[1]) == "Black"


# -- 4. prompt selection + voice -----------------------------------------------------------------

class _CapturingLLM:
    """Records every (system, user) it is asked to generate from, and returns a fixed beat."""

    def __init__(self):
        self.calls: list = []

    async def generate(self, messages, opts):
        system = "\n".join(m.content for m in messages if m.role == "system")
        user = "\n".join(m.content for m in messages if m.role != "system")
        self.calls.append({"system": system, "user": user})
        payload = {"mode": "tell", "text": "NARRATED"}
        return Completion(text=json.dumps(payload), json=payload, model="stub", usage=Usage(1, 1, 2))


class _FakeGround:
    """The read-only grounding surface, stubbed. `verdict` is what `evaluate` returns."""

    def __init__(self, verdict=None):
        self.verdict = verdict if verdict is not None else {"san": "e4", "class": "ok"}

    def evaluate(self, fen, moves, **kw):
        return self.verdict

    def analyze_and_show(self, fen, **kw):
        return {"fen": fen, "facts": ["Opening: King's Pawn Game"], "eval": {"win_pct": 52.0}}

    def get_hints(self, fen, **kw):
        return {"best": "d4", "hints": []}


class _FakeStore:
    def __init__(self, fens, sans=None):
        # History carries `san` because the real one does (play_move writes it) and the narration
        # prompt reads it. A fixture that omits it would let `_played_line` silently return "" and the
        # prompt assertions would pass against an empty line.
        sans = sans or [None] * len(fens)
        self._history = [{"n": i, "fen": f, "san": sans[i] if i < len(sans) else None}
                         for i, f in enumerate(fens)]
        self.board_view = fens[-1]
        self.session_unnamed = True
        self.named = None
        self.beats: list = []
        self._last_tree = None       # the drill-solved path reads it for the poisoned-line note

    def publish_status(self, *a, **k):
        pass

    def append_beats(self, beats):
        self.beats.extend(beats)

    def set_gate(self, *a, **k):
        pass

    def set_session_name(self, name):
        self.named = name


class _FakeCtx:
    def __init__(self, store, freeform=True):
        self.store = store
        self.freeform = freeform


def _sans_for(fens: list) -> list:
    """The SAN of each ply, derived through the real board core (ply 0 has none)."""
    out = [None]
    for i in range(1, len(fens)):
        b = Board(fens[i - 1])
        san = next((b.san(u) for u in b.legal_moves() if b.apply(u).fen == fens[i]), None)
        out.append(san)
    return out


def _run(fens, result, verdict=None, freeform=True):
    """Drive `_coach_move` with everything stubbed; return (llm, store)."""
    store = _FakeStore(fens, _sans_for(fens))
    llm = _CapturingLLM()
    orch = Orchestrator(ctx=_FakeCtx(store, freeform), model="stub", llm=llm,
                        ground_ctx=_FakeGround(verdict))
    pre = fens[-2] if len(fens) > 1 else START
    asyncio.run(orch._coach_move("e2e4", pre, result))
    return llm, store


FREEFORM = {"ok": True, "drill": False}
DRILL_CORRECT = {"ok": True, "drill": True, "correct": True, "finished": False}
DRILL_WRONG = {"ok": True, "drill": True, "correct": False, "finished": False}
DRILL_SOLVED = {"ok": True, "drill": True, "correct": True, "finished": True}
SUSPENDED = {"ok": True, "drill": "suspended"}


def test_an_in_book_freeform_move_gets_the_narration_prompt():
    llm, store = _run(_fens("e2e4"), FREEFORM)
    assert len(llm.calls) == 1
    system = llm.calls[0]["system"]
    assert "opening book" in system, "1.e4 did not get the narration voice"
    assert "short paragraph" in system, "narration must not inherit the 1-2 sentence cap"
    assert "King's Pawn Game" in llm.calls[0]["user"], "the opening name never reached the model"


def test_a_quiet_book_move_hides_the_best_move():
    """`hide_best` is the difference between narration and a verdict. Handed "best: d4", the model
    helpfully reports it — which reproduces "that's fine, though d4 is preferred", the exact line this
    feature exists to kill."""
    llm, _ = _run(_fens("e2e4"), FREEFORM, verdict={"san": "e4", "class": "ok", "best": {"san": "d4"}})
    assert "d4" not in llm.calls[0]["user"], \
        "the engine's preferred move was handed to a narration — it will be reported as a verdict"


def test_a_book_move_that_swings_keeps_the_class_and_is_told_to_explain_it():
    """The gambit: theory the engine dislikes. Here the class IS the subject, so it must survive."""
    kga = _fens("e2e4", "e7e5", "f2f4")
    llm, _ = _run(kga, FREEFORM, verdict={"san": "f4", "class": "dubious", "best": {"san": "Nf3"}})
    user = llm.calls[0]["user"]
    assert "dubious" in user, "the swing's class was hidden — the model cannot explain what it cannot see"
    assert "does NOT prefer" in user and "gives up" in user, \
        "the model was handed a swing with no instruction to explain the concession"


def test_a_silent_book_move_makes_no_llm_call_at_all():
    llm, store = _run(_fens("e2e4", "e7e5"), FREEFORM)
    assert llm.calls == [], "a book move with nothing new to say still called the LLM"
    assert store.beats == [], "a silent ply posted a beat"


def test_an_off_book_freeform_move_falls_back_to_normal_coaching():
    llm, _ = _run(_fens(*WANDER), FREEFORM, verdict={"san": "Ra7", "class": "ok"})
    assert "opening book" not in llm.calls[0]["system"], "an off-book move got the narration voice"
    assert "FREEFORM move" in llm.calls[0]["user"]


def test_the_freeform_paths_never_say_you():
    """The owner's constraint, asserted on the INPUT: in freeform nothing replies and the board has no
    side-to-move gate, so there is no "you" to address — only White and Black."""
    for label, fens, verdict in (
        ("narration", _fens("e2e4"), {"san": "e4", "class": "ok"}),
        ("swing", _fens("e2e4", "e7e5", "f2f4"), {"san": "f4", "class": "dubious"}),
        ("off-book", _fens(*WANDER), {"san": "Ra7", "class": "ok"}),
    ):
        llm, _ = _run(fens, FREEFORM, verdict=verdict)
        assert llm.calls, f"{label}: no LLM call to inspect"
        system, user = llm.calls[0]["system"].lower(), llm.calls[0]["user"].lower()
        assert "there is no 'you' here" in system, f"{label}: the freeform perspective block is missing"
        assert "address the player as 'you'" not in system, f"{label}: the DRILL perspective was installed"
        # The system prompt is asserted by WHICH block it carries, not by scanning for "you": the
        # freeform block necessarily contains the words it bans ("never say 'your opponent'"), so a
        # substring scan flags the instruction as the violation. The user prompt carries no such
        # instructions, so there it IS a straight scan — and it is the half that named the player.
        for banned in ("you played", "your opponent", "the player is playing"):
            assert banned not in user, f"{label}: the context said {banned!r}"


def test_every_drill_path_still_says_you():
    """Drills DO have a you: the engine plays the other side. This is the proof that the narration work
    did not leak into them."""
    for label, result in (("correct", DRILL_CORRECT), ("wrong", DRILL_WRONG),
                          ("solved", DRILL_SOLVED), ("suspended", SUSPENDED)):
        llm, _ = _run(_fens("e2e4"), result, verdict={"san": "e4", "class": "ok"})
        assert llm.calls, f"{label}: no LLM call — a drill move must always be coached"
        blob = llm.calls[0]["system"].lower()
        assert "address the player as 'you'" in blob, f"{label}: lost the drill voice"
        assert "opening book" not in blob, f"{label}: a drill move got the narration voice"


def test_a_suspended_drill_is_never_narrated():
    """The tri-state bug, pinned. `drill: "suspended"` is not False, so it is not freeform: the player
    is exploring their OWN drill line and the engine resumes replying when they return to it. Reading
    it as freeform would both narrate at them and drop the "you"."""
    llm, _ = _run(_fens("e2e4"), SUSPENDED, verdict={"san": "e4", "class": "ok"})
    assert "opening book" not in llm.calls[0]["system"]


def test_a_suspended_drill_is_not_told_it_is_freeform():
    """The system prompt and the CONTEXT head must agree on which mode this is.

    The tri-state has three cases and only two had branches: `is True` and `is False`. Suspended got the
    DRILL voice from the system prompt (correct) and then fell through the head's if/elif chain into
    "CONTEXT: FREEFORM move (no drill)" — the same prompt telling the model both things at once. Voice
    is only half of it; the CONTEXT has to match.
    """
    llm, _ = _run(_fens("e2e4"), SUSPENDED, verdict={"san": "e4", "class": "ok"})
    user = llm.calls[0]["user"]
    assert "FREEFORM move" not in user, "a suspended drill was announced to the model as freeform"
    assert "SUSPENDED" in user and "You played" in user, \
        "a suspended drill lost the drill framing in its context head"


def test_no_freeform_prompt_tells_the_model_to_say_you():
    """The whole system prompt, not just the perspective block.

    The perspective block is the OBVIOUS place the voice lives, so it is the place that gets fixed —
    and the fix stops there. The instruction that actually leaked was the "when you are unsure" line in
    the shared tail, which names a loser: "the position turns against you", sitting directly under a
    block that had just banned "you". It is also the line the model reaches for exactly when it is
    least sure, which makes it the MOST likely thing to be said, not the least. Scan the whole prompt.
    """
    # Scanned WITHOUT the perspective block: that block is the one place allowed to name the phrases
    # it bans ("never say 'your opponent'"), so including it makes the instruction look like the
    # violation. Everything AROUND it must be voice-neutral on its own — which is exactly the property
    # that was broken, and exactly what a scan of the whole string cannot see.
    around = (_MOVE_SYSTEM_HEAD + _move_system_tail(True)).lower()
    assert _perspective(True) not in around, "fixture drifted: the perspective block must be excluded here"
    for banned in ("against you", "you played", "your opponent", "the player's colour"):
        assert banned not in around, f"the freeform move prompt still instructs the model to say {banned!r}"
    assert "against you" in _move_system(False).lower(), \
        "the drill prompt lost its second-person fallback — drills DO have a you"


# Phrases that ADDRESS THE PLAYER. Not a bare "you" scan: these prompts say "you" to the MODEL all the
# time ("you may use your own knowledge", "you are a chess coach"), which is fine and unavoidable. What
# must never appear on a freeform path is a second person who is meant to be the player.
_ADDRESSES_THE_PLAYER = ("against you", "you played", "your opponent", "the player's colour",
                         "you are playing", "the player is playing", "your pieces", "your king")


def test_no_freeform_system_prompt_anywhere_addresses_a_player():
    """The generalisation of the finding above, applied to EVERY prompt a freeform turn can reach.

    Codex found the leak in the move prompt's tail. The useful lesson was not "fix that line" but "find
    every line shaped like that line" — the same voice ships from four prompt bodies, and the one that
    leaked was the one nobody thought of as a voice string at all. So this sweeps all of them by
    construction: a new freeform prompt is caught the moment it is added here.
    """
    for label, prompt in (
        ("move", _MOVE_SYSTEM_HEAD + _move_system_tail(True)),
        ("coach", _COACH_SYSTEM_HEAD + _COACH_SYSTEM_TAIL),
        ("narrate", _NARRATE_SYSTEM_HEAD + _NARRATE_SYSTEM_TAIL),
        ("endbook", _ENDBOOK_SYSTEM.replace(_perspective(True), "")),
    ):
        low = prompt.lower()
        for banned in _ADDRESSES_THE_PLAYER:
            assert banned not in low, f"the freeform {label} prompt addresses the player: {banned!r}"


def test_narration_titles_the_session_after_the_opening_family():
    """Deterministic, not asked of the model — move one is the most likely titling moment, and a JSON
    that omitted `name` would silently stop titling forever."""
    _, store = _run(_fens("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"), FREEFORM,
                    verdict={"san": "Bb5", "class": "ok"})
    assert store.named == "Ruy Lopez", f"expected the family as the title, got {store.named!r}"


# -- 5. the gambit, against the real engine ------------------------------------------------------

def _have_stockfish() -> bool:
    import os
    import shutil
    return bool(os.environ.get("LUCENA_STOCKFISH")) or bool(shutil.which("stockfish"))


@pytest.mark.engine
@pytest.mark.skipif(not _have_stockfish(), reason="no stockfish")
def test_the_kings_gambit_actually_triggers_the_swing_fork():
    """The feature's premise, measured through the REAL evaluate — not a reimplementation of it.

    The swing fork reuses the engine's own `classify` boundaries instead of inventing a threshold, on
    the theory that a BOOK move the engine classes as dubious-or-worse is a line that deliberately
    concedes something. That theory is worthless if the canonical example doesn't cross it. It does:
    2.f4 measures ~-8 win% against the engine's preference, which is `dubious` — so the offer narrates
    with its class intact ("what does White get for the pawn?").

    Driven through `ToolContext.evaluate` on purpose. An earlier version of this measurement
    reimplemented `classify` in the test, got the sign of the played-move score backwards, and
    confidently reported the exact opposite result. The real call path is the only oracle.
    """
    import tempfile

    from lucena_engine import Engine
    from lucena_backend.state import StateStore
    from lucena_backend.tools import ToolContext

    b = Board(START)
    with Engine(threads=1) as e:
        e.new_game()
        ctx = ToolContext(e, StateStore(tempfile.mkdtemp()), limit={"movetime_ms": 1200})
        classes = {}
        for u in ("e2e4", "e7e5", "f2f4", "e5f4"):
            classes[u] = (ctx.evaluate(b.fen, [b.san(u)]) or {}).get("class")
            b = b.apply(u)

    fens = _fens("e2e4", "e7e5", "f2f4")
    assert _book_route(fens, swing=False)[1] == "King's Gambit", "2.f4 must be in the book"
    assert _is_swing({"class": classes["f2f4"]}), (
        f"2.f4 classed {classes['f2f4']!r} — it no longer crosses the engine's own dubious boundary, so "
        f"the swing fork never fires on the canonical gambit and _SWING_CLASSES needs revisiting"
    )
    route, name, _ = _book_route(fens, swing=True)
    assert (route, name) == (_NARRATE, "King's Gambit")


def test_the_narration_is_told_exactly_which_moves_exist():
    """Caught end-to-end on the first real narration: given only 1.e4, the model wrote "Black responds
    with e5, a classic challenge…" — theory reported as HISTORY, for a move nobody had played.

    It is the carve-out's natural failure mode rather than a random miss: permission to explain an
    opening reads, from inside the model, as permission to recite its mainline, and the difference is
    one word ("usually answers" vs "responds"). The prompt-side fix is a tense rule; the fact-side fix
    is this — name the whole line and say that it IS the whole line, so the rule has something to be
    checked against instead of being aspirational.
    """
    llm, _ = _run(_fens("e2e4"), FREEFORM, verdict={"san": "e4", "class": "ok"})
    user, system = llm.calls[0]["user"], llm.calls[0]["system"]
    assert "MOVES SO FAR" in user and "1.e4" in user, "the model was not told what has been played"
    assert "ENTIRE game" in user, "nothing told the model the line was complete, so a continuation reads as fair game"
    assert "NEVER write a later move as though it happened" in system, "the tense rule is missing"


def test_the_played_line_is_numbered_from_the_history():
    from lucena_backend.orchestrator import Orchestrator
    fens = _fens("e2e4", "c7c5", "g1f3")
    store = _FakeStore(fens, _sans_for(fens))
    orch = Orchestrator(ctx=_FakeCtx(store), model="stub", llm=_CapturingLLM(), ground_ctx=_FakeGround())
    assert orch._played_line() == "1.e4 c5 2.Nf3"
