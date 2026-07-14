"""Deterministic instant-feedback beats for the app-driven drill walk.

The app owns the tree walk and posts a drill event (started / solved / wrong / new_line /
finished); the MCP turns that event into the matching feedback beat and pushes it on the beats
channel. So the app stays a pure renderer, the feedback text lives in ONE place, and — crucially —
it is never authored by the LLM, so it can't hallucinate (generic, no board claims). The coach's
*grounded* WHY still comes separately, on demand, via `push_beat`.

`started` produces no beat — the coach sets the scene. Every other event does.
"""

from __future__ import annotations

_CORRECT = [
    "That's right! Want to know more? Type explain.",
    "Yes — that's the move. Type explain to hear why.",
    "Correct! Curious what makes it work? Type explain.",
    "Nailed it. Type explain for the reasoning.",
]
_WRONG = [
    "That's not quite the right move. Type ask why below, or hit Retry.",
    "Not this one. Want to know why? Type ask why, or hit Retry.",
    "Close, but not it. Type ask why below, or hit Retry.",
    "Hmm, not quite. Type ask why below, or hit Retry.",
]
_BACKTRACK = [
    "That line's done — this time the opponent answers {d}. Your move.",
    "One defense down. Now the opponent tries {d} instead. How do you refute it?",
    "That branch is handled — the opponent plays {d} here. Find the win.",
    "That line's refuted. This time it's {d}. What's your answer?",
]


def _beat(tone: str, text: str) -> dict:
    return {"kind": "say", "stops": False, "tone": tone, "segments": [{"text": text}]}


def correct_beat(i: int) -> dict:
    return _beat("praise", _CORRECT[i % len(_CORRECT)])


def wrong_beat(i: int) -> dict:
    return _beat("correct", _WRONG[i % len(_WRONG)])


def backtrack_beat(i: int, defense: str | None) -> dict:
    return _beat("teach", _BACKTRACK[i % len(_BACKTRACK)].format(d=defense or "a new try"))


def finish_beat(tree: dict | None) -> dict:
    """The wrap-up verdict, from the tree the MCP built (the winning move + result)."""
    root = (tree or {}).get("root", {}) or {}
    first = root.get("expect_san")
    if not first:
        opts = root.get("options") or []
        first = opts[0].get("san") if opts else None
    mate = root.get("mate_in")
    if mate and first:
        text = f"Solved! {first} forces mate in {mate}. You refuted every defense — clean."
    elif first:
        text = f"Solved! {first} was the key — it wins, and you refuted every defense."
    else:
        text = "Solved — you refuted every defense. Well done."
    return _beat("verdict", text)
