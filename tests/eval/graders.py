"""SKETCH — verdict-gloss graders. Not run yet; not wired into CI.

Two kinds:
  • deterministic_graders(...) — cheap, no LLM, hard-fail on any hit.
  • judge_unsupported(...)      — one LLM call to a STRONGER model; returns claims not entailed by the
    facts (the causal-gloss check). Sample it J times and majority-vote in the runner.

Reuses the production grounding helpers so a grader can never drift from what the app enforces.
"""
from __future__ import annotations

import re

from lucena_backend.coaching.grounding import _invented_moves, _move_tokens

# --- deterministic graders --------------------------------------------------------------------------

# A bullet that opens with a taxonomy label — the presentation regression ('- THE REFUTATION —',
# '- The point:'). Flags an ALL-CAPS lead run, or a Titled label followed by ':' / ' —'.
_LABEL_PREFIX = re.compile(
    r"^\s*[-*]\s*(?:[A-Z][A-Z ]{2,}\s*[—:-]"           # '- THE REFUTATION —'
    r"|(?:The|Why|What)\b[^:\n]{0,24}:)",              # '- The point:' / '- Why it works:'
    re.MULTILINE,
)
_EVAL_NUMBERS = re.compile(r"\b\d{1,3}\s?%|\bcentipawn|\b[+-]\d+(?:\.\d+)?\b|\bcp\b", re.IGNORECASE)


def _bullets(text: str) -> list[str]:
    return [ln for ln in (text or "").splitlines() if re.match(r"\s*[-*]\s+", ln)]


def deterministic_graders(text: str, fx: dict) -> list[str]:
    """Every deterministic failure for one generated verdict. Empty list == clean."""
    fails: list[str] = []
    facts, played = fx.get("facts", ""), fx.get("san")

    invented = _invented_moves(text, facts, played)
    if invented:
        fails.append(f"invented_moves: {sorted(invented)}")

    # A WRONG verdict must never name the solution.
    if not fx.get("correct"):
        leaked = [s for s in (fx.get("solution_sans") or []) if s and s in text]
        if leaked:
            fails.append(f"solution_leak: {leaked}")

    if _LABEL_PREFIX.search(text):
        fails.append("label_prefix: a bullet opens with a taxonomy label")

    if _EVAL_NUMBERS.search(text):
        fails.append("eval_numbers: win%/cp/eval leaked into prose")

    # Shape: a lead prose line, then (optionally) a bulleted list — the first non-empty line is prose.
    first = next((ln for ln in (text or "").splitlines() if ln.strip()), "")
    if re.match(r"\s*[-*]\s+", first):
        fails.append("format_shape: starts with a bullet, missing the lead sentence")

    exp = fx.get("expect") or {}
    for bad in exp.get("must_not_contain") or []:
        if bad in text:
            fails.append(f"must_not_contain: '{bad}' present")
    any_of = exp.get("must_mention_any") or []
    if any_of and not any(m in text for m in any_of):
        fails.append(f"must_mention_any: none of {any_of} present")

    # Perspective smell test: the player's colour is 'you'; the OTHER colour must not be called 'your'.
    other = {"white": "Black", "black": "White"}.get((exp.get("color") or "").lower())
    if other and re.search(rf"\byour {other}\b", text, re.IGNORECASE):
        fails.append(f"perspective: '{other}' attributed to the player ('your {other}')")

    return fails


# --- entailment judge (LLM) -------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict grounding auditor for a chess coaching app. You are given FACTS (the ONLY ground "
    "truth the coach was allowed to use) and a VERDICT (what the coach wrote). Find every claim in the "
    "VERDICT that is NOT entailed by the FACTS.\n"
    "RULES:\n"
    "- The FACTS are the only source of truth. Do NOT use your own chess knowledge. A claim can be true "
    "in real chess yet UNSUPPORTED if the FACTS do not state or directly imply it — flag it (the coach "
    "is required to interpret handed-over facts, not recall).\n"
    "- SUPPORTED = the FACTS state it, or it is a trivial paraphrase/restatement.\n"
    "- FLAG: invented moves; invented CAUSAL mechanisms ('by removing the pawn that supports X' when the "
    "facts never say that); invented threats/motifs; invented evaluations; a claim attributed to the "
    "wrong colour.\n"
    "- Do NOT flag: warmth, encouragement, hedging, formatting, or a fact restated in other words.\n"
    "- Naming the move played is fine.\n"
    "Return JSON only: {\"unsupported\": [{\"claim\": \"<verbatim span>\", \"why\": \"<why not entailed>\"}], "
    "\"verdict\": \"PASS\" | \"FAIL\"}. PASS iff unsupported is empty."
)


def judge_prompt(facts: str, verdict_text: str) -> str:
    return f"FACTS:\n{facts}\n\nVERDICT:\n{verdict_text}\n\nReturn JSON."


async def judge_unsupported(llm, judge_model: str, facts: str, verdict_text: str) -> list[dict]:
    """One judge call → its list of unsupported claims (empty == PASS). The runner samples this J times
    and keeps a claim only if a majority of judges flag it (damps judge flakiness)."""
    from lucena_backend.llm.interface import GenerateOptions, Message
    comp = await llm.generate(
        [Message("system", JUDGE_SYSTEM), Message("user", judge_prompt(facts, verdict_text))],
        GenerateOptions(model=judge_model, max_tokens=500, temperature=0.0),
    )
    return ((comp.json or {}).get("unsupported")) or []
