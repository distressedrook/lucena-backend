"""SKETCH — the verdict-gloss eval runner. NOT wired into CI; NOT run yet.

  PYTHONPATH=python .venv/bin/python -m tests.eval.run_eval [--quick] [--live-facts]

Per fixture: generate G verdict samples → deterministic graders (hard) → judge each J times
(majority vote) → aggregate rates → print report → write eval_results.json → exit nonzero if a gate
fails. Skips cleanly (exit 0, prints a notice) when no API key is present, so a CI job can be a no-op
off-key rather than a red X.

Design + open decisions: see README.md. Numbers below (G/J/thresholds, judge model) are PLACEHOLDERS
pending your call — do not treat them as final.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib

from .graders import deterministic_graders, judge_unsupported

HERE = pathlib.Path(__file__).parent

# --- knobs (PLACEHOLDER — confirm before trusting) --------------------------------------------------
GEN_MODEL = "gemini-flash-lite-latest"    # what actually ships — eval the real generator
JUDGE_MODEL = "gemini-flash-latest"       # a STRONGER tier; independence is an open decision (README #1)
G_SAMPLES, J_SAMPLES = 3, 3               # generations per fixture, judges per generation
UNSUPPORTED_RATE_GATE = 0.15              # fraction of samples allowed to carry an unsupported claim


async def _one_fixture(gen_llm, judge_llm, fx: dict, g: int, j: int) -> dict:
    from lucena_backend.coaching.mode_prompts import VerdictPrompt
    from lucena_backend.llm.interface import GenerateOptions, Message

    sys = VerdictPrompt.system(correct=fx["correct"], player_color=fx.get("player_color"))
    usr = VerdictPrompt.prompt(attempt=fx["san"], facts=fx["facts"])

    det_fail_samples, unsupported_samples, texts = 0, 0, []
    for _ in range(g):
        comp = await gen_llm.generate(
            [Message("system", sys), Message("user", usr)],
            GenerateOptions(model=GEN_MODEL, max_tokens=400, temperature=0.4),
        )
        text = (comp.json or {}).get("text") or comp.text or ""
        texts.append(text)

        det = deterministic_graders(text, fx)
        if det:
            det_fail_samples += 1

        # J judges → majority vote per claim (keyed by verbatim claim span).
        votes: dict[str, int] = {}
        for _ in range(j):
            for c in await judge_unsupported(judge_llm, JUDGE_MODEL, fx["facts"], text):
                votes[c.get("claim", "")] = votes.get(c.get("claim", ""), 0) + 1
        if any(v >= (j // 2 + 1) for v in votes.values()):
            unsupported_samples += 1

    return {
        "id": fx["id"],
        "det_fail_rate": det_fail_samples / g,
        "unsupported_rate": unsupported_samples / g,
        "sample_texts": texts,
        "last_det_fails": deterministic_graders(texts[-1], fx) if texts else [],
    }


async def _run(quick: bool) -> int:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("[eval] no GEMINI_API_KEY — skipping (this is a no-op, not a failure).")
        return 0

    from lucena_backend.llm.gemini import GeminiAdapter
    gen_llm = GeminiAdapter(default_model=GEN_MODEL)
    judge_llm = GeminiAdapter(default_model=JUDGE_MODEL)
    g, j = (1, 1) if quick else (G_SAMPLES, J_SAMPLES)

    data = json.loads((HERE / "fixtures.json").read_text())
    fixtures = [f for f in data["fixtures"] if not str(f.get("fen", "")).startswith("PLACEHOLDER")]

    results = [await _one_fixture(gen_llm, judge_llm, fx, g, j) for fx in fixtures]

    print(f"\n{'fixture':32} {'det_fail':>9} {'unsupported':>12}")
    det_gate = unsup_gate = True
    for r in results:
        print(f"{r['id']:32} {r['det_fail_rate']:>9.0%} {r['unsupported_rate']:>12.0%}"
              + (f"   {r['last_det_fails']}" if r["last_det_fails"] else ""))
        if r["det_fail_rate"] > 0:
            det_gate = False
        if r["unsupported_rate"] > UNSUPPORTED_RATE_GATE:
            unsup_gate = False

    (HERE / "eval_results.json").write_text(json.dumps(results, indent=2))
    ok = det_gate and unsup_gate
    print(f"\nGATES: deterministic={'PASS' if det_gate else 'FAIL'}  "
          f"entailment(<= {UNSUPPORTED_RATE_GATE:.0%})={'PASS' if unsup_gate else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="G=J=1 for fast iteration")
    ap.add_argument("--live-facts", action="store_true", help="(TODO) build facts via _move_facts+engine")
    args = ap.parse_args()
    if args.live_facts:
        raise SystemExit("--live-facts not implemented in the sketch (see README 'Facts: frozen vs live')")
    return asyncio.run(_run(args.quick))


if __name__ == "__main__":
    raise SystemExit(main())
