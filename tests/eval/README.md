# Verdict-gloss eval harness (SKETCH — not wired into CI, not run yet)

## Why this exists

The unit tests (`test_prompts_spine`, `test_grounding_tiers`) assert on prompt *structure* and
grounding *functions* — they deliberately never assert on generated text ("did it write a good
paragraph" is ungroundable). That leaves the actual failure mode unguarded: **the facts are grounded,
but the model's connective gloss isn't.** Real example caught live, correct-move verdict:

> "It neutralizes Black's threat of Rxc3+ *by removing the pawn that supports the attack on your rook.*"

The facts were correct. The **causal clause is fabricated** — Rxh3 neutralises Rxc3+ by moving the rook
off c3, not by removing a pawn. A learner reads it as truth. This is the exact thing that undercuts
"engine-grounded, trustworthy coach", and it only surfaces today when a human tests live.

This harness turns "I noticed it live" into "the eval caught it."

## What it grades

Verdicts are non-deterministic (temp 0.4, no seed), so we grade **properties, not exact strings**, in
two layers:

### Layer 1 — deterministic graders (cheap, no extra LLM call)
Run on the generated text. Hard-fail the fixture on any hit.
- `invented_moves` — reuse `grounding._invented_moves(text, facts, played)`; must be empty.
- `solution_leak` — for a WRONG verdict, no solution SAN (from the tree) appears.
- `label_prefix` — no bullet starts with a taxonomy label (`THE REFUTATION`, `The point:`, `Why it
  works:`) — the presentation regression we just fixed.
- `format_shape` — a lead prose line, blank line, then `- ` bullets (or lead-only).
- `eval_numbers` — no win%, centipawns, `+3.4`.
- fixture-scoped `must_not_contain` / `must_mention_any` (e.g. Rc4 must NOT say "Rxc3", must mention
  the real refutation "Rb4").

### Layer 2 — entailment judge (the crux; one LLM call per sample)
A **stronger** model than the generator is handed the FACTS (the only ground truth) and the VERDICT,
and returns every claim NOT entailed by the facts. Critically it is told **not to use its own chess
knowledge** — a claim can be chess-true yet UNSUPPORTED if the facts don't state it, and per the
calculate-vs-interpret invariant that is still a leak (the model got lucky). This is what catches the
"removing the pawn that supports the attack" class.

To damp judge flakiness: J judge samples per verdict, majority vote; a claim counts only if ≥⌈J/2⌉
judges flag it.

## Non-determinism → sample and aggregate

Per fixture: generate **G** verdict samples; judge each **J** times. Report the *rate*, not a single
pass/fail — a solution leak in 1/5 samples is still a real defect. Suggested: `G=3, J=3` for the full
run (~12 calls/fixture), `G=1, J=1` for `--quick` iteration.

## Facts: frozen vs live

Two modes, because they answer different questions:
- **frozen (default)** — the fixture carries a pinned `facts` string; the eval isolates the
  **prompt + LLM** (the gloss). Reproducible; a failure means the *wording/model* slipped, not the
  engine.
- **--live-facts** — build facts via `CoachHandler._move_facts` + the real engine; tests the whole
  pipeline end-to-end. Slower, needs Stockfish; use pre-release.

Freeze by running the harness once in live mode and snapshotting the facts into the fixture (a
`--freeze` helper), reviewed by a human so a bad fact can't become the golden truth.

## Gates (what fails CI)

- **Deterministic:** ZERO failures across all samples. Non-negotiable (these are correctness/safety:
  spoilers, illegal moves, perspective).
- **Entailment:** unsupported-claim rate ≤ threshold `T` (start generous, e.g. `T=0.15` of samples,
  ratchet down as the prompt improves). Judge is imperfect, so this is a *budget*, not zero.
- Writes `eval_results.json` (per-fixture rates) for trend tracking.

## Where it runs (NOT per-commit)

LLM-in-the-loop → slow, costs tokens, flaky, quota-bound (see the `gemini-2.0-flash-lite` `limit:0`
gotcha — generator model is `gemini-flash-lite-latest`, judge should be a stronger tier). So:
- `@pytest.mark.eval`, skipped unless `RUN_EVAL=1` **and** a key is present — same pattern as the
  existing "end-to-end coaching tests need `GEMINI_API_KEY`".
- Nightly job + a pre-release gate + on-demand (`python -m tests.eval.run_eval`). NOT on every push.
- Optional: run automatically when a prompt file (`mode_prompts.py`, `prompts.py`, `grounding.py`)
  changes, since those are exactly what it protects.

## Files

- `fixtures.json`   — the golden set (seed: 3 fixtures; grow to ~15–20 covering wrong/right, stalemate,
  fork, created-threat, promotion, opening narration).
- `graders.py`      — deterministic graders + the entailment-judge prompt & call.
- `run_eval.py`     — the runner: generate → grade → judge → aggregate → report → gate.

## Open decisions (need your call)

1. **Judge model** — a stronger tier than the generator (`gemini-flash-latest`? a non-Gemini model for
   independence?). Independence matters: judging flash-lite with flash-lite risks shared blind spots.
2. **Thresholds** `T`, and `G`/`J` sample counts (cost vs signal).
3. **Frozen vs live default** for the CI gate (I lean frozen for a clean signal + a separate weekly
   live run).
4. **Scope** — verdicts only to start, or also freeform reads / opening narration (same harness,
   more fixtures)?
