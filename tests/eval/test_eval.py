"""Pytest entry for the eval harness — SKIPPED by default (LLM-in-the-loop, slow, quota-bound).

Runs only with RUN_EVAL=1 and a key present, mirroring the existing "end-to-end coaching tests need
GEMINI_API_KEY" convention. In normal CI this is a no-op; a nightly/pre-release job sets RUN_EVAL=1.

The deterministic graders ALSO deserve pure unit tests (no LLM) so the harness itself is trustworthy —
that TODO is called out below; those would run on every push.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EVAL") != "1"
    or not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="eval harness: set RUN_EVAL=1 and a Gemini key (nightly/pre-release only)",
)


def test_verdict_gloss_eval():
    from .run_eval import _run
    import asyncio
    assert asyncio.run(_run(quick=False)) == 0, "verdict-gloss eval gate failed — see eval_results.json"


# TODO(sketch): pure-unit tests for graders.deterministic_graders on hand-written strings (a leaked
# solution, a '- THE REFUTATION —' label, a 'your Black' perspective slip) — NO LLM, run every push,
# so a change to the graders can't silently stop catching regressions. These are the cheap, always-on
# half and should land first.
