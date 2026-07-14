"""CI gate (llm-adapter.md rule 1): the orchestrator imports the interface, never a
provider SDK. No `google.genai` / `openai` import outside the llm/ adapter package."""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent / "python" / "lucena_backend"
_FORBIDDEN = ("google.genai", "from google import genai", "import openai")


def test_no_provider_sdk_outside_llm_package():
    offenders = []
    for p in _ROOT.rglob("*.py"):
        if "llm" in p.relative_to(_ROOT).parts:
            continue                       # the adapter package is the ONLY place providers live
        text = p.read_text(encoding="utf-8")
        if any(tok in text for tok in _FORBIDDEN):
            offenders.append(str(p.relative_to(_ROOT)))
    assert not offenders, f"provider SDK imported outside llm/: {offenders}"
