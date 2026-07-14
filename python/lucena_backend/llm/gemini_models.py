"""Resolve a Gemini model this API key can actually call.

New keys list many pinned models but gate them ("not available to new users").
We probe candidates (best coaching-tier first) + rolling aliases and return the
first that generateContent succeeds on. Mirrors tools/p0_pick_model.py.
"""
from __future__ import annotations

import os

_CANDIDATES = [
    "gemini-3-pro-preview", "gemini-3-pro", "gemini-pro-latest",
    "gemini-3-flash-preview", "gemini-3-flash", "gemini-flash-latest",
    "gemini-2.5-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-001",
]


def resolve_model(preferred: str | None = None) -> str:
    """Return a callable model id. Honors `preferred` (or LUCENA_GEMINI_MODEL) if
    it works, else probes candidates + the key's rolling/gemini-3 models."""
    from google import genai
    from google.genai import types

    preferred = preferred or os.environ.get("LUCENA_GEMINI_MODEL") or ""
    client = genai.Client()

    def works(name: str) -> bool:
        try:
            client.models.generate_content(
                model=name, contents="hi",
                config=types.GenerateContentConfig(max_output_tokens=1))
            return True
        except Exception:
            return False

    if preferred and works(preferred):
        return preferred

    candidates = list(_CANDIDATES)
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            if ("generateContent" in (m.supported_actions or [])
                    and ("latest" in name or name.startswith("gemini-3"))
                    and name not in candidates):
                candidates.append(name)
    except Exception:
        pass
    for name in candidates:
        if works(name):
            return name
    raise RuntimeError("No callable Gemini model for this API key "
                       "(check billing / API restrictions / org policy).")
