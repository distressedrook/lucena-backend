"""Provider selection is configuration. The orchestrator calls make_adapter(config)
and depends only on the returned LLMAdapter — switching Gemini → OpenRouter is a
config change, not code.
"""

from __future__ import annotations

from .interface import LLMAdapter


def make_adapter(config: dict) -> LLMAdapter:
    """config: {provider: "gemini"|"openrouter", default_model: str, ...}."""
    provider = (config or {}).get("provider", "gemini")
    if provider == "gemini":
        from .gemini import GeminiAdapter
        return GeminiAdapter(default_model=config["default_model"])
    if provider == "openrouter":
        raise NotImplementedError("OpenRouterAdapter is a later drop-in")
    raise ValueError(f"unknown LLM provider: {provider!r}")
