"""The whole surface the orchestrator sees: messages in, one completion out.
OpenAI-chat-shaped so an OpenRouter adapter is a drop-in. No tools, no streaming,
no provider types leaking out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Message:
    role: str            # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerateOptions:
    model: str | None = None       # provider model id; None → the configured default
    schema: dict | None = None     # JSON Schema → force structured JSON output
    temperature: float = 0.4
    max_tokens: int = 400
    timeout_s: float = 30.0


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class Completion:
    text: str                      # raw text (or the JSON string when schema is set)
    json: dict | None              # parsed object when schema is set and parseable, else None
    model: str                     # the model actually used
    usage: Usage


class LLMAdapter(Protocol):
    async def generate(self, messages: list[Message], opts: GenerateOptions) -> Completion:
        ...
