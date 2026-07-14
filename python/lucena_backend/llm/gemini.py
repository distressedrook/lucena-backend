"""GeminiAdapter — wraps google-genai behind the generic LLMAdapter interface.

Retry-with-backoff on 429/503 lives here, fail-fast bounded (the ADK-hang lesson).
The client is built lazily (needs the API key at construction) and can be injected
for tests. google-genai is imported lazily so the package doesn't hard-require it.
"""

from __future__ import annotations

import json

from .interface import Message, GenerateOptions, Usage, Completion


class GeminiAdapter:
    def __init__(self, *, default_model: str, client=None,
                 attempts: int = 3, initial_delay: float = 1.0, max_delay: float = 8.0,
                 exp_base: float = 2.0, jitter: float = 1.0):
        self._default_model = default_model
        self._client = client                 # injectable (tests); else built lazily
        self._retry = dict(attempts=attempts, initial_delay=initial_delay,
                           max_delay=max_delay, exp_base=exp_base, jitter=jitter,
                           http_status_codes=[429, 503])

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(**self._retry)))
        return self._client

    async def generate(self, messages: list[Message], opts: GenerateOptions) -> Completion:
        from google.genai import types

        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        turns = [m for m in messages if m.role != "system"]
        contents = _to_contents(turns, types)

        cfg_kwargs = dict(
            system_instruction=system,
            max_output_tokens=opts.max_tokens,
            temperature=opts.temperature,
        )
        if opts.schema is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
        cfg = types.GenerateContentConfig(**cfg_kwargs)

        model = opts.model or self._default_model
        resp = await self._get_client().aio.models.generate_content(
            model=model, contents=contents, config=cfg)

        text = (resp.text or "").strip()
        parsed = None
        if opts.schema is not None:
            try:
                parsed = json.loads(text or "{}")
            except ValueError:
                parsed = None   # caller falls back to raw text
        return Completion(text=text, json=parsed, model=model, usage=_usage(resp))


def _to_contents(turns: list[Message], types):
    """user/assistant messages → Gemini contents. A single user turn passes as a plain
    string (the common hot-path case); multi-turn builds Content(role, parts)."""
    if len(turns) == 1 and turns[0].role == "user":
        return turns[0].content
    role_map = {"user": "user", "assistant": "model"}
    return [
        types.Content(role=role_map.get(m.role, "user"), parts=[types.Part(text=m.content)])
        for m in turns
    ]


def _usage(resp) -> Usage:
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return Usage()
    return Usage(
        input_tokens=getattr(um, "prompt_token_count", None),
        output_tokens=getattr(um, "candidates_token_count", None),
        total_tokens=getattr(um, "total_token_count", None),
    )
