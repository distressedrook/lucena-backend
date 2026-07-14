"""B2: the generic LLM adapter. Verifies the Gemini mapping (messages → contents +
system_instruction, schema → json mode, usage extraction) with a stubbed client —
no network, no API key. The live call was validated separately (P0)."""

import asyncio

import pytest

from lucena_backend.llm import Message, GenerateOptions
from lucena_backend.llm.gemini import GeminiAdapter

pytest.importorskip("google.genai")   # needs the SDK for types (not a network call)


class _Usage:
    prompt_token_count = 10
    candidates_token_count = 8
    total_token_count = 18


class _Resp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = _Usage()


class _Models:
    def __init__(self, text):
        self._text = text
        self.calls = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _Resp(self._text)


class _Client:
    def __init__(self, text):
        self.models = _Models(text)
        self.aio = self   # adapter calls client.aio.models.generate_content


def test_single_user_turn_is_a_plain_string_and_system_extracted():
    client = _Client('{"mode":"ask","text":"What does Black threaten?"}')
    ad = GeminiAdapter(default_model="gemini-flash-lite-latest", client=client)
    msgs = [Message("system", "You are a coach"), Message("user", "Position: ...")]
    comp = asyncio.run(ad.generate(msgs, GenerateOptions(schema={"type": "object"}, max_tokens=200)))

    call = client.models.calls[0]
    assert call["model"] == "gemini-flash-lite-latest"
    assert call["contents"] == "Position: ..."                 # single user turn → string
    assert call["config"].system_instruction == "You are a coach"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].max_output_tokens == 200


def test_schema_parses_json_and_usage_maps():
    client = _Client('{"mode":"tell","text":"Nice move."}')
    ad = GeminiAdapter(default_model="m", client=client)
    comp = asyncio.run(ad.generate([Message("user", "x")], GenerateOptions(schema={"type": "object"})))
    assert comp.json == {"mode": "tell", "text": "Nice move."}
    assert comp.text.startswith("{")
    assert comp.model == "m"
    assert comp.usage.input_tokens == 10 and comp.usage.output_tokens == 8
    assert comp.usage.total_tokens == 18


def test_no_schema_is_plain_text_no_json():
    client = _Client("Just prose.")
    ad = GeminiAdapter(default_model="m", client=client)
    comp = asyncio.run(ad.generate([Message("user", "x")], GenerateOptions()))
    call = client.models.calls[0]
    assert getattr(call["config"], "response_mime_type", None) in (None, "")
    assert comp.json is None
    assert comp.text == "Just prose."


def test_multi_turn_builds_content_list_with_model_role():
    client = _Client("ok")
    ad = GeminiAdapter(default_model="m", client=client)
    msgs = [Message("user", "hi"), Message("assistant", "hello"), Message("user", "more")]
    asyncio.run(ad.generate(msgs, GenerateOptions()))
    contents = client.models.calls[0]["contents"]
    assert isinstance(contents, list) and len(contents) == 3
    assert [c.role for c in contents] == ["user", "model", "user"]


def test_opts_model_overrides_default():
    client = _Client("ok")
    ad = GeminiAdapter(default_model="default-m", client=client)
    asyncio.run(ad.generate([Message("user", "x")], GenerateOptions(model="override-m")))
    assert client.models.calls[0]["model"] == "override-m"
