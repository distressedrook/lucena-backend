"""HandlerBase LLM plumbing — a rate-limit (429) or any LLM outage must DEGRADE, never break the turn
(a crash flashes the app's 'Uh oh' and eats a solve). `_gen_json` swallows the error and returns {} so
every caller falls back to its grounded text."""
import asyncio

from lucena_backend.coaching.handler_base import HandlerBase


class _RaisingLLM:
    async def generate(self, messages, opts):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, retry in 48s")


class _OkLLM:
    async def generate(self, messages, opts):
        from lucena_backend.llm import Completion
        return Completion(text='{"text": "hi"}', json={"text": "hi"}, model="stub", usage=None)


def test_gen_json_degrades_to_empty_on_llm_error():
    h = HandlerBase(ctx=object(), store=object(), llm=_RaisingLLM(), model="stub")
    assert asyncio.run(h._gen_json("sys", "usr")) == {}     # no exception, empty → callers fall back


def test_gen_json_returns_parsed_json_on_success():
    h = HandlerBase(ctx=object(), store=object(), llm=_OkLLM(), model="stub")
    assert asyncio.run(h._gen_json("sys", "usr")) == {"text": "hi"}
