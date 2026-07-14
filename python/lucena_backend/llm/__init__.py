"""Generic LLM adapter — the orchestrator depends ONLY on this interface, never on a
provider SDK. Provider + model are configuration (see make_adapter)."""

from .interface import Message, GenerateOptions, Usage, Completion, LLMAdapter
from .factory import make_adapter

__all__ = [
    "Message", "GenerateOptions", "Usage", "Completion", "LLMAdapter", "make_adapter",
]
