"""The reasoning layer — "our own XAI".

Derive the causal POINT of a move DETERMINISTICALLY from the board, one tactical motif at a time, so
the coach explains from grounded reasoning that the LLM only verbalizes — never narrates or invents.
This is deliberately its own project, not part of the grounding blob: each motif is its own module
(`undermine`, and next `deflection`, `fork`, `pin`, …) with a small shared vocabulary in `_pieces`.

Public surface: one function per motif, each `(pre_fen, uci) -> str | None`. A None means "this motif
does not explain this move" — callers move on to the next motif or fall back.
"""
from .line import describe_plan, pv_san_to_uci
from .undermine import undermines_defender

__all__ = ["undermines_defender", "describe_plan", "pv_san_to_uci"]
