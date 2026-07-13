# backend — closed-source application layer

The proprietary application stack:
- **Orchestrator** — the deterministic coaching pipeline over Gemini. Classifies the turn, calls the
  grounding engine to GROUND the facts, asks the model for ONE grounded generation, then acts. The
  LLM never tool-calls and never sets the board.
- **Mastery** — per-player mastery model (EWMA over honest observations).
- **Auth** — user authentication + sessions.
- **Memory** — parked in the current plan.

Reaches the grounding engine ONLY over its locked API contract (never by importing it).

Design-first: the API contract is locked in `docs/` before implementation. Reuses the orchestrator
prototyped in `legacy/agent/orchestrator.py`.
