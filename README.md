# backend — closed-source application layer

The proprietary application stack:
- **ConversationLoop** — a deterministic loop that routes every turn/move by mode: an active Lesson on
  the chat → coach mode (adjudicates the move), else freeform mode (explains the move / answers the
  question). It grounds the facts via the engine, then asks the model for ONE grounded generation. The
  LLM never tool-calls and never sets the board.
- **Mastery** — per-player mastery model (EWMA over honest observations).
- **Auth** — user authentication + sessions.
- **Memory** — parked in the current plan.

Reaches the grounding engine ONLY over its locked API contract (never by importing it).

Design-first: the API contract is locked in `docs/` before implementation.
