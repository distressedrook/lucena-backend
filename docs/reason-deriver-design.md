# Reason-deriver: relevance-filtered, structured verdict grounding (DESIGN)

Status: DESIGN — not implemented. Supersedes the "dump every motif" grounding for move verdicts.
Motivating defects: regression-log P1 (stale threat), P2 (fabricated structural why), P3 (unvalidated
fork "right idea" / "deal with d8 first"). Root cause: the verdict hands the LLM ALL detector motifs,
unstructured, with no relevance signal — so it force-fits an irrelevant one into a causal claim.

## Principle

Give the LLM **only facts on a line that matters** — the **refutation** (why this move fails) or the
**solution** (why the right move wins) — in **typed single-purpose fields**, one field → one bullet.
No triage burden, no motif dump. When a field can't be derived cleanly, OMIT it (empty > wrong).

## Output: a typed `MoveReason`

Rendered to the prompt as labelled fields; the prompt maps each PRESENT field to one bullet, absent →
no bullet. Everything here is derived on the board / from the engine lines — never guessed.

WRONG move:
```
MOVE:        <san> — <capture/check summary>
VERDICT:     <class> — <swing: winning→losing>
REFUTATION:  <refuting move> — <what it wins/achieves>          (from refutation_pv)
WHY:         <the ONE mechanism>                                 (see mechanisms below; may be absent)
IDEA:        <only if the played motif is genuinely part of the win>   (gated on solution; usually absent)
FIX:         <what the solution does FIRST that this move skipped>     (gated on solution; may be absent)
THREAT:      <decisive threat the move creates>                  (mate/material; may be absent)
```

RIGHT move:
```
MOVE:        <san> — <what it achieves>
VERDICT:     <class> (brilliant/best/…)
POINT:       <the ONE linchpin the solution turns on>            (relevance-filtered, not all motifs)
THREAT:      <decisive threat the move creates>
```

## Algorithm

**1. Losing mechanism (WRONG) — walk the refutation line.** Extend today's `_why_loses`:
- `guarded_destination` — moved onto a square the opponent still guards → recaptured. (exists)
- `deserted_defender` — moved the only/a defender off a piece that then hangs. (exists)
- `already_hanging` — ignored an already-undefended piece. (exists)
- `stalemate` — the win draws by stalemate. (exists, `_draws_by_stalemate`)
- `walks_into` — NEW: the refutation isn't a material recapture but a bigger threat against you (a
  fork, a mate, a decisive check). Covers king moves / quiet blunders (P2 Kg1) where today nothing is
  derived and the model invents. Derive by classifying pv[0]/pv on the board.
Pick ONE (priority: stalemate > guarded_destination > deserted_defender > walks_into > already_hanging).
If none classifies, WHY is absent — the REFUTATION line stands alone (P2 leniency, already shipped).

**2. Solution walk — extract the line, then two judgments.** Line-extractor handles both tree kinds
(`mate` → options[].then…, `solve` → expect + then…). From it:
- `IDEA` (gates the "right idea" credit): the played move has a motif (e.g. a fork on targets T). It's
  earned ONLY if the solution genuinely uses that motif — the same fork recurs on a solution move, OR
  the solution first removes the exact blocker and then the motif lands. P3: the forked rook (d8) is
  itself the defender, so the motif can never recur → IDEA absent. Err toward absent.
- `FIX`: the concrete thing the solution's FIRST player move does that the played move didn't — e.g.
  captures/deflects the guard, inserts a zwischenzug. Only when cleanly derivable from move 1. "first"
  ⇒ it is precisely about the solution's first move, so this is well-defined.

**3. Assemble + render.** Drop empty fields. Render labelled; the prompt maps field→bullet.

## Wiring / what it replaces

- Extends `_why_loses` → the mechanism field (+ `walks_into`).
- `_deep_tactics` (dump-all-motifs) → RETIRED from verdicts; replaced by the relevance-filtered
  `POINT` (right) and nothing on wrong unless FIX/IDEA are earned.
- `_created_threat` → the `THREAT` field (unchanged logic).
- `_swing_phrase`/`_brief_move` → MOVE/VERDICT/REFUTATION.
- `mode_prompts.VerdictPrompt._WRONG/_RIGHT` → simplified: "one bullet per PRESENT field, in order;
  never invent a field that isn't there" (much of the current conditional prose disappears — the
  fields already encode the conditionality).

## Phases — final status

1. **Line-extractor + tests** — ✅ DONE, as `_node_at` / `_node_solutions` (position-anchored, searches
   all branches, counts best moves). Better than the original root-walk plan.
3. **IDEA/FIX gated on the solution** — ✅ DONE, gated further on `single_solution` (a non-puzzle flow or
   several best moves stays permissive; a blunder's mechanism is always derived). Fixes P3.
   *(Retiring `_deep_tactics` from the wrong path — the noise that fed the force-fitting — shipped here
   too; fixes P2.)*

2. **`walks_into` mechanism + typed `_why_loses`** — ⏸️ NOT DONE, deliberately.
   - `walks_into` (a derived "why" for a non-capture refutation, e.g. a king move into a fork): the
     refutation LINE + the leniency (skip the beat when no mechanism) already convey this correctly and
     read well live (P2). A *derived* mechanism here would have to classify a fork/decisive-threat deep
     in the PV — fragile, and a WRONG mechanism is exactly the false-cause class this whole effort
     removed. Low marginal value, real regression risk → left out.
   - The "typed mechanism" refactor only pays off if Phase 4 lands (nothing else consumes the type), so
     it's coupled to Phase 4 below.
4. **Typed `MoveReason` + render + prompt rewrite** — ⏸️ SUPERSEDED. Its purpose was to end the model's
   *triage* of a motif dump. But Phase 3 already ended the dump (relevance filter), and the current
   sentence-facts render as clean, correctly-ordered bullets live. A full typed-field rewrite of the
   load-bearing verdict prompts is now diminishing-returns for real regression risk. Revisit only if a
   concrete formatting defect appears that the sentence facts can't fix.
5. **Eval fixtures (P1/P2/P3 → asserted fields)** — ⏸️ DEFERRED to the release checkpoint. This is the
   verdict-gloss eval (RELEASE_CHECKLIST §5), which the owner parked while pre-production. The P1/P2/P3
   cases live in `tests/eval/regression-log.md` ready to become fixtures when that harness is built.

**Net:** the phases that fix the logged defects (1 + 3, plus the `_deep_tactics` retirement) shipped and
are covered by unit tests + two local-review rounds. 2/4/5 are intentionally not built — superseded,
low-value/high-risk, or owner-deferred — not forgotten.

## Risks / open questions

- Solution-line extraction across tree kinds is the fragile foundation — Phase 1 must be rock-solid and
  well-tested before anything builds on it.
- IDEA/FIX are the hardest; the discipline is **absent-when-unsure**. A missing credit is fine; a false
  one is the bug we're removing.
- This is the LLD's "calculate vs interpret" line moving further toward *calculate*: we now derive the
  WHY deterministically instead of letting the model narrate it. Worth a note in LLD.md §9.
- Big surface change → every phase goes through `./review-loop.sh` (Codex must PASS).
