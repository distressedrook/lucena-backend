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

## Reasoning layer — the `reasoning/` package (beyond verdict grounding)

The relevance-filtered grounding above tells the coach WHICH facts to use. The `reasoning/` package is
the next step in — it DERIVES the causal point of a move so the LLM only verbalizes it. One function per
motif, `(...) -> str | None` (None = "this motif doesn't explain this move"), fed FIRST into the
correct-branch `point` in `coach._move_facts`, ahead of the static defender fact.

### Multi-ply plan — `line.describe_plan` — ✅ SHIPPED

The single-ply motifs (`undermine`) label ONE move, so they can only say a piece is *threatened* — they
don't see the reply. `describe_plan` walks the engine's **principal variation** (which already contains
the opponent's best defense), finds the **target the player actually wins**, and upgrades the hedge to a
verified statement: "threatening to win the knight" → "it undermines the only defender of Black's bishop
on a2 — it cannot be held, and falls." Spoiler-safe (names the target piece, never a future move).

Two guards, both hard-won from validation:
- **ABSOLUTE final material on STANDARD values** (not the compressed grounding scale R=4/N=2, and not the
  material *swing*). We count pieces on the actual final board and require the mover to both NET material
  over the line (`final > pre`) AND end up AHEAD (`final > 0`), plus the target square must not be
  recaptured. This took three cuts, each closing a hole the last let through: (a) the running-material
  *peak* credited the transient spike of a plain trade (Rxd8, Kxd8) as a won rook; (b) the *swing*
  credited an incidental pawn grabbed while still down a rook in a sacrificial attack; (c) `final > 0`
  alone was fooled by a *pre-existing surplus* — up a rook, trade it for a knight on a different square,
  still shows +2. `final > pre` is the invariant that rejects all three (the even trade included: `pre`
  and `final` are equal). Cuts (a) and (c) were caught by the local review loop, not by the scale run.
- **A lenient cp floor** (mover-POV eval ≥ +100, from `evaluate`'s `eval.cp`). The material test does the
  real work; cp only catches the rare "even on the board but the position is lost." A *coupled* cp gate
  (threshold ∝ target value) was tried and **rejected** — it over-filtered genuine wins (66%→37%) without
  fixing the real bug, which was in the material accounting, not cp.

Wired only when the played move IS the engine's best (its PV head: `pv_ucis[0] == inp.uci`), so the
"point of THIS move" always describes the played line. Fixtures + scale run: `test_reasoning_line.py`;
**150 Lichess puzzles → fired 67%, verified 101/101 (100%), 0 false.** Methodology note worth keeping:
the FIRST scale run reported 100% too, but its verifier reused the reasoner's own peak walk — it was
self-confirming and blind to exactly the bug a local review caught. The verifier was rebuilt to be
independent (count material on the real final board; confirm the last capture on the target square was
the player's) — that dropped the honest number to 87% and surfaced the false positives now fixed.
Fire-by-theme is a good honesty signal: high on `crushing`/`advantage`/`fork`/`pin`, low on `mate` (2/32)
and `mateIn2` (2/16) — a mate wins no material, so it correctly stays quiet instead of inventing one.

### Positional term-delta — ⏸️ PARKED (research), NOT built

The material reasoners are silent on a good *quiet* move (wins nothing, threatens nothing). The idea:
diff the engine-side **five-term positional read** (`analyze_positional` → material, king safety, piece
activity, pawn structure, centre, each in cp) before/after the move; the largest positive NON-material
term-delta names the reason ("it improves White's king safety"). Same shape as the material path —
derive deterministically, LLM verbalizes — for a new coverage class.

Prototyped and validated on tactical + positional Lichess slices. **Verdict: not shippable as-is, but
the engine cp makes it viable.** Findings, for whoever picks this up:
- **Raw single-ply term-delta mislabels tactics.** It fired on mates and sacrifices, slapping "improves
  king safety / piece activity" on a *tactic* the material reasoner missed (a mating move relocates a
  piece → the activity term jumps). Confident and wrong — the opposite of the layer's 0-false-positive
  bar. These five cp numbers are OUR piece-square heuristic, not Stockfish; their "why" ≠ the engine's.
- **The engine cp is the discriminator — specifically its CEILING.** Gating to a "good but not winning"
  band (≈ +50…+350cp mover-POV) dropped the mate/crushing misattributions (a mate/+485 eval ⇒ the point
  is a tactic, not a positional nudge) and kept the genuine positional moves, which clustered tightly
  (+237…+295). On the tactical slice 4 raw fires → 1 survived; on the positional slice 4 → 4 survived.
- **Two gates still needed before shipping:** the cp-ceiling band AND a *quietness* gate (non-capture,
  non-check, engine best line non-forcing) — a +161 fork slipped the cp band alone.
- **Lichess is the wrong corpus** — tactical by construction, so even the "positional/advantage/quietMove"
  tags are ~78% material tactics; it exercises the positional path only ~6% of the time. Real validation
  needs quiet-positional master games, not a tactics DB.

Prototype lives in the experiment scratch (not committed to the package). Lower precision than
`describe_plan` (heuristic, not ground truth) → if built, it is the LAST fallback, always cp-gated.
