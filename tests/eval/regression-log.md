# Manual regression log — puzzles from live testing

Append a puzzle here whenever a verdict reads wrong during a live walk. The user runs a manual
regression pass over these after ~20+ puzzles. When the eval harness (see `README.md`) is built, these
become `fixtures.json` entries (frozen-facts + `expect` invariants).

**Per entry:** pre-move FEN · move played · correct? · what the coach said · the defect · root cause ·
status (OPEN / FIXED / WONTFIX).

---

## P1 — rook endgame, `Rc4` — FIXED (commit d03a54d)

- **FEN:** `k7/2K5/1P6/8/7p/1rR4p/7P/8 w - - 0 7`  ·  **move:** `Rc4` (uci c3c4)  ·  **wrong**
- **Defect:** last bullet said "You must deal with the threat where Black plays Rxc3+" — but `Rc4`
  moved the rook off c3, so `Rxc3+` is impossible; contradicted the real refutation `Rb4`.
- **Root cause:** `_deep_tactics` grounded on the PRE-move board cited a threat the move neutralised.
- **Fix:** `_deep_tactics(..., live_fen=after_fen)` drops a clause whose every named move is illegal
  after the move. Regression test: `test_deep_tactics_drops_a_threat_the_move_made_impossible`.

## P2 — knight fork, `Kg1` — OPEN

- **FEN:** `5r1k/4q2p/3p1p1P/pp1P3P/3B1Q2/2P1R1p1/P4n2/7K w - - 1 2`  ·  **move:** `Kg1` (uci h1g1)
  ·  **wrong** (White is in check from Nf2; the real refutation of Kg1 is the fork below)
- **Refutation PV (grounded, correct):** `Nh3+ Kh1 g2+ Kxg2 Nxf4+ Kf3` — wins the queen.
- **What the coach said:**
  > Your move 1. Kg1 turns a winning position into a losing one…
  > - Your move creates the threat of 2. Bxf6+, but it is immediately refuted by 2... Nh3+.
  > - **The move fails because the pawn on g3, which was the only defender of the knight on f2, is
  >   bypassed by the check.**
  > - Following 2... Nh3+ 3. Kf1 g2+ 4. Kxg2, Black plays 4... Nxf4+, winning your queen…
- **Defect A (gloss):** bullet 2 is meaningless. The g3-defends-f2 fact is true but irrelevant to why
  `Kg1` fails; "is bypassed by the check" is fabricated causality.
- **Defect B (guard gap?):** bullet 3 says `3. Kf1`, but the grounded PV says `3. Kh1` (both are legal
  escapes from g1, so `Kf1` isn't illegal — it just deviates from the handed-over line). The catch:
  `_invented_moves` DOES flag `Kf1` against these facts (verified), so the `_verdict_text` guard should
  have regenerated. It shipped anyway → either the 00:21 backend predated a fix, or a runtime gap
  (facts string differs from the reconstruction, warm-engine PV differs, etc.). **Investigate during
  the regression pass — a hallucinated/deviating move slipping the invention guard is worse than a
  weak gloss.** Also minor: "leaving the position at 5. Kf3" is awkward filler.
- **Root cause:** `_deep_tactics` handed over three generic STRUCTURAL facts ("g3 defends f2", "f6
  pinned", "e7 defends d6"), none of which is why `Kg1` loses, under the framing "surface the one that
  explains why simple tries fail." The model force-fit the first into a false "the move fails
  because…". `_why_loses` correctly returned `None` (king move), so nothing grounded the *mechanism*,
  and the deep-tactics structural noise filled the gap.
- **Fix applied (leniency, commit TBD):** stop *forcing* a "why it fails". Two spots softened:
  - `_deep_tactics` reframed from "surface the one that explains why simple tries fail" to "BACKGROUND
    — use one ONLY if it genuinely explains the failure; else IGNORE, don't force a connection".
  - `_WRONG (c)` now skips when the facts give no exact mechanism: "the refutation line already shows
    why (wins material or mates)… NEVER manufacture 'the move fails because <structural fact>'".
  - Live re-run: the g3 bullet is gone; the verdict just shows the fork line. Defect A resolved.
- **Defect B still a watch item:** the `Kf1`-slips-the-invention-guard question is unproven — the
  fixed re-run didn't hallucinate a king move, so it may have been an older backend. Re-check if a
  deviating/illegal move appears in any future verdict.
- **The deeper fix remains parked:** derive WHY a combination wins deterministically ("tactic reason
  derivation") so the mechanism is grounded, not narrated.
- **Status:** Defect A FIXED (leniency); Defect B WATCH.
