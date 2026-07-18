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
- **Defect:** bullet 2 is meaningless. The g3-defends-f2 fact is true but irrelevant to why `Kg1`
  fails; "is bypassed by the check" is fabricated causality. (Also minor: bullet 3 says `3. Kf1` while
  the grounded PV says `3. Kh1`, and "leaving the position at 5. Kf3" is awkward.)
- **Root cause:** `_deep_tactics` handed over three generic STRUCTURAL facts ("g3 defends f2", "f6
  pinned", "e7 defends d6"), none of which is why `Kg1` loses, under the framing "surface the one that
  explains why simple tries fail." The model force-fit the first into a false "the move fails
  because…". `_why_loses` correctly returned `None` (king move), so nothing grounded the *mechanism*,
  and the deep-tactics structural noise filled the gap.
- **Candidate fixes (not yet done — logged for the regression pass):**
  1. Don't feed `_deep_tactics` structural facts to a WRONG verdict when the refutation is already a
     concrete winning line — the line IS the explanation; the linchpins are noise.
  2. Or: the `(c) why it works` beat should draw only from `_why_loses`/the refutation, and treat
     deep-tactics as optional colour, not "why the move fails".
  3. The real fix is the parked "tactic reason derivation" — derive WHY the fork wins deterministically
     instead of letting the model narrate the mechanism.
- **Status:** OPEN.
