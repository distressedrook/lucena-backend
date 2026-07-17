"""The book voice: opening narration, and the end-of-book hand-off.

Split out of orchestrator.py because it is a fully self-contained subsystem — `_book_route` is a
pure function with no engine/LLM/DB dependency, and nothing outside `Orchestrator.coach_move`'s
dispatch touches any of this. It has its own dedicated test file (test_opening_narration.py); this
module is what that file is actually testing. Follows the same split-out-a-topic precedent as
drill.py / drill_feedback.py elsewhere in this package.
"""

from __future__ import annotations

from lucena_engine import openings   # pure table lookups: no engine process, no I/O per call

# -- book routing ---------------------------------------------------------------------------------
# The three freeform book-voice routes (plus _COACH, the fall-through — see orchestrator.py, which
# uses it for drill/non-narration flows too). Named, because the choice is made once from a pure
# function and asserted in tests by name — a bare string at the branch would be a typo away from
# silently coaching.
_NARRATE, _ENDBOOK, _COACH = "narrate", "endbook", "coach"

# What "strong players" means when we ask Maia what is normally played here. NOT `player_rating`, which
# answers a different question ("who is sitting here", for 'a common mistake at your level'). Narration
# asks what the theory IS, and the theory is what strong players actually play — so it is a fixed
# property of the QUESTION, not of the user. 2200 is a strong club player: high enough that the replies
# are real theory, not so high that they are engine-correct novelties nobody plays.
_BOOK_RATING = 2200

# How many replies to hand over. Three is a shape decision: it is enough to show that a position has
# CHOICES (the point of an opening), and few enough that a short paragraph can say what each does.
_BOOK_REPLIES = 3

# Plies out of the book before we say so. FOUR, not three, and it is a measured heuristic rather than
# a round number: two-unnamed-ply gaps occur at ~12% of book positions (the table names positions, not
# lines, and re-attaches names unevenly), and the probe that measured it was capped at 3 — so 3-ply
# gaps cannot be excluded and 3 has zero margin. Announcing "you have left the book" to someone still
# in the Ruy Lopez is worse than announcing it a ply late.
_OFF_BOOK_AT = 4

# That 12%-gap measurement was made on deep trees — Ruy Lopez (234 rows in the table), Sicilian
# Defense (385) — where a position briefly missing from the table is a coverage artifact, not a
# sign theory ran out. It was never validated on a shallow one. 57 of the table's 148 families have
# only 1-2 rows: a single catalogued position with nothing documented past it. Caught live: 1.e4 h5
# ("Goldsmith Defense", 2 rows total) 2.a4 has ZERO table entries, yet at one ply unnamed it rode the
# same sticky tolerance into full book-voice narration — "Strong players typically respond to this
# position..." asserted with the exact confidence a real Ruy Lopez continuation would earn. Below
# this many rows in the family, there is no tree for a gap to occur IN.
_MIN_FAMILY_SIZE = 5

# The engine's own word for "this move concedes something", not a threshold of our own invention:
# `classify` already returns these for a win-% drop past _DUBIOUS (5.0) / _MISTAKE (10.0) / _BLUNDER
# (15.0), measured against the engine's best move. A BOOK move that crosses one is the gambit case —
# theory that the engine disagrees with — and that disagreement is the interesting part.
_SWING_CLASSES = frozenset({"dubious", "mistake", "blunder"})


def _is_swing(verdict: dict) -> bool:
    return ((verdict or {}).get("class") or "") in _SWING_CLASSES


def _book_route(fens: list, swing: bool) -> tuple:
    """Which voice a freeform move gets: `(route, name, prev_name)`. Pure — no engine, no LLM, no DB.

    Pure on purpose: this is the whole cadence policy, it has four interacting cases, and the way to
    test it is to replay real openings through it and assert the exact fire sequence — not to mock an
    LLM. The caller does I/O; this only decides.

    Every in-book ply narrates. There is NO silent case, and removing it was not a tuning tweak — the
    silent case was actively wrong, and the way it was wrong is worth keeping written down.

    It used to stay quiet when the folded name did not change, on the reasoning that a repeat name has
    no news in it. But the table names POSITIONS, not lines, and it re-attaches COARSER names at real
    branch points. Measured on 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 — the Open Sicilian:

        d6   -> "Sicilian Defense: Modern Variations"    narrate
        d4   -> "Sicilian Defense"  (coarser)            SILENT   <- the Open Sicilian. Nothing said.
        cxd4 -> "Sicilian Defense"  (coarser)            SILENT
        Nxd4 -> unnamed, sticky                          SILENT

    Three plies through one of the most important branches in chess, in total silence. The fold is
    right to refuse to ANNOUNCE a coarser name — that reads as the coach forgetting. The mistake was
    treating "this name is not news" as "nothing happened here". They are different questions, and
    "did the table print a new string" was never a good proxy for the second one.

    So the name now decides what to CALL the opening, never whether to speak:

        in book + name changed                          -> narrate, and write the delta from the previous name
        in book + name unchanged                        -> narrate the MOVE; the name is context, not news
        in book (exact position) + swing                -> the concession is the subject (class/best — caller)
        was in book, now swing, deep family              -> normal coaching; might still transpose back
        was in book, now shallow family (first ply)      -> hand off HERE — no tree to wait on
        was in book, now shallow family (later ply)      -> normal coaching; already said so once
        psn hits the threshold, family still deep        -> say so, once (the original off-book case)
        psn hits the threshold, family shallow            -> normal coaching; already said so above
        off book / never in it                            -> normal coaching
    """
    psn = openings.plies_since_named(fens)
    if psn is None:                       # never in the book: a drill, a pasted midgame FEN.
        return _COACH, None, None         # "you have left the book" is meaningless there.
    if psn < _OFF_BOOK_AT:
        name = openings.book_name(fens)
        # Two different reasons the sticky-unnamed tolerance (psn > 0: the table has no entry for
        # the CURRENT position, only a recent one) is not license to keep speaking with book-voice
        # confidence:
        #   - A swing only gets the "the engine disagrees, yet it IS theory" framing when the
        #     position it LANDS ON is itself in the table (psn == 0) — the one case where "this is
        #     a known line" is a fact the table actually confirms. Caught live: 3.h4 then 3...Nh6 in
        #     a Vienna Game the table had lost track of got narrated as "a recognized theoretical
        #     attempt" for a move that was simply a mistake.
        #   - The tolerance itself assumes a deep tree with real coverage gaps (~12% of positions,
        #     measured on trees like Ruy Lopez's 234 table rows) — not an opening that barely has a
        #     tree. Caught live: "Goldsmith Defense" (1.e4 h5) has 2 rows total; 2.a4 has none of
        #     them, yet at psn=1 it still narrated "Strong players typically respond to this
        #     position..." with the same confidence a real continuation would earn.
        # Either way the honest coaching voice is the correct fallback — not a fabricated "it's
        # theory" excuse.
        shallow = openings.family_size(name) < _MIN_FAMILY_SIZE
        if psn > 0 and (swing or shallow):
            # A SHALLOW family gets the hand-off immediately, not the silent _COACH a swing gets.
            # The two look similar (both bail out of book-voice narration) but differ in whether
            # there is anything left to WAIT for: a swing might still be inside a deep, well-covered
            # tree (Ruy Lopez) where the next ply could easily transpose back into a named position
            # — silently coaching one ply and letting the psn == _OFF_BOOK_AT threshold below decide
            # is still the right call there. A shallow family has no such tree; `family_size` won't
            # change no matter how the game continues, so there is nothing to wait FOR. Saying so
            # immediately, at the precise ply the position ONE BACK (psn_prev) was still confirmed
            # in the table, is strictly more honest than staying quiet for up to 3 more plies.
            #
            # Caught live: 2.a4 after "Goldsmith Defense" (2 rows total) just started coaching
            # plainly with no "we're out of theory" moment at all — the DECISION (not book-confident
            # any more) and the HAND-OFF (_ENDBOOK, gated purely on the fixed threshold) had come
            # apart. One ply later psn_prev is already > 0 (already unnamed last time too), so this
            # doesn't refire — same no-latch shape as the psn == _OFF_BOOK_AT case below, and a
            # transposition back into a named position (psn -> 0) re-arms it the same way too.
            if shallow:
                psn_prev = openings.plies_since_named(fens[:-1]) if len(fens) > 1 else None
                if psn_prev == 0:
                    return _ENDBOOK, name, None
            return _COACH, None, None
        prev = openings.book_name(fens[:-1]) if len(fens) > 1 else None
        changed = name is not None and name != prev
        # `prev` is passed ONLY when the name changed — it is there to make the model write the DELTA
        # ("Nc6 -> King's Knight Opening: Normal Variation" deserves a clause, not a paragraph). When
        # the name is unchanged there is no delta to write, and the subject is simply the move.
        return _NARRATE, name, (prev if changed else None)
    if psn == _OFF_BOOK_AT:
        # EXACTLY at the threshold, so this fires once per exit with no latch to store or resync. A
        # transposition back into the book returns psn to 0 and re-arms it — correct: the book was
        # left twice. `book_name` still returns the name here (its fold has no OFF_BOOK_AT gate of
        # its own) — that's what the hand-off summarises, so it travels in the `name` slot same as
        # the narrate case, not as a third None.
        #
        # UNLESS the family is shallow — then the hand-off already fired above, back at psn == 1,
        # and `family_size` is a property of the (unchanged, still-sticky) name, not of how far psn
        # has since climbed: if it was shallow then, it is exactly as shallow now. Firing again here
        # would be a second "we're out of theory" for a theory the coach already said goodbye to.
        endbook_name = openings.book_name(fens)
        if openings.family_size(endbook_name) < _MIN_FAMILY_SIZE:
            return _COACH, None, None
        return _ENDBOOK, endbook_name, None
    return _COACH, None, None
