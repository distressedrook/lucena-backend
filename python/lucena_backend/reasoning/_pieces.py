"""The piece vocabulary shared by every reasoning motif — kept here so a motif module never has to
reach back into the coaching/grounding code."""

PIECE_WORD = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight", "P": "pawn"}
PIECE_VAL = {"K": 6, "Q": 5, "R": 4, "B": 3, "N": 2, "P": 1}


def word(piece: str) -> str:
    """A piece letter → its English word ('N' → 'knight'). Unknown/empty → 'pawn'."""
    return PIECE_WORD.get((piece or "").upper(), "pawn")


def value(piece: str) -> int:
    """A piece letter → its rough exchange value (K=6 … P=1). Unknown/empty → 0."""
    return PIECE_VAL.get((piece or "").upper(), 0)
