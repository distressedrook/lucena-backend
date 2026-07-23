"""The /margin content builder — JSON-inspection mode (owner, 2026-07-23):
the margin shows the plans layer's raw pre/post-verify JSON. The former
card builder (epigraph/theory/position cards) lives at backend fb4b2d7."""
import pytest

from lucena_backend import margin
from lucena_backend.margin import build

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUT_OF_BOOK = "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2NBPN2/PPP2PPP/R2Q1RK1 w - - 4 9"


def test_unconfigured_margin_is_bare_and_never_pends():
    m = build(OUT_OF_BOOK)                     # no pool configured, not live
    assert m["raw"] is None and m["plansPending"] is False


def test_cached_result_is_served_raw():
    key = " ".join(OUT_OF_BOOK.split()[:4])
    margin._cache(key, '{\n  "schema": "lucena-plans/sheet@1"\n}',
                  "POST-VERIFY", False)
    try:
        m = build(OUT_OF_BOOK)
        assert m["statusLine"] == "POST-VERIFY"
        assert '"schema"' in m["raw"]
        assert m["plansPending"] is False
    finally:
        margin._deep_cache.clear()


def test_pending_pre_keeps_polling_alive():
    key = " ".join(START.split()[:4])
    margin._cache(key, "{}", "PRE-VERIFY · VERIFYING…", True)
    try:
        m = build(START)
        assert m["plansPending"] is True and m["raw"] == "{}"
    finally:
        margin._deep_cache.clear()


def test_bad_fen_raises():
    with pytest.raises(ValueError):
        build("not a fen")
