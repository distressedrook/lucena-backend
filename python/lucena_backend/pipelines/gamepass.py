"""Moved from lucena-engine (core-migration Phase 5, 2026-07-23): the PGN ->
analysis.json import pass is product ingestion — backend machinery composing
core truth (board, pgn) with the tactical layer (facts, brilliant).

Import-time engine pass (LLD §3): PGN -> analysis.json digests.

`run_pass` takes a game and produces the `analysis.json` structure (§3.2): every
ply classified (lichess-verbatim glyphs from M2), the decisive moment, motif
tags on the key moments (from the M3 fact sheet), and a summary with weakness
hypotheses. Claude is never involved — zero tokens (LLD §3).

Two passes (§3.1). The **fast pass** analyses every distinct position once and
classifies every ply; it lands in seconds and can be written out immediately
(`status:"fast-pass"`). The **deep pass** re-analyses the key moments
(mistakes + the eval-curve inflection) at a higher limit for a stable class,
the best line, the refutation, and motifs (`status:"complete"`).

Determinism split (as everywhere): production callers pass movetime limits;
tests pass node limits. `fast_limit` / `deep_limit` are dicts splatted straight
into `Engine.analyse` (e.g. `{"movetime_ms": 150}` or `{"nodes": 40000}`).
"""

from __future__ import annotations

from dataclasses import dataclass

from lucena_core.board import Board
from ..grounding_tools.brilliant import is_brilliant
from lucena_engine.evalmodel import Glyph, Score, classify, win_pct_from_score
from ..grounding_tools.facts import build_fact_sheet
from lucena_core.pgn import Game, parse_pgn

SCHEMA = 1
DEFAULT_FAST_LIMIT = {"movetime_ms": 150}
DEFAULT_DEEP_LIMIT = {"movetime_ms": 2000}

# lichess glyph -> analysis.json class string
_CLASS = {
    Glyph.OK: "ok",
    Glyph.ONLY_MOVE: "only_move",
    Glyph.DUBIOUS: "dubious",
    Glyph.MISTAKE: "mistake",
    Glyph.BLUNDER: "blunder",
}
_MISTAKE_CLASSES = {"dubious", "mistake", "blunder"}
_ALREADY_LOST_WP = 25.0  # decisive moment must not have been already lost


@dataclass(frozen=True)
class PositionEval:
    best_uci: str | None       # None only for a terminal position
    best_score: Score          # side-to-move POV
    best_pv: list[str]
    second_score: Score | None  # 2nd MultiPV line, if any


def _eval_position(engine, fen, limit, multipv) -> PositionEval:
    """Analyse one position (fresh hash for determinism). Terminal positions are
    scored without the engine: checkmate -> side to move is mated; stalemate ->
    drawn."""
    board = Board(fen)
    if not board.legal_moves():
        score = Score(mate=0) if board.in_check else Score(cp=0)
        return PositionEval(None, score, [], None)
    engine.new_game()
    an = engine.analyse(fen, multipv=multipv, **limit)
    second = an.lines[1].score if len(an.lines) > 1 else None
    return PositionEval(an.best.pv[0], an.best.score, list(an.best.pv), second)


def _pv_to_san(fen: str, pv: list[str], max_plies: int = 6) -> list[str]:
    out, b = [], Board(fen)
    for uci in pv[:max_plies]:
        try:
            out.append(b.san(uci))
            b = b.apply(uci)
        except Exception:
            break
    return out


def _classify_ply(p, before: PositionEval, after_score: Score | None
                  ) -> tuple[str, float, float, int]:
    """Return (class, win_pct_mover, delta_win_pct, eval_cp_mover) for one ply.

    `after_score` is the engine eval of `fen_after` (opponent POV), or None when
    that position is terminal (checkmate/stalemate)."""
    best_from_mover = before.best_score              # fen_before stm = the mover
    best_wp = win_pct_from_score(best_from_mover)

    if after_score is None:
        after_board = Board(p.fen_after)
        if after_board.in_check:                     # the move delivered mate
            return "ok", 100.0, round(100.0 - best_wp, 1), 1000
        played_result = Score(cp=0)                  # stalemate -> drawn
    else:
        played_result = after_score                  # opponent POV

    glyph, _drop = classify(
        best_from_mover,
        played_result,
        second_best_from_mover=before.second_score,
        played_is_best=(p.uci == before.best_uci),
    )
    mover_view = played_result.negated()
    played_wp = win_pct_from_score(mover_view)
    cls = _CLASS[glyph]
    if cls in ("ok", "only_move"):   # a sound sacrifice upgrades to brilliant (!!)
        second_wp = (win_pct_from_score(before.second_score)
                     if before.second_score is not None else None)
        if is_brilliant(Board(p.fen_before), p.uci, best_win=best_wp,
                        played_win=played_wp, second_win=second_wp,
                        played_is_best=(p.uci == before.best_uci)):
            cls = "brilliant"
    return (
        cls,
        round(played_wp, 1),
        round(played_wp - best_wp, 1),
        mover_view.to_ceiled_cp(),
    )


def _after_score(pos_eval: dict[str, PositionEval], fen_after: str) -> Score | None:
    pe = pos_eval.get(fen_after)
    return pe.best_score if pe is not None else None


def _fast_pass(game: Game, engine, limit) -> tuple[list[dict], dict[str, PositionEval]]:
    """Analyse every distinct position once; classify every ply."""
    pos_eval: dict[str, PositionEval] = {}
    for p in game.plies:
        if p.fen_before not in pos_eval:
            pos_eval[p.fen_before] = _eval_position(engine, p.fen_before, limit, 2)
    # the final position (for the last move's result), only if non-terminal
    if game.plies and Board(game.plies[-1].fen_after).legal_moves():
        last = game.plies[-1].fen_after
        if last not in pos_eval:
            pos_eval[last] = _eval_position(engine, last, limit, 2)

    plies_out = []
    for p in game.plies:
        before = pos_eval[p.fen_before]
        cls, win, delta, cp = _classify_ply(p, before, _after_score(pos_eval, p.fen_after))
        plies_out.append({
            "ply": p.ply, "move_no": p.move_no, "side": p.side,
            "san": p.san, "uci": p.uci, "fen_after": p.fen_after,
            "eval_cp": cp, "win_pct": win, "delta_win_pct": delta, "class": cls,
            "best": {
                "san": (_pv_to_san(p.fen_before, before.best_pv, 1) or [""])[0],
                "pv_san": _pv_to_san(p.fen_before, before.best_pv),
                "eval_cp": before.best_score.to_ceiled_cp(),
            },
            "refutation_pv": [],
            "motifs": [],
        })
    return plies_out, pos_eval


def _key_moment_indices(plies_out: list[dict]) -> list[int]:
    """Mistakes plus the single biggest eval-curve swing (inflection)."""
    keys = {i for i, pr in enumerate(plies_out) if pr["class"] in _MISTAKE_CLASSES}
    if plies_out:
        inflection = max(range(len(plies_out)),
                         key=lambda i: abs(plies_out[i]["delta_win_pct"]))
        keys.add(inflection)
    return sorted(keys)


def _deep_pass(game: Game, plies_out: list[dict], engine, limit) -> None:
    """Re-analyse the key moments: stable class, best line, refutation, motifs."""
    for i in _key_moment_indices(plies_out):
        p = game.plies[i]
        before = _eval_position(engine, p.fen_before, limit, 3)
        after_score = None
        if Board(p.fen_after).legal_moves():
            after = _eval_position(engine, p.fen_after, limit, 1)
            after_score = after.best_score
            plies_out[i]["refutation_pv"] = _pv_to_san(p.fen_after, after.best_pv)
        cls, win, delta, cp = _classify_ply(p, before, after_score)
        plies_out[i].update(eval_cp=cp, win_pct=win, delta_win_pct=delta, **{"class": cls})
        plies_out[i]["best"] = {
            "san": (_pv_to_san(p.fen_before, before.best_pv, 1) or [""])[0],
            "pv_san": _pv_to_san(p.fen_before, before.best_pv),
            "eval_cp": before.best_score.to_ceiled_cp(),
        }
        plies_out[i]["motifs"] = _motifs(p.fen_before, engine, limit)


def _motifs(fen: str, engine, limit) -> list[dict]:
    facts = build_fact_sheet(Board(fen), engine, top_n=5, **limit)
    return [{
        "motif": f.kind, "confidence": f.salience,
        "squares": list(f.squares), "concept_id": f.concept_id,
    } for f in facts]


def _decisive_ply(plies_out: list[dict], pos_eval: dict) -> int | None:
    """Mover-perspective ply with the largest win% drop that was not already
    lost (mover win% > 25 before the move)."""
    best_i, best_drop = None, 0.0
    for i, pr in enumerate(plies_out):
        before_wp = pr["win_pct"] - pr["delta_win_pct"]  # win% before the move
        drop = -pr["delta_win_pct"]
        if before_wp > _ALREADY_LOST_WP and drop > best_drop:
            best_i, best_drop = i, drop
    return None if best_i is None else plies_out[best_i]["ply"]


def _phase_of(pr: dict) -> str:
    if pr["move_no"] <= 12:
        return "opening"
    if pr["move_no"] <= 30:
        return "middlegame"
    return "endgame"


def _summary(game: Game, plies_out: list[dict], pos_eval: dict) -> dict:
    counts: dict[str, int] = {}
    phase_losses = {"opening": 0, "middlegame": 0, "endgame": 0}
    motif_tally: dict[tuple, dict] = {}
    user_side = _user_side(game)
    for pr in plies_out:
        counts[pr["class"]] = counts.get(pr["class"], 0) + 1
        if pr["class"] in _MISTAKE_CLASSES and pr["side"] == user_side:
            phase_losses[_phase_of(pr)] += 1
        for m in pr["motifs"]:
            key = (m["motif"], m["concept_id"])
            slot = motif_tally.setdefault(
                key, {"motif": m["motif"], "concept_id": m["concept_id"], "occurrences": 0})
            slot["occurrences"] += 1
    weakness = [v for v in motif_tally.values() if v["occurrences"] >= 2]
    weakness.sort(key=lambda v: (-v["occurrences"], v["motif"]))
    return {
        "decisive_ply": _decisive_ply(plies_out, pos_eval),
        "counts": {k: counts[k] for k in sorted(counts)},
        "phase_losses": phase_losses,
        "weakness_hypotheses": weakness,
    }


def _user_side(game: Game) -> str | None:
    us = (game.headers.get("user_side") or game.headers.get("UserSide") or "").lower()
    if us in ("w", "b"):
        return us
    if us in ("white", "black"):
        return us[0]
    return None


def _game_block(game: Game) -> dict:
    h = game.headers
    return {
        "id": h.get("id") or h.get("Site") or "game",
        "white": h.get("White", "?"), "black": h.get("Black", "?"),
        "result": game.result, "user_side": _user_side(game),
        "source": h.get("source") or h.get("Source"),
        "date": h.get("Date"),
    }


def run_pass(
    pgn_text: str,
    engine,
    *,
    out_path: str | None = None,
    fast_limit: dict | None = None,
    deep_limit: dict | None = None,
    fast_only: bool = False,
) -> dict:
    """Analyse one PGN game into the `analysis.json` structure (§3.2).

    Writes progressively to `out_path` if given (fast-pass then complete),
    always returns the final structure. `fast_limit`/`deep_limit` are splatted
    into `Engine.analyse` (movetime in prod, nodes in tests). `fast_only` stops
    after the fast pass.
    """
    fast_limit = fast_limit or DEFAULT_FAST_LIMIT
    deep_limit = deep_limit or DEFAULT_DEEP_LIMIT
    game = parse_pgn(pgn_text)

    plies_out, pos_eval = _fast_pass(game, engine, fast_limit)

    def assemble(status: str, seq: int) -> dict:
        return {
            "schema": SCHEMA, "seq": seq, "status": status,
            "game": _game_block(game), "plies": plies_out,
            "summary": _summary(game, plies_out, pos_eval),
        }

    fast = assemble("fast-pass", 1)
    if fast_only:
        return fast

    _deep_pass(game, plies_out, engine, deep_limit)
    complete = assemble("complete", 2)
    return complete


def main(argv=None):
    """`python -m lucena.engine.gamepass <game.pgn> --out analysis.json [--fast]`.

    The M4 stand-in for the frozen binary's `lucena-mcp pass` subcommand (M5).
    Production limits (movetime); Claude is not involved.
    """
    import argparse

    from lucena_engine.uci import Engine

    ap = argparse.ArgumentParser(description="Lucena import pass (PGN -> analysis.json)")
    ap.add_argument("pgn", help="path to a PGN file")
    ap.add_argument("--out", required=True, help="output analysis.json path")
    ap.add_argument("--fast", action="store_true", help="fast pass only")
    args = ap.parse_args(argv)

    with open(args.pgn, encoding="utf-8") as fh:
        pgn_text = fh.read()
    with Engine() as engine:
        result = run_pass(pgn_text, engine, out_path=args.out, fast_only=args.fast)
    s = result["summary"]
    print(f"{result['status']}: {len(result['plies'])} plies, "
          f"decisive ply {s['decisive_ply']}, counts {s['counts']} -> {args.out}")


if __name__ == "__main__":
    main()

