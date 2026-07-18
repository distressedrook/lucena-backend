"""Server-side drill walk. The MCP owns the forcing-win tree and adjudicates every move: the app
just pushes the raw move; the server decides correct/wrong, advances the board, plays the
opponent's reply, and returns the side effects (board / feedback beat / drill event) to push. So
the app is a pure renderer and the coach reads real move context.

Ported from the app-side walker (the debugged logic): a solve node has one required move; a reply
node holds the opponent's defenses; each defense's `then` is the next solve/done. Only defenses
that lead to another move to find are auto-played; a defense straight to `done` is a one-move win.
"""

from __future__ import annotations

from . import drill_feedback


def _same_move(in_san: str | None, in_uci: str | None,
               exp_san: str | None, exp_uci: str | None) -> bool:
    """Is the played move the expected one? SAN is the canonical form — compare it first; fall back
    to a UCI prefix match only if SAN is unavailable/mismatched (defensive, e.g. a stale conversion)."""
    if in_san and exp_san and in_san == exp_san:
        return True
    return bool(in_uci) and bool(exp_uci) and exp_uci.startswith(in_uci)


def _is_student(node: dict) -> bool:
    return node.get("kind") in ("solve", "mate")


def _solve_count(node: dict) -> int:
    k = node.get("kind")
    if k == "solve":
        return 1 + (_solve_count(node["after"]) if node.get("after") else 0)
    if k == "mate":
        opts = node.get("options") or []
        return 1 + (_solve_count(opts[0]["then"]) if opts else 0)
    if k == "reply":
        return sum(_solve_count(d["then"]) for d in (node.get("defenses") or []))
    return 0


def _children(node: dict):
    """The (step, child-node) pairs out of a node — the tree's navigable edges. Steps are structural
    (a string or ["opt"|"def", i]) so a node can be addressed by a path that is STABLE across a
    reload (unlike an object identity). Used to (de)serialize the walker (P2c)."""
    k = node.get("kind")
    if k == "solve":
        a = node.get("after")
        return [("after", a)] if a else []
    if k == "mate":
        return [(["opt", i], o["then"]) for i, o in enumerate(node.get("options") or [])]
    if k == "reply":
        return [(["def", i], d["then"]) for i, d in enumerate(node.get("defenses") or [])]
    return []


def _node_path(root: dict, target: dict):
    """The structural path (list of steps) from `root` to `target` by object identity, or None."""
    if root is target:
        return []
    for step, child in _children(root):
        sub = _node_path(child, target)
        if sub is not None:
            return [step] + sub
    return None


def _resolve_path(root: dict, path):
    """Walk a structural path (from `_node_path`, possibly round-tripped through JSON so tuples are
    lists) back to its node in a freshly-loaded tree."""
    node = root
    for step in (path or []):
        if step == "after":
            node = node["after"]
        elif step[0] == "opt":
            node = node["options"][step[1]]["then"]
        elif step[0] == "def":
            node = node["defenses"][step[1]]["then"]
    return node


class DrillState:
    """Walks one forcing-win tree. `play(uci)` returns the side effects for the MCP to apply."""

    def __init__(self, tree: dict):
        self.tree = tree
        self.current = tree["root"]
        self.total = _solve_count(tree["root"])
        self.solved = 0
        self.finished = False
        # (node, defense-san, defense-uci, branch_len) — branch_len = the line length at the point
        # this sibling branches from, so a backtrack truncates the line back to it.
        self.stack: list[tuple[dict, str | None, str | None, int]] = []
        self._i = {"correct": 0, "wrong": 0, "backtrack": 0}
        # The move line currently on the board, ply 0 = the starting position. Each ply is
        # {n, san, uci, fen (after the ply)}. Truncated + rewritten on a backtrack so it always
        # matches the board (the move navigator renders this).
        start_fen = tree.get("fen") or self.current.get("fen")
        self.line: list[dict] = [{"n": 0, "san": None, "uci": None, "fen": start_fen}]

    def _bump(self, key: str) -> int:
        v = self._i[key]
        self._i[key] = v + 1
        return v

    def _add_ply(self, san: str | None, uci: str | None, fen: str | None) -> None:
        self.line.append({"n": len(self.line), "san": san, "uci": uci, "fen": fen})

    def play(self, uci: str, san: str | None = None) -> dict:
        """Adjudicate a move. Returns {correct, board, feedback, extra, event, finished, plies}
        where `plies` is the full move line (ply 0 = start) currently on the board.

        Matching is on **SAN** — the single canonical internal notation. The app's UCI is converted to
        SAN at the /move boundary before it reaches here (`play_move`), so a move is compared as SAN
        vs the node's `expect_san`; UCI is kept only as a defensive fallback. This kills the UCI/SAN
        mixing that made a correct move ("bxc4" == "b5c4") non-deterministically read as wrong."""
        cur = self.current
        kind = cur.get("kind")
        chosen_then: dict | None = None
        chosen_uci: str | None = None
        chosen_san: str | None = None
        matched = False
        if kind == "solve":
            matched = _same_move(san, uci, cur.get("expect_san"), cur.get("expect_uci"))
            chosen_then, chosen_uci, chosen_san = cur.get("after"), cur.get("expect_uci"), cur.get("expect_san")
        elif kind == "mate":
            for o in cur.get("options") or []:
                if _same_move(san, uci, o.get("san"), o.get("uci")):
                    matched, chosen_then, chosen_uci, chosen_san = True, o.get("then"), o.get("uci"), o.get("san")
                    break

        if not matched:
            return {
                "correct": False, "board": None,
                "feedback": drill_feedback.wrong_beat(self._bump("wrong")),
                "extra": None, "finished": False,
                "event": {"kind": "drill_wrong", "tried": uci, "fen": cur.get("fen")},
                "plies": list(self.line),   # unchanged — the wrong move isn't recorded
            }

        self.solved += 1
        # Record the player's move (fen = position after it, from the tree).
        self._add_ply(san or chosen_san, chosen_uci or uci, (chosen_then or {}).get("fen"))
        feedback = drill_feedback.correct_beat(self._bump("correct"))
        board, event, extra, finished = self._advance(chosen_then)
        return {"correct": True, "board": board, "feedback": feedback,
                "extra": extra, "finished": finished, "event": event, "plies": list(self.line),
                "reply": (event or {}).get("reply")}   # the opponent's auto-played reply, if any

    def _advance(self, node: dict | None):
        # `node` is the position after your move. For a non-reply node (a one-move win → `done`),
        # show that position, then finish.
        if not node or node.get("kind") != "reply":
            return self._next_or_finish(board=(node.get("fen") if node else None))
        live = [d for d in (node.get("defenses") or []) if _is_student(d["then"])]
        branch_len = len(self.line)               # siblings branch from here (after the player ply)
        for d in reversed(live[1:]):
            self.stack.append((d["then"], d.get("san"), d.get("uci"), branch_len))
        if live:
            first = live[0]
            self.current = first["then"]
            fen = first["then"].get("fen")
            self._add_ply(first.get("san"), first.get("uci"), fen)   # the opponent's defense
            # Surface the reply so the coach can voice it as its own beat: the move, and the position
            # it was played FROM (this reply node's fen — after the player's move, before the reply).
            reply = {"san": first.get("san"), "uci": first.get("uci"), "from_fen": node.get("fen")}
            return fen, {"kind": "drill", "event": "solved", "fen": fen, "reply": reply}, None, False
        # No further move to find — your move was the last. Show the position after it, then finish.
        return self._next_or_finish(board=node.get("fen"))

    def _next_or_finish(self, board: str | None = None):
        if self.stack:
            node, defense, defense_uci, branch_len = self.stack.pop()
            self.current = node
            fen = node.get("fen")
            self.line = self.line[:branch_len]        # rewind to the branch point …
            branch_fen = self.line[-1]["fen"] if self.line else None   # position the sibling is played FROM
            self._add_ply(defense, defense_uci, fen)  # … then take the sibling defense
            # Surface the sibling as a reply flagged `new_line`, so the coach ANNOUNCES the backtrack
            # ("that defence is handled — now the opponent tries a different one") instead of the board
            # silently jumping back to the branch point (the "weird state" the player saw).
            reply = {"san": defense, "uci": defense_uci, "from_fen": branch_fen, "new_line": True}
            return (fen,
                    {"kind": "drill", "event": "new_line", "defense": defense, "fen": fen, "reply": reply},
                    drill_feedback.backtrack_beat(self._bump("backtrack"), defense),
                    False)
        self.finished = True
        # The whole tree is solved — tell the coach so it can give a grounded closing (the app also
        # posts this on `finished`; the single-slot input mailbox de-dupes the two identical writes).
        return board, {"kind": "drill_solved", "fen": board}, drill_feedback.finish_beat(self.tree), True

    # -- serialization (P2c): persist the FULL walker so restart is a deterministic deserialize,
    #    not a fragile "replay history and hope it fits the tree". `tree` + `line` live elsewhere in
    #    the document (last_tree + history); this captures the position within the walk. --
    @property
    def counters(self) -> dict:
        """Per-drill tallies {correct, wrong, backtrack} — the deterministic basis for a solve grade
        (the MCP banks mastery on solve without asking the LLM)."""
        return dict(self._i)

    def to_state(self) -> dict:
        """The walker's resumable state — nodes encoded as structural paths from the tree root, so
        `current` and the backtrack `stack` (and thus `solved`/`held_wrong` progress) survive a
        restart even mid-line. Pure. Does NOT include the move line: that has ONE home, the document's
        `history`, handed back to `restore` — a second copy here would be a rival that drifts the
        moment any writer touches history without the walker."""
        root = self.tree["root"]
        return {
            "current": _node_path(root, self.current),
            "solved": self.solved,
            "finished": self.finished,
            "stack": [{"path": _node_path(root, node), "san": san, "uci": uci, "branch_len": bl}
                      for (node, san, uci, bl) in self.stack],
            "counters": dict(self._i),
        }

    @classmethod
    def restore(cls, tree: dict, state: dict, line: list | None = None) -> "DrillState":
        """Rebuild a walker from `to_state()` output against a freshly-loaded `tree` — resolving each
        stored path back to its node. Deterministic; no move replay. The move `line` comes from its ONE
        home (the document's `history`), passed in by the caller; a legacy state that still embeds a
        `line` is honoured only as a migration fallback when no `line` is given."""
        d = cls(tree)
        root = tree["root"]
        d.current = _resolve_path(root, state.get("current"))
        d.solved = int(state.get("solved") or 0)
        d.finished = bool(state.get("finished"))
        d.stack = [(_resolve_path(root, e.get("path")), e.get("san"), e.get("uci"), e.get("branch_len"))
                   for e in (state.get("stack") or [])]
        if state.get("counters"):
            d._i = dict(state["counters"])
        if line is not None:
            d.line = list(line)
        elif state.get("line"):        # legacy documents persisted the line inside the walker state
            d.line = list(state["line"])
        return d
