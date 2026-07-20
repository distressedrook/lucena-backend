"""Durable per-player Lesson-PROGRESS store (LLD §7) — a file store, mirroring the
per-user `memory/` pattern mastery uses (NOT Postgres: every DB table is session-scoped,
but progress is per-user × per-lesson, cross-session). This avoids a schema bump.

Stores ONLY progress (`LessonProgress`) — the shared CONTENT (`LessonSpec`) comes from
the library (the static/appended `*.jsonl` files). A full `Lesson` is paired at read
time by the caller (StateStore integration). `meta` is stored write-once here; the
`state`/`meta` queries (`active`, `open_items`) back the §2.2 surfaces.

One JSON file per user under `<home>/lessons/<bucket>.json`, `bucket = user_id or "anon"`,
holding `{lesson_id: <progress dict>}`. Writes are atomic (temp + os.replace).
"""

from __future__ import annotations

import json
import os
import tempfile

from .bits import BitProgress
from .lesson import ACTIVE, OPEN, SUSPENDED, LessonProgress


def _bucket(user_id: str | None) -> str:
    return user_id or "anon"


def _progress_to_dict(p: LessonProgress) -> dict:
    return {
        "lesson_id": p.lesson_id, "state": p.state, "meta": p.meta,
        "currentBit": p.currentBit, "chat_id": p.chat_id,
        "bits": [{"cleared": b.cleared, "attempts": b.attempts,
                  "strategy_state": b.strategy_state} for b in p.bits],
    }


def _progress_from_dict(d: dict) -> LessonProgress:
    return LessonProgress(
        lesson_id=d["lesson_id"], state=d["state"], meta=d.get("meta"),
        currentBit=int(d.get("currentBit") or 0), chat_id=d.get("chat_id"),
        bits=[BitProgress(cleared=bool(b.get("cleared")), attempts=int(b.get("attempts") or 0),
                          strategy_state=b.get("strategy_state")) for b in (d.get("bits") or [])],
    )


class LessonStore:
    """Per-user progress persistence. All methods take the resolved `user_id` (None → anonymous)."""

    def __init__(self, home: str):
        self._dir = os.path.join(home, "lessons")
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, user_id: str | None) -> str:
        return os.path.join(self._dir, f"{_bucket(user_id)}.json")

    def _load_raw(self, user_id: str | None) -> dict:
        try:
            with open(self._path(user_id), encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, ValueError):
            return {}

    def _write_raw(self, user_id: str | None, data: dict) -> None:
        # Atomic: write a temp file in the same dir then os.replace, so a crash never leaves a
        # half-written progress file (a corrupt file would silently lose a user's whole history).
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self._path(user_id))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- reads -----------------------------------------------------------------
    def get(self, user_id: str | None, lesson_id: str) -> LessonProgress | None:
        raw = self._load_raw(user_id).get(lesson_id)
        return _progress_from_dict(raw) if raw else None

    def active(self, user_id: str | None, chat_id: str | None) -> LessonProgress | None:
        """The live lesson FOR THIS CHAT: `state == active AND meta is None AND chat_id == chat_id`.
        Per-chat, not per-user — a lesson active in another chat must not force coach mode here."""
        for d in self._load_raw(user_id).values():
            if (d.get("state") == ACTIVE and d.get("meta") is None
                    and d.get("chat_id") == chat_id):
                return _progress_from_dict(d)
        return None

    def suspended(self, user_id: str | None, chat_id: str | None) -> LessonProgress | None:
        """The SUSPENDED lesson for THIS CHAT — a drill parked by a what-if excursion. `set_state`
        keeps chat_id, so a suspension stays bound to its chat and can be resumed when the player plays
        a move (back to solving). Same shape as `active`, state==suspended."""
        for d in self._load_raw(user_id).values():
            if (d.get("state") == SUSPENDED and d.get("meta") is None
                    and d.get("chat_id") == chat_id):
                return _progress_from_dict(d)
        return None

    def open_items(self, user_id: str | None) -> list[LessonProgress]:
        """Resumable items for 'pick up an open item' — `state == open AND meta is None`."""
        return [_progress_from_dict(d) for d in self._load_raw(user_id).values()
                if d.get("state") == OPEN and d.get("meta") is None]

    def is_solved(self, user_id: str | None, lesson_id: str) -> bool:
        """For the library dedup (§2.2 surface 2) — durable, so a solved lesson stays solved."""
        d = self._load_raw(user_id).get(lesson_id)
        return bool(d and d.get("meta") == "solved")

    # -- writes ----------------------------------------------------------------
    def put(self, user_id: str | None, progress: LessonProgress) -> None:
        data = self._load_raw(user_id)
        data[progress.lesson_id] = _progress_to_dict(progress)
        self._write_raw(user_id, data)

    def set_state(self, user_id: str | None, lesson_id: str, state: str) -> None:
        data = self._load_raw(user_id)
        if lesson_id in data:
            data[lesson_id]["state"] = state
            self._write_raw(user_id, data)

    def activate(self, user_id: str | None, lesson_id: str, chat_id: str | None) -> None:
        """Bind a lesson ACTIVE to a specific chat (state=active + chat_id). Used on coach entry."""
        data = self._load_raw(user_id)
        if lesson_id in data:
            data[lesson_id]["state"] = ACTIVE
            data[lesson_id]["chat_id"] = chat_id
            self._write_raw(user_id, data)
