"""The Lesson-CONTENT library (LLD §7) — file-based, append-only, shared across players.

Holds full `LessonSpec`s (including a `move_line` bit's derived forcing-line `tree`, which
the curated `content/puzzles/*.jsonl` does NOT carry — that only has fen/themes). "Append
to files" (the chosen shape): a player-discovered position, once classified + its tree
derived (the lesson-creation seam), is written here so it becomes durable, reusable
content. Curated puzzles are converted to specs at creation time and cached here too.

One JSON file, `<home>/lessons/library.json`, `{lesson_id: <spec dict>}`. Pure persistence
— derivation (tree-building via the engine) happens in the caller, never here. Atomic writes.
"""

from __future__ import annotations

import json
import os
import tempfile

from .bits import BitSpec
from .lesson import LessonSpec


def _spec_to_dict(s: LessonSpec) -> dict:
    return {
        "id": s.id, "type": s.type, "fen": s.fen, "motif": list(s.motif),
        "concept_id": s.concept_id,
        "bits": [{"strategy": b.strategy, "params": b.params, "challenge": b.challenge,
                  "grounding_req": b.grounding_req, "required": b.required,
                  "concept_id": b.concept_id} for b in s.bits],
    }


def _spec_from_dict(d: dict) -> LessonSpec:
    return LessonSpec(
        id=d["id"], type=d["type"], fen=d["fen"],
        motif=list(d.get("motif") or ["user_generated"]), concept_id=d.get("concept_id"),
        bits=[BitSpec(strategy=b["strategy"], params=b.get("params") or {},
                      challenge=b.get("challenge"), grounding_req=b.get("grounding_req") or [],
                      required=bool(b.get("required", True)), concept_id=b.get("concept_id"))
              for b in (d.get("bits") or [])],
    )


class Library:
    """Shared Lesson-content store. `get`/`put` full specs by id; `has` for the dedup path."""

    def __init__(self, home: str):
        self._dir = os.path.join(home, "lessons")
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, "library.json")

    def _load(self) -> dict:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get(self, lesson_id: str) -> LessonSpec | None:
        raw = self._load().get(lesson_id)
        return _spec_from_dict(raw) if raw else None

    def has(self, lesson_id: str) -> bool:
        return lesson_id in self._load()

    def put(self, spec: LessonSpec) -> None:
        """Append/overwrite a spec. Idempotent on id — re-deriving the same lesson is a no-op write."""
        data = self._load()
        data[spec.id] = _spec_to_dict(spec)
        self._write(data)
