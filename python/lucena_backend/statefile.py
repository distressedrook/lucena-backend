"""Atomic single-writer JSON state files (LLD §1/§5 invariant).

Every JSON state file the product writes (`analysis.json`, later `board.json` /
`beats.json` / `learner.json`) has **exactly one writer** and is written
**atomically** — temp file in the same directory, `flush` + `fsync`, then
`os.replace` (an atomic rename on POSIX). A reader watching the file therefore
never sees a torn or partial write: it sees either the old file or the whole new
one.

Every state file carries `schema` (format version) and `seq` (a monotonically
increasing write counter) so a watcher can detect updates and order them. The
writer enforces their presence.
"""

from __future__ import annotations

import json
import os
import tempfile


def write_state(path: str, obj: dict) -> None:
    """Atomically write `obj` as JSON to `path`.

    `obj` must carry `"schema"` and `"seq"` keys (the single-writer invariant);
    otherwise `ValueError`. The write is temp+fsync+rename, so `path` is only
    ever observed as a complete file.
    """
    if "schema" not in obj or "seq" not in obj:
        raise ValueError("state files must carry 'schema' and 'seq'")
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _atomic_write(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)  # atomic rename needs the dir to exist
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic rename on POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
