"""The Sessions list — sourced entirely from OUR DB, the single source of truth for which sessions
exist (the app inserts each session it creates). Nuking the DB clears the rail.

Historically this module also read Claude Code's per-session transcripts
(`~/.claude/projects/.../<id>.jsonl`) to cache Claude's auto-title into our DB. The ADK/Gemini
pivot has no such transcript, so that coupling is gone: names come from the DB alone.
`refresh_session_names` is kept as a no-op for call-site stability until a pivot-native titling
source lands (e.g. a coach-set or first-message title).
"""

from __future__ import annotations


def refresh_session_names(home: str, db=None) -> None:
    """No-op in the ADK path (no Claude transcript to read). Kept for call-site stability; the DB's
    `name` column is authoritative. A future titling source (coach-set / first-message) would write
    the DB here."""
    return


def list_sessions(home: str, db=None) -> list[dict]:
    """Every session for `home` (from OUR DB), newest first: `{session_id, name, updated_at}`. A PURE
    read of the DB rows — no I/O, no writes — so it is safe on the event loop and inside a snapshot.
    The DB is the source of truth. (`home` is kept in the signature for call-site stability.)"""
    if db is None:
        return []
    return sorted(db.list_sessions(), key=lambda s: s["updated_at"], reverse=True)
