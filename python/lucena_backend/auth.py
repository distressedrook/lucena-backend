"""Accounts and login sessions.

Vocabulary (see CLAUDE.md — these are three different things and are never conflated):
  login session = an authenticated user; a bearer token in `auth_token`. THIS module.
  chat session  = a coaching conversation; MANY per user; the `session` table.
  active chat   = which chat a user currently has open; ONE per user; `session.is_active`.

Opaque DB-backed tokens rather than JWT: there is one process and a database already in hand, so
stateless verification buys nothing, while revocation (logout, ban) actually works. Only the sha256
of a token is stored, so a database dump is not a credential dump.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
import uuid

TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30      # 30 days


def _hasher():
    """argon2id. Imported lazily so the module can be imported without the optional dep present."""
    from argon2 import PasswordHasher
    return PasswordHasher()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(32)


async def hash_password(password: str) -> str:
    # argon2 is deliberately expensive; on the event loop it would block every coaching turn in the
    # process for its whole duration. Always off-loop.
    return await asyncio.to_thread(lambda: _hasher().hash(password))


async def verify_password(password_hash: str, password: str) -> bool:
    def _verify() -> bool:
        from argon2.exceptions import VerificationError, VerifyMismatchError
        try:
            return bool(_hasher().verify(password_hash, password))
        except (VerifyMismatchError, VerificationError):
            return False
    return await asyncio.to_thread(_verify)


class RegisterError(ValueError):
    """A register failure the CLIENT is allowed to distinguish, carrying a stable code.

    The code is the contract, not the message. This used to raise a bare ValueError whose prose went
    onto the wire as `error`, which made the client choose between rendering the server's raw string
    (so any unhandled exception on this path prints itself into the login box) and showing one generic
    failure for everything. A closed set of codes gives it a third option: map what it knows, generalise
    what it does not. The message stays for logs and for a client with no mapping.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# A floor, not a policy: argon2id handles slow hashing, and length is the only rule that reliably
# helps without pushing people toward "Passw0rd!". Stated here because register is the only writer.
_MIN_PASSWORD = 8


async def register(db, email: str, password: str) -> str:
    """Create a user, returning its id. Raises RegisterError (with a stable `code`) on refusal."""
    email = (email or "").strip()
    if not email or not password:
        raise RegisterError("missing_fields", "email and password are required")
    if len(password) < _MIN_PASSWORD:
        raise RegisterError("weak_password", f"password must be at least {_MIN_PASSWORD} characters")
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise RegisterError("invalid_email", "that does not look like an email address")
    if db.get_user_by_email(email) is not None:
        raise RegisterError("email_taken", "that email is already registered")
    uid = str(uuid.uuid4())
    await asyncio.to_thread(db.create_user, uid, email, await hash_password(password), time.time())
    return uid


async def login(db, email: str, password: str) -> str | None:
    """A fresh bearer token, or None if the credentials do not match."""
    user = await asyncio.to_thread(db.get_user_by_email, email)
    if user is None:
        # Still hash, so a missing account and a wrong password cost the same — otherwise response
        # time enumerates which emails are registered.
        await verify_password(
            "$argon2id$v=19$m=65536,t=3,p=4$"
            "c29tZXNhbHRzb21lc2FsdA$4Y0F6xUvKz8mB1nJZ0Jz4qJ0j0Q1nJ0Z5xY9wQ0Q0Q0", password)
        return None
    if not await verify_password(user["password_hash"], password):
        return None
    tok = new_token()
    now = time.time()
    await asyncio.to_thread(db.put_token, token_hash(tok), user["id"], now, now + TOKEN_TTL_SECONDS)
    return tok


async def user_for_bearer(db, authorization: str | None) -> str | None:
    """Resolve `Authorization: Bearer <token>` to a user id, or None."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return await asyncio.to_thread(db.user_for_token, token_hash(parts[1]), time.time())


class AuthMiddleware:
    """Authenticate every request in ONE place, and bind the user for the whole request.

    Why middleware and not a check inside each route: per-route auth is fail-OPEN — a new route is
    unguarded until someone remembers to add the gate, and the gate's position within the handler is
    up to whoever wrote it (three routes here were validating the request body, and answering 400,
    before they ever looked at the token). Here the default is refusal and `PUBLIC` is an explicit,
    reviewable list, so forgetting a route makes it 401 rather than open.

    A pure ASGI middleware, NOT Starlette's BaseHTTPMiddleware: BaseHTTPMiddleware runs the rest of
    the app in a separate task, which makes propagating the ContextVar we bind here fragile. This runs
    in the same task, so `store.current_user` is simply true downstream — including inside
    `asyncio.to_thread` (which copies the context) and any task the endpoint spawns.

    Handles websockets too, and rejects them BEFORE accept: accepting first would leave an
    unauthenticated socket alive. Browsers cannot set headers on a WS handshake, so a socket presents
    its token as `?token=`.
    """

    PUBLIC = frozenset({"/health", "/auth/register", "/auth/login", "/auth/logout"})

    def __init__(self, app, *, db, store, required: bool):
        self._app = app
        self._db = db
        self._store = store
        self._required = required

    @staticmethod
    def _bearer_from(scope) -> str | None:
        if scope["type"] == "websocket":
            from urllib.parse import parse_qs
            qs = parse_qs(scope.get("query_string", b"").decode())
            tok = (qs.get("token") or [""])[0]
            return f"Bearer {tok}" if tok else None
        for k, v in scope.get("headers") or []:
            if k == b"authorization":
                return v.decode()
        return None

    async def _reject(self, scope, receive, send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})    # policy violation, pre-accept
            return
        from starlette.responses import JSONResponse
        await JSONResponse({"error": "unauthenticated"}, status_code=401)(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not self._required \
                or scope.get("path") in self.PUBLIC:
            await self._app(scope, receive, send)
            return
        uid = await user_for_bearer(self._db, self._bearer_from(scope))
        if uid is None:
            await self._reject(scope, receive, send)
            return
        with self._store.as_user(uid):
            await self._app(scope, receive, send)
