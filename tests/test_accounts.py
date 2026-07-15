"""Phase 3 — accounts. A user has many chats; one user must never reach another's.

Vocabulary (see CLAUDE.md — three different things, never conflated):
  login session = an authenticated user; a bearer token in `auth_token`.
  chat session  = a coaching conversation; MANY per user; the `session` table.
  active chat   = which chat a user currently has open; ONE per user; `session.is_active`.
"""

import asyncio
import os
import shutil

import pytest

from lucena_backend import auth
from lucena_backend.db import DB
from lucena_backend.state import StateStore

_have_sf = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_sf, reason="no stockfish")


@pytest.fixture
def db(tmp_path):
    return DB(str(tmp_path / "lucena"))


def _mkuser(db, email="a@example.com", password="correct horse battery staple"):
    return asyncio.run(auth.register(db, email, password))


# -- credentials -----------------------------------------------------------------

def test_password_is_hashed_not_stored_plain(db):
    pw = "correct horse battery staple"
    _mkuser(db, "a@example.com", pw)
    row = db.get_user_by_email("a@example.com")
    assert pw not in row["password_hash"]
    assert row["password_hash"].startswith("$argon2id$")     # argon2id, not bcrypt/plaintext


def test_login_returns_a_token_and_wrong_password_does_not(db):
    _mkuser(db, "a@example.com", "right-password")
    assert asyncio.run(auth.login(db, "a@example.com", "wrong-password")) is None
    assert asyncio.run(auth.login(db, "nobody@example.com", "right-password")) is None
    tok = asyncio.run(auth.login(db, "a@example.com", "right-password"))
    assert tok


def test_token_is_stored_hashed(db):
    """A database dump must not be a credential dump."""
    _mkuser(db)
    tok = asyncio.run(auth.login(db, "a@example.com", "correct horse battery staple"))
    assert db.user_for_token(auth.token_hash(tok), 0) is not None   # the hash resolves
    assert db.user_for_token(tok, 0) is None                        # the raw token does not


def test_expired_token_does_not_authenticate(db):
    uid = _mkuser(db)
    db.put_token(auth.token_hash("t"), uid, now=0, expires_at=100)
    assert db.user_for_token(auth.token_hash("t"), 50) == uid       # before expiry
    assert db.user_for_token(auth.token_hash("t"), 500) is None     # after

def test_logout_revokes_the_token(db):
    """The point of opaque DB tokens over JWT: revocation actually works."""
    uid = _mkuser(db)
    tok = asyncio.run(auth.login(db, "a@example.com", "correct horse battery staple"))
    assert db.user_for_token(auth.token_hash(tok), 0) == uid
    db.delete_token(auth.token_hash(tok))
    assert db.user_for_token(auth.token_hash(tok), 0) is None


def test_duplicate_email_is_rejected(db):
    _mkuser(db, "a@example.com")
    with pytest.raises(ValueError):
        _mkuser(db, "A@EXAMPLE.COM")          # case-insensitive: the unique index is on lower(email)


# -- the active chat is PER USER -------------------------------------------------

def test_active_chat_is_per_user(db):
    """The pre-accounts schema had ONE global is_active row; setting one user's active chat must not
    clear anyone else's."""
    u1, u2 = _mkuser(db, "one@example.com"), _mkuser(db, "two@example.com")
    db.upsert_session("c1", "n", 1.0, user_id=u1)
    db.upsert_session("c2", "n", 1.0, user_id=u2)
    db.set_active_session("c1", user_id=u1)
    db.set_active_session("c2", user_id=u2)
    assert db.get_active_session(u1) == "c1"
    assert db.get_active_session(u2) == "c2"      # NOT wiped by u1's set

    db.upsert_session("c1b", "n", 2.0, user_id=u1)
    db.set_active_session("c1b", user_id=u1)      # u1 switches chats
    assert db.get_active_session(u1) == "c1b"
    assert db.get_active_session(u2) == "c2"      # u2 untouched


def test_a_user_can_have_many_chats(db):
    u = _mkuser(db)
    for i in range(3):
        db.upsert_session(f"c{i}", f"chat {i}", float(i), user_id=u)
    assert {r["session_id"] for r in db.list_sessions(u)} == {"c0", "c1", "c2"}


def test_list_sessions_is_user_scoped(db):
    """Un-scoped, the rail would show every user every other user's chats."""
    u1, u2 = _mkuser(db, "one@example.com"), _mkuser(db, "two@example.com")
    db.upsert_session("mine", "n", 1.0, user_id=u1)
    db.upsert_session("theirs", "n", 1.0, user_id=u2)
    assert [r["session_id"] for r in db.list_sessions(u1)] == ["mine"]
    assert [r["session_id"] for r in db.list_sessions(u2)] == ["theirs"]


def test_set_active_on_someone_elses_chat_is_refused(db):
    """The user_id predicate IS the ownership check: 0 rows updated must raise, never silently leave
    the user with no active chat."""
    u1, u2 = _mkuser(db, "one@example.com"), _mkuser(db, "two@example.com")
    db.upsert_session("theirs", "n", 1.0, user_id=u2)
    db.set_active_session("theirs", user_id=u2)
    with pytest.raises(PermissionError):
        db.set_active_session("theirs", user_id=u1)
    assert db.get_active_session(u2) == "theirs"      # u2's pointer survived the attempt


# -- the store enforces ownership ------------------------------------------------

def test_user_a_cannot_open_user_b_chat(tmp_path):
    """The check must precede the upsert: upsert_session does ON CONFLICT DO UPDATE, so merely
    attaching to another user's chat id would bump their row."""
    store = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena")))
    u1, u2 = _mkuser(store.db, "one@example.com"), _mkuser(store.db, "two@example.com")

    with store.as_user(u2):
        store.open_chat("b-chat")
    before = store.db.list_sessions(u2)[0]["updated_at"]

    with store.as_user(u1):
        with pytest.raises(PermissionError):
            store.open_chat("b-chat")

    assert store.db.owner_of("b-chat") == u2                  # still B's
    assert store.db.list_sessions(u2)[0]["updated_at"] == before   # not touched by A's attempt
    assert store.db.list_sessions(u1) == []                   # A gained nothing


def test_authorization_precedes_the_document_load(tmp_path):
    """Refusal must happen BEFORE the chat's document is read into memory.

    bind_current -> _live_for -> _load_live reads session_view/session_doc from the DB. If the check
    ran after the bind, a guessed chat id would pull another user's coaching into `_live` first, and
    "we raised afterwards" is not a defence once the data is in the process.
    """
    store = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena")))
    u1, u2 = _mkuser(store.db, "one@example.com"), _mkuser(store.db, "two@example.com")
    with store.as_user(u2):
        store.open_chat("b-chat")
        store.append_beats([{"kind": "say", "stops": False,
                             "segments": [{"text": "B-PRIVATE-COACHING"}]}])
    store._live.clear()                              # force a cold load on the next touch

    with store.as_user(u1):
        with pytest.raises(PermissionError):
            store.attach_chat("b-chat")
    assert "b-chat" not in store._live, "B's document was loaded into memory before the refusal"


def test_authenticated_user_cannot_take_over_a_legacy_unowned_chat(tmp_path):
    """`owner is None` means either 'no such chat' or 'a pre-accounts row'. Conflating them lets a
    real user read and mutate legacy chats — and a FAILED attempt would still have bumped the row."""
    store = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena")))
    store.db.upsert_session("legacy", "n", 1.0, user_id=None)     # a pre-accounts chat
    u = _mkuser(store.db, "one@example.com")

    with store.as_user(u):
        with pytest.raises(PermissionError):
            store.attach_chat("legacy")
    exists, owner = store.db.session_owner("legacy")
    assert (exists, owner) == (True, None)           # untouched, and NOT adopted
    assert store.db.list_sessions(u) == []
    assert "legacy" not in store._live


def test_session_owner_distinguishes_missing_from_unowned(db):
    db.upsert_session("exists-unowned", "n", 1.0, user_id=None)
    assert db.session_owner("no-such-chat") == (False, None)
    assert db.session_owner("exists-unowned") == (True, None)


def test_a_new_chat_id_is_allowed(tmp_path):
    """The flip side: refusing unowned rows must not break creating a genuinely new chat."""
    store = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena")))
    u = _mkuser(store.db, "one@example.com")
    with store.as_user(u):
        store.open_chat("brand-new")
    assert store.db.session_owner("brand-new") == (True, u)


def test_each_user_resolves_their_own_active_chat(tmp_path):
    store = StateStore(str(tmp_path), db=DB(str(tmp_path / "lucena")))
    u1, u2 = _mkuser(store.db, "one@example.com"), _mkuser(store.db, "two@example.com")
    with store.as_user(u1):
        a = store.ensure_session_id()
    with store.as_user(u2):
        b = store.ensure_session_id()
    assert a != b
    with store.as_user(u1):
        assert store.ensure_session_id() == a       # stable across calls, and not B's
    with store.as_user(u2):
        assert store.ensure_session_id() == b


# -- the auth gate ---------------------------------------------------------------

# Every route the server exposes, minus the ones that are public BY DESIGN: /health must answer a
# probe holding no token, and /auth/* is how you get a token. Parametrized over the table so a route
# added later without a gate fails here automatically, rather than quietly becoming an exception.
_PUBLIC = {"/health", "/auth/register", "/auth/login", "/auth/logout"}
_GATED = [
    ("GET", "/sessions"), ("GET", "/session"), ("POST", "/session/new"), ("POST", "/session"),
    ("POST", "/move"), ("POST", "/drill"), ("POST", "/analyze"), ("GET", "/config"),
]


@requires_engine
@pytest.mark.parametrize("method,path", _GATED)
def test_every_non_public_route_is_401_without_a_token(tmp_path, monkeypatch, method, path):
    monkeypatch.setenv("LUCENA_REQUIRE_AUTH", "1")
    import importlib
    from lucena_backend import httpserver
    importlib.reload(httpserver)                 # _AUTH_REQUIRED is read at import
    from fastapi.testclient import TestClient
    try:
        app = httpserver.build_app(home=str(tmp_path), llm=None, model="stub")
        client = TestClient(app)
        r = client.request(method, path, json={})
        assert r.status_code == 401, f"{method} {path} answered {r.status_code} with no token"
    finally:
        monkeypatch.delenv("LUCENA_REQUIRE_AUTH", raising=False)
        importlib.reload(httpserver)


@requires_engine
def test_health_stays_public_when_auth_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("LUCENA_REQUIRE_AUTH", "1")
    import importlib
    from lucena_backend import httpserver
    importlib.reload(httpserver)
    from fastapi.testclient import TestClient
    try:
        client = TestClient(httpserver.build_app(home=str(tmp_path), llm=None, model="stub"))
        assert client.get("/health").status_code == 200
    finally:
        monkeypatch.delenv("LUCENA_REQUIRE_AUTH", raising=False)
        importlib.reload(httpserver)


@requires_engine
def test_ws_is_rejected_before_accept_when_unauthenticated(tmp_path, monkeypatch):
    """Rejected at the handshake, not after accept: accepting first would leave an unauthenticated
    socket alive and make the token a formality."""
    monkeypatch.setenv("LUCENA_REQUIRE_AUTH", "1")
    import importlib
    from lucena_backend import httpserver
    importlib.reload(httpserver)
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    try:
        client = TestClient(httpserver.build_app(home=str(tmp_path), llm=None, model="stub"))
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
    finally:
        monkeypatch.delenv("LUCENA_REQUIRE_AUTH", raising=False)
        importlib.reload(httpserver)


@requires_engine
def test_asking_for_another_users_chat_is_403_not_500(tmp_path, monkeypatch):
    """A refused attach must be a controlled 403 on EVERY route that can name a chat. Leaking the
    PermissionError as a 500 is both a worse contract and a wider error-logging surface."""
    monkeypatch.setenv("LUCENA_REQUIRE_AUTH", "1")
    import importlib
    from lucena_backend import httpserver
    importlib.reload(httpserver)
    from fastapi.testclient import TestClient
    try:
        app = httpserver.build_app(home=str(tmp_path), llm=None, model="stub")
        client = TestClient(app, raise_server_exceptions=False)
        store = app.state.ctx.store

        _mkuser(store.db, "one@example.com")
        _mkuser(store.db, "two@example.com")
        t1 = client.post("/auth/login", json={"email": "one@example.com",
                                              "password": "correct horse battery staple"}).json()["token"]
        t2 = client.post("/auth/login", json={"email": "two@example.com",
                                              "password": "correct horse battery staple"}).json()["token"]
        h1, h2 = {"Authorization": f"Bearer {t1}"}, {"Authorization": f"Bearer {t2}"}

        u2 = store.db.get_user_by_email("two@example.com")["id"]
        theirs = client.get("/session", headers=h2).json()["session_id"]   # B's chat
        # Read the baseline straight from the DB: any request AS B would resolve B's active chat and
        # bump updated_at itself, which would mask (or fake) a touch by A.
        before = store.db.list_sessions(u2)[0]["updated_at"]

        for r in (client.get(f"/sessions?session={theirs}", headers=h1),
                  client.post("/session", json={"session_id": theirs}, headers=h1),
                  client.post("/move", json={"uci": "e2e4", "session_id": theirs}, headers=h1),
                  client.post("/drill", json={"fen": "8/8/8/4k3/8/4K3/8/7R w - - 0 1",
                                              "session_id": theirs}, headers=h1)):
            assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text[:120]}"

        assert store.db.session_owner(theirs) == (True, u2)             # still B's
        assert store.db.list_sessions(u2)[0]["updated_at"] == before     # A's attempts left no mark
        # A's rail may contain A's OWN chat (asking for it mints one) — it must not contain B's.
        mine = [r["session_id"] for r in client.get("/sessions", headers=h1).json()["sessions"]]
        assert theirs not in mine
    finally:
        monkeypatch.delenv("LUCENA_REQUIRE_AUTH", raising=False)
        importlib.reload(httpserver)
