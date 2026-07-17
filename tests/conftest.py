"""Postgres test isolation. Tests run against a SEPARATE database (lucena_test), NOT the dev
database the running backend uses — so `pg_terminate_backend` during schema cleanup can never
kill a live backend's connection. Each DB(path) maps its path to a unique schema (s_*); those
schemas are dropped around the session.
"""

import os

# Point every DB() in the test process at the test database, before anything imports/creates one.
_TEST_DSN = os.environ.get("LUCENA_TEST_DSN", "postgresql:///lucena_test")
os.environ["LUCENA_PG_DSN"] = _TEST_DSN

import psycopg  # noqa: E402
import pytest   # noqa: E402


def _ensure_test_db():
    try:
        psycopg.connect(_TEST_DSN).close()
        return
    except Exception:
        pass
    admin = psycopg.connect("postgresql:///postgres", autocommit=True)
    try:
        admin.execute("CREATE DATABASE lucena_test")
    finally:
        admin.close()


def _drop_test_schemas():
    try:
        conn = psycopg.connect(_TEST_DSN, autocommit=True)
    except Exception:
        return
    with conn.cursor() as cur:
        # Only OTHER connections to THIS (test) database — never the dev backend.
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND pid <> pg_backend_pid()")
        cur.execute("SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name LIKE 's\\_%%'")
        for (s,) in cur.fetchall():
            cur.execute(f'DROP SCHEMA IF EXISTS "{s}" CASCADE')
    conn.close()


@pytest.fixture(scope="session", autouse=True)
def _clean_pg_schemas():
    _ensure_test_db()
    _drop_test_schemas()
    yield
    _drop_test_schemas()


@pytest.fixture(autouse=True)
def _reset_bound_chat():
    """Unbind the chat cursor around every test.

    `StateStore.bind_current` sets a ContextVar without resetting it — correct for the server, where
    each connection/task gets its own copied context, but pytest runs every test in ONE context, so a
    chat bound by one test would leak into the next and silently mask an unbound-cursor bug.
    """
    from lucena_backend.persistence.state import _current_sid
    token = _current_sid.set("")
    yield
    _current_sid.reset(token)
