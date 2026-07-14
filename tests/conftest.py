"""Postgres test isolation: each DB(path) maps its path to a unique schema (s_*) in the
configured database. Drop all such schemas around the session so runs start clean and
don't accumulate."""

import os

import psycopg
import pytest

_DSN = os.environ.get("LUCENA_PG_DSN", "postgresql:///lucena_dev")


def _drop_test_schemas():
    try:
        conn = psycopg.connect(_DSN, autocommit=True)
    except Exception:
        return  # no Postgres → DB-backed tests will error/skip on their own
    with conn.cursor() as cur:
        # Tests create DB() without closing, so a schema can still have open backends;
        # terminate them first or DROP SCHEMA CASCADE blocks on their locks.
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND pid <> pg_backend_pid()")
        cur.execute("SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name LIKE 's\\_%%'")
        for (s,) in cur.fetchall():
            cur.execute(f'DROP SCHEMA IF EXISTS "{s}" CASCADE')
    conn.close()


@pytest.fixture(scope="session", autouse=True)
def _clean_pg_schemas():
    _drop_test_schemas()
    yield
    _drop_test_schemas()
