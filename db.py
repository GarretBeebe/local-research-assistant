from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

_DB_PATH = Path(__file__).parent / "research_assistant.db"
_SESSION_TTL_HOURS = 8


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                query         TEXT    NOT NULL,
                answer        TEXT    NOT NULL,
                confidence    INTEGER,
                critic_passed INTEGER,
                re_planned    INTEGER,
                created_at    TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                csrf_token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)


def save_query(
    query: str,
    answer: str,
    confidence: int | None = None,
    critic_passed: bool | None = None,
    re_planned: bool | None = None,
) -> int:
    now = _utcnow_iso()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO queries (query, answer, confidence, critic_passed, re_planned, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                query,
                answer,
                confidence,
                int(critic_passed) if critic_passed is not None else None,
                int(re_planned) if re_planned is not None else None,
                now,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_history(limit: int = 50) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, query, answer, confidence, critic_passed, re_planned, created_at"
            " FROM queries ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_query(query_id: int) -> bool:
    with _db() as conn:
        cur = conn.execute("DELETE FROM queries WHERE id = ?", (query_id,))
        return cur.rowcount > 0


def create_session() -> tuple[str, str]:
    token = secrets.token_hex(32)
    csrf_token = secrets.token_hex(16)
    now = datetime.now(tz=UTC)
    expires = now + timedelta(hours=_SESSION_TTL_HOURS)
    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, csrf_token, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, csrf_token, now.isoformat(), expires.isoformat()),
        )
    return token, csrf_token


def get_session(token: str) -> dict | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT token, csrf_token, created_at, expires_at FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def delete_session(token: str) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def prune_expired_sessions() -> None:
    now = _utcnow_iso()
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
