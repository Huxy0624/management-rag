#!/usr/bin/env python
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_DB_PATH = Path("db/session.sqlite3")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_generation_log_columns(conn: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(generation_logs)")
    }
    if "metadata_json" not in columns:
        conn.execute("ALTER TABLE generation_logs ADD COLUMN metadata_json TEXT")


def ensure_demo_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS demo_request_logs (
            request_id TEXT PRIMARY KEY,
            session_id TEXT,
            question TEXT NOT NULL,
            success INTEGER NOT NULL DEFAULT 1 CHECK (success IN (0, 1)),
            selected_from TEXT,
            fallback_triggered INTEGER NOT NULL DEFAULT 0 CHECK (fallback_triggered IN (0, 1)),
            total_latency_ms INTEGER,
            error_message TEXT,
            client_ip TEXT,
            user_mode TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS demo_feedback (
            feedback_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            session_id TEXT,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            comment TEXT,
            client_ip TEXT,
            created_at TEXT NOT NULL
        )
        """
    )


def create_session(
    title: str,
    user_id: str | None = None,
    status: str = "active",
    session_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    session_id = session_id or str(uuid4())
    now = utc_now_iso()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, title, user_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, title, user_id, status, now, now),
        )
        conn.commit()

    return session_id


def add_message(
    session_id: str,
    role: str,
    content: str,
    turn_index: int,
    message_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    message_id = message_id or str(uuid4())
    now = utc_now_iso()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO messages (
                message_id, session_id, role, content, turn_index, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, role, content, turn_index, now),
        )
        conn.execute(
            """
            UPDATE sessions
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (now, session_id),
        )
        conn.commit()

    return message_id


def add_retrieval_log(
    session_id: str,
    message_id: str,
    raw_query: str,
    retrieval_query: str,
    top_k: int,
    retrieved_items_json: str | list[dict[str, Any]] | dict[str, Any],
    context_length: int,
    rerank_applied: bool,
    retrieval_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    retrieval_id = retrieval_id or str(uuid4())
    now = utc_now_iso()

    if not isinstance(retrieved_items_json, str):
        retrieved_items_json = json.dumps(retrieved_items_json, ensure_ascii=False)

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO retrieval_logs (
                retrieval_id,
                session_id,
                message_id,
                raw_query,
                retrieval_query,
                top_k,
                retrieved_items_json,
                context_length,
                rerank_applied,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                retrieval_id,
                session_id,
                message_id,
                raw_query,
                retrieval_query,
                top_k,
                retrieved_items_json,
                context_length,
                int(rerank_applied),
                now,
            ),
        )
        conn.commit()

    return retrieval_id


def add_generation_log(
    session_id: str,
    message_id: str,
    provider: str,
    model_name: str,
    prompt_text: str,
    prompt_length: int,
    answer: str | None,
    latency_ms: int | None,
    retry_count: int,
    success: bool,
    error_message: str | None = None,
    metadata_json: str | list[dict[str, Any]] | dict[str, Any] | None = None,
    generation_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    generation_id = generation_id or str(uuid4())
    now = utc_now_iso()

    if metadata_json is not None and not isinstance(metadata_json, str):
        metadata_json = json.dumps(metadata_json, ensure_ascii=False)

    with get_connection(db_path) as conn:
        ensure_generation_log_columns(conn)
        conn.execute(
            """
            INSERT INTO generation_logs (
                generation_id,
                session_id,
                message_id,
                provider,
                model_name,
                prompt_text,
                prompt_length,
                answer,
                latency_ms,
                retry_count,
                success,
                error_message,
                metadata_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generation_id,
                session_id,
                message_id,
                provider,
                model_name,
                prompt_text,
                prompt_length,
                answer,
                latency_ms,
                retry_count,
                int(success),
                error_message,
                metadata_json,
                now,
            ),
        )
        conn.commit()

    return generation_id


def add_demo_request_log(
    request_id: str,
    question: str,
    success: bool,
    session_id: str | None = None,
    selected_from: str | None = None,
    fallback_triggered: bool = False,
    total_latency_ms: int | None = None,
    error_message: str | None = None,
    client_ip: str | None = None,
    user_mode: str = "user",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        ensure_demo_tables(conn)
        conn.execute(
            """
            INSERT INTO demo_request_logs (
                request_id,
                session_id,
                question,
                success,
                selected_from,
                fallback_triggered,
                total_latency_ms,
                error_message,
                client_ip,
                user_mode,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                session_id,
                question,
                int(success),
                selected_from,
                int(fallback_triggered),
                total_latency_ms,
                error_message,
                client_ip,
                user_mode,
                now,
            ),
        )
        conn.commit()
    return request_id


def add_demo_feedback(
    request_id: str,
    rating: str,
    comment: str | None = None,
    session_id: str | None = None,
    client_ip: str | None = None,
    feedback_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    feedback_id = feedback_id or str(uuid4())
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        ensure_demo_tables(conn)
        conn.execute(
            """
            INSERT INTO demo_feedback (
                feedback_id,
                request_id,
                session_id,
                rating,
                comment,
                client_ip,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                request_id,
                session_id,
                rating,
                comment,
                client_ip,
                now,
            ),
        )
        conn.commit()
    return feedback_id


def list_recent_demo_requests(
    limit: int = 50,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        ensure_demo_tables(conn)
        rows = conn.execute(
            """
            SELECT
                r.request_id,
                r.session_id,
                r.question,
                r.success,
                r.selected_from,
                r.fallback_triggered,
                r.total_latency_ms,
                r.error_message,
                r.client_ip,
                r.user_mode,
                r.created_at,
                f.rating AS feedback_rating,
                f.comment AS feedback_comment,
                f.created_at AS feedback_created_at
            FROM demo_request_logs r
            LEFT JOIN demo_feedback f
              ON f.request_id = r.request_id
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]
