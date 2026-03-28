#!/usr/bin/env python
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("db/session.sqlite3")
DEFAULT_SCHEMA_PATH = Path("schema.sql")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the SQLite database for chat sessions and logs.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Target SQLite database path.")
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH, help="SQL schema file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {args.schema_path}")

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = args.schema_path.read_text(encoding="utf-8")

    with sqlite3.connect(args.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_sql)
        conn.commit()

    print(f"Database initialized: {args.db_path}")


if __name__ == "__main__":
    main()
