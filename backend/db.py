from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    thread_type TEXT NOT NULL CHECK (thread_type IN ('finalized', 'trace')),
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(conversation_id, thread_type)
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (trim(role) <> ''),
    kind TEXT NOT NULL CHECK (trim(kind) <> ''),
    content TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    UNIQUE(thread_id, turn_index)
);

CREATE TABLE IF NOT EXISTS message_sentences (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (trim(role) <> ''),
    sentence_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    qdrant_point_id TEXT,
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    UNIQUE(message_id, sentence_index)
);

CREATE INDEX IF NOT EXISTS idx_threads_conversation_updated_at
ON threads (conversation_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_messages_thread_turn_index
ON messages (thread_id, turn_index);

CREATE INDEX IF NOT EXISTS idx_messages_thread_created_at
ON messages (thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_message_sentences_message_index
ON message_sentences (message_id, sentence_index);

CREATE INDEX IF NOT EXISTS idx_message_sentences_thread_role_created_at
ON message_sentences (thread_id, role, created_at);
"""

_REQUIRED_TABLES = ("threads", "messages", "message_sentences")


@dataclass(frozen=True)
class SQLiteDatabaseStatus:
    ready: bool
    path: str
    error: str | None = None


class SQLiteDatabase:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def initialize(self) -> None:
        with self.transaction() as connection:
            connection.executescript(SCHEMA_SQL)

    def status(self) -> SQLiteDatabaseStatus:
        try:
            if self._path != ":memory:" and not self._path.startswith("file:"):
                db_path = Path(self._path).expanduser().resolve()
                if not db_path.exists():
                    return SQLiteDatabaseStatus(
                        ready=False,
                        path=str(db_path),
                        error=f"SQLite database file does not exist: {db_path}",
                    )

            with self.connection() as connection:
                existing_tables = {
                    str(row["name"])
                    for row in connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        """
                    ).fetchall()
                }
                missing_tables = [
                    table_name
                    for table_name in _REQUIRED_TABLES
                    if table_name not in existing_tables
                ]
                if missing_tables:
                    return SQLiteDatabaseStatus(
                        ready=False,
                        path=self._resolved_path_label(),
                        error=(
                            "SQLite schema is incomplete. Missing tables: "
                            f"{', '.join(missing_tables)}"
                        ),
                    )

                connection.execute("SELECT 1").fetchone()
                return SQLiteDatabaseStatus(
                    ready=True,
                    path=self._resolved_path_label(),
                )
        except Exception as exc:
            return SQLiteDatabaseStatus(
                ready=False,
                path=self._resolved_path_label(),
                error=f"SQLite is unavailable at {self._resolved_path_label()}: {exc}",
            )

    def reset(self) -> None:
        if self._path == ":memory:" or self._path.startswith("file:"):
            with self.transaction() as connection:
                connection.executescript(
                    """
                    DELETE FROM message_sentences;
                    DELETE FROM messages;
                    DELETE FROM threads;
                    """
                )
            return

        db_path = Path(self._path).expanduser().resolve()
        if db_path.exists():
            db_path.unlink()

        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self._ensure_parent_directory()
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._ensure_parent_directory()
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_parent_directory(self) -> None:
        if self._path == ":memory:" or self._path.startswith("file:"):
            return

        db_path = Path(self._path).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolved_path_label(self) -> str:
        if self._path == ":memory:" or self._path.startswith("file:"):
            return self._path

        return str(Path(self._path).expanduser().resolve())
