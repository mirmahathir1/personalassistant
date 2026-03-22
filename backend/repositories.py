from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Mapping, Sequence
from uuid import uuid4

from .db import SQLiteDatabase

THREAD_TYPE_FINALIZED = "finalized"
THREAD_TYPE_TRACE = "trace"
FINALIZED_USER_KIND = "finalized_user_message"
FINALIZED_ASSISTANT_KIND = "finalized_assistant_response"


@dataclass(frozen=True)
class ThreadRecord:
    id: str
    conversation_id: str
    thread_type: str
    title: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MessageRecord:
    id: str
    thread_id: str
    role: str
    kind: str
    content: str
    turn_index: int
    created_at: str


@dataclass(frozen=True)
class MessageSentenceRecord:
    id: str
    message_id: str
    thread_id: str
    role: str
    sentence_index: int
    text: str
    created_at: str
    qdrant_point_id: str | None


@dataclass(frozen=True)
class ConversationThreads:
    conversation_id: str
    finalized: ThreadRecord
    trace: ThreadRecord


@dataclass(frozen=True)
class ConversationSummaryRecord:
    conversation_id: str
    finalized_thread_id: str
    trace_thread_id: str
    title: str | None
    created_at: str
    updated_at: str
    message_count: int
    last_message_preview: str | None


class ConversationIntegrityError(RuntimeError):
    """Raised when SQLite no longer contains a valid finalized/trace pair."""


class _RepositoryBase:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    @contextmanager
    def _read_connection(
        self,
        connection: sqlite3.Connection | None = None,
    ) -> Iterator[sqlite3.Connection]:
        if connection is not None:
            yield connection
            return

        with self._database.connection() as managed_connection:
            yield managed_connection

    @contextmanager
    def _write_connection(
        self,
        connection: sqlite3.Connection | None = None,
    ) -> Iterator[sqlite3.Connection]:
        if connection is not None:
            yield connection
            return

        with self._database.transaction() as managed_connection:
            yield managed_connection

    def _get_thread_row(
        self,
        connection: sqlite3.Connection,
        thread_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT id, conversation_id, thread_type, title, created_at, updated_at
            FROM threads
            WHERE id = ?
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown thread_id: {thread_id}")
        return row

    def _get_message_row(
        self,
        connection: sqlite3.Connection,
        message_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT id, thread_id, role, kind, content, turn_index, created_at
            FROM messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown message_id: {message_id}")
        return row

    def _touch_thread(
        self,
        connection: sqlite3.Connection,
        thread_id: str,
        updated_at: str,
    ) -> None:
        connection.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (updated_at, thread_id),
        )


class ThreadRepository(_RepositoryBase):
    def list_conversations(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> list[ConversationSummaryRecord]:
        with self._read_connection(connection) as active_connection:
            rows = active_connection.execute(
                """
                SELECT
                    finalized.conversation_id AS conversation_id,
                    finalized.id AS finalized_thread_id,
                    trace.id AS trace_thread_id,
                    COALESCE(finalized.title, trace.title) AS title,
                    finalized.created_at AS created_at,
                    CASE
                        WHEN finalized.updated_at >= trace.updated_at
                            THEN finalized.updated_at
                        ELSE trace.updated_at
                    END AS updated_at,
                    (
                        SELECT COUNT(*)
                        FROM messages AS messages
                        WHERE messages.thread_id = finalized.id
                    ) AS message_count,
                    (
                        SELECT messages.content
                        FROM messages AS messages
                        WHERE messages.thread_id = finalized.id
                        ORDER BY messages.turn_index DESC
                        LIMIT 1
                    ) AS last_message_preview
                FROM threads AS finalized
                JOIN threads AS trace
                    ON trace.conversation_id = finalized.conversation_id
                    AND trace.thread_type = 'trace'
                WHERE finalized.thread_type = 'finalized'
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
            return [_conversation_summary_from_row(row) for row in rows]

    def get_thread(
        self,
        thread_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ThreadRecord | None:
        with self._read_connection(connection) as active_connection:
            row = active_connection.execute(
                """
                SELECT id, conversation_id, thread_type, title, created_at, updated_at
                FROM threads
                WHERE id = ?
                """,
                (thread_id,),
            ).fetchone()
            return _thread_from_row(row) if row is not None else None

    def get_conversation(
        self,
        conversation_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ConversationThreads | None:
        with self._read_connection(connection) as active_connection:
            return self._get_conversation_locked(active_connection, conversation_id)

    def get_conversation_for_thread(
        self,
        thread_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ConversationThreads | None:
        with self._read_connection(connection) as active_connection:
            row = active_connection.execute(
                "SELECT conversation_id FROM threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                return None

            return self._get_conversation_locked(
                active_connection,
                str(row["conversation_id"]),
            )

    def get_or_create_conversation(
        self,
        *,
        conversation_id: str | None = None,
        title: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> ConversationThreads:
        with self._write_connection(connection) as active_connection:
            if conversation_id is not None:
                existing = self._get_conversation_locked(
                    active_connection,
                    conversation_id,
                )
                if existing is not None:
                    return existing

            normalized_title = _normalize_optional_text(title)
            created_at = _utc_now()
            resolved_conversation_id = conversation_id or str(uuid4())
            finalized = ThreadRecord(
                id=str(uuid4()),
                conversation_id=resolved_conversation_id,
                thread_type=THREAD_TYPE_FINALIZED,
                title=normalized_title,
                created_at=created_at,
                updated_at=created_at,
            )
            trace = ThreadRecord(
                id=str(uuid4()),
                conversation_id=resolved_conversation_id,
                thread_type=THREAD_TYPE_TRACE,
                title=normalized_title,
                created_at=created_at,
                updated_at=created_at,
            )

            active_connection.executemany(
                """
                INSERT INTO threads (
                    id,
                    conversation_id,
                    thread_type,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        finalized.id,
                        finalized.conversation_id,
                        finalized.thread_type,
                        finalized.title,
                        finalized.created_at,
                        finalized.updated_at,
                    ),
                    (
                        trace.id,
                        trace.conversation_id,
                        trace.thread_type,
                        trace.title,
                        trace.created_at,
                        trace.updated_at,
                    ),
                ],
            )

            return ConversationThreads(
                conversation_id=resolved_conversation_id,
                finalized=finalized,
                trace=trace,
            )

    def set_conversation_title(
        self,
        conversation_id: str,
        title: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        normalized_title = _require_text(title, "title")

        with self._write_connection(connection) as active_connection:
            result = active_connection.execute(
                """
                UPDATE threads
                SET title = ?
                WHERE conversation_id = ?
                """,
                (normalized_title, conversation_id),
            )
            if result.rowcount == 0:
                raise ValueError(f"Unknown conversation_id: {conversation_id}")

    def _get_conversation_locked(
        self,
        connection: sqlite3.Connection,
        conversation_id: str,
    ) -> ConversationThreads | None:
        rows = connection.execute(
            """
            SELECT id, conversation_id, thread_type, title, created_at, updated_at
            FROM threads
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()
        if not rows:
            return None

        threads_by_type = {
            thread.thread_type: thread for thread in (_thread_from_row(row) for row in rows)
        }

        finalized = threads_by_type.get(THREAD_TYPE_FINALIZED)
        trace = threads_by_type.get(THREAD_TYPE_TRACE)
        if finalized is None or trace is None:
            raise ConversationIntegrityError(
                f"Conversation {conversation_id} does not have both finalized and trace threads."
            )

        return ConversationThreads(
            conversation_id=conversation_id,
            finalized=finalized,
            trace=trace,
        )


class MessageRepository(_RepositoryBase):
    def add_finalized_user_message(
        self,
        thread_id: str,
        content: str,
        *,
        connection: sqlite3.Connection | None = None,
        created_at: str | None = None,
    ) -> MessageRecord:
        return self._insert_message(
            thread_id=thread_id,
            role="user",
            kind=FINALIZED_USER_KIND,
            content=content,
            expected_thread_type=THREAD_TYPE_FINALIZED,
            connection=connection,
            created_at=created_at,
        )

    def add_finalized_assistant_response(
        self,
        thread_id: str,
        content: str,
        *,
        connection: sqlite3.Connection | None = None,
        created_at: str | None = None,
    ) -> MessageRecord:
        return self._insert_message(
            thread_id=thread_id,
            role="assistant",
            kind=FINALIZED_ASSISTANT_KIND,
            content=content,
            expected_thread_type=THREAD_TYPE_FINALIZED,
            connection=connection,
            created_at=created_at,
        )

    def add_trace_message(
        self,
        thread_id: str,
        *,
        role: str,
        kind: str,
        content: str,
        connection: sqlite3.Connection | None = None,
        created_at: str | None = None,
    ) -> MessageRecord:
        return self._insert_message(
            thread_id=thread_id,
            role=role,
            kind=kind,
            content=content,
            expected_thread_type=THREAD_TYPE_TRACE,
            connection=connection,
            created_at=created_at,
        )

    def get_message(
        self,
        message_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> MessageRecord | None:
        with self._read_connection(connection) as active_connection:
            row = active_connection.execute(
                """
                SELECT id, thread_id, role, kind, content, turn_index, created_at
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
            return _message_from_row(row) if row is not None else None

    def list_messages(
        self,
        thread_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> list[MessageRecord]:
        with self._read_connection(connection) as active_connection:
            rows = active_connection.execute(
                """
                SELECT id, thread_id, role, kind, content, turn_index, created_at
                FROM messages
                WHERE thread_id = ?
                ORDER BY turn_index ASC
                """,
                (thread_id,),
            ).fetchall()
            return [_message_from_row(row) for row in rows]

    def get_finalized_user_messages_by_ids(
        self,
        message_ids: Sequence[str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> list[MessageRecord]:
        normalized_message_ids = [message_id.strip() for message_id in message_ids if message_id.strip()]
        if not normalized_message_ids:
            return []

        placeholders = ", ".join("?" for _ in normalized_message_ids)
        with self._read_connection(connection) as active_connection:
            rows = active_connection.execute(
                f"""
                SELECT
                    m.id,
                    m.thread_id,
                    m.role,
                    m.kind,
                    m.content,
                    m.turn_index,
                    m.created_at
                FROM messages AS m
                JOIN threads AS t
                    ON t.id = m.thread_id
                WHERE m.id IN ({placeholders})
                    AND m.role = 'user'
                    AND t.thread_type = 'finalized'
                """,
                tuple(normalized_message_ids),
            ).fetchall()

        messages_by_id = {
            message.id: message for message in (_message_from_row(row) for row in rows)
        }
        return [
            messages_by_id[message_id]
            for message_id in normalized_message_ids
            if message_id in messages_by_id
        ]

    def _insert_message(
        self,
        *,
        thread_id: str,
        role: str,
        kind: str,
        content: str,
        expected_thread_type: str,
        connection: sqlite3.Connection | None,
        created_at: str | None,
    ) -> MessageRecord:
        normalized_role = _require_text(role, "role")
        normalized_kind = _require_text(kind, "kind")
        if expected_thread_type == THREAD_TYPE_FINALIZED and normalized_role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                "Finalized threads only accept user messages and final assistant responses."
            )

        if not content.strip():
            raise ValueError("content must not be empty.")

        with self._write_connection(connection) as active_connection:
            thread = _thread_from_row(
                self._get_thread_row(active_connection, thread_id)
            )
            if thread.thread_type != expected_thread_type:
                raise ValueError(
                    f"Thread {thread_id} is {thread.thread_type}, expected {expected_thread_type}."
                )

            turn_index = int(
                active_connection.execute(
                    """
                    SELECT COALESCE(MAX(turn_index), -1) + 1
                    FROM messages
                    WHERE thread_id = ?
                    """,
                    (thread_id,),
                ).fetchone()[0]
            )
            timestamp = created_at or _utc_now()
            message = MessageRecord(
                id=str(uuid4()),
                thread_id=thread_id,
                role=normalized_role,
                kind=normalized_kind,
                content=content,
                turn_index=turn_index,
                created_at=timestamp,
            )

            active_connection.execute(
                """
                INSERT INTO messages (
                    id,
                    thread_id,
                    role,
                    kind,
                    content,
                    turn_index,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.thread_id,
                    message.role,
                    message.kind,
                    message.content,
                    message.turn_index,
                    message.created_at,
                ),
            )
            self._touch_thread(active_connection, thread_id, message.created_at)
            return message


class MessageSentenceRepository(_RepositoryBase):
    def add_finalized_message_sentences(
        self,
        *,
        message_id: str,
        thread_id: str,
        role: str,
        sentences: Sequence[str],
        qdrant_point_ids: Sequence[str | None] | None = None,
        connection: sqlite3.Connection | None = None,
        created_at: str | None = None,
    ) -> list[MessageSentenceRecord]:
        normalized_role = _require_text(role, "role")
        if normalized_role not in {"user", "assistant"}:
            raise ValueError("Sentence rows only support user and assistant roles.")

        normalized_sentences = [_require_text(sentence, "sentence") for sentence in sentences]
        if not normalized_sentences:
            return []

        resolved_qdrant_point_ids = list(qdrant_point_ids or [None] * len(normalized_sentences))
        if len(resolved_qdrant_point_ids) != len(normalized_sentences):
            raise ValueError(
                "qdrant_point_ids must match the sentence count when provided."
            )

        if normalized_role != "user" and any(
            point_id is not None for point_id in resolved_qdrant_point_ids
        ):
            raise ValueError("Assistant sentence rows cannot carry qdrant_point_id values.")

        with self._write_connection(connection) as active_connection:
            thread = _thread_from_row(
                self._get_thread_row(active_connection, thread_id)
            )
            if thread.thread_type != THREAD_TYPE_FINALIZED:
                raise ValueError("Trace-thread messages must not create sentence rows.")

            message = _message_from_row(
                self._get_message_row(active_connection, message_id)
            )
            if message.thread_id != thread_id:
                raise ValueError(
                    f"Message {message_id} does not belong to thread {thread_id}."
                )
            if message.role != normalized_role:
                raise ValueError(
                    f"Message {message_id} has role {message.role}, expected {normalized_role}."
                )

            timestamp = created_at or _utc_now()
            sentence_rows: list[MessageSentenceRecord] = []
            for sentence_index, text in enumerate(normalized_sentences):
                sentence = MessageSentenceRecord(
                    id=str(uuid4()),
                    message_id=message_id,
                    thread_id=thread_id,
                    role=normalized_role,
                    sentence_index=sentence_index,
                    text=text,
                    created_at=timestamp,
                    qdrant_point_id=resolved_qdrant_point_ids[sentence_index],
                )
                active_connection.execute(
                    """
                    INSERT INTO message_sentences (
                        id,
                        message_id,
                        thread_id,
                        role,
                        sentence_index,
                        text,
                        created_at,
                        qdrant_point_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sentence.id,
                        sentence.message_id,
                        sentence.thread_id,
                        sentence.role,
                        sentence.sentence_index,
                        sentence.text,
                        sentence.created_at,
                        sentence.qdrant_point_id,
                    ),
                )
                sentence_rows.append(sentence)

            return sentence_rows

    def list_sentences_for_message(
        self,
        message_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> list[MessageSentenceRecord]:
        with self._read_connection(connection) as active_connection:
            rows = active_connection.execute(
                """
                SELECT
                    id,
                    message_id,
                    thread_id,
                    role,
                    sentence_index,
                    text,
                    created_at,
                    qdrant_point_id
                FROM message_sentences
                WHERE message_id = ?
                ORDER BY sentence_index ASC
                """,
                (message_id,),
            ).fetchall()
            return [_message_sentence_from_row(row) for row in rows]

    def set_qdrant_point_ids(
        self,
        point_ids_by_sentence_id: Mapping[str, str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if not point_ids_by_sentence_id:
            return

        with self._write_connection(connection) as active_connection:
            for sentence_id, qdrant_point_id in point_ids_by_sentence_id.items():
                normalized_point_id = _require_text(qdrant_point_id, "qdrant_point_id")
                row = active_connection.execute(
                    """
                    SELECT id, role, qdrant_point_id
                    FROM message_sentences
                    WHERE id = ?
                    """,
                    (sentence_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Unknown sentence_id: {sentence_id}")
                if str(row["role"]) != "user":
                    raise ValueError(
                        "Only finalized user sentence rows may carry qdrant_point_id values."
                    )

                active_connection.execute(
                    """
                    UPDATE message_sentences
                    SET qdrant_point_id = ?
                    WHERE id = ?
                    """,
                    (normalized_point_id, sentence_id),
                )


class ConversationRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database
        self.threads = ThreadRepository(database)
        self.messages = MessageRepository(database)
        self.sentences = MessageSentenceRepository(database)

    def initialize(self) -> None:
        self._database.initialize()

    def delete_all(self) -> None:
        self._database.reset()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self._database.connection() as connection:
            yield connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._database.transaction() as connection:
            yield connection


def _thread_from_row(row: sqlite3.Row) -> ThreadRecord:
    return ThreadRecord(
        id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        thread_type=str(row["thread_type"]),
        title=str(row["title"]) if row["title"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _message_from_row(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        id=str(row["id"]),
        thread_id=str(row["thread_id"]),
        role=str(row["role"]),
        kind=str(row["kind"]),
        content=str(row["content"]),
        turn_index=int(row["turn_index"]),
        created_at=str(row["created_at"]),
    )


def _message_sentence_from_row(row: sqlite3.Row) -> MessageSentenceRecord:
    return MessageSentenceRecord(
        id=str(row["id"]),
        message_id=str(row["message_id"]),
        thread_id=str(row["thread_id"]),
        role=str(row["role"]),
        sentence_index=int(row["sentence_index"]),
        text=str(row["text"]),
        created_at=str(row["created_at"]),
        qdrant_point_id=(
            str(row["qdrant_point_id"]) if row["qdrant_point_id"] is not None else None
        ),
    )


def _conversation_summary_from_row(row: sqlite3.Row) -> ConversationSummaryRecord:
    return ConversationSummaryRecord(
        conversation_id=str(row["conversation_id"]),
        finalized_thread_id=str(row["finalized_thread_id"]),
        trace_thread_id=str(row["trace_thread_id"]),
        title=str(row["title"]) if row["title"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        message_count=int(row["message_count"]),
        last_message_preview=(
            str(row["last_message_preview"])
            if row["last_message_preview"] is not None
            else None
        ),
    )


def _require_text(value: str, field_name: str) -> str:
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized_value


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized_value = value.strip()
    return normalized_value or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
