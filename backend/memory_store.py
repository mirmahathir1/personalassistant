from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from time import monotonic, sleep
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import Settings


@dataclass(frozen=True)
class MemoryStoreStatus:
    ready: bool
    url: str
    collection: str
    embedding_model: str
    error: str | None = None


@dataclass(frozen=True)
class IndexedUserSentence:
    qdrant_point_id: str
    sentence_id: str
    thread_id: str
    message_id: str
    sentence_index: int
    text: str
    created_at: str


@dataclass(frozen=True)
class MemorySearchHit:
    qdrant_point_id: str
    sentence_id: str
    thread_id: str
    message_id: str
    role: str
    sentence_index: int
    created_at: str
    score: float


class MemoryStoreUnavailableError(RuntimeError):
    """Raised when Qdrant-backed memory operations cannot be completed."""


class MemoryStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._client: QdrantClient | None = None

    def prepare(self) -> MemoryStoreStatus:
        self._configure_fastembed_cache()
        return self._check_qdrant()

    def status(self) -> MemoryStoreStatus:
        self._configure_fastembed_cache()
        return self._check_qdrant()

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> MemoryStoreStatus:
        deadline = monotonic() + max(0.0, timeout_seconds)
        last_status = self.status()
        while not last_status.ready and monotonic() < deadline:
            sleep(max(0.0, poll_interval_seconds))
            last_status = self.status()

        return last_status

    def upsert_user_sentences(
        self,
        sentences: list[IndexedUserSentence],
    ) -> None:
        if not sentences:
            return

        self._configure_fastembed_cache()
        try:
            with self._lock:
                client = self._ensure_ready_client()
                client.upload_collection(
                    collection_name=self._settings.qdrant_collection,
                    vectors=[
                        models.Document(
                            text=sentence.text,
                            model=self._settings.fastembed_model_name,
                        )
                        for sentence in sentences
                    ],
                    payload=[
                        {
                            "sentence_id": sentence.sentence_id,
                            "thread_id": sentence.thread_id,
                            "message_id": sentence.message_id,
                            "role": "user",
                            "sentence_index": sentence.sentence_index,
                            "created_at": sentence.created_at,
                        }
                        for sentence in sentences
                    ],
                    ids=[self._validate_point_id(sentence.qdrant_point_id) for sentence in sentences],
                    wait=True,
                )
        except Exception as exc:
            raise self._unavailable_error(
                "upsert user sentence points",
                exc,
            ) from exc

    def search_user_sentences(
        self,
        query_text: str,
        *,
        limit: int,
        exclude_message_ids: list[str] | None = None,
        thread_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[MemorySearchHit]:
        normalized_query = query_text.strip()
        if not normalized_query:
            raise ValueError("query_text must not be empty.")
        if limit <= 0:
            return []

        self._configure_fastembed_cache()
        try:
            with self._lock:
                client = self._ensure_ready_client()
                query_kwargs = {
                    "collection_name": self._settings.qdrant_collection,
                    "query": models.Document(
                        text=normalized_query,
                        model=self._settings.fastembed_model_name,
                    ),
                    "query_filter": self._build_query_filter(
                        exclude_message_ids=exclude_message_ids,
                        thread_ids=thread_ids,
                    ),
                    "limit": limit,
                    "with_payload": True,
                }
                if score_threshold is not None:
                    query_kwargs["score_threshold"] = score_threshold

                response = client.query_points(
                    **query_kwargs,
                )
        except Exception as exc:
            raise self._unavailable_error(
                "search user sentence points",
                exc,
            ) from exc

        hits: list[MemorySearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                MemorySearchHit(
                    qdrant_point_id=str(point.id),
                    sentence_id=str(payload["sentence_id"]),
                    thread_id=str(payload["thread_id"]),
                    message_id=str(payload["message_id"]),
                    role=str(payload["role"]),
                    sentence_index=int(payload["sentence_index"]),
                    created_at=str(payload["created_at"]),
                    score=float(point.score or 0.0),
                )
            )

        return hits

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = [text.strip() for text in texts if text.strip()]
        if not normalized_texts:
            return []

        self._configure_fastembed_cache()
        try:
            with self._lock:
                client = self._ensure_ready_client()
                embed_documents = getattr(client, "_embed_documents", None)
                if embed_documents is None:
                    raise RuntimeError(
                        "The configured qdrant-client build does not expose FastEmbed document encoding."
                    )

                return [
                    vector
                    for _, vector in embed_documents(
                        normalized_texts,
                        embedding_model_name=self._settings.fastembed_model_name,
                    )
                ]
        except Exception as exc:
            raise self._unavailable_error("embed message texts", exc) from exc

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return

        self._configure_fastembed_cache()
        try:
            with self._lock:
                client = self._ensure_ready_client()
                client.delete(
                    collection_name=self._settings.qdrant_collection,
                    points_selector=models.PointIdsList(
                        points=[self._validate_point_id(point_id) for point_id in point_ids]
                    ),
                    wait=True,
                )
        except Exception as exc:
            raise self._unavailable_error("delete sentence points", exc) from exc

    def delete_all(self) -> None:
        self._configure_fastembed_cache()
        try:
            with self._lock:
                client = self._get_client()
                client.get_collections()
                if client.collection_exists(self._settings.qdrant_collection):
                    client.delete_collection(self._settings.qdrant_collection)
        except Exception as exc:
            raise self._unavailable_error("delete all memory", exc) from exc

    def _configure_fastembed_cache(self) -> None:
        os.environ["FASTEMBED_CACHE_PATH"] = self._settings.fastembed_cache_path

    def _check_qdrant(self) -> MemoryStoreStatus:
        try:
            with self._lock:
                self._ensure_ready_client()
        except Exception as exc:
            return MemoryStoreStatus(
                ready=False,
                url=self._settings.qdrant_url,
                collection=self._settings.qdrant_collection,
                embedding_model=self._settings.fastembed_model_name,
                error=f"Qdrant is unavailable at {self._settings.qdrant_url}: {exc}",
            )

        return MemoryStoreStatus(
            ready=True,
            url=self._settings.qdrant_url,
            collection=self._settings.qdrant_collection,
            embedding_model=self._settings.fastembed_model_name,
        )

    def _ensure_ready_client(self) -> QdrantClient:
        client = self._get_client()
        client.set_model(
            self._settings.fastembed_model_name,
            cache_dir=self._settings.fastembed_cache_path,
            lazy_load=True,
        )
        client.get_collections()

        if not client.collection_exists(self._settings.qdrant_collection):
            vector_size = client.get_embedding_size(self._settings.fastembed_model_name)
            client.create_collection(
                collection_name=self._settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        return client

    def _build_query_filter(
        self,
        *,
        exclude_message_ids: list[str] | None,
        thread_ids: list[str] | None,
    ) -> models.Filter:
        must_conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="role",
                match=models.MatchValue(value="user"),
            )
        ]
        must_not_conditions: list[models.FieldCondition] = []

        if thread_ids:
            must_conditions.append(
                self._match_condition(
                    key="thread_id",
                    values=thread_ids,
                )
            )

        if exclude_message_ids:
            must_not_conditions.append(
                self._match_condition(
                    key="message_id",
                    values=exclude_message_ids,
                )
            )

        return models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
        )

    def _match_condition(
        self,
        *,
        key: str,
        values: list[str],
    ) -> models.FieldCondition:
        normalized_values = [value.strip() for value in values if value.strip()]
        if not normalized_values:
            raise ValueError(f"{key} filter values must not be empty.")

        if len(normalized_values) == 1:
            return models.FieldCondition(
                key=key,
                match=models.MatchValue(value=normalized_values[0]),
            )

        return models.FieldCondition(
            key=key,
            match=models.MatchAny(any=normalized_values),
        )

    def _validate_point_id(self, point_id: str) -> str:
        normalized_point_id = point_id.strip()
        if not normalized_point_id:
            raise ValueError("qdrant_point_id must not be empty.")

        UUID(normalized_point_id)
        return normalized_point_id

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self._settings.qdrant_url)
        return self._client

    def _unavailable_error(
        self,
        action: str,
        exc: Exception,
    ) -> MemoryStoreUnavailableError:
        return MemoryStoreUnavailableError(
            f"Failed to {action} in Qdrant at {self._settings.qdrant_url}: {exc}"
        )
