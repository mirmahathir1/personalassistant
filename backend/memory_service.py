from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from .config import Settings
from .memory_store import MemoryStore
from .repositories import (
    ConversationRepository,
    ConversationThreads,
    THREAD_TYPE_FINALIZED,
)
from .sentence_splitter import normalize_whitespace, split_sentences

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class RetrievedMemoryMessage:
    message_id: str
    thread_id: str
    content: str
    created_at: str
    best_score: float
    estimated_tokens: int
    matched_sentence_ids: tuple[str, ...]
    matched_point_ids: tuple[str, ...]


@dataclass(frozen=True)
class MemoryRetrievalResult:
    conversation: ConversationThreads | None
    query_sentences: tuple[str, ...]
    candidates: tuple[RetrievedMemoryMessage, ...]
    memories: tuple[RetrievedMemoryMessage, ...]
    memory_block: str | None


@dataclass
class _AggregatedMessageHit:
    message_id: str
    thread_id: str
    created_at: str
    best_score: float
    sentence_ids: list[str]
    point_ids: list[str]


@dataclass
class _CandidateWithEmbedding:
    memory: RetrievedMemoryMessage
    embedding: list[float]


class MemoryService:
    _CANDIDATE_POOL_MULTIPLIER = 3

    def __init__(
        self,
        *,
        settings: Settings,
        conversation_repository: ConversationRepository,
        memory_store: MemoryStore,
    ) -> None:
        self._settings = settings
        self._conversation_repository = conversation_repository
        self._memory_store = memory_store

    def retrieve_relevant_memories(
        self,
        user_message: str,
        *,
        finalized_thread_id: str | None = None,
        exclude_message_ids: Sequence[str] | None = None,
    ) -> MemoryRetrievalResult:
        normalized_user_message = user_message.strip()
        if not normalized_user_message:
            raise ValueError("A message is required for retrieval.")

        conversation = self._resolve_conversation(finalized_thread_id)
        query_sentences = tuple(
            fragment.text for fragment in split_sentences(normalized_user_message)
        )
        if not query_sentences:
            return MemoryRetrievalResult(
                conversation=conversation,
                query_sentences=(),
                candidates=(),
                memories=(),
                memory_block=None,
            )

        merged_hits = self._merge_sentence_hits(
            query_sentences=query_sentences,
            exclude_message_ids=exclude_message_ids,
        )
        if not merged_hits:
            return MemoryRetrievalResult(
                conversation=conversation,
                query_sentences=query_sentences,
                candidates=(),
                memories=(),
                memory_block=None,
            )

        candidates = self._expand_hits_to_full_messages(merged_hits)
        selected_memories = self._select_memories(candidates)

        return MemoryRetrievalResult(
            conversation=conversation,
            query_sentences=query_sentences,
            candidates=tuple(candidates),
            memories=tuple(selected_memories),
            memory_block=self._build_memory_block(selected_memories),
        )

    def _resolve_conversation(
        self,
        finalized_thread_id: str | None,
    ) -> ConversationThreads | None:
        if finalized_thread_id is None:
            return None

        thread = self._conversation_repository.threads.get_thread(finalized_thread_id)
        if thread is None:
            raise ValueError(f"Unknown thread_id: {finalized_thread_id}")
        if thread.thread_type != THREAD_TYPE_FINALIZED:
            raise ValueError(
                f"Expected a finalized thread_id, got {thread.thread_type}: {finalized_thread_id}"
            )

        conversation = self._conversation_repository.threads.get_conversation_for_thread(
            finalized_thread_id
        )
        if conversation is None:
            raise ValueError(
                f"Could not resolve the paired conversation for thread_id: {finalized_thread_id}"
            )
        return conversation

    def _merge_sentence_hits(
        self,
        *,
        query_sentences: Sequence[str],
        exclude_message_ids: Sequence[str] | None,
    ) -> list[_AggregatedMessageHit]:
        aggregated_hits: dict[str, _AggregatedMessageHit] = {}
        normalized_exclude_ids = [
            message_id.strip()
            for message_id in (exclude_message_ids or [])
            if message_id.strip()
        ]
        candidate_limit = max(
            self._settings.memory_sentence_hits_per_sentence,
            self._settings.memory_max_full_messages * self._CANDIDATE_POOL_MULTIPLIER,
        )

        for sentence in query_sentences:
            sentence_hits = self._memory_store.search_user_sentences(
                sentence,
                limit=candidate_limit,
                exclude_message_ids=normalized_exclude_ids or None,
                score_threshold=None,
            )
            for hit in sentence_hits:
                if hit.message_id in normalized_exclude_ids:
                    continue

                entry = aggregated_hits.get(hit.message_id)
                if entry is None:
                    entry = _AggregatedMessageHit(
                        message_id=hit.message_id,
                        thread_id=hit.thread_id,
                        created_at=hit.created_at,
                        best_score=hit.score,
                        sentence_ids=[],
                        point_ids=[],
                    )
                    aggregated_hits[hit.message_id] = entry

                entry.best_score = max(entry.best_score, hit.score)
                if hit.sentence_id not in entry.sentence_ids:
                    entry.sentence_ids.append(hit.sentence_id)
                if hit.qdrant_point_id not in entry.point_ids:
                    entry.point_ids.append(hit.qdrant_point_id)

        return sorted(
            aggregated_hits.values(),
            key=lambda hit: hit.best_score,
            reverse=True,
        )

    def _expand_hits_to_full_messages(
        self,
        aggregated_hits: Sequence[_AggregatedMessageHit],
    ) -> list[RetrievedMemoryMessage]:
        ordered_message_ids = [hit.message_id for hit in aggregated_hits]
        messages = self._conversation_repository.messages.get_finalized_user_messages_by_ids(
            ordered_message_ids
        )
        messages_by_id = {message.id: message for message in messages}

        candidates: list[RetrievedMemoryMessage] = []
        for hit in aggregated_hits:
            message = messages_by_id.get(hit.message_id)
            if message is None:
                continue

            candidates.append(
                RetrievedMemoryMessage(
                    message_id=message.id,
                    thread_id=message.thread_id,
                    content=message.content,
                    created_at=message.created_at,
                    best_score=hit.best_score,
                    estimated_tokens=self._estimate_tokens(message.content),
                    matched_sentence_ids=tuple(hit.sentence_ids),
                    matched_point_ids=tuple(hit.point_ids),
                )
            )

        return sorted(
            candidates,
            key=lambda memory: (-memory.best_score, memory.estimated_tokens),
        )

    def _select_memories(
        self,
        candidates: Sequence[RetrievedMemoryMessage],
    ) -> list[RetrievedMemoryMessage]:
        if not candidates:
            return []

        embeddings = self._memory_store.embed_texts(
            [candidate.content for candidate in candidates]
        )
        embedded_candidates = [
            _CandidateWithEmbedding(memory=candidate, embedding=embedding)
            for candidate, embedding in zip(candidates, embeddings)
        ]

        selected_candidates: list[_CandidateWithEmbedding] = []
        used_tokens = 0
        for candidate in embedded_candidates:
            if len(selected_candidates) >= self._settings.memory_max_full_messages:
                break

            if candidate.memory.estimated_tokens > self._settings.memory_block_max_tokens:
                continue
            if used_tokens + candidate.memory.estimated_tokens > self._settings.memory_block_max_tokens:
                continue
            if self._is_semantic_duplicate(candidate, selected_candidates):
                continue

            selected_candidates.append(candidate)
            used_tokens += candidate.memory.estimated_tokens

        return [candidate.memory for candidate in selected_candidates]

    def _is_semantic_duplicate(
        self,
        candidate: _CandidateWithEmbedding,
        selected_candidates: Sequence[_CandidateWithEmbedding],
    ) -> bool:
        for selected in selected_candidates:
            if (
                self._cosine_similarity(candidate.embedding, selected.embedding)
                >= self._settings.memory_semantic_dedupe_threshold
            ):
                return True
        return False

    def _build_memory_block(
        self,
        memories: Sequence[RetrievedMemoryMessage],
    ) -> str | None:
        if not memories:
            return None

        lines = ["Relevant memory:"]
        for memory in memories:
            lines.append(
                f"- User message: {normalize_whitespace(memory.content)}"
            )
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        normalized_text = normalize_whitespace(text)
        if not normalized_text:
            return 0
        return len(_TOKEN_RE.findall(normalized_text))

    def _cosine_similarity(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
        return dot_product / (left_norm * right_norm)
