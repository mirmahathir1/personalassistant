from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    thread_id: str
    turn_count: int
    used_tokens: int
    context_window: int


class MessageResponse(BaseModel):
    id: str
    thread_id: str
    role: str
    kind: str
    content: str
    turn_index: int
    created_at: str


class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    finalized_thread_id: str
    trace_thread_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message_preview: str | None = None


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    finalized_thread_id: str
    trace_thread_id: str
    title: str
    created_at: str
    updated_at: str
    finalized_messages: list[MessageResponse]
    trace_messages: list[MessageResponse]


class DeleteAllMemoryResponse(BaseModel):
    status: str
    deleted_conversations: int
    deleted_messages: int


class HealthResponse(BaseModel):
    status: str
    model: str
    model_path: str | None
    model_loaded: bool
    session_initialized: bool
    turn_count: int
    sqlite_ready: bool
    sqlite_path: str
    sqlite_error: str | None = None
    qdrant_ready: bool
    qdrant_url: str
    qdrant_collection: str
    qdrant_embedding_model: str
    qdrant_error: str | None = None


class SessionResponse(BaseModel):
    initialized: bool
    turn_count: int
    used_tokens: int
    context_window: int
