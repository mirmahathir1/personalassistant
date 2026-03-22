from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import SQLiteDatabase
from .llm_service import ChatTurnResult, LlamaService, SessionCapacityError
from .memory_store import MemoryStore, MemoryStoreUnavailableError
from .model_catalog import DEFAULT_MODEL
from .repositories import (
    ConversationRepository,
    ConversationSummaryRecord,
    ConversationThreads,
    MessageRecord,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetailResponse,
    ConversationSummaryResponse,
    DeleteAllMemoryResponse,
    HealthResponse,
    MessageResponse,
    SessionResponse,
)

settings = get_settings()
database = SQLiteDatabase(settings.sqlite_path)
conversation_repository = ConversationRepository(database)
memory_store = MemoryStore(settings=settings)
llm_service = LlamaService(
    settings=settings,
    model_spec=DEFAULT_MODEL,
    conversation_repository=conversation_repository,
    memory_store=memory_store,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    conversation_repository.initialize()
    sqlite_status = database.status()
    if not sqlite_status.ready:
        raise RuntimeError(sqlite_status.error or "SQLite initialization failed.")

    qdrant_status = memory_store.wait_until_ready(
        timeout_seconds=settings.qdrant_startup_timeout_seconds,
        poll_interval_seconds=settings.qdrant_startup_poll_interval_seconds,
    )
    if not qdrant_status.ready:
        raise RuntimeError(
            qdrant_status.error
            or "Qdrant did not become ready during backend startup."
        )
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _conversation_title(
    title: str | None,
    preview: str | None = None,
) -> str:
    normalized_title = (title or "").strip()
    if normalized_title:
        return normalized_title

    normalized_preview = _preview_text(preview)
    if normalized_preview is not None:
        return normalized_preview

    return "Untitled conversation"


def _preview_text(content: str | None, *, max_length: int = 120) -> str | None:
    if content is None:
        return None

    normalized = " ".join(content.split())
    if not normalized:
        return None
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def _message_response(message: MessageRecord) -> MessageResponse:
    return MessageResponse(
        id=message.id,
        thread_id=message.thread_id,
        role=message.role,
        kind=message.kind,
        content=message.content,
        turn_index=message.turn_index,
        created_at=message.created_at,
    )


def _conversation_summary_response(
    summary: ConversationSummaryRecord,
) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        conversation_id=summary.conversation_id,
        finalized_thread_id=summary.finalized_thread_id,
        trace_thread_id=summary.trace_thread_id,
        title=_conversation_title(summary.title, summary.last_message_preview),
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        message_count=summary.message_count,
        last_message_preview=_preview_text(summary.last_message_preview),
    )


def _conversation_detail_response(
    conversation: ConversationThreads,
) -> ConversationDetailResponse:
    finalized_messages = [
        _message_response(message)
        for message in conversation_repository.messages.list_messages(
            conversation.finalized.id
        )
    ]
    trace_messages = [
        _message_response(message)
        for message in conversation_repository.messages.list_messages(
            conversation.trace.id
        )
    ]
    updated_at = max(conversation.finalized.updated_at, conversation.trace.updated_at)
    preview_source = None
    if finalized_messages:
        preview_source = finalized_messages[-1].content

    return ConversationDetailResponse(
        conversation_id=conversation.conversation_id,
        finalized_thread_id=conversation.finalized.id,
        trace_thread_id=conversation.trace.id,
        title=_conversation_title(conversation.finalized.title, preview_source),
        created_at=conversation.finalized.created_at,
        updated_at=updated_at,
        finalized_messages=finalized_messages,
        trace_messages=trace_messages,
    )


@app.get("/api/health", response_model=HealthResponse)
def healthcheck(response: Response) -> HealthResponse:
    session = llm_service.session_status()
    sqlite_status = database.status()
    qdrant_status = memory_store.status()

    if not sqlite_status.ready or not qdrant_status.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if sqlite_status.ready and qdrant_status.ready else "degraded",
        model=llm_service.model_name,
        model_path=llm_service.model_path,
        model_loaded=llm_service.is_loaded,
        session_initialized=session.initialized,
        turn_count=session.turn_count,
        sqlite_ready=sqlite_status.ready,
        sqlite_path=sqlite_status.path,
        sqlite_error=sqlite_status.error,
        qdrant_ready=qdrant_status.ready,
        qdrant_url=qdrant_status.url,
        qdrant_collection=qdrant_status.collection,
        qdrant_embedding_model=qdrant_status.embedding_model,
        qdrant_error=qdrant_status.error,
    )


@app.get("/api/session", response_model=SessionResponse)
def session_status() -> SessionResponse:
    session = llm_service.session_status()
    return SessionResponse(
        initialized=session.initialized,
        turn_count=session.turn_count,
        used_tokens=session.used_tokens,
        context_window=session.context_window,
    )


@app.post("/api/reset", response_model=SessionResponse)
def reset_session() -> SessionResponse:
    session = llm_service.reset_session()
    return SessionResponse(
        initialized=session.initialized,
        turn_count=session.turn_count,
        used_tokens=session.used_tokens,
        context_window=session.context_window,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result: ChatTurnResult = llm_service.generate_reply(
            payload.message,
            finalized_thread_id=payload.thread_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SessionCapacityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MemoryStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate a response from the local model: {exc}",
        ) from exc

    return ChatResponse(
        reply=result.reply,
        model=result.model,
        thread_id=result.thread_id,
        turn_count=result.turn_count,
        used_tokens=result.used_tokens,
        context_window=result.context_window,
    )


@app.get("/api/threads", response_model=list[ConversationSummaryResponse])
def list_threads() -> list[ConversationSummaryResponse]:
    summaries = conversation_repository.threads.list_conversations()
    return [_conversation_summary_response(summary) for summary in summaries]


@app.post("/api/threads", response_model=ConversationDetailResponse)
def create_thread() -> ConversationDetailResponse:
    conversation = conversation_repository.threads.get_or_create_conversation()
    return _conversation_detail_response(conversation)


@app.get("/api/threads/{thread_id}", response_model=ConversationDetailResponse)
def get_thread(thread_id: str) -> ConversationDetailResponse:
    conversation = conversation_repository.threads.get_conversation_for_thread(thread_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")

    return _conversation_detail_response(conversation)


@app.post("/api/delete-all-memory", response_model=DeleteAllMemoryResponse)
def delete_all_memory() -> DeleteAllMemoryResponse:
    with conversation_repository.connection() as connection:
        deleted_conversations = int(
            connection.execute(
                "SELECT COUNT(DISTINCT conversation_id) FROM threads"
            ).fetchone()[0]
        )
        deleted_messages = int(
            connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )

    try:
        memory_store.delete_all()
        conversation_repository.delete_all()
        llm_service.reset_session()
    except MemoryStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete all stored memory: {exc}",
        ) from exc

    return DeleteAllMemoryResponse(
        status="ok",
        deleted_conversations=deleted_conversations,
        deleted_messages=deleted_messages,
    )
