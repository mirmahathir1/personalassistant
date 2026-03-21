from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .llm_service import ChatTurnResult, LlamaService, SessionCapacityError
from .model_catalog import DEFAULT_MODEL
from .schemas import ChatRequest, ChatResponse, HealthResponse, SessionResponse

settings = get_settings()
llm_service = LlamaService(settings=settings, model_spec=DEFAULT_MODEL)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.preload_model:
        llm_service.prepare_session()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    session = llm_service.session_status()
    return HealthResponse(
        status="ok",
        model=llm_service.model_name,
        model_path=llm_service.model_path,
        model_loaded=llm_service.is_loaded,
        session_initialized=session.initialized,
        turn_count=session.turn_count,
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
        result: ChatTurnResult = llm_service.generate_reply(payload.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SessionCapacityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        turn_count=result.turn_count,
        used_tokens=result.used_tokens,
        context_window=result.context_window,
    )
