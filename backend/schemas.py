from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str
    model: str
    turn_count: int
    used_tokens: int
    context_window: int


class HealthResponse(BaseModel):
    status: str
    model: str
    model_path: str | None
    model_loaded: bool
    session_initialized: bool
    turn_count: int


class SessionResponse(BaseModel):
    initialized: bool
    turn_count: int
    used_tokens: int
    context_window: int
