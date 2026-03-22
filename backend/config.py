from __future__ import annotations

import json
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

SQLITE_PATH = "/data/app.db"
QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION = "conversation_sentences"
FASTEMBED_CACHE_PATH = "/data/fastembed"
FASTEMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MEMORY_RETRIEVAL_MODE = "sentence_to_full_message"
MEMORY_SENTENCE_HITS_PER_SENTENCE = 5
MEMORY_MAX_FULL_MESSAGES = 5
MEMORY_SEMANTIC_DEDUPE_THRESHOLD = 0.92
MEMORY_BLOCK_MAX_TOKENS = 256
QDRANT_STARTUP_TIMEOUT_SECONDS = 30.0
QDRANT_STARTUP_POLL_INTERVAL_SECONDS = 1.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Personal Assistant"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )
    model_cache_dir: str = "./models"
    model_path: str | None = None
    model_n_ctx: int = 4096
    model_n_threads: int = max(1, (os.cpu_count() or 2) - 1)
    model_n_gpu_layers: int = 0
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    chat_format: str | None = None
    verbose: bool = False

    @property
    def sqlite_path(self) -> str:
        return SQLITE_PATH

    @property
    def qdrant_url(self) -> str:
        return QDRANT_URL

    @property
    def qdrant_collection(self) -> str:
        return QDRANT_COLLECTION

    @property
    def fastembed_cache_path(self) -> str:
        return FASTEMBED_CACHE_PATH

    @property
    def fastembed_model_name(self) -> str:
        return FASTEMBED_MODEL_NAME

    @property
    def memory_retrieval_mode(self) -> str:
        return MEMORY_RETRIEVAL_MODE

    @property
    def memory_sentence_hits_per_sentence(self) -> int:
        return MEMORY_SENTENCE_HITS_PER_SENTENCE

    @property
    def memory_max_full_messages(self) -> int:
        return MEMORY_MAX_FULL_MESSAGES

    @property
    def memory_semantic_dedupe_threshold(self) -> float:
        return MEMORY_SEMANTIC_DEDUPE_THRESHOLD

    @property
    def memory_block_max_tokens(self) -> int:
        return MEMORY_BLOCK_MAX_TOKENS

    @property
    def qdrant_startup_timeout_seconds(self) -> float:
        return QDRANT_STARTUP_TIMEOUT_SECONDS

    @property
    def qdrant_startup_poll_interval_seconds(self) -> float:
        return QDRANT_STARTUP_POLL_INTERVAL_SECONDS

    @property
    def cors_origins_list(self) -> list[str]:
        raw_value = self.cors_origins.strip()
        if not raw_value:
            return []

        if raw_value.startswith("["):
            parsed_value = json.loads(raw_value)
            return [str(origin).strip() for origin in parsed_value if str(origin).strip()]

        return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
