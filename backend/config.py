from __future__ import annotations

import json
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from .model_catalog import DEFAULT_MODEL


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
    preload_model: bool = False
    model_repo_id: str = DEFAULT_MODEL.repo_id
    model_filename: str = DEFAULT_MODEL.filename
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
    system_prompt: str = DEFAULT_MODEL.system_prompt

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
