from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    filename: str
    download_url: str
    system_prompt: str


DEFAULT_MODEL = ModelSpec(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    download_url=(
        "https://huggingface.co/bartowski/"
        "Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/"
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true"
    ),
    system_prompt="",
)
