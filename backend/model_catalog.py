from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    filename: str
    download_url: str
    system_prompt: str


DEFAULT_MODEL = ModelSpec(
    repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    filename="llama-3.2-3b-instruct-q4_k_m.gguf",
    download_url=(
        "https://huggingface.co/hugging-quants/"
        "Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/"
        "llama-3.2-3b-instruct-q4_k_m.gguf?download=true"
    ),
    system_prompt=(
        "You are a concise, helpful assistant. Answer directly and stay grounded "
        "in the user's request."
    ),
)
