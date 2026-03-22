from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    repo_id: str
    filename: str
    download_url: str
    system_prompt: str


AVAILABLE_MODELS = (
    ModelSpec(
        id="llama-3.1-8b-q4-k-m",
        label="Llama 3.1 8B Instruct Q4_K_M",
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        download_url=(
            "https://huggingface.co/bartowski/"
            "Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/"
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true"
        ),
        system_prompt="",
    ),
    ModelSpec(
        id="llama-3.1-8b-q5-k-m",
        label="Llama 3.1 8B Instruct Q5_K_M",
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        download_url=(
            "https://huggingface.co/bartowski/"
            "Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/"
            "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf?download=true"
        ),
        system_prompt="",
    ),
    ModelSpec(
        id="llama-3.2-3b-q4-k-m",
        label="Llama 3.2 3B Instruct Q4_K_M",
        repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
        filename="llama-3.2-3b-instruct-q4_k_m.gguf",
        download_url=(
            "https://huggingface.co/hugging-quants/"
            "Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/"
            "llama-3.2-3b-instruct-q4_k_m.gguf?download=true"
        ),
        system_prompt="",
    ),
)

MODEL_SPECS_BY_ID = {model.id: model for model in AVAILABLE_MODELS}

DEFAULT_MODEL = AVAILABLE_MODELS[0]


def get_model_spec(model_id: str) -> ModelSpec | None:
    return MODEL_SPECS_BY_ID.get(model_id)
