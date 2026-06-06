"""Call your ChatGPT/Codex subscription through the local CLIProxyAPI server.

This helper does not query /v1/models or try to choose the best model. Pick the
model and reasoning effort explicitly for each request.
"""

import os
from typing import Literal

from openai import OpenAI

# Base URL of the local proxy (note the trailing /v1).
BASE_URL = os.getenv("CLIPROXY_BASE_URL", "http://localhost:8317/v1")

# Must match one of the values under `api-keys:` in config.yaml.
API_KEY = os.getenv("CLIPROXY_API_KEY", "sk-codexproxy-local-7f3a9c2e8b14d05f")

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


def build_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


client = build_client()


def chat(prompt: str, *, model: str, reasoning_effort: ReasoningEffort) -> str:
    resp = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[
            {"role": "system", "content": "You are a concise coding assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def stream(prompt: str, *, model: str, reasoning_effort: ReasoningEffort) -> None:
    s = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in s:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
