"""Call your ChatGPT/Codex subscription through the local CLIProxyAPI server.

The model is NOT hardcoded. At startup the client asks the proxy which models your
account actually exposes (GET /v1/models) and automatically selects the strongest
one. This works on any plan (Free or paid) because it only ever chooses from what
your own authentication grants.
"""

import os
import re
from openai import OpenAI

# Base URL of the local proxy (note the trailing /v1).
BASE_URL = os.getenv("CLIPROXY_BASE_URL", "http://localhost:8317/v1")

# Must match one of the values under `api-keys:` in config.yaml.
API_KEY = os.getenv("CLIPROXY_API_KEY", "sk-codexproxy-local-7f3a9c2e8b14d05f")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Lightweight / non-flagship variants to skip when picking the "strongest" model.
_WEAK_MARKERS = ("mini", "spark", "nano", "lite", "micro", "instant", "flash")


def _score(model_id: str) -> float:
    """Higher score = stronger model. Version number dominates; small tie-breakers
    favor 'max' variants and (for this coding use case) 'codex' variants."""
    name = model_id.lower()
    if any(marker in name for marker in _WEAK_MARKERS):
        return -1.0  # exclude small/fast variants from "strongest"
    match = re.search(r"(\d+(?:\.\d+)?)", name)  # e.g. 5.5, 5.1, 5
    version = float(match.group(1)) if match else 0.0
    score = version * 100.0
    if "max" in name:
        score += 5.0
    if "codex" in name:
        score += 2.0
    return score


def pick_strongest_model() -> str:
    """Query the proxy and return the strongest available model id.

    Override with the CLIPROXY_MODEL env var if you ever want to force a specific one.
    """
    forced = os.getenv("CLIPROXY_MODEL")
    if forced:
        return forced

    models = [m.id for m in client.models.list().data]
    if not models:
        raise RuntimeError("No models returned by the proxy. Is it logged in and running?")

    ranked = sorted(models, key=_score, reverse=True)
    best = ranked[0]
    if _score(best) < 0:  # everything was a 'weak' variant; just take the top one
        best = sorted(models, reverse=True)[0]
    return best


MODEL = pick_strongest_model()
print(f"[client] using model: {MODEL}")


def chat(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise coding assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def stream(prompt: str) -> None:
    s = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in s:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    print(chat("Write a one-line Python function to reverse a string."))
    print("\n--- streaming ---")
    stream("Explain what a Python decorator is in two sentences.")
