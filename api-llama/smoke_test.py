#!/usr/bin/env python3
"""Smoke test for the local Llama 3.1 8B server.

Self-contained: it talks straight to http://localhost:8317/v1 and deliberately
mirrors the exact request shape the assistant sends (note `reasoning_effort`,
which llama-server harmlessly ignores). If this passes, the unmodified assistant
works against this server too.
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI

BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8317/v1")
# llama-server ignores the key when started without --api-key; any value works.
API_KEY = os.getenv("LLAMA_API_KEY", "sk-llama-local")
# llama-server ignores the requested model name and uses the loaded GGUF.
MODEL = os.getenv("LLAMA_TEST_MODEL", "llama-3.1-8b-lexi-uncensored")

QUESTION = "what is 768+1234? reply with just the number."
EXPECTED = "2002"


def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="low",
        messages=[{"role": "user", "content": QUESTION}],
        max_completion_tokens=64,
        max_tokens=64,
    )
    output = response.choices[0].message.content or ""

    print(f"question: {QUESTION}")
    print(f"expected_substring: {EXPECTED}")
    print(f"actual: {output}")

    if EXPECTED not in output:
        print(f"Llama smoke test failed: expected {EXPECTED!r} in response.", file=sys.stderr)
        return 1

    print("Llama smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
