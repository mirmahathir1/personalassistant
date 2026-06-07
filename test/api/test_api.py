#!/usr/bin/env python3
"""API smoke test for the local Codex proxy client."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.codex_client import build_client

QUESTION = "what is 768+1234?"
EXPECTED = "2002"
MODEL = os.getenv("CODEX_TEST_MODEL", "gpt-5.5")
REASONING_EFFORT = os.getenv("CODEX_TEST_REASONING_EFFORT", "low")


def main() -> int:
    client = build_client()
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[{"role": "user", "content": QUESTION}],
    )
    output = response.choices[0].message.content or ""

    print(f"question: {QUESTION}")
    print(f"expected_substring: {EXPECTED}")
    print(f"actual: {output}")

    if EXPECTED not in output:
        print(
            f"API test failed: expected {EXPECTED!r} in response.",
            file=sys.stderr,
        )
        return 1

    print("API test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
