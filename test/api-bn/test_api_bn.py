#!/usr/bin/env python3
"""Bangla translation + API smoke test."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.codex_client import build_client

QUESTION_BN = "৭৬৮ এবং ২০৩১ এর যোগফল কত? শুধু সংখ্যাটি উত্তর দিন।"
EXPECTED_BN_SUM = "২৭৯৯"
MODEL = os.getenv("CODEX_TEST_MODEL", "gpt-5.5")
REASONING_EFFORT = os.getenv("CODEX_TEST_REASONING_EFFORT", "low")
ASCII_TO_BANGLA_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

translate_bn_to_en = importlib.import_module("translate-bn-to-en")
translate_en_to_bn = importlib.import_module("translate-en-to-bn")


def bangla_digit_text(text: str) -> str:
    return text.translate(ASCII_TO_BANGLA_DIGITS).replace(",", "")


def main() -> int:
    question_en = translate_bn_to_en.translate_bangla_to_english(QUESTION_BN)

    client = build_client()
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[{"role": "user", "content": question_en}],
    )
    api_output_en = response.choices[0].message.content or ""
    output_bn = translate_en_to_bn.translate_english_to_bangla(api_output_en)
    searchable_output = bangla_digit_text(output_bn)

    print(f"question_bn: {QUESTION_BN}")
    print(f"question_en: {question_en}")
    print(f"api_output_en: {api_output_en}")
    print(f"translated_output_bn: {output_bn}")
    print(f"expected_bangla_sum: {EXPECTED_BN_SUM}")

    if EXPECTED_BN_SUM not in searchable_output:
        print(
            f"API Bangla test failed: expected {EXPECTED_BN_SUM!r} in "
            f"{output_bn!r}.",
            file=sys.stderr,
        )
        return 1

    print("API Bangla test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
