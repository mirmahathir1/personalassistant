#!/usr/bin/env python3
"""Round-trip English text through English->Bangla->English translation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEXT = "I'm a boy."

translate_en_to_bn = importlib.import_module("translate-en-to-bn")
translate_bn_to_en = importlib.import_module("translate-bn-to-en")


def main() -> int:
    bangla = translate_en_to_bn.translate_english_to_bangla(TEXT)
    english = translate_bn_to_en.translate_bangla_to_english(bangla)

    print(f"input_en: {TEXT}")
    print(f"translated_bn: {bangla}")
    print(f"roundtrip_en: {english}")
    print(f"exact_match: {english == TEXT}")

    if english != TEXT:
        print(
            f"Translation round-trip failed: expected {TEXT!r}, got {english!r}.",
            file=sys.stderr,
        )
        return 1

    print("Translation round-trip test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
