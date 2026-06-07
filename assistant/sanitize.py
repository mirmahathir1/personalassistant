"""Sanitize text returned by LLM calls."""

from __future__ import annotations


def sanitize_llm_text(text: str) -> str:
    return text.replace("*", "").strip()
