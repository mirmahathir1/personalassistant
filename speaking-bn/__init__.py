"""Bangla text-to-speech helpers."""

from .speech import (
    DEFAULT_MODEL,
    Speaker,
    extract_sentence,
    speak_text,
    synthesize_audio,
)

__all__ = [
    "DEFAULT_MODEL",
    "Speaker",
    "extract_sentence",
    "speak_text",
    "synthesize_audio",
]
