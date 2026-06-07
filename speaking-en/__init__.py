"""English text-to-speech helpers."""

from .speech import DEFAULT_VOICE, Speaker, extract_sentence, speak_text, synthesize_audio

__all__ = [
    "DEFAULT_VOICE",
    "Speaker",
    "extract_sentence",
    "speak_text",
    "synthesize_audio",
]
