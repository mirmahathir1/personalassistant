#!/usr/bin/env python3
"""Round-trip Bangla TTS audio through Bangla speech-to-text."""

from __future__ import annotations

import importlib
import string
import sys
import unicodedata
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEXT = "আমি ভালো আছি"
AUDIO_PATH = Path(__file__).resolve().parent / "generated-bn.wav"

speaking_bn = importlib.import_module("speaking-bn")
listening_bn = importlib.import_module("listening-bn")


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    source_times = np.arange(audio.size, dtype=np.float64) / source_rate
    target_size = max(1, round(audio.size * target_rate / source_rate))
    target_times = np.arange(target_size, dtype=np.float64) / target_rate
    return np.interp(target_times, source_times, audio).astype(np.float32)


def normalize_text(text: str) -> str:
    punctuation = string.punctuation + "।"
    normalized = unicodedata.normalize("NFKC", text)
    without_punctuation = "".join(
        char for char in normalized if char not in punctuation
    )
    return " ".join(without_punctuation.split())


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main() -> int:
    print(f"expected: {TEXT}")

    audio, sample_rate = speaking_bn.synthesize_audio(TEXT)
    audio = np.asarray(audio, dtype=np.float32)
    audio = resample(audio, sample_rate, listening_bn.SAMPLE_RATE)
    write_wav(AUDIO_PATH, audio, listening_bn.SAMPLE_RATE)
    print(f"audio:    {AUDIO_PATH}")

    transcript = listening_bn.transcribe(audio, status_callback=status)
    print(f"actual:   {transcript}")

    exact_match = transcript.strip() == TEXT
    normalized_match = normalize_text(transcript) == normalize_text(TEXT)
    print(f"exact_match: {exact_match}")
    print(f"normalized_match: {normalized_match}")

    return 0 if normalized_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
