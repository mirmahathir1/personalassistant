#!/usr/bin/env python3
"""English speech -> text -> API reply pipeline test."""

from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.codex_client import build_client

INPUT_EN = "What is the sum of 213 and 435 ?"
EXPECTED = "648"
AUDIO_PATH = Path(__file__).resolve().parent / "generated-en.wav"
MODEL = "gpt-5.5"
REASONING_EFFORT = "low"

speaking_en = importlib.import_module("speaking-en")
listening_en = importlib.import_module("listening-en")


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


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main() -> int:
    print(f"input_en: {INPUT_EN}")
    print(f"expected_substring: {EXPECTED}")

    audio, sample_rate = speaking_en.synthesize_audio(INPUT_EN)
    audio = np.asarray(audio, dtype=np.float32)
    audio = resample(audio, sample_rate, listening_en.SAMPLE_RATE)
    write_wav(AUDIO_PATH, audio, listening_en.SAMPLE_RATE)
    print(f"audio: {AUDIO_PATH}")

    heard_en = listening_en.transcribe(audio, status_callback=status)
    print(f"heard_en: {heard_en}")

    client = build_client()
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[{"role": "user", "content": heard_en}],
    )
    reply = response.choices[0].message.content or ""
    print(f"reply: {reply}")

    if EXPECTED not in reply:
        print(
            f"Pipeline test failed: expected {EXPECTED!r} in reply.",
            file=sys.stderr,
        )
        return 1

    print("Speaking/listening/API pipeline test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
