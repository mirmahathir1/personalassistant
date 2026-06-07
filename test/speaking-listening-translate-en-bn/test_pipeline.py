#!/usr/bin/env python3
"""Bangla speech -> text -> English speech -> text -> Bangla pipeline test."""

from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INPUT_BN = "আমি একটা ছেলে।"
BN_AUDIO_PATH = Path(__file__).resolve().parent / "generated-bn.wav"
EN_AUDIO_PATH = Path(__file__).resolve().parent / "generated-en.wav"

speaking_bn = importlib.import_module("speaking-bn")
listening_bn = importlib.import_module("listening-bn")
translate_bn_to_en = importlib.import_module("translate-bn-to-en")
speaking_en = importlib.import_module("speaking-en")
listening_en = importlib.import_module("listening-en")
translate_en_to_bn = importlib.import_module("translate-en-to-bn")


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


def synthesize_for_listener(
    text: str,
    speaker_module,
    listener_module,
    audio_path: Path,
    **synthesize_kwargs,
) -> np.ndarray:
    audio, sample_rate = speaker_module.synthesize_audio(text, **synthesize_kwargs)
    audio = np.asarray(audio, dtype=np.float32)
    audio = resample(audio, sample_rate, listener_module.SAMPLE_RATE)
    write_wav(audio_path, audio, listener_module.SAMPLE_RATE)
    return audio


def main() -> int:
    print(f"input_bn: {INPUT_BN}")

    bangla_audio = synthesize_for_listener(
        INPUT_BN,
        speaking_bn,
        listening_bn,
        BN_AUDIO_PATH,
        seed=2,
    )
    heard_bn = listening_bn.transcribe(
        bangla_audio,
        beam_size=1,
        status_callback=status,
    )
    english = translate_bn_to_en.translate_bangla_to_english(heard_bn)

    english_audio = synthesize_for_listener(
        english,
        speaking_en,
        listening_en,
        EN_AUDIO_PATH,
    )
    heard_en = listening_en.transcribe(english_audio, status_callback=status)
    output_bn = translate_en_to_bn.translate_english_to_bangla(heard_en)

    print(f"bangla_audio: {BN_AUDIO_PATH}")
    print(f"heard_bn: {heard_bn}")
    print(f"translated_en: {english}")
    print(f"english_audio: {EN_AUDIO_PATH}")
    print(f"heard_en: {heard_en}")
    print(f"output_bn: {output_bn}")
    print(f"exact_match: {output_bn == INPUT_BN}")

    if output_bn != INPUT_BN:
        print(
            f"Pipeline test failed: expected {INPUT_BN!r}, got {output_bn!r}.",
            file=sys.stderr,
        )
        return 1

    print("Speaking/listening/translation pipeline test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
