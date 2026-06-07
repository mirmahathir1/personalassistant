#!/usr/bin/env python3
"""Bangla speech -> text -> API (in Bangla) -> speech -> text pipeline test."""

from __future__ import annotations

import importlib
import os
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from banglanum2words.num_convert import number_to_bangla_words

from api.codex_client import build_client

# Ask for a short sentence (not a bare digit): a one-word answer makes too short
# a TTS clip and Whisper hallucinates on it.
QUESTION_BN = "৩ এবং ৪ এর যোগফল কত? একটি ছোট বাংলা বাক্যে উত্তর দিন।"
EXPECTED_BN_SUM = "৭"
# MMS-TTS cannot speak numerals, so the answer is spoken (and thus heard) as
# Bangla words. Compare on the spelled-out form rather than the digits.
EXPECTED_BN_WORDS = number_to_bangla_words(EXPECTED_BN_SUM)
QUESTION_AUDIO_PATH = Path(__file__).resolve().parent / "question-bn.wav"
ANSWER_AUDIO_PATH = Path(__file__).resolve().parent / "answer-bn.wav"
MODEL = os.getenv("CODEX_TEST_MODEL", "gpt-5.4-mini")
REASONING_EFFORT = os.getenv("CODEX_TEST_REASONING_EFFORT", "low")

speaking_bn = importlib.import_module("speaking-bn")
listening_bn = importlib.import_module("listening-bn")


# Bangla ASR freely confuses near-homophones (সাত vs শাথ): sibilants all sound
# like স, and aspirated stops like their unaspirated pair. Fold them and drop
# whitespace so the comparison matches on pronunciation, not exact spelling.
_PHONETIC_FOLD = str.maketrans(
    {
        "শ": "স", "ষ": "স",
        "থ": "ত", "ঠ": "ত",
        "ধ": "দ", "ঢ": "দ",
        "ঝ": "জ", "খ": "ক",
        "ঘ": "গ", "ফ": "প",
        "ভ": "ব", "ণ": "ন",
    }
)


def phonetic_fold(text: str) -> str:
    return "".join(text.split()).translate(_PHONETIC_FOLD)


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
    print(f"question_bn: {QUESTION_BN}")
    print(f"expected_bangla_sum: {EXPECTED_BN_SUM}")

    # Bangla question -> audio -> heard Bangla text.
    question_audio = synthesize_for_listener(
        QUESTION_BN,
        speaking_bn,
        listening_bn,
        QUESTION_AUDIO_PATH,
        seed=2,
    )
    heard_bn = listening_bn.transcribe(
        question_audio,
        beam_size=1,
        status_callback=status,
    )

    # Ask the API directly in Bangla; the model answers in Bangla.
    client = build_client()
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[{"role": "user", "content": heard_bn}],
    )
    answer_bn = response.choices[0].message.content or ""

    # Bangla answer -> audio -> heard Bangla text.
    answer_audio = synthesize_for_listener(
        answer_bn,
        speaking_bn,
        listening_bn,
        ANSWER_AUDIO_PATH,
        seed=2,
    )
    heard_answer_bn = listening_bn.transcribe(
        answer_audio,
        beam_size=1,
        status_callback=status,
    )

    # The answer is spoken as Bangla words; match on folded pronunciation.
    heard_compact = phonetic_fold(heard_answer_bn)
    expected_compact = phonetic_fold(EXPECTED_BN_WORDS)

    print(f"question_audio: {QUESTION_AUDIO_PATH}")
    print(f"heard_bn: {heard_bn}")
    print(f"answer_bn: {answer_bn}")
    print(f"answer_audio: {ANSWER_AUDIO_PATH}")
    print(f"heard_answer_bn: {heard_answer_bn}")
    print(f"expected_bangla_words: {EXPECTED_BN_WORDS}")

    if expected_compact not in heard_compact:
        print(
            f"Bangla pipeline test failed: expected {EXPECTED_BN_WORDS!r} "
            f"(sum {EXPECTED_BN_SUM}) in {heard_answer_bn!r}.",
            file=sys.stderr,
        )
        return 1

    print("Speaking/listening/API Bangla pipeline test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
