"""Listen to the user: local push-to-talk speech-to-text using faster-whisper.

Records microphone audio until Enter is pressed, then transcribes it locally
(no network, no API key).
"""

from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DEFAULT_MODEL = "base.en"

# Selectable faster-whisper sizes, fastest to most accurate. The ".en" variants
# are English-only and faster; the plain names are multilingual.
AVAILABLE_MODELS = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v3",
)

# Cache one loaded model per name so switching sizes does not reload needlessly.
_models: dict = {}


def get_model(name: str = DEFAULT_MODEL):
    if name not in _models:
        # int8 on CPU is fast and light; good enough for dictation.
        _models[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _models[name]


def record_until_enter():
    """Capture mono float32 audio until the user presses Enter."""
    frames: list = []

    def callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        frames.append(indata.copy())

    print("Recording... press Enter to stop.", flush=True)
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        input()

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def transcribe(audio, model_name: str = DEFAULT_MODEL) -> str:
    if audio.size == 0:
        return ""
    model = get_model(model_name)
    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=1,  # greedy decode: much faster than the default beam_size=5
        vad_filter=True,  # skip silence so only speech is decoded
    )
    return " ".join(segment.text for segment in segments).strip()


def warm_up(model_name: str = DEFAULT_MODEL) -> None:
    """Load (and download if needed) the model up front, before recording.

    Done at startup so the first real transcription does not pay the load cost
    mid-conversation.
    """
    get_model(model_name)


def listen(model_name: str = DEFAULT_MODEL) -> str:
    """Record then transcribe; returns the recognized text."""
    audio = record_until_enter()
    print("Recording finished. Making transcript...", flush=True)
    return transcribe(audio, model_name)
