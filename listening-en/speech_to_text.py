"""English microphone speech-to-text using faster-whisper.

After listening starts, recording begins when speech or other sound is detected.
Recording stops after a silence period, then transcribes locally (no network,
no API key after the selected model is cached).
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable

import huggingface_hub.constants as hub_constants
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel, utils

SAMPLE_RATE = 16000
DEFAULT_MODEL = "base.en"
SILENCE_SECONDS = 3.0
CHUNK_SECONDS = 0.1
PRE_ROLL_SECONDS = 0.3
START_RMS_THRESHOLD = 0.015
STOP_RMS_THRESHOLD = 0.01
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

StatusCallback = Callable[[str], None]

# Cache one loaded model per runtime setting so model switching is not reloaded
# needlessly.
_models: dict[tuple[str, str, str], Any] = {}


def print_status(message: str) -> None:
    print(message, flush=True)


def get_model(
    name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
    model_path: str | None = None,
    status_callback: StatusCallback = print_status,
):
    key = (name, device, compute_type)
    if key not in _models:
        if model_path is None:
            model_path = ensure_model_available(name, status_callback=status_callback)
        _models[key] = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )
    return _models[key]


def disable_huggingface_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    hub_constants.HF_HUB_DISABLE_XET = True


def model_repo_id(size_or_id: str) -> str | None:
    if Path(size_or_id).expanduser().exists():
        return None
    if "/" in size_or_id:
        return size_or_id

    disable_huggingface_xet()
    repo_id = utils._MODELS.get(size_or_id)
    if repo_id is None:
        raise ValueError(
            "Invalid model size '%s', expected one of: %s"
            % (size_or_id, ", ".join(utils._MODELS.keys()))
        )
    return repo_id


def ensure_model_available(
    model_name: str = DEFAULT_MODEL,
    status_callback: StatusCallback = print_status,
) -> str:
    """Check/download the faster-whisper model and return a local load path."""
    disable_huggingface_xet()
    repo_id = model_repo_id(model_name)
    if repo_id is None:
        status_callback(f"Using local model path: {model_name}")
        return str(Path(model_name).expanduser())

    status_callback(f"Checking speech model cache: {model_name} ({repo_id})")
    try:
        model_path = utils.download_model(model_name, local_files_only=True)
        status_callback("Speech model already downloaded.")
        return model_path
    except Exception:
        status_callback("Speech model is not fully cached. Downloading now...")

    model_path = utils.download_model(model_name, local_files_only=False)
    status_callback("Speech model download complete.")
    return model_path


def audio_rms(audio) -> float:
    if audio.size == 0:
        return 0.0
    as_float = audio.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(as_float * as_float)))


def record_until_silence(
    silence_seconds: float = SILENCE_SECONDS,
    chunk_seconds: float = CHUNK_SECONDS,
    start_threshold: float = START_RMS_THRESHOLD,
    stop_threshold: float = STOP_RMS_THRESHOLD,
    pre_roll_seconds: float = PRE_ROLL_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    status_callback: StatusCallback = print_status,
):
    """Capture mono float32 audio after sound starts, then stop on silence."""
    if silence_seconds <= 0:
        raise ValueError("silence_seconds must be positive.")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    blocksize = max(1, int(sample_rate * chunk_seconds))
    max_silent_chunks = max(1, math.ceil(silence_seconds / chunk_seconds))
    pre_roll_chunks = max(0, math.ceil(pre_roll_seconds / chunk_seconds))
    pre_roll: deque = deque(maxlen=pre_roll_chunks)
    frames = []
    recording = False
    silent_chunks = 0

    status_callback("Listening... start speaking when ready.")
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
    ) as stream:
        while True:
            chunk, overflowed = stream.read(blocksize)
            if overflowed:
                print(
                    "Audio input overflowed; some audio may be missing.",
                    file=sys.stderr,
                )

            audio = chunk.flatten().copy()
            level = audio_rms(audio)

            if not recording:
                if pre_roll.maxlen:
                    pre_roll.append(audio)

                if level >= start_threshold:
                    status_callback(
                        "Sound detected. Recording until "
                        f"{silence_seconds:g} seconds of silence..."
                    )
                    recording = True
                    if pre_roll.maxlen:
                        frames.extend(pre_roll)
                    else:
                        frames.append(audio)
                continue

            frames.append(audio)
            if level < stop_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= max_silent_chunks:
                status_callback("Silence detected. Making transcript...")
                break

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames)


def transcribe(
    audio,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
    status_callback: StatusCallback = print_status,
) -> str:
    if audio.size == 0:
        return ""

    model = get_model(
        model_name,
        device=device,
        compute_type=compute_type,
        status_callback=status_callback,
    )
    segments, _info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text for segment in segments).strip()


def warm_up(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
    status_callback: StatusCallback = print_status,
) -> None:
    """Load and download the model up front before recording."""
    model_path = ensure_model_available(model_name, status_callback=status_callback)
    status_callback("Loading speech model...")
    get_model(
        model_name,
        device=device,
        compute_type=compute_type,
        model_path=model_path,
        status_callback=status_callback,
    )
    status_callback("Speech model ready.")


def listen(
    model_name: str = DEFAULT_MODEL,
    status_callback: StatusCallback = print_status,
) -> str:
    """Record then transcribe; returns the recognized text."""
    audio = record_until_silence(status_callback=status_callback)
    return transcribe(
        audio,
        model_name=model_name,
        status_callback=status_callback,
    )
