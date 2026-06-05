"""Local push-to-talk voice input using faster-whisper.

Records microphone audio until Enter is pressed, then transcribes it locally
(no network, no API key). Heavy dependencies are imported lazily so the rest of
the assistant still runs in text-only mode when they are not installed.
"""

from __future__ import annotations

import sys

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


class VoiceUnavailable(RuntimeError):
    """Raised when voice dependencies or a microphone are not available."""


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise VoiceUnavailable(
            "Voice input needs extra packages. Install them with:\n"
            "    pip install faster-whisper sounddevice numpy"
        ) from exc


def get_model(name: str = DEFAULT_MODEL):
    if name not in _models:
        faster_whisper = _require("faster_whisper")
        # int8 on CPU is fast and light; good enough for dictation.
        _models[name] = faster_whisper.WhisperModel(
            name, device="cpu", compute_type="int8"
        )
    return _models[name]


def record_until_enter():
    """Capture mono float32 audio until the user presses Enter."""
    sd = _require("sounddevice")
    np = _require("numpy")

    frames: list = []

    def callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        frames.append(indata.copy())

    print("Recording... press Enter to stop.", flush=True)
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            input()
    except Exception as exc:  # microphone unavailable, etc.
        raise VoiceUnavailable(f"Could not record audio: {exc}") from exc

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
    mid-conversation. Failures are surfaced as VoiceUnavailable by get_model.
    """
    get_model(model_name)


def listen(model_name: str = DEFAULT_MODEL) -> str:
    """Record then transcribe; returns the recognized text."""
    audio = record_until_enter()
    print("Recording finished. Making transcript...", flush=True)
    return transcribe(audio, model_name)
