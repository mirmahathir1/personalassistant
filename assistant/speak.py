"""Speak to the user: local neural text-to-speech using Piper.

Speaks the assistant's reply sentence-by-sentence on a background thread, so
audio plays while the text keeps streaming. Everything runs locally (no network,
no API key).
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import sounddevice as sd
from piper import PiperVoice

DEFAULT_VOICE = "en_US-lessac-medium"
VOICES_DIR = Path(".assistant") / "voices"


def voice_path(name: str = DEFAULT_VOICE) -> Path:
    path = VOICES_DIR / f"{name}.onnx"
    if not path.exists():
        raise FileNotFoundError(
            f"Voice model not found: {path}\n"
            f"Download it with:\n"
            f"    python -m piper.download_voices {name} --download-dir {VOICES_DIR}"
        )
    return path


class Speaker:
    """Queue text and have it spoken aloud in order on a worker thread."""

    def __init__(self, voice_name: str = DEFAULT_VOICE):
        self._voice = PiperVoice.load(str(voice_path(voice_name)))
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break
            try:
                self._speak_now(text)
            except Exception as exc:  # don't let a playback hiccup kill the thread
                print(f"TTS playback error: {exc}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def _speak_now(self, text: str) -> None:
        for chunk in self._voice.synthesize(text):
            sd.play(chunk.audio_int16_array, chunk.sample_rate)
            sd.wait()

    def say(self, text: str) -> None:
        text = text.strip()
        if text:
            self._queue.put(text)

    def wait(self) -> None:
        """Block until everything queued so far has been spoken."""
        self._queue.join()

    def stop(self) -> None:
        """Drop anything pending and cut off current playback (e.g. on Ctrl-C)."""
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass
        try:
            sd.stop()
        except Exception:
            pass

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
