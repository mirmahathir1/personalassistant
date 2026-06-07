"""English local neural text-to-speech using Piper."""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import numpy as np
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


def synthesize_audio(text: str, voice_name: str = DEFAULT_VOICE):
    """Return synthesized English speech audio and its sample rate."""
    voice = PiperVoice.load(str(voice_path(voice_name)))
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.zeros(0, dtype=np.float32), 16000

    sample_rate = chunks[0].sample_rate
    audio = np.concatenate(
        [
            chunk.audio_int16_array.astype(np.float32, copy=False) / 32768.0
            for chunk in chunks
        ]
    )
    return audio, sample_rate


class Speaker:
    """Queue English text and speak it aloud in order on a worker thread."""

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
            except Exception as exc:
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
        """Drop anything pending and cut off current playback."""
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


def extract_sentence(buffer: str) -> tuple[str | None, str]:
    """Split one complete sentence off the front of buffer for speaking."""
    for index, char in enumerate(buffer):
        if char not in ".!?\n":
            continue
        end = index + 1
        while end < len(buffer) and buffer[end] in '")]’”':
            end += 1
        if end >= len(buffer):
            if char == "\n":
                return buffer[:end].strip(), buffer[end:]
            return None, buffer
        if buffer[end].isspace():
            return buffer[:end].strip(), buffer[end:].lstrip()
    return None, buffer


def speak_text(text: str, speaker: Speaker) -> None:
    pending = text
    while True:
        sentence, pending = extract_sentence(pending)
        if sentence is None:
            break
        speaker.say(sentence)
    if pending.strip():
        speaker.say(pending)
