"""Bangla local neural text-to-speech using Facebook MMS-TTS."""

from __future__ import annotations

import queue
import re
import sys
import threading
from contextlib import nullcontext

import sounddevice as sd
import torch
from banglanum2words.num_convert import number_to_bangla_words
from transformers import AutoTokenizer, VitsModel

DEFAULT_MODEL = "facebook/mms-tts-ben"

# The MMS-TTS Bangla tokenizer has no numeral glyphs (০–৯), so digits are
# silently dropped from the waveform. Spell numbers out as Bangla words first.
_ASCII_TO_BANGLA_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_NUMBER_RUN = re.compile(r"[0-9০-৯]+")


def spell_numbers_bn(text: str) -> str:
    """Replace digit runs with their Bangla word spelling so TTS can speak them."""

    def replace(match: re.Match) -> str:
        bangla_digits = match.group().translate(_ASCII_TO_BANGLA_DIGITS)
        words = number_to_bangla_words(bangla_digits)
        return words or match.group()

    return _NUMBER_RUN.sub(replace, text)


def synthesize_audio(
    text: str,
    model_name: str = DEFAULT_MODEL,
    seed: int | None = None,
):
    """Return synthesized Bangla speech audio and its sample rate."""
    model = VitsModel.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(spell_numbers_bn(text), return_tensors="pt")
    context = torch.random.fork_rng() if seed is not None else nullcontext()
    with context:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            output = model(**inputs).waveform
    audio = output.squeeze().detach().cpu().numpy()
    return audio, model.config.sampling_rate


class Speaker:
    """Queue Bangla text and speak it aloud in order on a worker thread."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._torch = torch
        self._model = VitsModel.from_pretrained(model_name)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._sample_rate = self._model.config.sampling_rate
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
        audio = self._synthesize(text)
        sd.play(audio, self._sample_rate)
        sd.wait()

    def _synthesize(self, text: str):
        inputs = self._tokenizer(spell_numbers_bn(text), return_tensors="pt")
        with self._torch.no_grad():
            output = self._model(**inputs).waveform
        return output.squeeze().detach().cpu().numpy()

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
        if char not in ".!?\n।":
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
