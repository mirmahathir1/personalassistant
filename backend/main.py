"""Single-thread voice chat backend.

Chat and TTS providers are both chosen per request from the frontend
(Groq cloud or local Ollama for chat; Groq Orpheus or local Piper for TTS);
STT (Whisper) always runs on Groq. The whole app keeps exactly one
conversation thread in memory, persisted to a JSON file across restarts.

CHAT_PROVIDER / TTS_PROVIDER env vars only set the dropdowns' defaults.
"""

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from groq import Groq
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Chat model per provider; the frontend picks the provider per request.
CHAT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    # uncensored Llama-3.1 fine-tune (Lexi V2), pulled from HuggingFace via Ollama
    "ollama": "hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M",
}
# Friendly labels for the dropdown.
CHAT_LABELS = {"groq": "Groq Llama 70B", "ollama": "Local Lexi (uncensored)"}

# Default provider used when a request doesn't specify one.
DEFAULT_CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "groq").strip().lower()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# STT is Groq-only (Whisper).
STT_MODEL = "whisper-large-v3"

# Default TTS provider when a request doesn't pick a specific voice:
# "groq" (cloud Orpheus) or "piper" (fully offline, on-CPU). Either way, the
# dropdown lets the user pick any voice from either provider per request.
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "groq").strip().lower()
TTS_MODEL = "canopylabs/orpheus-v1-english"  # groq

# Online Groq Orpheus voices to expose (female-leaning subset of the 6 available).
GROQ_VOICES = ["diana", "hannah", "autumn"]
TTS_VOICE = GROQ_VOICES[0]  # default Groq voice

# Hardcoded offline voice used for *asterisk-wrapped* segments (always Piper).
# "Sofia" = en_US-libritts_r-medium.
ASTERISK_VOICE = os.environ.get("ASTERISK_VOICE", "en_US-libritts_r-medium")

# Piper (offline) voices live in backend/voices/ as <id>.onnx (+ .onnx.json).
# Selectable voices are those whose .onnx model is present on disk.
VOICES_DIR = Path(__file__).resolve().parent / "voices"

# Friendly display names; ids are the .onnx filename stems.
_VOICE_LABELS = {
    "en_US-amy-medium": "Amy",
    "en_US-hfc_female-medium": "Hannah",
    "en_US-libritts_r-medium": "Sofia",
    "en_US-kristin-medium": "Kristin",
    "en_US-kathleen-low": "Kathleen",
    "en_US-lessac-medium": "Lessac",
}


SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# Map a voice id to its short sample-file stem (used to curate the menu).
_VOICE_SAMPLE = {
    "en_US-amy-medium": "amy",
    "en_US-hfc_female-medium": "hfc_female",
    "en_US-libritts_r-medium": "libritts_r",
    "en_US-kristin-medium": "kristin",
    "en_US-kathleen-low": "kathleen",
    "en_US-lessac-medium": "lessac",
}


def _piper_voices() -> list[dict]:
    """Offline Piper voices: model present on disk AND a kept sample WAV.

    Deleting a sample under backend/samples/ removes that voice from the menu,
    so the curated set is whatever samples you keep.
    """
    out = []
    for onnx in sorted(VOICES_DIR.glob("*.onnx")):
        vid = onnx.stem
        sample = _VOICE_SAMPLE.get(vid, vid)
        if not (SAMPLES_DIR / f"{sample}.wav").exists():
            continue
        out.append(
            {"id": f"piper:{vid}", "label": _VOICE_LABELS.get(vid, vid), "online": False}
        )
    return sorted(out, key=lambda v: v["label"])


def _groq_voices() -> list[dict]:
    """Online Groq Orpheus voices."""
    return [
        {"id": f"groq:{v}", "label": v.capitalize(), "online": True} for v in GROQ_VOICES
    ]


def _available_voices() -> list[dict]:
    """All selectable voices (online Groq + offline Piper), online listed first."""
    return _groq_voices() + _piper_voices()


# Namespaced default voice id (e.g. "groq:diana" or "piper:en_US-amy-medium"),
# chosen by TTS_PROVIDER. PIPER_VOICE optionally overrides the offline default.
_PIPER_DEFAULT_MODEL = os.environ.get("PIPER_VOICE", "")


def _default_voice_id() -> str:
    avail = _available_voices()
    if TTS_PROVIDER == "piper":
        if _PIPER_DEFAULT_MODEL:
            return f"piper:{_PIPER_DEFAULT_MODEL}"
        piper = [v for v in avail if not v["online"]]
        return piper[0]["id"] if piper else f"groq:{TTS_VOICE}"
    return f"groq:{TTS_VOICE}"


DEFAULT_VOICE = _default_voice_id()

SYSTEM_PROMPT = "You are a helpful, concise voice assistant. Keep replies natural and to the point."


def _load_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    # Fall back to api_key.txt at the project root (one level up from backend/).
    key_file = Path(__file__).resolve().parent.parent / "api_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    raise RuntimeError(
        "No Groq API key found. Set GROQ_API_KEY or create api_key.txt at the project root."
    )


# Groq client: always used for STT/TTS.
groq_key = _load_groq_key()
groq_client = Groq(api_key=groq_key)

# One OpenAI-compatible chat client per provider (both Groq and Ollama speak it).
# The frontend chooses which to use per request via a dropdown.
chat_clients = {
    "groq": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key),
    "ollama": OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"),  # key ignored by Ollama
}

if DEFAULT_CHAT_PROVIDER not in chat_clients:
    DEFAULT_CHAT_PROVIDER = "groq"

print(f"[startup] chat providers={list(chat_clients)} default={DEFAULT_CHAT_PROVIDER}")

# Offline TTS: load each Piper voice once, lazily, and cache it by id.
_piper_cache: dict[str, object] = {}


def _get_piper_voice(voice_id: str):
    if voice_id not in _piper_cache:
        from piper import PiperVoice  # imported lazily so groq-only runs don't need it

        onnx = VOICES_DIR / f"{voice_id}.onnx"
        if not onnx.exists():
            raise RuntimeError(
                f"Piper voice {voice_id!r} not found at {onnx}. "
                "Download it into backend/voices/ (see README)."
            )
        _piper_cache[voice_id] = PiperVoice.load(str(onnx))
    return _piper_cache[voice_id]


print(f"[startup] tts default provider={TTS_PROVIDER} default voice={DEFAULT_VOICE}")

# ---------------------------------------------------------------------------
# Single conversation thread, persisted to a JSON file
# ---------------------------------------------------------------------------

# Stored next to this file (override with CHAT_HISTORY_FILE).
HISTORY_FILE = Path(
    os.environ.get("CHAT_HISTORY_FILE", Path(__file__).resolve().parent / "chat_history.json")
)


def _fresh_conversation() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _load_conversation() -> list[dict]:
    """Load the saved thread, or start fresh if there's nothing valid on disk."""
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list) and data:
                # Keep whatever was saved; ensure a system prompt leads the thread.
                if data[0].get("role") != "system":
                    data.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                return data
        except (json.JSONDecodeError, OSError, AttributeError) as exc:
            print(f"[startup] could not read {HISTORY_FILE}: {exc}; starting fresh")
    return _fresh_conversation()


def _save_conversation() -> None:
    """Persist the thread atomically so a crash mid-write can't corrupt it."""
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(conversation, ensure_ascii=False, indent=2))
    tmp.replace(HISTORY_FILE)


conversation: list[dict] = _load_conversation()
print(f"[startup] loaded {len([m for m in conversation if m['role'] != 'system'])} saved messages")

# ---------------------------------------------------------------------------
# UI settings (dropdown selections), persisted to a JSON file
# ---------------------------------------------------------------------------

SETTINGS_FILE = Path(
    os.environ.get("SETTINGS_FILE", Path(__file__).resolve().parent / "settings.json")
)

# Only these keys are accepted/persisted, with their fallback defaults.
_SETTINGS_DEFAULTS = {
    "chatProvider": DEFAULT_CHAT_PROVIDER,
    "voice": DEFAULT_VOICE,
}


def _load_settings() -> dict:
    """Load saved UI settings merged over defaults; ignore unknown keys."""
    data = {}
    if SETTINGS_FILE.exists():
        try:
            raw = json.loads(SETTINGS_FILE.read_text())
            if isinstance(raw, dict):
                data = {k: raw[k] for k in _SETTINGS_DEFAULTS if k in raw}
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[startup] could not read {SETTINGS_FILE}: {exc}; using defaults")
    return {**_SETTINGS_DEFAULTS, **data}


def _save_settings() -> None:
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    tmp.replace(SETTINGS_FILE)


settings: dict = _load_settings()
print(f"[startup] settings {settings}")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Openclaw Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    provider: str | None = None  # "groq" or "ollama"; defaults to DEFAULT_CHAT_PROVIDER


class ChatResponse(BaseModel):
    reply: str


class TTSRequest(BaseModel):
    message: str
    voice: str | None = None  # Piper voice id; ignored by the Groq TTS provider


class SettingsRequest(BaseModel):
    chatProvider: str | None = None
    voice: str | None = None


@app.get("/api/settings")
def get_settings():
    """Return the persisted dropdown selections."""
    return settings


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    """Persist any provided dropdown selections; unknown/None fields are ignored."""
    incoming = {k: v for k, v in req.model_dump().items() if v is not None}
    settings.update({k: v for k, v in incoming.items() if k in _SETTINGS_DEFAULTS})
    _save_settings()
    return settings


@app.get("/api/history")
def history():
    """Return the visible conversation (everything except the system prompt)."""
    return {"messages": [m for m in conversation if m["role"] != "system"]}


@app.post("/api/reset")
def reset():
    """Clear the single thread back to just the system prompt."""
    global conversation
    conversation = _fresh_conversation()
    _save_conversation()
    return {"ok": True}


@app.get("/api/providers")
def providers():
    """List selectable chat providers (for the frontend dropdown)."""
    return {
        "default": DEFAULT_CHAT_PROVIDER,
        "providers": [
            {"id": p, "label": CHAT_LABELS.get(p, p), "model": CHAT_MODELS.get(p, "")}
            for p in chat_clients
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    provider = (req.provider or DEFAULT_CHAT_PROVIDER).strip().lower()
    client = chat_clients.get(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"Unknown chat provider {provider!r}")

    conversation.append({"role": "user", "content": text})
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODELS[provider],
            messages=conversation,
            temperature=0.7,
        )
    except Exception as exc:  # surface backend errors to the client
        conversation.pop()  # roll back the user turn we optimistically added
        raise HTTPException(
            status_code=502, detail=f"Chat failed ({provider}): {exc}"
        ) from exc

    reply = completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})
    _save_conversation()
    return ChatResponse(reply=reply)


@app.post("/api/stt")
async def stt(file: UploadFile = File(...)):
    """Transcribe uploaded audio with Whisper."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    try:
        result = groq_client.audio.transcriptions.create(
            file=(file.filename or "audio.webm", audio_bytes),
            model=STT_MODEL,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq STT failed: {exc}") from exc
    return {"text": result.text}


def _tts_groq(text: str, voice: str) -> bytes:
    speech = groq_client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        response_format="wav",
    )
    return speech.read()


def _tts_piper(text: str, voice_id: str) -> bytes:
    """Synthesize WAV bytes fully offline with Piper."""
    import io
    import wave

    voice = _get_piper_voice(voice_id)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()


# --- asterisk-segment splicing -------------------------------------------------

# Matches *...* spans (non-greedy, no nested asterisks, must contain something).
_ASTERISK_RE = re.compile(r"\*([^*]+?)\*")


def _split_asterisk_segments(text: str) -> list[tuple[str, bool]]:
    """Split text into (chunk, is_asterisk) pieces, in order.

    "Hello *he sighs* there" -> [("Hello ", False), ("he sighs", True), (" there", False)]
    """
    segments: list[tuple[str, bool]] = []
    pos = 0
    for m in _ASTERISK_RE.finditer(text):
        if m.start() > pos:
            segments.append((text[pos : m.start()], False))
        segments.append((m.group(1), True))
        pos = m.end()
    if pos < len(text):
        segments.append((text[pos:], False))
    # Drop blank/whitespace-only chunks so we don't synthesize silence.
    return [(s, a) for (s, a) in segments if s.strip()]


def _synthesize(text: str, voice_id: str, asterisk: bool) -> bytes:
    """Synthesize one segment's WAV; asterisk segments force the Sofia Piper voice."""
    if asterisk:
        return _tts_piper(text, ASTERISK_VOICE)
    provider, _, name = voice_id.partition(":")
    if not name:
        provider, name = TTS_PROVIDER, voice_id
    if provider == "piper":
        return _tts_piper(text, name)
    if provider == "groq":
        return _tts_groq(text, name)
    raise RuntimeError(f"Unknown voice provider {provider!r} in {voice_id!r}.")


def _wav_params(wav_bytes: bytes):
    """Return (n_channels, sampwidth, framerate, frames) for a WAV blob."""
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())


def _concat_wavs(wav_blobs: list[bytes]) -> bytes:
    """Concatenate WAVs into one, resampling/normalizing to a common format.

    Segments can come from different engines (Groq 24kHz vs Piper 22.05kHz), so
    each is converted to the first segment's channels/width/rate via audioop.
    """
    import audioop
    import io
    import wave

    if len(wav_blobs) == 1:
        return wav_blobs[0]

    ch0, w0, r0, _ = _wav_params(wav_blobs[0])
    out_frames = bytearray()
    for blob in wav_blobs:
        ch, w, r, frames = _wav_params(blob)
        if w != w0:
            frames = audioop.lin2lin(frames, w, w0)
        if ch != ch0:
            frames = audioop.tomono(frames, w0, 0.5, 0.5) if ch0 == 1 else audioop.tostereo(frames, w0, 1, 1)
        if r != r0:
            frames, _ = audioop.ratecv(frames, w0, ch0, r, r0, None)
        out_frames += frames

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch0)
        wf.setsampwidth(w0)
        wf.setframerate(r0)
        wf.writeframes(bytes(out_frames))
    return buf.getvalue()


@app.get("/api/voices")
def voices():
    """List all selectable voices (online Groq + offline Piper) and the default."""
    return {"default": DEFAULT_VOICE, "voices": _available_voices()}


@app.post("/api/tts")
def tts(req: TTSRequest):
    """Synthesize speech, routing by the voice id's provider prefix.

    Voice ids are namespaced: "groq:<orpheus_voice>" (online) or
    "piper:<model_id>" (offline). A bare/empty voice falls back to DEFAULT_VOICE.

    *Asterisk-wrapped* segments are spoken in the hardcoded Sofia Piper voice
    and spliced back into the selected voice's audio, in order, as one clip.
    """
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    voice_id = req.voice or DEFAULT_VOICE
    segments = _split_asterisk_segments(text) or [(text, False)]

    try:
        clips = [_synthesize(chunk, voice_id, is_ast) for (chunk, is_ast) in segments]
        audio_bytes = _concat_wavs(clips)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}") from exc
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "chat_default": DEFAULT_CHAT_PROVIDER,
        "chat_providers": list(chat_clients),
        "tts_default_provider": TTS_PROVIDER,
    }
