"""Single-thread voice chat backend.

Chat runs on a selectable LLM backend (Groq or a local Ollama model); speech
(STT via Whisper, TTS via Orpheus) always runs on Groq. The whole app keeps
exactly one conversation thread in memory — restart the server to clear it.

Configure the chat backend with the CHAT_PROVIDER env var: "groq" (default)
or "ollama".
"""

import json
import os
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

CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "groq").strip().lower()

# Chat model per provider (override with CHAT_MODEL).
_DEFAULT_CHAT_MODEL = {
    "groq": "llama-3.3-70b-versatile",
    # uncensored Llama-3.1 fine-tune (Lexi V2), pulled from HuggingFace via Ollama
    "ollama": "hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M",
}
CHAT_MODEL = os.environ.get("CHAT_MODEL", _DEFAULT_CHAT_MODEL.get(CHAT_PROVIDER, ""))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# STT is Groq-only (Whisper).
STT_MODEL = "whisper-large-v3"

# TTS provider: "groq" (cloud Orpheus) or "piper" (fully offline, on-CPU).
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "groq").strip().lower()
TTS_MODEL = "canopylabs/orpheus-v1-english"  # groq
TTS_VOICE = "diana"  # groq

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


def _available_voices() -> list[dict]:
    """Selectable Piper voices: model present on disk AND a kept sample WAV.

    Deleting a sample under backend/samples/ removes that voice from the menu,
    so the curated set is whatever samples you keep.
    """
    out = []
    for onnx in sorted(VOICES_DIR.glob("*.onnx")):
        vid = onnx.stem
        sample = _VOICE_SAMPLE.get(vid, vid)
        if not (SAMPLES_DIR / f"{sample}.wav").exists():
            continue
        out.append({"id": vid, "label": _VOICE_LABELS.get(vid, vid)})
    return sorted(out, key=lambda v: v["label"])


# Default selected voice: env override, else first available, else lessac.
DEFAULT_PIPER_VOICE = os.environ.get("PIPER_VOICE", "")
if not DEFAULT_PIPER_VOICE:
    _avail = _available_voices()
    DEFAULT_PIPER_VOICE = _avail[0]["id"] if _avail else "en_US-lessac-medium"

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


# Groq client: always used for STT/TTS, and for chat when CHAT_PROVIDER=groq.
groq_key = _load_groq_key()
groq_client = Groq(api_key=groq_key)

# Chat client (OpenAI-compatible). Both Groq and Ollama speak this protocol,
# so we point one OpenAI client at whichever backend was selected.
if CHAT_PROVIDER == "ollama":
    chat_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")  # key is ignored by Ollama
elif CHAT_PROVIDER == "groq":
    chat_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
else:
    raise RuntimeError(f"Unknown CHAT_PROVIDER={CHAT_PROVIDER!r}; expected 'groq' or 'ollama'.")

print(f"[startup] chat provider={CHAT_PROVIDER} model={CHAT_MODEL}")

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


print(f"[startup] tts provider={TTS_PROVIDER} default voice={DEFAULT_PIPER_VOICE}")

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


class ChatResponse(BaseModel):
    reply: str


class TTSRequest(BaseModel):
    message: str
    voice: str | None = None  # Piper voice id; ignored by the Groq TTS provider


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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    conversation.append({"role": "user", "content": text})
    try:
        completion = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=conversation,
            temperature=0.7,
        )
    except Exception as exc:  # surface backend errors to the client
        conversation.pop()  # roll back the user turn we optimistically added
        raise HTTPException(
            status_code=502, detail=f"Chat failed ({CHAT_PROVIDER}): {exc}"
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


def _tts_groq(text: str) -> bytes:
    speech = groq_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
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


@app.get("/api/voices")
def voices():
    """List selectable voices for the active TTS provider, plus the default."""
    if TTS_PROVIDER == "piper":
        avail = _available_voices()
        default = DEFAULT_PIPER_VOICE
    else:  # groq exposes a single fixed Orpheus voice here
        avail = [{"id": TTS_VOICE, "label": TTS_VOICE.capitalize()}]
        default = TTS_VOICE
    return {"provider": TTS_PROVIDER, "default": default, "voices": avail}


@app.post("/api/tts")
def tts(req: TTSRequest):
    """Synthesize speech for the given text, returning WAV audio."""
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        if TTS_PROVIDER == "piper":
            voice_id = req.voice or DEFAULT_PIPER_VOICE
            audio_bytes = _tts_piper(text, voice_id)
        elif TTS_PROVIDER == "groq":
            audio_bytes = _tts_groq(text)  # voice selection is fixed for Groq
        else:
            raise RuntimeError(f"Unknown TTS_PROVIDER={TTS_PROVIDER!r}; expected 'groq' or 'piper'.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS failed ({TTS_PROVIDER}): {exc}") from exc
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "chat_provider": CHAT_PROVIDER,
        "model": CHAT_MODEL,
        "tts_provider": TTS_PROVIDER,
    }
