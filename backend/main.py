"""Single-thread voice chat backend — fully offline.

Chat runs on local Ollama, text-to-speech on local Piper, and speech-to-text on
a local faster-whisper model. No cloud calls and no API key. The whole app keeps
exactly one conversation thread in memory, persisted to a JSON file across
restarts.
"""

import json
import os
import re
import threading
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Chat runs on local Ollama only. Kept as a single-entry dict so the per-request
# `provider` field and the frontend dropdown keep working unchanged.
CHAT_MODELS = {
    # uncensored Llama-3.1 fine-tune (Lexi V2), pulled from HuggingFace via Ollama
    "ollama": "hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M",
    # lighter uncensored Llama-3.2 3B (bartowski abliterated build), ~2GB vs ~4.9GB
    "ollama-3b": "hf.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF:Q4_K_M",
    # lightest: abliterated Llama-3.2 1B (huihui_ai), ~0.8GB — direct Ollama pull
    "ollama-1b": "huihui_ai/llama3.2-abliterate:1b",
}
# Friendly labels for the dropdown.
CHAT_LABELS = {
    "ollama": "Local Lexi 8B (uncensored)",
    "ollama-3b": "Local Llama 3B (uncensored, lighter)",
    "ollama-1b": "Local Llama 1B (uncensored, lightest)",
}

# Default provider used when a request doesn't specify one.
DEFAULT_CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "ollama").strip().lower()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Ollama's native REST API (model list/pull) lives at the host root, not under
# the OpenAI-compatible /v1 prefix used for chat. Derive it from the same base.
OLLAMA_HOST = OLLAMA_BASE_URL.rstrip("/")
if OLLAMA_HOST.endswith("/v1"):
    OLLAMA_HOST = OLLAMA_HOST[: -len("/v1")]

# STT runs locally with faster-whisper. Model size: tiny/base/small/medium/large-v3.
STT_MODEL = os.environ.get("STT_MODEL", "base.en")

# TTS is local Piper only.
TTS_PROVIDER = "piper"

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
        out.append({"id": f"piper:{vid}", "label": _VOICE_LABELS.get(vid, vid)})
    return sorted(out, key=lambda v: v["label"])


def _available_voices() -> list[dict]:
    """All selectable (offline Piper) voices."""
    return _piper_voices()


# Default offline voice id, e.g. "piper:en_US-amy-medium". PIPER_VOICE optionally
# overrides which model is the default (filename stem).
_PIPER_DEFAULT_MODEL = os.environ.get("PIPER_VOICE", "")


def _default_voice_id() -> str:
    if _PIPER_DEFAULT_MODEL:
        return f"piper:{_PIPER_DEFAULT_MODEL}"
    avail = _available_voices()
    return avail[0]["id"] if avail else f"piper:{ASTERISK_VOICE}"


DEFAULT_VOICE = _default_voice_id()

SYSTEM_PROMPT = "You are a helpful, concise voice assistant. Keep replies natural and to the point."


# OpenAI-compatible chat client(s). Ollama speaks the OpenAI protocol; the key
# is ignored. Kept as a dict so the per-request `provider` field keeps working.
_ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")  # key ignored by Ollama
chat_clients = {
    # All providers talk to the same local Ollama; only the model id differs.
    "ollama": _ollama_client,
    "ollama-3b": _ollama_client,
    "ollama-1b": _ollama_client,
}

if DEFAULT_CHAT_PROVIDER not in chat_clients:
    DEFAULT_CHAT_PROVIDER = "ollama"

print(f"[startup] chat providers={list(chat_clients)} default={DEFAULT_CHAT_PROVIDER}")

# Local STT: load the faster-whisper model once, lazily, on first transcription.
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel  # imported lazily

        # int8 on CPU keeps it light; faster-whisper auto-downloads the model.
        _whisper_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


# Offline TTS: load each Piper voice once, lazily, and cache it by id.
_piper_cache: dict[str, object] = {}


def _get_piper_voice(voice_id: str):
    if voice_id not in _piper_cache:
        from piper import PiperVoice  # imported lazily

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
# Hybrid retrieval over the full archive
# ---------------------------------------------------------------------------
# The conversation list above is the complete, never-truncated archive. To keep
# prompts inside the model's context window without ever discarding history, we
# send the model a bounded slice per request: the system prompt, the most
# relevant *older* turns (retrieved by BM25 + local embeddings), then the recent
# turns verbatim. Embeddings use the Ollama client we already have.
import retrieval
import memory

_RETRIEVAL_VECTORS = Path(
    os.environ.get("RETRIEVAL_VECTORS_FILE", Path(__file__).resolve().parent / "chat_vectors.npy")
)
_FACTS_FILE = Path(
    os.environ.get("MEMORY_FACTS_FILE", Path(__file__).resolve().parent / "facts.json")
)

# The LLM-driven memory steps (query rewrite, chunk rerank, fact extraction) use
# the local chat provider's client+model. Override the helper provider/model with
# MEMORY_LLM if you add more providers.
_LLM_PROVIDER = os.environ.get("MEMORY_LLM", DEFAULT_CHAT_PROVIDER).strip().lower()
if _LLM_PROVIDER not in chat_clients:
    _LLM_PROVIDER = DEFAULT_CHAT_PROVIDER
_LLM_CLIENT = chat_clients.get(_LLM_PROVIDER)
_LLM_MODEL = CHAT_MODELS.get(_LLM_PROVIDER, "")

conv_index = None
fact_store = None
if retrieval.ENABLED:
    _embed_client = chat_clients.get("ollama")  # embeddings always via local Ollama
    conv_index = retrieval.ConversationIndex(
        _RETRIEVAL_VECTORS,
        embed_client=_embed_client,
        rerank_client=_LLM_CLIENT,
        rerank_model=_LLM_MODEL,
    )
    conv_index.rebuild(conversation)
    fact_store = memory.FactStore(_FACTS_FILE, client=_LLM_CLIENT, model=_LLM_MODEL)
    print(
        f"[startup] retrieval enabled: window={retrieval.WINDOW}/{retrieval.STRIDE} "
        f"recent_n={retrieval.RECENT_N} top_k={retrieval.TOP_K} "
        f"embed_model={retrieval.EMBED_MODEL} rerank={retrieval.RERANK_ENABLED} "
        f"llm={_LLM_PROVIDER} facts={len(fact_store.facts)}"
    )
else:
    print("[startup] retrieval disabled (RETRIEVAL_ENABLED=0)")


def _build_chat_messages(message: str) -> list[dict]:
    """Assemble the bounded message list to send for this turn.

    system prompt + [long-term facts] + [reranked retrieved context]
    + last N turns verbatim. Falls back to the whole thread if retrieval is off.
    """
    system = [m for m in conversation if m.get("role") == "system"][:1]
    body = [m for m in conversation if m.get("role") != "system"]

    if conv_index is None:
        return system + body  # retrieval disabled: legacy behavior

    recent = body[-retrieval.RECENT_N:] if retrieval.RECENT_N else body

    # B. Rewrite the (conversational) message into a standalone retrieval query.
    query = memory.rewrite_query(_LLM_CLIENT, _LLM_MODEL, recent, message)
    # A + C. Hybrid candidates over windowed chunks, LLM-reranked to the best few.
    chunks = conv_index.retrieve(query, retrieval.RECENT_N, retrieval.TOP_K)
    # Drop any chunk that overlaps the recent tail we're already sending verbatim.
    recent_ids = {id(m) for m in recent}
    chunks = [c for c in chunks if not any(id(t) in recent_ids for t in c)]

    messages = list(system)
    if fact_store is not None:  # D. always-on long-term facts
        block = fact_store.render_block()
        if block:
            messages.append({"role": "system", "content": block})
    if chunks:
        messages.append({"role": "system", "content": retrieval.render_context_block(chunks)})
    messages.extend(recent)
    return messages

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


# Serializes settings read-modify-write+save. The frontend can POST provider +
# voice near-simultaneously, and FastAPI runs sync endpoints in a threadpool, so
# concurrent saves would otherwise race (and a shared tmp path would have its
# source renamed away mid-replace, raising FileNotFoundError).
_settings_lock = threading.Lock()


def _save_settings() -> None:
    # Unique tmp name per write so concurrent saves never share a tmp path.
    tmp = SETTINGS_FILE.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.tmp")
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
    provider: str | None = None  # "ollama"; defaults to DEFAULT_CHAT_PROVIDER


class ChatResponse(BaseModel):
    reply: str


class TTSRequest(BaseModel):
    message: str
    voice: str | None = None  # Piper voice id; falls back to DEFAULT_VOICE


class SettingsRequest(BaseModel):
    chatProvider: str | None = None
    voice: str | None = None


class DeleteRequest(BaseModel):
    # Indices into the visible conversation (the list /api/history returns,
    # i.e. excluding the system prompt), of the messages to forget.
    indices: list[int]


@app.get("/api/settings")
def get_settings():
    """Return the persisted dropdown selections."""
    return settings


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    """Persist any provided dropdown selections; unknown/None fields are ignored."""
    incoming = {k: v for k, v in req.model_dump().items() if v is not None}
    with _settings_lock:
        settings.update({k: v for k, v in incoming.items() if k in _SETTINGS_DEFAULTS})
        _save_settings()
        return dict(settings)


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
    if conv_index is not None:
        conv_index.rebuild(conversation)
    if fact_store is not None:
        fact_store.clear()
    return {"ok": True}


@app.post("/api/delete")
def delete_messages(req: DeleteRequest):
    """Forget specific messages: remove them, then prune derived memory.

    `indices` are positions in the visible (system-excluded) conversation. We
    delete those turns, rebuild + prune the retrieval index/vector cache, and ask
    the fact store to drop any long-term facts derived from the deleted text.
    """
    global conversation
    # Map visible indices -> positions in the full thread (which has the system prompt).
    visible_positions = [i for i, m in enumerate(conversation) if m.get("role") != "system"]
    to_remove = set()
    deleted_texts: list[str] = []
    for vi in req.indices:
        if 0 <= vi < len(visible_positions):
            pos = visible_positions[vi]
            to_remove.add(pos)
            deleted_texts.append(conversation[pos].get("content") or "")
    if not to_remove:
        raise HTTPException(status_code=400, detail="No valid message indices to delete")

    conversation = [m for i, m in enumerate(conversation) if i not in to_remove]
    _save_conversation()
    if conv_index is not None:
        conv_index.rebuild(conversation)  # rebuild + prune orphaned vectors
    removed_facts = fact_store.prune_facts(deleted_texts) if fact_store is not None else []
    return {
        "ok": True,
        "deleted": len(to_remove),
        "removed_facts": removed_facts,
        "messages": [m for m in conversation if m["role"] != "system"],
    }


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


# ---------------------------------------------------------------------------
# Ollama model management: check whether a provider's model is downloaded, and
# pull it (streaming progress) if not. Uses Ollama's native REST API on the
# host root (not the /v1 OpenAI-compat prefix used for chat).
# ---------------------------------------------------------------------------


def _ollama_installed_models() -> set[str]:
    """Names of models currently present locally (via Ollama's /api/tags)."""
    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    resp.raise_for_status()
    return {m.get("name", "") for m in resp.json().get("models", [])}


def _model_is_present(model: str, installed: set[str] | None = None) -> bool:
    """True if `model` is already downloaded.

    Ollama appends ``:latest`` to a tagless name in /api/tags, so match either
    the exact id or its ``:latest`` form.
    """
    if installed is None:
        installed = _ollama_installed_models()
    return model in installed or f"{model}:latest" in installed


def _resolve_model(provider: str | None) -> str:
    """Map a provider id to its model string, or raise 400."""
    p = (provider or DEFAULT_CHAT_PROVIDER).strip().lower()
    if p not in CHAT_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown chat provider {p!r}")
    return CHAT_MODELS[p]


@app.get("/api/model/status")
def model_status(provider: str | None = None):
    """Report whether the selected provider's Ollama model is downloaded."""
    model = _resolve_model(provider)
    try:
        present = _model_is_present(model)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")
    return {"provider": provider or DEFAULT_CHAT_PROVIDER, "model": model, "present": present}


@app.post("/api/model/pull")
def model_pull(req: SettingsRequest):
    """Ensure the selected provider's model is downloaded, streaming progress.

    Returns an SSE stream of JSON events:
      {"status": "..."}                          progress text from Ollama
      {"completed": N, "total": M}               byte counts during download
      {"done": true, "model": "..."}             finished (already-present or pulled)
      {"error": "..."}                           failure
    Each event is one ``data: <json>\\n\\n`` SSE frame.
    """
    model = _resolve_model(req.chatProvider)

    def event_stream():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        try:
            if _model_is_present(model):
                yield sse({"status": "already downloaded", "model": model})
                yield sse({"done": True, "model": model})
                return
        except requests.RequestException as exc:
            yield sse({"error": f"Ollama unreachable: {exc}"})
            return

        try:
            with requests.post(
                f"{OLLAMA_HOST}/api/pull",
                json={"model": model, "stream": True},
                stream=True,
                timeout=(10, None),  # connect timeout; no read timeout (pulls are long)
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("error"):
                        yield sse({"error": evt["error"]})
                        return
                    out = {}
                    if "status" in evt:
                        out["status"] = evt["status"]
                    if "completed" in evt and "total" in evt:
                        out["completed"] = evt["completed"]
                        out["total"] = evt["total"]
                    if out:
                        yield sse(out)
            yield sse({"done": True, "model": model})
        except requests.RequestException as exc:
            yield sse({"error": f"Pull failed: {exc}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
    # Build the bounded slice (system + retrieved older context + recent turns).
    # The new user turn is now in the thread, so it's part of the recent tail;
    # retrieval uses `text` as the query and looks only at older turns.
    messages = _build_chat_messages(text)
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODELS[provider],
            messages=messages,
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
    # Re-window the archive so the new turns are retrievable next time (cheap,
    # incremental: only new chunks are embedded), then update long-term facts.
    if conv_index is not None:
        conv_index.rebuild(conversation)
    if fact_store is not None:
        fact_store.update(text, reply)
    return ChatResponse(reply=reply)


@app.post("/api/stt")
async def stt(file: UploadFile = File(...)):
    """Transcribe uploaded audio locally with faster-whisper."""
    import tempfile

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    tmp = None
    try:
        # faster-whisper decodes via ffmpeg, so write the upload to a temp file.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(audio_bytes)
            tmp = fh.name
        segments, _ = _get_whisper().transcribe(tmp)
        text = "".join(seg.text for seg in segments).strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Local STT failed: {exc}") from exc
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return {"text": text}


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
        provider, name = "piper", voice_id
    if provider == "piper":
        return _tts_piper(text, name)
    raise RuntimeError(f"Unknown voice provider {provider!r} in {voice_id!r}.")


def _wav_params(wav_bytes: bytes):
    """Return (n_channels, sampwidth, framerate, frames) for a WAV blob."""
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())


def _concat_wavs(wav_blobs: list[bytes]) -> bytes:
    """Concatenate WAVs into one, resampling/normalizing to a common format.

    Different Piper voices can have different sample rates (e.g. 22.05kHz vs
    24kHz), so each is converted to the first segment's channels/width/rate.
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
    """List all selectable (offline Piper) voices and the default."""
    return {"default": DEFAULT_VOICE, "voices": _available_voices()}


@app.post("/api/tts")
def tts(req: TTSRequest):
    """Synthesize speech with Piper, routing by the voice id's provider prefix.

    Voice ids are namespaced "piper:<model_id>". A bare/empty voice falls back
    to DEFAULT_VOICE.

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
        "tts_provider": TTS_PROVIDER,
        "stt_model": STT_MODEL,
    }
