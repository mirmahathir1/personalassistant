"""Single-thread voice chat backend — fully offline.

Chat runs on local Ollama, text-to-speech on local Kokoro (neural ONNX), and
speech-to-text on a local faster-whisper model. No cloud calls and no API key.
The app holds many **characters**, each its own conversation thread with its own
voice + intelligence (chat model) and its own history/retrieval/facts, persisted
as namespaced JSON under data/<char_id>/ across restarts.
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
    # lightest + most uncensored sub-3B: abliterated Qwen3 1.7B (huihui_ai),
    # ~1.1GB — lower refusal rate than the old Llama-3.2 1B abliterate, still tiny.
    "ollama-1b": "huihui_ai/qwen3-abliterated:1.7b",
}
# Friendly labels for the dropdown.
CHAT_LABELS = {
    "ollama": "Local Lexi 8B (uncensored)",
    "ollama-3b": "Local Llama 3B (uncensored, lighter)",
    "ollama-1b": "Local Qwen3 1.7B (most uncensored, lightest)",
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

# Keep the downloaded faster-whisper models inside the project (project-local
# `models/whisper/`) instead of the default ~/.cache/huggingface, so the app
# stays self-contained and portable — same rationale as the Ollama models dir.
# Override with STT_MODELS_DIR. Path is project-root/models/whisper.
STT_CACHE_DIR = os.environ.get(
    "STT_MODELS_DIR",
    str(Path(__file__).resolve().parent.parent / "models" / "whisper"),
)

# TTS is local Kokoro (neural, ONNX on CPU) — noticeably more natural than the
# previous Piper engine. Fully offline: the model + voice styles are downloaded
# once into a project-local folder (same rationale as the Llama/Whisper dirs).
TTS_PROVIDER = "kokoro"

# Kokoro model + voice-style files live in project-root/models/kokoro/. Override
# the dir with KOKORO_MODELS_DIR; override the filenames with KOKORO_MODEL /
# KOKORO_VOICES. run.sh downloads them on first run.
KOKORO_DIR = Path(
    os.environ.get(
        "KOKORO_MODELS_DIR",
        str(Path(__file__).resolve().parent.parent / "models" / "kokoro"),
    )
)
KOKORO_MODEL_PATH = KOKORO_DIR / os.environ.get("KOKORO_MODEL", "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = KOKORO_DIR / os.environ.get("KOKORO_VOICES", "voices-v1.0.bin")

# Curated subset of Kokoro's ~50 built-in voices, with friendly display names.
# Ids are Kokoro's own voice keys (af_=US female, am_=US male, bf_/bm_=British).
# Only ids present in this map are offered in the dropdown; extend it to expose
# more of the bundled voices (the full set is in voices-v1.0.bin).
_VOICE_LABELS = {
    "af_heart": "Aria (US, warm)",
    "af_bella": "Bella (US)",
    "af_nicole": "Nicole (US, soft)",
    "af_sarah": "Sarah (US)",
    "am_michael": "Michael (US, male)",
    "am_fenrir": "Fenrir (US, male)",
    "bf_emma": "Emma (UK)",
    "bm_george": "George (UK, male)",
}

# A default female Kokoro voice id, used only as the fallback voice when folding
# a legacy gender-less thread into the "Emily" character (see migration below).
# *Asterisk* asides are no longer spoken (they become silence), so this is not a
# TTS voice for them. Override with ASTERISK_VOICE.
ASTERISK_VOICE = os.environ.get("ASTERISK_VOICE", "af_nicole")


def _available_voices(gender: str | None = None) -> list[dict]:
    """Selectable (offline Kokoro) voices, in label order.

    If `gender` ("female"/"male") is given, only voices of that gender are
    returned (af_/bf_ = female, am_/bm_ = male), so the creation flow can offer
    voices matching the chosen gender.
    """
    out = []
    for vid, label in _VOICE_LABELS.items():
        vg = "female" if len(vid) >= 2 and vid[1] == "f" else "male" if len(vid) >= 2 and vid[1] == "m" else None
        if gender and vg != gender:
            continue
        out.append({"id": f"kokoro:{vid}", "label": label, "gender": vg})
    return sorted(out, key=lambda v: v["label"])


# Default voice id, e.g. "kokoro:af_nicole". KOKORO_VOICE optionally overrides
# which Kokoro voice key is the default.
_KOKORO_DEFAULT = os.environ.get("KOKORO_VOICE", "af_nicole")


def _default_voice_id() -> str:
    return f"kokoro:{_KOKORO_DEFAULT}"


DEFAULT_VOICE = _default_voice_id()

SYSTEM_PROMPT = "You are a helpful, concise voice assistant. Keep replies natural and to the point."


def _system_prompt_for(char: dict | None) -> str:
    """Build a character's system prompt: identity + its persona core.

    The generic SYSTEM_PROMPT is the fallback (e.g. retrieval-disabled threads or
    characters created before bios existed). The persona core is short and always
    present; the long bio is surfaced separately via retrieval (see BioIndex).
    """
    if not char:
        return SYSTEM_PROMPT
    name = char.get("name") or "the assistant"
    gender = char.get("gender") or ""
    core = (char.get("persona_core") or "").strip()
    base = (
        f"You are {name}, a {gender} character. Stay in character at all times; "
        "never say you are an AI or break character. Keep replies natural and "
        "conversational."
    )
    return f"{base}\n\n{core}" if core else base


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

        # int8 on CPU keeps it light; faster-whisper auto-downloads the model
        # into the project-local STT_CACHE_DIR (not ~/.cache/huggingface).
        os.makedirs(STT_CACHE_DIR, exist_ok=True)
        _whisper_model = WhisperModel(
            STT_MODEL, device="cpu", compute_type="int8", download_root=STT_CACHE_DIR
        )
    return _whisper_model


# Offline TTS: load the Kokoro model once, lazily (all voice styles ship in the
# single voices file, so one engine instance serves every voice id).
_kokoro = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro  # imported lazily (heavy: onnxruntime)

        if not KOKORO_MODEL_PATH.exists() or not KOKORO_VOICES_PATH.exists():
            raise RuntimeError(
                f"Kokoro model/voices not found in {KOKORO_DIR}. "
                "Download them into models/kokoro/ (see README / run.sh)."
            )
        _kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    return _kokoro


print(f"[startup] tts default provider={TTS_PROVIDER} default voice={DEFAULT_VOICE}")

# ---------------------------------------------------------------------------
# Per-character chat threads (each its own history + retrieval index + facts)
# ---------------------------------------------------------------------------
# The app holds many characters; each is one never-truncated conversation thread
# with its own JSON files under data/<char_id>/. To keep prompts inside the
# model's context window without discarding history, we send a bounded slice per
# request: system prompt, relevant older turns (BM25 + local embeddings), then
# recent turns verbatim. Threads are loaded lazily and cached per character.
import retrieval
import memory
import characters as characters_mod

# The LLM-driven memory steps (query rewrite, chunk rerank, fact extraction) use
# the local chat provider's client+model. Override the helper provider/model with
# MEMORY_LLM if you add more providers.
_LLM_PROVIDER = os.environ.get("MEMORY_LLM", DEFAULT_CHAT_PROVIDER).strip().lower()
if _LLM_PROVIDER not in chat_clients:
    _LLM_PROVIDER = DEFAULT_CHAT_PROVIDER
_LLM_CLIENT = chat_clients.get(_LLM_PROVIDER)
_LLM_MODEL = CHAT_MODELS.get(_LLM_PROVIDER, "")

char_store = characters_mod.CharacterStore()


def _fresh_conversation() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


class CharacterThread:
    """One character's loaded thread: conversation list + retrieval index + facts.

    Wraps the same retrieval/memory machinery the app used to keep as global
    singletons, but pointed at the character's namespaced files. Loaded lazily
    and cached in `_threads`.
    """

    def __init__(self, char_id: str):
        self.id = char_id
        self.char = char_store.get(char_id)  # may be None (e.g. mid-delete); falls back to generic prompt
        self.system_prompt = _system_prompt_for(self.char)
        p = characters_mod.char_paths(char_id)
        p["dir"].mkdir(parents=True, exist_ok=True)
        self.history_file = p["history"]
        self.conversation = self._load_conversation()
        self.conv_index = None
        self.bio_index = None
        self.fact_store = None
        if retrieval.ENABLED:
            self.conv_index = retrieval.ConversationIndex(
                p["vectors"],
                embed_client=chat_clients.get("ollama"),  # embeddings via local Ollama
                rerank_client=_LLM_CLIENT,
                rerank_model=_LLM_MODEL,
            )
            self.conv_index.rebuild(self.conversation)
            bio = (self.char or {}).get("bio", "")
            if bio.strip():
                self.bio_index = retrieval.BioIndex(
                    p["bio_vectors"],
                    bio,
                    embed_client=chat_clients.get("ollama"),
                    rerank_client=_LLM_CLIENT,
                    rerank_model=_LLM_MODEL,
                )
            self.fact_store = memory.FactStore(p["facts"], client=_LLM_CLIENT, model=_LLM_MODEL)

    def _load_conversation(self) -> list[dict]:
        """Load history; always keep a single system turn that reflects the
        character's *current* persona (rebuilt from the record, not whatever was
        frozen on disk), so editing the persona later takes effect."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                if isinstance(data, list) and data:
                    body = [m for m in data if m.get("role") != "system"]
                    return [{"role": "system", "content": self.system_prompt}] + body
            except (json.JSONDecodeError, OSError, AttributeError) as exc:
                print(f"[chars] could not read {self.history_file}: {exc}; starting fresh")
        return [{"role": "system", "content": self.system_prompt}]

    def save(self) -> None:
        """Persist the thread atomically so a crash mid-write can't corrupt it."""
        tmp = self.history_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.conversation, ensure_ascii=False, indent=2))
        tmp.replace(self.history_file)

    def build_chat_messages(self, message: str) -> list[dict]:
        """Assemble the bounded message list to send for this turn.

        system prompt + [long-term facts] + [reranked retrieved context]
        + last N turns verbatim. Falls back to the whole thread if retrieval is off.
        """
        system = [m for m in self.conversation if m.get("role") == "system"][:1]
        body = [m for m in self.conversation if m.get("role") != "system"]

        if self.conv_index is None:
            return system + body  # retrieval disabled: legacy behavior

        recent = body[-retrieval.RECENT_N:] if retrieval.RECENT_N else body

        # Rewrite the message into a standalone retrieval query, gather hybrid
        # candidates over windowed chunks, LLM-rerank to the best few.
        query = memory.rewrite_query(_LLM_CLIENT, _LLM_MODEL, recent, message)
        chunks = self.conv_index.retrieve(query, retrieval.RECENT_N, retrieval.TOP_K)
        # Drop any chunk overlapping the recent tail we already send verbatim.
        recent_ids = {id(m) for m in recent}
        chunks = [c for c in chunks if not any(id(t) in recent_ids for t in c)]

        messages = list(system)
        if self.bio_index is not None:  # relevant slice of the character's own bio
            bio_chunks = self.bio_index.retrieve(query, retrieval.TOP_K)
            if bio_chunks:
                messages.append(
                    {"role": "system", "content": retrieval.render_bio_block(bio_chunks)}
                )
        if self.fact_store is not None:  # always-on long-term facts
            block = self.fact_store.render_block()
            if block:
                messages.append({"role": "system", "content": block})
        if chunks:
            messages.append({"role": "system", "content": retrieval.render_context_block(chunks)})
        messages.extend(recent)
        return messages


# Lazily-loaded, cached CharacterThread per character id.
_threads: dict[str, CharacterThread] = {}


def _get_thread(char_id: str) -> CharacterThread:
    if char_id not in _threads:
        _threads[char_id] = CharacterThread(char_id)
    return _threads[char_id]


def _require_character(char_id: str | None) -> dict:
    """Resolve a character id to its record, or raise 400/404."""
    if not char_id:
        raise HTTPException(status_code=400, detail="character id is required")
    char = char_store.get(char_id)
    if char is None:
        raise HTTPException(status_code=404, detail=f"Unknown character {char_id!r}")
    return char

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


# --- Non-destructive migration of the legacy single thread into "Emily" -------
# If there's no character index yet but a legacy chat_history.json exists, fold
# that thread into a new character "Emily" (Female, high intelligence, current
# voice). The legacy files are COPIED, never moved/deleted — they stay on disk.
def _migrate_legacy_thread() -> None:
    import shutil

    if char_store.exists():
        return  # already migrated, or a fresh install that already has an index
    legacy_history = Path(
        os.environ.get("CHAT_HISTORY_FILE", Path(__file__).resolve().parent / "chat_history.json")
    )
    if not legacy_history.exists():
        char_store._save()  # write an empty index so we don't re-scan every boot
        return

    voice = settings.get("voice", DEFAULT_VOICE)
    if characters_mod.voice_gender(voice) != "female":
        voice = f"kokoro:{ASTERISK_VOICE}"  # ensure Emily gets a female voice
    char = char_store.create(
        name="Emily",
        gender="female",
        voice=voice,
        intelligence="high",
        created_at="1970-01-01T00:00:00Z",  # sorts oldest; no clock access at import
    )
    dst = characters_mod.char_paths(char["id"])
    legacy_facts = Path(
        os.environ.get("MEMORY_FACTS_FILE", Path(__file__).resolve().parent / "facts.json")
    )
    legacy_vectors = Path(
        os.environ.get(
            "RETRIEVAL_VECTORS_FILE", Path(__file__).resolve().parent / "chat_vectors.npy"
        )
    )
    for src, dest in [
        (legacy_history, dst["history"]),
        (legacy_facts, dst["facts"]),
        (legacy_vectors, dst["vectors"]),
        (legacy_vectors.with_suffix(".keys.json"), dst["vectors"].with_suffix(".keys.json")),
    ]:
        if src.exists():
            shutil.copy2(src, dest)  # copy — never move or delete the originals
    print(
        f"[startup] migrated legacy thread into character 'Emily' ({char['id']}); "
        "legacy files kept on disk"
    )


_migrate_legacy_thread()
print(f"[startup] {len(char_store.characters)} character(s) available")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="WhatsApp Uncensored")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    character: str  # character id; its intelligence picks the chat model


class ChatResponse(BaseModel):
    reply: str


class TTSRequest(BaseModel):
    message: str
    voice: str | None = None  # Kokoro voice id; falls back to the character's / default
    character: str | None = None  # if given, use that character's voice


class SettingsRequest(BaseModel):
    chatProvider: str | None = None
    voice: str | None = None


class CharacterCreateRequest(BaseModel):
    name: str
    gender: str          # "female" | "male"
    voice: str           # Kokoro voice id; must match gender
    intelligence: str    # "low" | "medium" | "high"
    persona_core: str = ""  # short identity blurb, injected into the system prompt
    bio: str = ""           # long-form lore, indexed for retrieval (surfaced on demand)


class DeleteRequest(BaseModel):
    character: str  # which character's thread to forget messages from
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


# ---------------------------------------------------------------------------
# Characters: create / list / delete. Each is one chat thread; its settings
# (name, gender, voice, intelligence) are fixed at creation time.
# ---------------------------------------------------------------------------


def _public_character(char: dict) -> dict:
    """Character record + its derived chat provider, for the frontend."""
    return {
        **{k: char.get(k) for k in (
            "id", "name", "gender", "voice", "intelligence",
            "persona_core", "bio", "created_at",
        )},
        "provider": characters_mod.INTELLIGENCE_PROVIDER.get(char.get("intelligence", ""), ""),
    }


@app.get("/api/characters")
def list_characters():
    """List all characters (newest first)."""
    return {"characters": [_public_character(c) for c in char_store.list()]}


@app.post("/api/characters")
def create_character(req: CharacterCreateRequest):
    """Create a character (and its empty thread). Settings are immutable after."""
    from datetime import datetime, timezone

    try:
        char = char_store.create(
            name=req.name,
            gender=req.gender,
            voice=req.voice,
            intelligence=req.intelligence,
            persona_core=req.persona_core,
            bio=req.bio,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _get_thread(char["id"])  # materialize the (empty) thread + index now
    return _public_character(char)


@app.delete("/api/characters/{char_id}")
def delete_character(char_id: str):
    """Delete a character and all of its thread data."""
    if not char_store.get(char_id):
        raise HTTPException(status_code=404, detail=f"Unknown character {char_id!r}")
    char_store.delete(char_id)
    _threads.pop(char_id, None)
    return {"ok": True, "deleted": char_id}


@app.get("/api/history")
def history(character: str):
    """Return one character's visible conversation (excluding the system prompt)."""
    _require_character(character)
    thread = _get_thread(character)
    return {"messages": [m for m in thread.conversation if m["role"] != "system"]}


@app.post("/api/reset")
def reset(character: str):
    """Clear one character's thread back to just the system prompt."""
    _require_character(character)
    thread = _get_thread(character)
    thread.conversation = [{"role": "system", "content": thread.system_prompt}]
    thread.save()
    if thread.conv_index is not None:
        thread.conv_index.rebuild(thread.conversation)
    if thread.fact_store is not None:
        thread.fact_store.clear()
    return {"ok": True}


@app.post("/api/delete")
def delete_messages(req: DeleteRequest):
    """Forget specific messages from a character's thread, then prune derived memory.

    `indices` are positions in the visible (system-excluded) conversation. We
    delete those turns, rebuild + prune the retrieval index/vector cache, and ask
    the fact store to drop any long-term facts derived from the deleted text.
    """
    _require_character(req.character)
    thread = _get_thread(req.character)
    convo = thread.conversation
    # Map visible indices -> positions in the full thread (which has the system prompt).
    visible_positions = [i for i, m in enumerate(convo) if m.get("role") != "system"]
    to_remove = set()
    deleted_texts: list[str] = []
    for vi in req.indices:
        if 0 <= vi < len(visible_positions):
            pos = visible_positions[vi]
            to_remove.add(pos)
            deleted_texts.append(convo[pos].get("content") or "")
    if not to_remove:
        raise HTTPException(status_code=400, detail="No valid message indices to delete")

    thread.conversation = [m for i, m in enumerate(convo) if i not in to_remove]
    thread.save()
    if thread.conv_index is not None:
        thread.conv_index.rebuild(thread.conversation)  # rebuild + prune orphaned vectors
    removed_facts = (
        thread.fact_store.prune_facts(deleted_texts) if thread.fact_store is not None else []
    )
    return {
        "ok": True,
        "deleted": len(to_remove),
        "removed_facts": removed_facts,
        "messages": [m for m in thread.conversation if m["role"] != "system"],
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

    char = _require_character(req.character)
    # Intelligence picks the chat model; the character can't change it after creation.
    provider = characters_mod.INTELLIGENCE_PROVIDER.get(char.get("intelligence", ""), "")
    client = chat_clients.get(provider)
    if client is None:
        raise HTTPException(
            status_code=400, detail=f"Character has invalid intelligence {char.get('intelligence')!r}"
        )

    thread = _get_thread(char["id"])
    convo = thread.conversation
    convo.append({"role": "user", "content": text})
    # Build the bounded slice (system + retrieved older context + recent turns).
    # The new user turn is now in the thread, so it's part of the recent tail;
    # retrieval uses `text` as the query and looks only at older turns.
    messages = thread.build_chat_messages(text)
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODELS[provider],
            messages=messages,
            temperature=0.7,
        )
    except Exception as exc:  # surface backend errors to the client
        convo.pop()  # roll back the user turn we optimistically added
        raise HTTPException(
            status_code=502, detail=f"Chat failed ({provider}): {exc}"
        ) from exc

    reply = completion.choices[0].message.content
    convo.append({"role": "assistant", "content": reply})
    thread.save()
    # Re-window the archive so the new turns are retrievable next time (cheap,
    # incremental: only new chunks are embedded), then update long-term facts.
    if thread.conv_index is not None:
        thread.conv_index.rebuild(convo)
    if thread.fact_store is not None:
        thread.fact_store.update(text, reply)
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


def _tts_kokoro(text: str, voice_id: str) -> bytes:
    """Synthesize WAV bytes fully offline with Kokoro.

    Kokoro returns float32 samples + a sample rate; we encode them to a 16-bit
    PCM WAV (the format the splicing/concat code and browser expect).
    """
    import io
    import wave

    import numpy as np

    samples, sample_rate = _get_kokoro().create(text, voice=voice_id, lang="en-us")
    # float32 in [-1, 1] -> 16-bit signed PCM.
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
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


def _synthesize(text: str, voice_id: str) -> bytes:
    """Synthesize one spoken segment's WAV in the given voice.

    Voice ids are namespaced "<engine>:<id>". Only the Kokoro engine exists today;
    the prefix is kept so a second engine can be added without changing callers.
    (*Asterisk* asides are never synthesized — they become silence in the caller.)
    """
    provider, _, name = voice_id.partition(":")
    if not name:
        provider, name = "kokoro", voice_id
    if provider == "kokoro":
        return _tts_kokoro(text, name)
    raise RuntimeError(f"Unknown voice provider {provider!r} in {voice_id!r}.")


def _wav_params(wav_bytes: bytes):
    """Return (n_channels, sampwidth, framerate, frames) for a WAV blob."""
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())


# Kokoro's native output format (mono, 16-bit PCM, 24kHz). Used to synthesize
# standalone silence when there's no neighbouring clip to copy the format from
# (e.g. a message that is entirely an *asterisk* aside).
_KOKORO_CHANNELS, _KOKORO_SAMPWIDTH, _KOKORO_RATE = 1, 2, 24000


def _silence_wav(seconds: float, like: bytes | None = None) -> bytes:
    """A silent WAV of `seconds`.

    Copies the channels/width/rate of `like` when given; otherwise falls back to
    Kokoro's native format so silence can stand on its own.
    """
    import io
    import wave

    if like is not None:
        ch, w, r, _ = _wav_params(like)
    else:
        ch, w, r = _KOKORO_CHANNELS, _KOKORO_SAMPWIDTH, _KOKORO_RATE
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(w)
        wf.setframerate(r)
        wf.writeframes(b"\x00" * int(seconds * r) * ch * w)
    return buf.getvalue()


# *Asterisk* asides (stage directions, e.g. *he sighs*) are NOT spoken — they're
# replaced by a pause. The pause length scales with the aside's word count so a
# longer action reads as a longer beat, clamped to [MIN, MAX] seconds. Tune the
# per-word rate and bounds via the env vars below.
ASTERISK_PAUSE_PER_WORD = float(os.environ.get("ASTERISK_PAUSE_PER_WORD", "0.35"))
ASTERISK_PAUSE_MIN_SECONDS = float(os.environ.get("ASTERISK_PAUSE_MIN_SECONDS", "0.6"))
ASTERISK_PAUSE_MAX_SECONDS = float(os.environ.get("ASTERISK_PAUSE_MAX_SECONDS", "3"))


def _asterisk_pause_seconds(aside_text: str) -> float:
    """How long a silent pause to leave in place of an (unspoken) asterisk aside."""
    words = len(aside_text.split())
    secs = words * ASTERISK_PAUSE_PER_WORD
    return max(ASTERISK_PAUSE_MIN_SECONDS, min(ASTERISK_PAUSE_MAX_SECONDS, secs))


def _concat_wavs(wav_blobs: list[bytes]) -> bytes:
    """Concatenate WAVs into one, resampling/normalizing to a common format.

    Kokoro voices all share one format (24kHz mono), but the normalization is
    kept so segments still splice cleanly if a future engine differs — each is
    converted to the first segment's channels/width/rate.
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
def voices(gender: str | None = None):
    """List selectable (offline Kokoro) voices and the default.

    Pass ?gender=female|male to restrict to voices of that gender (used by the
    character-creation flow once a gender is picked).
    """
    g = (gender or "").strip().lower() or None
    if g and g not in characters_mod.GENDERS:
        raise HTTPException(status_code=400, detail=f"gender must be one of {characters_mod.GENDERS}")
    return {"default": DEFAULT_VOICE, "voices": _available_voices(g)}


@app.post("/api/tts")
def tts(req: TTSRequest):
    """Synthesize speech with Kokoro, routing by the voice id's provider prefix.

    Voice ids are namespaced "kokoro:<voice_key>". A bare/empty voice falls back
    to DEFAULT_VOICE.

    The voice is resolved as: explicit `voice` > the character's stored voice
    (if `character` is given) > DEFAULT_VOICE.

    *Asterisk-wrapped* segments (stage directions, e.g. *he sighs*) are NOT
    spoken: each is replaced by a silent pause whose length scales with the
    aside's word count, spliced into the spoken audio in order, as one clip.
    """
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    voice_id = req.voice
    if not voice_id and req.character:
        char = char_store.get(req.character)
        if char:
            voice_id = char.get("voice")
    voice_id = voice_id or DEFAULT_VOICE
    segments = _split_asterisk_segments(text) or [(text, False)]

    try:
        clips: list[bytes] = []
        for chunk, is_ast in segments:
            if is_ast:
                # Don't voice the aside — leave a pause in its place.
                clips.append(_silence_wav(_asterisk_pause_seconds(chunk)))
            else:
                clips.append(_synthesize(chunk, voice_id))
        # A message that's entirely an aside collapses to just silence.
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
        "characters": len(char_store.characters),
    }
