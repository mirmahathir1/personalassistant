# Openclaw Assistant

A multi-character voice chat assistant — **fully offline**, no cloud calls, no
API key. Create characters; each is its own chat thread with its own voice,
intelligence, and memory.

- **Chat:** local Ollama (uncensored Qwen3 1.7B / Llama 3B / Lexi 8B, picked per character)
- **Speech-to-text:** local faster-whisper (`base.en` by default, on CPU)
- **Text-to-speech:** local Kokoro neural TTS (ONNX on CPU) — noticeably more
  natural than the old Piper engine
- **Frontend:** Vue 3 + Vite
- **Backend:** FastAPI

Everything runs on your machine. The only network access is the one-time
download of the Ollama models and the Kokoro/Whisper model files.

## Characters

The home page is a **contact list** (empty at first). Create a character with:

- **Name**
- **Gender** (female / male) — gates the voice options
- **Voice** — a Kokoro voice matching the chosen gender (the picker only appears
  after a gender is selected, and lists only matching voices)
- **Intelligence** — a 3-stop slider mapping to the three local chat models:
  **Quick** → Qwen3 1.7B (abliterated), **Balanced** → Llama 3B, **Smart** → Lexi 8B

Creating a character adds a row to the list; tapping it opens that character's
chat thread. **These settings are fixed at creation** — the contact pane shows
them read-only. Each character has its own history, retrieval index, and
long-term facts, stored under `backend/data/<char_id>/` (the index of all
characters is `backend/data/characters.json`).

On first run after upgrading from the old single-thread build, the existing
conversation is folded into a character named **Emily** (Female, Smart, current
voice). The legacy files (`backend/chat_history.json` etc.) are **copied, never
deleted** — they stay on disk as a backup.

- **Voices** (Kokoro): a single model + voices file (`models/kokoro/`) ship ~50
  built-in voices; the creation picker exposes a curated subset defined by
  `_VOICE_LABELS` in `backend/main.py` (gender inferred from the `af_/bf_` =
  female, `am_/bm_` = male id prefix). Add a `"<kokoro_id>": "<label>"` entry to
  expose more.

**Asterisk segments:** any text wrapped in `*asterisks*` (e.g. `*she whispers*`)
is spoken in a hardcoded distinct **aside** voice (`af_nicole` / "Nicole")
and spliced — in order — into the selected voice's audio as a single clip.
Override it with the `ASTERISK_VOICE` env var.

## Layout

```
backend/        FastAPI app (chat / stt / tts / reset)
frontend/       Vue 3 + Vite SPA
run.sh          Starts backend + frontend together
```

## Quick start

```bash
./run.sh
```

Then open <http://localhost:5173>.

No arguments and no API key. On first run the script auto-starts Ollama, pulls
the Lexi Uncensored chat model and the embedding model, and downloads the
Kokoro TTS model + voices (~340 MB total, into `models/kokoro/`). The Whisper
STT model downloads on first mic use. Requires [Ollama](https://ollama.com)
installed.

All models live in a **project-local `models/` folder** so the app stays self-
contained and portable rather than scattering downloads across `~/.ollama` and
`~/.cache/huggingface`. Llama (chat) + embedding models go to `models/` (via
`OLLAMA_MODELS`, set by run.sh); the faster-whisper STT model goes to
`models/whisper/` (via `download_root`, set in `backend/main.py`); the Kokoro
TTS model + voices go to `models/kokoro/` (downloaded by run.sh). The folder is
git-ignored — the models are large and re-downloaded on first run. If an Ollama
server is already running, stop it (`pkill ollama`) before `./run.sh` so the
project-local dir is used. Override locations with `OLLAMA_MODELS` /
`STT_MODELS_DIR` / `KOKORO_MODELS_DIR`.

### Configuration (env vars)

| Var              | Default                        | Meaning                                   |
| ---------------- | ------------------------------ | ----------------------------------------- |
| `CHAT_MODEL`     | Lexi Uncensored                | Override the Ollama chat model id pulled by run.sh |
| `OLLAMA_MODELS`  | `./models`                     | Project-local dir Ollama stores all models in (keeps the app self-contained) |
| `OLLAMA_BASE_URL`| `http://localhost:11434/v1`    | Ollama OpenAI-compatible endpoint         |
| `STT_MODEL`      | `base.en`                      | faster-whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`, `.en` variants) |
| `STT_MODELS_DIR` | `./models/whisper`             | Project-local dir faster-whisper downloads STT models into (instead of `~/.cache/huggingface`) |
| `KOKORO_VOICE`   | `af_heart`                     | Default Kokoro voice key (the dropdown default) |
| `KOKORO_MODELS_DIR` | `./models/kokoro`           | Project-local dir holding the Kokoro `.onnx` + voices file |
| `ASTERISK_VOICE` | `af_nicole`                    | Kokoro voice used for `*asterisk*` aside segments |
| `RETRIEVAL_ENABLED` | `1`                         | Set `0` to send the whole thread (legacy) instead of a retrieved slice |
| `RETRIEVAL_RECENT_N`| `8`                         | Recent turns always sent verbatim         |
| `RETRIEVAL_TOP_K`   | `5`                         | Reranked chunks of older context injected per request |
| `RETRIEVAL_WINDOW` / `RETRIEVAL_STRIDE` | `4` / `2`   | Chunk size (turns) and step; overlap = window − stride |
| `RETRIEVAL_RERANK`  | `1`                         | LLM-rerank fused candidates (`0` = use fused order) |
| `EMBED_MODEL`       | `nomic-embed-text`          | Local Ollama embedding model for semantic search |
| `MEMORY_FACTS_ENABLED` | `1`                      | Extract & inject durable facts about the user (`0` to disable) |
| `MEMORY_REWRITE_ENABLED` | `1`                    | Rewrite the message into a standalone retrieval query (`0` to disable) |
| `MEMORY_LLM`        | the chat provider           | Which provider runs rewrite/rerank/fact-extraction (`ollama`) |

`CHAT_MODEL` defaults to `hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M`.

The voice dropdown selection is saved to `backend/settings.json` and restored on
the next page load. The env-var defaults above only apply until a selection is
saved.

## Run the pieces manually

Backend:

```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn main:app --port 8000
```

Make sure Ollama is running and the models are pulled:

```bash
export OLLAMA_MODELS="$PWD/models"   # keep models project-local (run.sh does this for you)
ollama serve &              # if not already running
ollama pull hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M
ollama pull nomic-embed-text   # for semantic retrieval (offline, ~270MB)
```

The embedding model powers the semantic half of conversation retrieval. If it
isn't pulled (or Ollama is down), retrieval automatically falls back to BM25
keyword search — and to the recent turns alone if `rank-bm25` is unavailable.
Nothing breaks; the only difference is recall quality.

The STT model (faster-whisper) downloads automatically the first time you use
the mic; no manual pull needed.

Frontend (separate terminal):

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api/*` to the backend on port 8000.

## How it works

- The backend keeps **one** in-memory conversation (`conversation` list in
  `backend/main.py`), persisted to `chat_history.json`. It is shared by everyone
  and survives restarts. Use the **Clear** button (or `POST /api/reset`) to wipe it.
- **Long conversations never lose context.** The full thread is the archive and
  is never truncated. Per request the prompt is assembled as: system prompt +
  long-term facts + reranked relevant older context + the last
  `RETRIEVAL_RECENT_N` turns verbatim — bounded no matter how long the thread
  grows. Four layers drive recall quality (each degrades gracefully if its
  dependency is missing — e.g. Ollama down):
  - **Windowed chunks** (`backend/retrieval.py`): the history is indexed in
    overlapping multi-turn windows, not lone messages, so each unit carries
    enough context to embed and to be useful when injected.
  - **Hybrid search**: BM25 (lexical) + embeddings (semantic, local Ollama),
    fused with Reciprocal Rank Fusion. Embeddings cached in `chat_vectors.npy`.
  - **Query rewrite + LLM rerank** (`backend/memory.py` + `retrieval.py`): the
    message is rewritten into a standalone search query (resolving "it"/"that"),
    candidates are gathered broadly, then the chat model reranks them down to the
    best `RETRIEVAL_TOP_K`.
  - **Long-term facts** (`backend/memory.py`): after each exchange the model
    extracts durable facts about the user ("dog named Rex", "prefers offline
    TTS"), deduped into `facts.json` and injected on every request — so the
    assistant *remembers* even after raw turns age out of retrieval.
  Set `RETRIEVAL_ENABLED=0` to revert to sending the whole thread.
- Typing a message → `POST /api/chat` → Llama reply, which is then auto-spoken
  via `POST /api/tts`.
- The 🎤 mic button records audio in the browser (`MediaRecorder`), uploads it
  to `POST /api/stt`, and drops the transcript into the text box for you to send.

## API

Chat/history/tts/reset/delete are **scoped to a character** (by id). Voices can
be filtered by gender for the creation flow.

| Method | Path                       | Body / query                                   | Returns              |
| ------ | -------------------------- | ---------------------------------------------- | -------------------- |
| GET    | `/api/health`              | —                                              | status + char count  |
| GET    | `/api/characters`          | —                                              | character list       |
| POST   | `/api/characters`          | `{ name, gender, voice, intelligence }`        | created character    |
| DELETE | `/api/characters/{id}`     | —                                              | `{ "ok": true }`     |
| GET    | `/api/voices`              | `?gender=female\|male` (optional)              | voices + default     |
| GET    | `/api/history`             | `?character=<id>`                              | visible messages     |
| POST   | `/api/chat`                | `{ "message": "...", "character": "<id>" }`    | `{ "reply": "..." }` |
| POST   | `/api/tts`                 | `{ "message": "...", "character": "<id>" }`    | `audio/wav`          |
| POST   | `/api/stt`                 | multipart file field `file`                    | `{ "text": "..." }`  |
| POST   | `/api/reset`               | `?character=<id>`                              | `{ "ok": true }`     |
| POST   | `/api/delete`              | `{ "character": "<id>", "indices": [...] }`    | updated messages     |

## Notes

- Mic capture requires a secure context: `localhost` works; on a remote host
  you'll need HTTPS for the browser to grant microphone access.
- Change the system prompt, chat model, or default voice at the top of
  `backend/main.py`. Expose more of Kokoro's ~50 built-in voices by adding
  `"<kokoro_id>": "<label>"` entries to `_VOICE_LABELS` in `backend/main.py` —
  no extra downloads needed (every voice ships in `models/kokoro/voices-v1.0.bin`).
