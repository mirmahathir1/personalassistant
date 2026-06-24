# Openclaw Assistant

A single-thread voice chat assistant — **fully offline**, no cloud calls, no API
key. One conversation, no thread management.

- **Chat:** local Ollama `Llama-3.1-8B-Lexi-Uncensored-V2` (uncensored)
- **Speech-to-text:** local faster-whisper (`base.en` by default, on CPU)
- **Text-to-speech:** local Piper (`en_US-amy-medium` by default, on CPU)
- **Frontend:** Vue 3 + Vite
- **Backend:** FastAPI

Everything runs on your machine. The only network access is the one-time
download of the Ollama models and Piper/Whisper model files.

The frontend header has a **voice dropdown** of the installed Piper voices. The
chosen voice is sent per request and routed by its `piper:` prefix.

- **Voices** (Piper): curated by the sample WAVs in `backend/samples/` — a voice
  appears only if both its model (`backend/voices/<id>.onnx`) and its sample
  (`backend/samples/<name>.wav`) exist; delete a sample to drop it.

**Asterisk segments:** any text wrapped in `*asterisks*` (e.g. `*she whispers*`)
is spoken in the hardcoded **Sofia** Piper voice (`en_US-libritts_r-medium`)
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
default Piper voices (~60 MB each). The Whisper STT model downloads on first
mic use. Requires [Ollama](https://ollama.com) installed.

### Configuration (env vars)

| Var              | Default                        | Meaning                                   |
| ---------------- | ------------------------------ | ----------------------------------------- |
| `CHAT_MODEL`     | Lexi Uncensored                | Override the Ollama chat model id pulled by run.sh |
| `OLLAMA_BASE_URL`| `http://localhost:11434/v1`    | Ollama OpenAI-compatible endpoint         |
| `STT_MODEL`      | `base.en`                      | faster-whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`, `.en` variants) |
| `PIPER_VOICE`    | first available voice          | Default voice model id (filename stem)    |
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

| Method | Path           | Body                          | Returns            |
| ------ | -------------- | ----------------------------- | ------------------ |
| GET    | `/api/health`  | —                             | status + model     |
| GET    | `/api/history` | —                             | visible messages   |
| POST   | `/api/chat`    | `{ "message": "..." }`        | `{ "reply": "..." }` |
| POST   | `/api/tts`     | `{ "message": "..." }`        | `audio/wav`        |
| POST   | `/api/stt`     | multipart file field `file`   | `{ "text": "..." }`  |
| POST   | `/api/reset`   | —                             | `{ "ok": true }`   |

## Notes

- Mic capture requires a secure context: `localhost` works; on a remote host
  you'll need HTTPS for the browser to grant microphone access.
- Change the system prompt, chat model, or default voice at the top of
  `backend/main.py`. Add more Piper voices by dropping `<id>.onnx` (+ `.onnx.json`)
  into `backend/voices/` and a matching sample WAV into `backend/samples/`.
