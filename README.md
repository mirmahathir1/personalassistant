# Personal Assistant

A minimal local chat app with:

- a Vue 3 frontend
- a FastAPI backend
- a Dockerized Qdrant service for the sentence-memory foundation
- the GGUF model `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`

The backend now assembles prompts per request from SQLite-backed finalized chat history plus retrieved Qdrant memory. It no longer treats rolling `llama.cpp` state snapshots as the canonical conversation store.

Backend startup now waits for the shared Dockerized Qdrant service to become ready. If SQLite initialization fails or Qdrant never becomes ready, the backend fails startup instead of serving without retrieval memory.

The UI is organized into three columns:

- finalized chat
- paired trace activity
- previous conversations and global memory controls

Each `/api/chat` request sends only:

- the current user message
- the finalized `thread_id` for the active conversation

For each request, the backend runs a fixed three-pass flow:

- a draft pass from recent finalized thread history plus the current user message
- a memory-analysis pass from retrieved long-term memory plus the current thread context
- a final synthesis pass that produces the user-visible assistant reply

Only the final user message and final assistant reply are persisted to the finalized thread. Retrieved memory and intermediate pass artifacts are stored in the paired trace thread.

## Model

Default Hugging Face model:

- Repo: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`
- File: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- URL: `https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true`

By default, the backend downloads the model into `./models` the first time a chat request needs it, unless `MODEL_PATH` is set.

## Local Run

```bash
chmod +x run.sh
./run.sh
```

Then open `http://localhost:5173`.

The script will:

- create `.venv`
- install Python dependencies
- install frontend dependencies
- start the FastAPI backend on port `8000`
- start the Vue app on port `5173`

The phase-1 memory runtime is Compose-first. `run.sh` still starts the app stack, but the Qdrant-backed memory foundation and healthy `/api/health` response require `docker compose up`.

## Docker Run

```bash
docker compose up --build
```

Then open `http://localhost:5173`.

Compose now bind-mounts `./backend` and `./frontend`, so Python and Vue code reload inside the running containers as soon as you edit them. Rebuild only when you change Dockerfiles or dependency manifests like `backend/requirements.txt` or `frontend/package.json`.

The backend is exposed on `http://localhost:8000`.
Qdrant is exposed on `http://localhost:6333`.

Docker persists the runtime memory foundation under `./data`:

- SQLite path in the backend container: `/data/app.db`
- FastEmbed cache in the backend container: `/data/fastembed`
- Qdrant storage in the Qdrant container: `/qdrant/storage`

## Backend Configuration

Environment variables you can override:

- `MODEL_PATH`: use an already-downloaded `.gguf` file instead of downloading
- `MODEL_CACHE_DIR`: model download directory, defaults to `./models`
- `MODEL_N_CTX`: context window, defaults to `4096`
- `MODEL_N_THREADS`: CPU threads used by `llama.cpp`
- `MODEL_N_GPU_LAYERS`: set to `-1` to offload all layers when your local `llama-cpp-python` build supports GPU or Metal
- `MAX_TOKENS`: max response tokens, defaults to `512`
- `TEMPERATURE`: sampling temperature, defaults to `0.7`
- `TOP_P`: nucleus sampling, defaults to `0.95`
- `CHAT_FORMAT`: optional explicit chat format if you want to override GGUF metadata detection
- `CORS_ORIGINS`: comma-separated list or JSON array of frontend origins

Repository-fixed memory foundation settings live in `backend/config.py`:

- `SQLITE_PATH = /data/app.db`
- `QDRANT_URL = http://qdrant:6333`
- `QDRANT_COLLECTION = conversation_sentences`
- `FASTEMBED_CACHE_PATH = /data/fastembed`
- `MEMORY_SENTENCE_HITS_PER_SENTENCE = 5`
- `MEMORY_MAX_FULL_MESSAGES = 5`
- `MEMORY_BLOCK_MAX_TOKENS = 256`

Operational readiness is also fixed in `backend/config.py`:

- `QDRANT_STARTUP_TIMEOUT_SECONDS = 30.0`
- `QDRANT_STARTUP_POLL_INTERVAL_SECONDS = 1.0`

## API

`GET /api/health`

Returns backend, model, conversation, SQLite, and Qdrant status. SQLite and Qdrant readiness are reported separately, and the endpoint returns `503` with a degraded payload if either storage layer is unavailable.

`GET /api/session`

Returns the currently active conversation status for the running backend process.

`POST /api/reset`

Clears the active conversation pointer so the next chat starts a new conversation.

`POST /api/chat`

Continues the selected finalized thread and returns the finalized `thread_id`.

`GET /api/threads`

Lists previous conversations using the finalized thread as the conversation key.

`POST /api/threads`

Creates a new blank conversation with a finalized thread and a paired trace thread.

`GET /api/threads/{thread_id}`

Loads both the finalized thread and the paired trace thread for the selected conversation.

`POST /api/delete-all-memory`

Hard-deletes all stored SQLite and Qdrant memory.

Request:

```json
{
  "message": "Explain what this app does.",
  "thread_id": "finalized-thread-id"
}
```

Response:

```json
{
  "reply": "This app assembles each reply from stored chat history plus retrieved memory.",
  "model": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
  "thread_id": "finalized-thread-id",
  "turn_count": 1,
  "used_tokens": 142,
  "context_window": 4096
}
```
