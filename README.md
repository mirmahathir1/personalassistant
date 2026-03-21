# Personal Assistant

A minimal local chat app with:

- a Vue 3 frontend
- a FastAPI backend
- a single stateful conversation backed by the GGUF model `hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF`

The backend keeps one live conversation in memory and advances it with `llama.cpp` state snapshots. It does not manage multiple chat threads.

Each `/api/chat` request sends only:

- the current user message

The backend restores the current model state, appends the new turn, generates the reply, and saves the updated state again. The live conversation is lost if the backend process restarts or you call the reset endpoint.

## Model

Default Hugging Face model:

- Repo: `hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF`
- File: `llama-3.2-3b-instruct-q4_k_m.gguf`
- URL: `https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf?download=true`

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

## Docker Run

```bash
docker compose up --build
```

Then open `http://localhost:5173`.

The backend is exposed on `http://localhost:8000`.

## Backend Configuration

Environment variables you can override:

- `MODEL_PATH`: use an already-downloaded `.gguf` file instead of downloading
- `PRELOAD_MODEL`: set to `true` if you want the backend to load the model during startup
- `MODEL_CACHE_DIR`: model download directory, defaults to `./models`
- `MODEL_N_CTX`: context window, defaults to `4096`
- `MODEL_N_THREADS`: CPU threads used by `llama.cpp`
- `MODEL_N_GPU_LAYERS`: set to `-1` to offload all layers when your local `llama-cpp-python` build supports GPU or Metal
- `MAX_TOKENS`: max response tokens, defaults to `512`
- `TEMPERATURE`: sampling temperature, defaults to `0.7`
- `TOP_P`: nucleus sampling, defaults to `0.95`
- `CHAT_FORMAT`: optional explicit chat format if you want to override GGUF metadata detection
- `CORS_ORIGINS`: comma-separated list or JSON array of frontend origins

## API

`GET /api/health`

Returns backend, model, and conversation status.

`GET /api/session`

Returns the current single-session memory status.

`POST /api/reset`

Resets the live conversation back to just the system prompt.

`POST /api/chat`

Request:

```json
{
  "message": "Explain what this app does."
}
```

Response:

```json
{
  "reply": "This app keeps one live conversation in backend memory.",
  "model": "llama-3.2-3b-instruct-q4_k_m.gguf",
  "turn_count": 1,
  "used_tokens": 142,
  "context_window": 4096
}
```
