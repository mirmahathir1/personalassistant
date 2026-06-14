# Openclaw Assistant

A single-thread voice chat assistant. One conversation, no thread management.

- **Chat:** selectable — Groq `llama-3.3-70b-versatile` (cloud) **or** local
  Ollama `Llama-3.1-8B-Lexi-Uncensored-V2` (uncensored)
- **Speech-to-text:** Groq `whisper-large-v3` (always)
- **Text-to-speech:** selectable — Groq `canopylabs/orpheus-v1-english` (cloud)
  **or** Piper `en_US-lessac-medium` (fully offline, on-CPU)
- **Frontend:** Vue 3 + Vite
- **Backend:** FastAPI

Chat and TTS each have a swappable provider, so you can run chat **and** TTS
fully offline (Ollama + Piper) while only STT still calls Groq.

## Layout

```
backend/        FastAPI app (chat / stt / tts / reset)
frontend/       Vue 3 + Vite SPA
api_key.txt     Groq API key (gsk_...) — read by the backend if GROQ_API_KEY is unset
run.sh          Starts backend + frontend together
```

## Quick start

`run.sh` takes a required chat provider and an optional TTS provider:

```bash
./run.sh <groq|ollama> [groq|piper]

./run.sh groq               # cloud chat, cloud TTS
./run.sh ollama             # local uncensored chat, cloud TTS
./run.sh ollama piper       # local chat + offline TTS (only STT hits Groq)
```

Then open <http://localhost:5173>.

When you pass `ollama`, the script auto-starts the Ollama server and pulls
the Lexi Uncensored model if needed. When you pass `piper`, it downloads the
offline Piper voice (~60 MB) into `backend/voices/` on first run.

The Groq API key (used for STT/TTS, and for chat when provider is `groq`) is
read from the `GROQ_API_KEY` environment variable, or falls back to
`api_key.txt` at the project root.

### Configuration (env vars)

| Var              | Default                        | Meaning                                   |
| ---------------- | ------------------------------ | ----------------------------------------- |
| `CHAT_PROVIDER`  | `groq`                         | `groq` or `ollama` (set by `run.sh` arg)  |
| `CHAT_MODEL`     | per-provider default           | Override the chat model id                |
| `OLLAMA_BASE_URL`| `http://localhost:11434/v1`    | Ollama OpenAI-compatible endpoint         |
| `TTS_PROVIDER`   | `groq`                         | `groq` (cloud) or `piper` (offline)       |
| `PIPER_VOICE`    | `backend/voices/en_US-lessac-medium.onnx` | Path to the Piper voice model  |
| `GROQ_API_KEY`   | falls back to `api_key.txt`    | Groq key                                  |

Provider defaults for `CHAT_MODEL`: `groq` → `llama-3.3-70b-versatile`,
`ollama` → `hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M`.

## Run the pieces manually

Backend:

```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
CHAT_PROVIDER=groq ./.venv/bin/uvicorn main:app --port 8000
# or: CHAT_PROVIDER=ollama ./.venv/bin/uvicorn main:app --port 8000
```

For the `ollama` provider, make sure Ollama is running and the model is pulled:

```bash
ollama serve &              # if not already running
ollama pull hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M
```

Frontend (separate terminal):

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api/*` to the backend on port 8000.

## How it works

- The backend keeps **one** in-memory conversation (`conversation` list in
  `backend/main.py`). It is shared by everyone and lives until the server
  restarts. Use the **Clear** button (or `POST /api/reset`) to wipe it.
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
- Change the system prompt, model, or TTS voice at the top of
  `backend/main.py` (valid Orpheus voices: `autumn diana hannah austin daniel troy`).
