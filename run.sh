#!/usr/bin/env bash
# Start backend (FastAPI/uvicorn) and frontend (Vite) together.
#
# Usage: ./run.sh <groq|ollama> [groq|piper]
#   arg 1 (chat, required): groq   = Groq llama-3.3-70b-versatile (cloud)
#                           ollama = local Ollama (uncensored Llama-3.1 Lexi V2)
#   arg 2 (tts, optional):  groq   = Groq Orpheus (cloud, default)
#                           piper  = fully offline, on-CPU
#
# STT always runs on Groq (Whisper). Ctrl-C stops both servers.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# --- require & validate the chat provider argument ---
PROVIDER="${1:-}"
if [ "$PROVIDER" != "groq" ] && [ "$PROVIDER" != "ollama" ]; then
  echo "Usage: ./run.sh <groq|ollama> [groq|piper]" >&2
  echo "  e.g. ./run.sh ollama piper" >&2
  exit 1
fi
export CHAT_PROVIDER="$PROVIDER"

# --- optional TTS provider argument (default groq) ---
TTS="${2:-groq}"
if [ "$TTS" != "groq" ] && [ "$TTS" != "piper" ]; then
  echo "Usage: ./run.sh <groq|ollama> [groq|piper]   (TTS must be groq or piper)" >&2
  exit 1
fi
export TTS_PROVIDER="$TTS"

# --- piper preflight: ensure an offline voice model is present ---
if [ "$TTS" = "piper" ]; then
  VOICE_DIR="$ROOT/backend/voices"
  VOICE="$VOICE_DIR/en_US-lessac-medium.onnx"
  if [ ! -f "$VOICE" ]; then
    echo "Downloading Piper voice (en_US-lessac-medium, ~60MB)..."
    mkdir -p "$VOICE_DIR"
    BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
    curl -sL -o "$VOICE" "$BASE/en_US-lessac-medium.onnx"
    curl -sL -o "$VOICE.json" "$BASE/en_US-lessac-medium.onnx.json"
  fi
fi

# --- ollama preflight: ensure server is up and the model is present ---
if [ "$PROVIDER" = "ollama" ]; then
  OLLAMA_MODEL="${CHAT_MODEL:-hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M}"
  if ! curl -s -o /dev/null http://localhost:11434/api/tags 2>/dev/null; then
    echo "Starting ollama server..."
    ollama serve > /tmp/ollama.log 2>&1 &
    until curl -s -o /dev/null http://localhost:11434/api/tags 2>/dev/null; do sleep 0.3; done
  fi
  if ! ollama list | grep -qF "$OLLAMA_MODEL"; then
    echo "Pulling $OLLAMA_MODEL..."
    ollama pull "$OLLAMA_MODEL"
  fi
fi

# --- backend ---
cd "$ROOT/backend"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -r requirements.txt
fi
./.venv/bin/uvicorn main:app --port 8000 &
BACK_PID=$!

# --- frontend ---
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  npm install
fi
npm run dev &
FRONT_PID=$!

trap "kill $BACK_PID $FRONT_PID 2>/dev/null" EXIT
echo "Chat provider: $PROVIDER | TTS provider: $TTS"
echo "Backend  -> http://localhost:8000"
echo "Frontend -> http://localhost:5173"
wait
