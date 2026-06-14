#!/usr/bin/env bash
# Start backend (FastAPI/uvicorn) and frontend (Vite) together.
#
# Usage: ./run.sh <groq|ollama>
#   groq    chat via Groq llama-3.3-70b-versatile (cloud)
#   ollama  chat via local Ollama (uncensored Llama-3.1 Lexi V2)
#
# Speech (STT/TTS) always runs on Groq regardless of the chat provider.
# Ctrl-C stops both servers.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# --- require & validate the provider argument ---
PROVIDER="${1:-}"
if [ "$PROVIDER" != "groq" ] && [ "$PROVIDER" != "ollama" ]; then
  echo "Usage: ./run.sh <groq|ollama>" >&2
  echo "  e.g. ./run.sh ollama" >&2
  exit 1
fi
export CHAT_PROVIDER="$PROVIDER"

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
echo "Chat provider: $PROVIDER"
echo "Backend  -> http://localhost:8000"
echo "Frontend -> http://localhost:5173"
wait
