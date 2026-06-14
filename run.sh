#!/usr/bin/env bash
# Start backend (FastAPI/uvicorn) and frontend (Vite) together.
#
# Usage: ./run.sh
#
# No arguments. The chat provider (Groq | Ollama) and TTS voice (online Groq |
# offline Piper) are both chosen from dropdowns in the UI. STT always uses Groq.
# This script makes both providers ready, then starts the servers. Ctrl-C stops.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

OLLAMA_MODEL="${CHAT_MODEL:-hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M}"

# --- ollama: ensure the server is up and the model is pulled (for the dropdown) ---
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s -o /dev/null http://localhost:11434/api/tags 2>/dev/null; then
    echo "Starting ollama server..."
    ollama serve > /tmp/ollama.log 2>&1 &
    until curl -s -o /dev/null http://localhost:11434/api/tags 2>/dev/null; do sleep 0.3; done
  fi
  if ! ollama list | grep -qF "$OLLAMA_MODEL"; then
    echo "Pulling $OLLAMA_MODEL (first run only)..."
    ollama pull "$OLLAMA_MODEL"
  fi
else
  echo "Note: 'ollama' not found — the Local provider in the dropdown won't work until it's installed." >&2
fi

# --- piper: ensure the default offline voice exists (for the offline voices) ---
# Default offline female voice + the male voice used for *asterisk* segments.
VOICE_DIR="$ROOT/backend/voices"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"
mkdir -p "$VOICE_DIR"
download_voice() {  # $1=voice id, $2=path under en_US (e.g. amy/medium)
  if [ ! -f "$VOICE_DIR/$1.onnx" ]; then
    echo "Downloading Piper voice $1 (~60MB)..."
    curl -sL -o "$VOICE_DIR/$1.onnx" "$PIPER_BASE/$2/$1.onnx"
    curl -sL -o "$VOICE_DIR/$1.onnx.json" "$PIPER_BASE/$2/$1.onnx.json"
  fi
}
download_voice en_US-amy-medium amy/medium
download_voice en_US-joe-medium joe/medium   # male, for *asterisk* segments

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
echo "Backend  -> http://localhost:8000"
echo "Frontend -> http://localhost:5173"
echo "Pick chat provider and voice from the dropdowns in the UI."
wait
