#!/usr/bin/env bash
# Start backend (FastAPI/uvicorn) and frontend (Vite) together.
#
# Usage: ./run.sh
#
# No arguments. Fully offline: chat via local Ollama, TTS via local Piper, STT
# via local faster-whisper. This script makes everything ready (Ollama server +
# models, Piper voices, Python deps), then starts the servers. Ctrl-C stops.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

OLLAMA_MODEL="${CHAT_MODEL:-hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M}"

# --- ollama: ensure the server is up and the chat + embedding models are pulled ---
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
  # Embedding model for the semantic half of conversation retrieval. Without it,
  # retrieval falls back to BM25 keyword search (still works, lower recall).
  EMBED_MODEL_ID="${EMBED_MODEL:-nomic-embed-text}"
  if ! ollama list | grep -qF "$EMBED_MODEL_ID"; then
    echo "Pulling $EMBED_MODEL_ID for retrieval (first run only)..."
    ollama pull "$EMBED_MODEL_ID"
  fi
else
  echo "Error: 'ollama' not found — chat requires it. Install from https://ollama.com" >&2
  exit 1
fi

# --- piper: ensure the default offline voice exists (for the offline voices) ---
# Default offline voice + Sofia (used for *asterisk* segments).
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
download_voice en_US-libritts_r-medium libritts_r/medium   # Sofia, for *asterisk* segments

# --- backend ---
cd "$ROOT/backend"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# Always sync deps (idempotent and fast when satisfied) so new requirements —
# e.g. rank-bm25/numpy for retrieval — get installed even on an existing venv.
./.venv/bin/pip install -q -r requirements.txt
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
echo "Fully offline. Pick a voice from the dropdown in the UI."
wait
