#!/usr/bin/env bash
# Start backend (FastAPI/uvicorn) and frontend (Vite) together.
#
# Usage: ./run.sh
#
# No arguments. Fully offline: chat via local Ollama, TTS via local Kokoro, STT
# via local faster-whisper. This script makes everything ready (Ollama server +
# models, Kokoro model+voices, Python deps), then starts the servers. Ctrl-C stops.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

OLLAMA_MODEL="${CHAT_MODEL:-hf.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M}"

# Keep all Llama (chat) + embedding models in a project-local folder so the app
# is self-contained and portable. Ollama reads this via OLLAMA_MODELS; we export
# it for both `ollama serve` and the `ollama pull`/`ollama list` calls below.
# Models are large and stay out of git (see .gitignore).
export OLLAMA_MODELS="${OLLAMA_MODELS:-$ROOT/models}"
mkdir -p "$OLLAMA_MODELS"

# --- ollama: ensure the server is up and the chat + embedding models are pulled ---
if command -v ollama >/dev/null 2>&1; then
  # If a server is already running, it may be using a different models dir than
  # our project-local one. Warn so models don't silently land in ~/.ollama.
  if curl -s -o /dev/null http://localhost:11434/api/tags 2>/dev/null; then
    echo "Note: an ollama server is already running; it may not use OLLAMA_MODELS=$OLLAMA_MODELS." >&2
    echo "      Stop it (e.g. 'pkill ollama') and re-run to use the project-local model dir." >&2
  else
    echo "Starting ollama server (models in $OLLAMA_MODELS)..."
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

# --- kokoro: ensure the neural TTS model + voice styles exist (offline) ---
# One ONNX model (~310MB) + one voices file (~27MB, ~50 bundled voices) serve
# every selectable voice. Kept project-local in models/kokoro/ (out of git).
KOKORO_DIR="$ROOT/models/kokoro"
KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
mkdir -p "$KOKORO_DIR"
download_kokoro() {  # $1=filename, $2=approx size for the message
  if [ ! -f "$KOKORO_DIR/$1" ]; then
    echo "Downloading Kokoro $1 ($2, first run only)..."
    curl -fL --retry 3 -o "$KOKORO_DIR/$1" "$KOKORO_BASE/$1"
  fi
}
download_kokoro kokoro-v1.0.onnx "~310MB"
download_kokoro voices-v1.0.bin "~27MB"

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
