#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/backend/requirements.txt"
npm --prefix "$ROOT_DIR/frontend" install

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

python -m uvicorn backend.main:app --host "$HOST" --port "$PORT" &
BACKEND_PID=$!

npm --prefix "$ROOT_DIR/frontend" run dev -- --host 0.0.0.0 --port 5173
