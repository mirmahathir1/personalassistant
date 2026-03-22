FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential cmake libgomp1 \
    && python -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["sh", "-c", "pip install -r /app/backend/requirements.txt && uvicorn backend.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
