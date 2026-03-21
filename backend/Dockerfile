FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY backend ./backend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
