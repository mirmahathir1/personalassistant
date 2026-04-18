FROM python:3.12-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_APP=backend.app

CMD ["sh", "-c", "python -m pip install --no-cache-dir -r backend/requirements.txt && flask run --debug --host 0.0.0.0 --port 5000"]
