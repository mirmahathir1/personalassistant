# Minimal Setup

From a fresh clone:

```bash
git clone <repo-url>
cd personalassistant
mkdir -p api/auth
docker compose -f api/docker-compose.yml pull
docker compose -f api/docker-compose.yml run --rm --service-ports cliproxy /CLIProxyAPI/CLIProxyAPI --codex-login --no-browser
```

Open the printed OpenAI URL in your browser, sign in, and finish OAuth. Then
start the proxy:

```bash
docker compose -f api/docker-compose.yml up -d
```

The local OpenAI-compatible API will be available at:

```text
http://localhost:8317/v1
```

## Call it from Python (no Docker)

`example/` is a minimal client that calls the running proxy with plain Python —
no container needed:

```bash
python3 -m venv example/.venv
source example/.venv/bin/activate
pip install -r example/requirements.txt
python -m example.example
```

## Run the assistant

The assistant is a listen-and-speak CLI that talks to the proxy above, so make
sure the proxy is up first (`docker compose -f api/docker-compose.yml up -d`). It
listens through your microphone and speaks its replies, so run it on the host:

```bash
python3 -m venv assistant/.venv
source assistant/.venv/bin/activate
pip install -r api/requirements.txt
pip install -r assistant/requirements.txt
python -m piper.download_voices en_US-lessac-medium --download-dir .assistant/voices
python -m assistant.assistant --model gpt-5.4-mini --effort low
```

At the prompt press Enter to record from your microphone (or type `t` to type a
turn instead); replies stream to the terminal and are spoken aloud with Piper.
