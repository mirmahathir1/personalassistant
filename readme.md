# Minimal Setup

From a fresh clone:

```bash
git clone <repo-url>
cd personalassistant
mkdir -p auth
docker compose pull
docker compose run --rm --service-ports cliproxy /CLIProxyAPI/CLIProxyAPI --codex-login --no-browser
```

Open the printed OpenAI URL in your browser, sign in, and finish OAuth. Then run:

```bash
docker compose up -d
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python example.py
```

The local OpenAI-compatible API will be available at:

```text
http://localhost:8317/v1
```
