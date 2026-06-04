# Codex-as-API via CLIProxyAPI (Docker)

Use your **ChatGPT / Codex subscription** as a local, OpenAI-compatible API.

This project runs [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) inside
Docker. The proxy logs into your ChatGPT account using Codex's OAuth flow and exposes
a standard OpenAI `/v1` API on `http://localhost:8317`. Any OpenAI client library can
then talk to it — no OpenAI API key required, just your existing subscription.

Nothing is installed on the host except Docker. Everything lives in this folder.

---

## Implementation checklist (for the coding agent)

1. Create the directory layout shown in **Project structure**.
2. Create `config.yaml` exactly as shown.
3. Create `docker-compose.yml` exactly as shown.
4. Create `client/codex_client.py` and `client/requirements.txt`.
5. Run the **one-time login** command and complete OAuth in a browser.
6. Start the server with `docker compose up -d`.
7. Verify with the **health check** (`curl /v1/models`).
8. Run the Python client to confirm end-to-end.

Do **not** commit `auth/` (it contains access tokens). A `.gitignore` is included.

---

## Prerequisites

- Docker and Docker Compose installed and running.
- A **paid ChatGPT account** (Plus / Pro / Business / Edu / Enterprise). The proxy
  uses your subscription; it does not work with a free account.
- A browser available on the machine where you run the login step (the OAuth
  redirect returns to `localhost:1455`).

---

## Project structure

```
codex-proxy/
├── config.yaml            # proxy configuration
├── docker-compose.yml     # server definition
├── .gitignore
├── auth/                  # OAuth tokens are written here (DO NOT COMMIT)
└── client/
    ├── codex_client.py    # Python example client
    └── requirements.txt
```

Create the empty `auth/` directory up front:

```bash
mkdir -p auth client
```

---

## Configuration

### `config.yaml`

```yaml
# Bind to all interfaces inside the container; Docker maps it to localhost:8317.
host: "0.0.0.0"
port: 8317

# Where OAuth tokens are stored inside the container (mounted to ./auth on host).
auth-dir: "/root/.cli-proxy-api"

# Client API keys. These are local secrets YOU invent. Your Python client sends
# one of these as its Bearer token. They are NOT OpenAI keys.
api-keys:
  - "sk-codexproxy-local-7f3a9c2e8b14d05f"

# Number of automatic retries on transient upstream errors (403/408/5xx).
request-retry: 3

# Strongest-output setting: force maximum reasoning effort on every request so the
# model thinks as hard as it can. "xhigh" is the top tier on current flagship/codex
# models; drop to "high" if a model rejects "xhigh".
payload:
  override:
    - models:
        - name: "gpt-*"
          protocol: "codex"
      params:
        "reasoning.effort": "xhigh"
```

> This project uses a fixed, hardcoded client key:
> `sk-codexproxy-local-7f3a9c2e8b14d05f`. It is used everywhere in this README — in
> `config.yaml`, the curl tests, and the Python client — so nothing needs to be
> generated. It is a **local secret**, not an OpenAI key. Keep it as-is, or replace
> every occurrence with your own value if you prefer.

### `.gitignore`

```gitignore
auth/
*.log
__pycache__/
client/.venv/
```

---

## Server (Docker Compose)

### `docker-compose.yml`

```yaml
services:
  cliproxy:
    image: eceasy/cli-proxy-api:latest
    container_name: codex-proxy
    restart: unless-stopped
    ports:
      - "8317:8317"     # OpenAI-compatible API
      - "1455:1455"     # OAuth callback (only needed during login)
    volumes:
      - ./config.yaml:/CLIProxyAPI/config.yaml
      - ./auth:/root/.cli-proxy-api
```

---

## Step 1 — One-time login (Codex / ChatGPT OAuth)

Authenticate once. Tokens are written to `./auth/` and persist across restarts.

```bash
docker compose run --rm --service-ports cliproxy \
  /CLIProxyAPI/CLIProxyAPI --codex-login --no-browser
```

This prints a URL. Open it in your browser, sign in to ChatGPT, and approve.
The redirect completes back to `localhost:1455` (forwarded into the container),
and `auth/` will now contain your token file.

> `--no-browser` is used because the container has no browser of its own; you open
> the printed URL on your host machine instead.

---

## Step 2 — Start the server

```bash
docker compose up -d
```

Check logs:

```bash
docker compose logs -f cliproxy
```

The API is now live at `http://localhost:8317/v1`.

---

## Step 3 — Health check

List the models your subscription exposes (use a key from `config.yaml`):

```bash
curl http://localhost:8317/v1/models \
  -H "Authorization: Bearer sk-codexproxy-local-7f3a9c2e8b14d05f"
```

A quick chat test:

```bash
curl http://localhost:8317/v1/chat/completions \
  -H "Authorization: Bearer sk-codexproxy-local-7f3a9c2e8b14d05f" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "Say hello in one word."}]
  }'
```

---

## Step 4 — Connect from Python

### `client/requirements.txt`

```
openai>=1.0.0
requests
```

### `client/codex_client.py`

```python
"""Call your ChatGPT/Codex subscription through the local CLIProxyAPI server."""

import os
from openai import OpenAI

# Base URL of the local proxy (note the trailing /v1).
BASE_URL = os.getenv("CLIPROXY_BASE_URL", "http://localhost:8317/v1")

# Must match one of the values under `api-keys:` in config.yaml.
API_KEY = os.getenv("CLIPROXY_API_KEY", "sk-codexproxy-local-7f3a9c2e8b14d05f")

# Model served through your subscription. We default to the STRONGEST available
# model. See the "Model selection" section — confirm with GET /v1/models.
MODEL = os.getenv("CLIPROXY_MODEL", "gpt-5.5")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def chat(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise coding assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def stream(prompt: str) -> None:
    s = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in s:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    print(chat("Write a one-line Python function to reverse a string."))
    print("\n--- streaming ---")
    stream("Explain what a Python decorator is in two sentences.")
```

### Run it

```bash
cd client
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
export CLIPROXY_API_KEY="sk-codexproxy-local-7f3a9c2e8b14d05f"
python codex_client.py
```

---

## Common operations

| Action | Command |
| --- | --- |
| Start server | `docker compose up -d` |
| Stop server | `docker compose down` |
| View logs | `docker compose logs -f cliproxy` |
| Update image | `docker compose pull && docker compose up -d` |
| Re-login | rerun the Step 1 login command |
| Full removal | `docker compose down && docker rmi eceasy/cli-proxy-api:latest` then delete the folder |

---

## Model selection — use the strongest available

**Goal: always run the most capable model the subscription exposes.**

Default model for this project is **`gpt-5.5`** — OpenAI's current flagship and the
strongest for complex coding, agentic workflows, reasoning, and tool use. The
`config.yaml` above also forces maximum reasoning effort (`xhigh`) so the model
thinks as hard as it can on every request.

Because model availability changes over time and depends on subscription tier, do
not hard-assume a name. Pick the strongest at runtime:

1. List what your account actually exposes:
   ```bash
   curl http://localhost:8317/v1/models \
     -H "Authorization: Bearer sk-codexproxy-local-7f3a9c2e8b14d05f"
   ```
2. Choose the strongest from the returned list, preferring in this order:
   `gpt-5.5` → `gpt-5.4` → newest `*-codex` (e.g. `gpt-5.3-codex`) →
   `gpt-5.1-codex-max`. Avoid `mini` / `spark` variants — those are the small,
   fast models, not the strongest.
3. Set it once via the environment variable so you never edit code:
   ```bash
   export CLIPROXY_MODEL="gpt-5.5"
   ```

> Note: the very strongest models are sometimes gated to higher tiers (for example,
> top Codex models have launched as ChatGPT Pro–only before wider rollout). If a
> model name 404s, it isn't available on your plan — fall back to the next one in
> the list above.

---

## Troubleshooting

- **401 / "invalid api key"** — The `Authorization` bearer token must match an entry
  in `config.yaml` `api-keys`. This is your local secret, not an OpenAI key.
- **Empty `auth/` after login** — Confirm the `./auth:/root/.cli-proxy-api` volume is
  mounted and that you completed the browser OAuth. Re-run Step 1.
- **OAuth never completes** — Ensure port `1455` is published during login and not in
  use by another process. The login command above publishes it via `--service-ports`.
- **Model not found** — Run the `/v1/models` health check and use a name from that list.
- **Connection refused on 8317** — Check `docker compose ps` and the logs; make sure
  the container is running and the port is mapped.

---

## Important caveats

- This routes through ChatGPT's **undocumented backend** that Codex CLI uses. It can
  change or break without notice and has no SLA.
- Your normal **subscription rate limits** still apply, regardless of how you call it.
- This is an **unofficial** use of your account and is not affiliated with OpenAI.
  Treat it as a personal/dev convenience, not production infrastructure.
- Keep `auth/` and your `api-keys` private — they grant access to your subscription.