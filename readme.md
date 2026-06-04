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
4. Create `codex_client.py` and `requirements.txt`.
5. Run the **one-time login** command and complete OAuth in a browser.
6. Start the server with `docker compose up -d`.
7. Verify with the **health check** (`curl /v1/models`).
8. Run the Python client to confirm end-to-end.

Do **not** commit `auth/` (it contains access tokens). A `.gitignore` is included.

---

## Prerequisites

- Docker and Docker Compose installed and running.
- A **ChatGPT account**. Codex is included on all tiers — Free, Go, Plus, Pro,
  Business, Edu, Enterprise — so this works without paying. Free has tight usage
  limits and may not expose the very top models, so a paid plan is recommended for
  sustained use of the strongest model, but it is not required to get started.
- A browser available on the machine where you run the login step (the OAuth
  redirect returns to `localhost:1455`).

---

## Project structure

```
codex-proxy/
├── config.yaml            # proxy configuration
├── docker-compose.yml     # server definition
├── .gitignore
├── codex_client.py        # Python example client
├── requirements.txt
└── auth/                  # OAuth tokens are written here (DO NOT COMMIT)
```

Create the empty `auth/` directory up front:

```bash
mkdir -p auth
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
.venv/
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

### `requirements.txt`

```
openai>=1.0.0
requests
```

### `codex_client.py`

```python
"""Call your ChatGPT/Codex subscription through the local CLIProxyAPI server.

The model is NOT hardcoded. At startup the client asks the proxy which models your
account actually exposes (GET /v1/models) and automatically selects the strongest
one. This works on any plan (Free or paid) because it only ever chooses from what
your own authentication grants.
"""

import os
import re
from openai import OpenAI

# Base URL of the local proxy (note the trailing /v1).
BASE_URL = os.getenv("CLIPROXY_BASE_URL", "http://localhost:8317/v1")

# Must match one of the values under `api-keys:` in config.yaml.
API_KEY = os.getenv("CLIPROXY_API_KEY", "sk-codexproxy-local-7f3a9c2e8b14d05f")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Lightweight / non-flagship variants to skip when picking the "strongest" model.
_WEAK_MARKERS = ("mini", "spark", "nano", "lite", "micro", "instant", "flash")


def _score(model_id: str) -> float:
    """Higher score = stronger model. Version number dominates; small tie-breakers
    favor 'max' variants and (for this coding use case) 'codex' variants."""
    name = model_id.lower()
    if any(marker in name for marker in _WEAK_MARKERS):
        return -1.0  # exclude small/fast variants from "strongest"
    match = re.search(r"(\d+(?:\.\d+)?)", name)  # e.g. 5.5, 5.1, 5
    version = float(match.group(1)) if match else 0.0
    score = version * 100.0
    if "max" in name:
        score += 5.0
    if "codex" in name:
        score += 2.0
    return score


def pick_strongest_model() -> str:
    """Query the proxy and return the strongest available model id.

    Override with the CLIPROXY_MODEL env var if you ever want to force a specific one.
    """
    forced = os.getenv("CLIPROXY_MODEL")
    if forced:
        return forced

    models = [m.id for m in client.models.list().data]
    if not models:
        raise RuntimeError("No models returned by the proxy. Is it logged in and running?")

    ranked = sorted(models, key=_score, reverse=True)
    best = ranked[0]
    if _score(best) < 0:  # everything was a 'weak' variant; just take the top one
        best = sorted(models, reverse=True)[0]
    return best


MODEL = pick_strongest_model()
print(f"[client] using model: {MODEL}")


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
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
export CLIPROXY_API_KEY="sk-codexproxy-local-7f3a9c2e8b14d05f"
python codex_client.py   # auto-selects the strongest model and prints which one
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

## Model selection — strongest available, auto-detected

**The model is not hardcoded.** The client asks the proxy what your account exposes
and picks the strongest at runtime, so it works on any plan (Free or paid) and
automatically upgrades itself when OpenAI ships a newer model.

How `pick_strongest_model()` ranks candidates:

1. It calls `GET /v1/models` (via `client.models.list()`), which returns only the
   models your authenticated subscription can actually use.
2. It drops lightweight/fast variants (`mini`, `spark`, `nano`, `lite`, `micro`,
   `instant`, `flash`) — those are the small models, not the strongest.
3. It scores the rest, with the **version number dominating** (e.g. `gpt-5.5` beats
   `gpt-5.1-codex-max`), plus small tie-breakers favoring `max` and `codex` variants.
4. It selects the top-scoring model and prints which one it chose.

This naturally adapts by tier: a Pro account might resolve to `gpt-5.5`, while a Free
account might resolve to whatever lighter flagship it's allowed — without any code
changes. The combination of "strongest available model" + the `xhigh` reasoning
effort forced in `config.yaml` gives you the most capable output your plan permits.

### Forcing a specific model (optional)

Auto-selection is the default. If you ever want to pin one explicitly, set an env var
and the client uses it as-is:

```bash
export CLIPROXY_MODEL="gpt-5.5"
```

To inspect what's available yourself:

```bash
curl http://localhost:8317/v1/models \
  -H "Authorization: Bearer sk-codexproxy-local-7f3a9c2e8b14d05f"
```

> Note: the very strongest models are sometimes gated to higher tiers (for example,
> some top Codex models have launched as ChatGPT Pro–only before wider rollout) and
> Free has tight usage limits. Auto-selection handles this gracefully — it only ever
> picks from models your account is actually granted.

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
