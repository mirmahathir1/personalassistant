# api-llama

A drop-in replacement for `api/`. Instead of proxying your Codex/ChatGPT
subscription, this runs **Llama 3.1 8B Lexi Uncensored V2** (an uncensored
fine-tune of Llama 3.1 8B Instruct) locally and exposes the same
OpenAI-compatible endpoint on the same port:

```text
http://localhost:8317/v1
```

Because the assistant only knows about that URL (see `api/codex_client.py`,
which the assistant imports), **no assistant code changes are needed** — start a
server from this folder instead of `api/` and run the assistant as usual.

The model name you pass to the assistant (`--model ...`) is ignored by
llama-server, which always uses the loaded GGUF, so any value works.

---

## Option A — Native (recommended on macOS, uses Metal GPU)

Docker on macOS cannot access the Mac GPU, so an 8B model runs CPU-only and
slowly. Running `llama-server` natively gets Metal acceleration.

```bash
brew install llama.cpp

# Downloads the GGUF once (cached under ~/.cache/llama.cpp), then serves it.
llama-server \
  -hf bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8317 \
  --jinja --ctx-size 8192
```

Leave that running in its own terminal.

## Option B — Docker (parity with `api/`, CPU-only on macOS)

```bash
docker compose -f api-llama/docker-compose.yml up -d
```

The first start downloads the GGUF into `api-llama/models/` (git-ignored), so it
is fetched only once. Stop it with:

```bash
docker compose -f api-llama/docker-compose.yml down
```

---

## Smoke test

With a server running (Option A or B), confirm it answers the exact request
shape the assistant uses:

```bash
source assistant/.venv/bin/activate   # for the `openai` package
python3 api-llama/smoke_test.py
```

It asks a simple arithmetic question and checks the answer contains `2002`.

## Run the assistant against Llama

Identical to the main readme — just make sure a server from this folder is up
instead of the Codex proxy:

```bash
source assistant/.venv/bin/activate
python -m assistant.assistant --model llama-3.1-8b-lexi-uncensored --effort low --lang en
```

## Notes

- The assistant always sends `reasoning_effort` and `max_tokens`/
  `max_completion_tokens`. llama-server ignores unknown fields like
  `reasoning_effort`, so these are harmless.
- No API key is required: this server is started without `--api-key`, so the
  Bearer token the assistant sends is accepted and ignored. To require one, add
  `--api-key <key>` to the server command and export `CLIPROXY_API_KEY=<key>`.
- `Q4_K_M` is a good speed/quality balance (~4.9 GB). For higher quality use
  `:Q5_K_M` or `:Q6_K`; for less RAM use `:Q3_K_M`.
- Quality is lower than the Codex models. If the assistant's token guard ever
  complains, the model's context is set with `--ctx-size` above.
```
