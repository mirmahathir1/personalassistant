# CLI Assistant Plan

## Goal

Build a Python CLI chat tool that uses the same local OpenAI-compatible proxy interface shown in `example.py`:

```python
from codex_client import MODEL, build_client

client = build_client()
client.chat.completions.create(...)
```

The CLI should behave like one continuing chat thread. When the user exits and later starts the tool again, it loads the last saved conversation and keeps going from that context.

## Key Constraint

The `chat.completions.create` interface used by `example.py` is stateless. It does not create or remember a server-side thread. To continue from the last session, the CLI must store the message history locally and send that full history with each new request.

## Proposed File

Create a new executable Python entry point:

```text
assistant.py
```

It should import the existing client helper rather than duplicating proxy configuration:

```python
from codex_client import MODEL, build_client
```

## Conversation Storage

Use one local session file:

```text
.assistant/session.json
```

The file should contain:

```json
{
  "model": "selected-model-id",
  "messages": [
    {"role": "system", "content": "You are a warm, supportive counselor..."},
    {"role": "user", "content": "First user message"},
    {"role": "assistant", "content": "First assistant reply"}
  ]
}
```

Rules:

- Create `.assistant/` if it does not exist.
- Load `.assistant/session.json` by default at startup.
- If no session exists, start a new one with the default system message.
- Save the file after every assistant response.
- Keep only one active thread.

## CLI Behavior

Default command:

```bash
python assistant.py
```

Behavior:

1. Load the existing session if present.
2. Print the selected model.
3. Start an interactive prompt.
4. For each user message:
   - Append `{"role": "user", "content": text}`.
   - Call `client.chat.completions.create(model=MODEL, messages=messages, stream=True)`.
   - Stream the assistant response to stdout.
   - Append `{"role": "assistant", "content": full_response}`.
   - Save the updated session.
5. Exit cleanly on Ctrl-D or Ctrl-C while waiting at the prompt.

## Implementation Outline

Functions:

```python
def load_session(path: Path, system_prompt: str) -> dict:
    ...

def save_session(path: Path, session: dict) -> None:
    ...

def stream_reply(client, messages: list[dict[str, str]]) -> str:
    ...

def main() -> None:
    ...
```

`stream_reply` should follow the streaming pattern already present in `codex_client.py`:

```python
stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    stream=True,
)

parts = []
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
        parts.append(delta)
print()
return "".join(parts)
```

## Error Handling

Handle these cases cleanly:

- Proxy is not running: print that `http://localhost:8317/v1` must be available.
- Proxy auth expired: tell the user to repeat the `--codex-login` step from `readme.md`.
- Invalid session JSON: move the bad file aside to `session.json.bak` and start a new session.
- Empty user input: ignore it.
- Keyboard interrupt while waiting at the prompt: exit.
- Keyboard interrupt during streaming: do not save a partial assistant response unless it completed.

## Context Growth

The simplest version can resend the full message history forever. After that works, add a limit:

- Keep the system message.
- Keep the most recent N turns.
- Optionally summarize older turns into one assistant-maintained memory message.

Do not add summarization in the first pass unless context length becomes a real problem.

## Acceptance Criteria

- Running `python assistant.py` starts an interactive chat.
- The tool uses `build_client()` and `MODEL` from `codex_client.py`.
- It calls `client.chat.completions.create` with the same OpenAI SDK interface used in `example.py`.
- Replies stream to the terminal.
- Exiting and restarting continues from `.assistant/session.json`.
- No real OpenAI platform API key is required; authentication still goes through the local CLIProxyAPI server.
