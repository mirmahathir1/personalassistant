"""Single-thread CLI assistant using the local Codex proxy client."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SESSION_PATH = Path(".assistant") / "session.json"
DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, supportive counselor. Listen carefully and reflect back "
    "what you hear so the person feels understood. Ask gentle, open-ended "
    "questions to help them explore their thoughts and feelings, and validate "
    "their emotions without judgment. Offer perspective and coping strategies "
    "when helpful, but guide the person toward their own insights rather than "
    "dictating answers. Keep a calm, patient, and non-judgmental tone. You are "
    "not a replacement for professional mental health care; if someone is in "
    "crisis or at risk of harming themselves or others, gently encourage them "
    "to contact a qualified professional or emergency services."
)
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def new_session(model: str, reasoning_effort: str) -> dict[str, Any]:
    return {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        ],
    }


def backup_path(path: Path) -> Path:
    candidate = path.with_suffix(path.suffix + ".bak")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        numbered = path.with_suffix(path.suffix + f".bak.{index}")
        if not numbered.exists():
            return numbered
        index += 1


def valid_messages(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("role"), str):
            return False
        if not isinstance(item.get("content"), str):
            return False
    return True


def load_session(path: Path, model: str, reasoning_effort: str) -> dict[str, Any]:
    if not path.exists():
        return new_session(model, reasoning_effort)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup = backup_path(path)
        path.rename(backup)
        print(f"Invalid session JSON moved to {backup}", file=sys.stderr)
        return new_session(model, reasoning_effort)

    if not isinstance(data, dict) or not valid_messages(data.get("messages")):
        backup = backup_path(path)
        path.rename(backup)
        print(f"Invalid session shape moved to {backup}", file=sys.stderr)
        return new_session(model, reasoning_effort)

    data["model"] = model
    data["reasoning_effort"] = reasoning_effort
    return data


def save_session(path: Path, session: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(session, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def stream_reply(
    client: Any,
    model: str,
    reasoning_effort: str,
    messages: list[dict[str, str]],
) -> str:
    stream = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=messages,
        stream=True,
    )

    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            parts.append(delta)
    print()
    return "".join(parts)


def print_api_error(exc: Exception) -> None:
    detail = f"{type(exc).__name__}: {exc}"
    lower = detail.lower()

    if "connection" in lower or "localhost:8317" in lower:
        print(
            "Could not reach the local proxy at http://localhost:8317/v1. "
            "Start it with `docker compose up -d`.",
            file=sys.stderr,
        )
    elif "401" in lower or "403" in lower or "unauthorized" in lower:
        print(
            "The local proxy rejected the request. If auth expired, repeat the "
            "`--codex-login` step from readme.md.",
            file=sys.stderr,
        )
    else:
        print("The local proxy request failed.", file=sys.stderr)

    print(detail, file=sys.stderr)


def load_client() -> Any:
    try:
        from codex_client import build_client
    except Exception as exc:
        print_api_error(exc)
        raise SystemExit(1) from exc

    return build_client()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-thread CLI assistant using the local Codex proxy.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use for every request (e.g. gpt-5-codex).",
    )
    parser.add_argument(
        "--effort",
        required=True,
        choices=REASONING_EFFORTS,
        help="Reasoning effort to use for every request.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    model = args.model
    reasoning_effort = args.effort

    client = load_client()
    session = load_session(SESSION_PATH, model, reasoning_effort)
    messages = session["messages"]

    print(f"model: {model}")
    print(f"reasoning effort: {reasoning_effort}")
    print(f"session: {SESSION_PATH}")
    print("Press Ctrl-D or Ctrl-C at the prompt to exit.")

    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not user_text:
            continue

        user_message = {"role": "user", "content": user_text}
        messages.append(user_message)

        try:
            assistant_text = stream_reply(client, model, reasoning_effort, messages)
        except KeyboardInterrupt:
            messages.pop()
            print("\nInterrupted; response was not saved.", file=sys.stderr)
            continue
        except Exception as exc:
            messages.pop()
            print_api_error(exc)
            return 1

        messages.append({"role": "assistant", "content": assistant_text})

        try:
            save_session(SESSION_PATH, session)
        except OSError as exc:
            print(f"Could not save session: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
