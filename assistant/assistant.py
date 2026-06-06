"""Single-thread CLI assistant using the local Codex proxy client."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from api.codex_client import build_client

from .listen import listen, warm_up
from .speak import Speaker


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


def extract_sentence(buffer: str) -> tuple[str | None, str]:
    """Split one complete sentence off the front of buffer for speaking.

    Returns (sentence, rest). When no complete sentence is present yet, returns
    (None, buffer) unchanged so the caller can keep accumulating deltas.
    """
    for index, char in enumerate(buffer):
        if char not in ".!?\n":
            continue
        end = index + 1
        # Absorb trailing closing quotes/brackets so they go with the sentence.
        while end < len(buffer) and buffer[end] in '")]’”':
            end += 1
        if end >= len(buffer):
            # More text may still be arriving; only flush hard line breaks now.
            if char == "\n":
                return buffer[:end].strip(), buffer[end:]
            return None, buffer
        if buffer[end].isspace():
            return buffer[:end].strip(), buffer[end:].lstrip()
    return None, buffer


def stream_reply(
    client: Any,
    model: str,
    reasoning_effort: str,
    messages: list[dict[str, str]],
    speaker: Speaker,
) -> str:
    print("Waiting for reply from OpenAI...", end="", flush=True)
    stream = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=messages,
        stream=True,
    )

    parts: list[str] = []
    pending = ""
    waiting_shown = True
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if waiting_shown:
                # Clear the "Waiting..." line before the reply starts.
                print("\r\033[K", end="", flush=True)
                waiting_shown = False
            print(delta, end="", flush=True)
            parts.append(delta)
            pending += delta
            while True:
                sentence, pending = extract_sentence(pending)
                if sentence is None:
                    break
                speaker.say(sentence)
    if waiting_shown:
        # Empty reply: clear the leftover "Waiting..." line.
        print("\r\033[K", end="", flush=True)
    print()
    if pending.strip():
        speaker.say(pending)
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
    parser.add_argument(
        "--listen-model",
        default="base.en",
        help=(
            "faster-whisper model used to listen/transcribe (default: base.en). "
            "Common: tiny.en, base.en, small.en, medium.en, large-v3. The plain "
            "names (e.g. small) are multilingual; .en variants are English-only "
            "and faster."
        ),
    )
    parser.add_argument(
        "--speak-voice",
        default="en_US-lessac-medium",
        help=(
            "Piper voice used to speak replies (default: en_US-lessac-medium). "
            "Download others with `python -m piper.download_voices <name> "
            "--download-dir .assistant/voices`."
        ),
    )
    return parser.parse_args(argv)


def read_user_text(listen_model: str = "base.en") -> str:
    """Read the next user turn by listening, with a typed fallback.

    Raises EOFError / KeyboardInterrupt like input() so the main loop can exit.
    """
    choice = input("[Enter]=record  [t]=type  > ").strip()
    if choice.lower() == "t":
        return input("type> ").strip()
    if choice:
        # Anything else typed at the prompt is treated as text input.
        return choice

    text = listen(listen_model)
    print(f"you said: {text}")
    return text


def main() -> int:
    args = parse_args()
    model = args.model
    reasoning_effort = args.effort

    client = build_client()
    session = load_session(SESSION_PATH, model, reasoning_effort)
    messages = session["messages"]

    speaker = Speaker(args.speak_voice)

    print(f"model: {model}")
    print(f"reasoning effort: {reasoning_effort}")
    print(f"session: {SESSION_PATH}")
    print(f"listen: on (faster-whisper {args.listen_model}, local)")
    print("Loading listening model...", flush=True)
    warm_up(args.listen_model)
    print(f"speak: on (Piper {args.speak_voice}, local)")
    print("Press Ctrl-D or Ctrl-C at the prompt to exit.")

    try:
        while True:
            try:
                user_text = read_user_text(args.listen_model)
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
                assistant_text = stream_reply(
                    client, model, reasoning_effort, messages, speaker
                )
                speaker.wait()
            except KeyboardInterrupt:
                speaker.stop()
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
    finally:
        speaker.close()


if __name__ == "__main__":
    raise SystemExit(main())
