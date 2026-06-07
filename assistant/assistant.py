"""Single-thread CLI assistant using the local Codex proxy client."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken

from api.codex_client import build_client

from . import summary
from .sanitize import sanitize_llm_text


translate_en_to_bn = importlib.import_module("translate-en-to-bn")
translate_bn_to_en = importlib.import_module("translate-bn-to-en")
listening_en = importlib.import_module("listening-en")
listening_bn = importlib.import_module("listening-bn")
speaking_en = importlib.import_module("speaking-en")
speaking_bn = importlib.import_module("speaking-bn")

ENV_PATH = Path(__file__).resolve().parent / ".env"
SESSION_PATH = Path(".assistant") / "session.json"
TIKTOKEN_CACHE_DIR = Path(".assistant") / "tiktoken-cache"
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
# Conversation languages. The assistant itself always thinks in English; for "bn"
# the user's Bengali speech is transcribed then translated to English on the way
# in, and the English reply is translated back to Bengali on the way out.
LANGUAGES = ("en", "bn")
# faster-whisper default per language: ".en" models are English-only, so Bengali
# needs a multilingual model. Small multilingual models transcribe Bengali poorly
# (often transliterating to Latin or producing another script), so Bengali
# defaults to large-v3 for usable accuracy; use medium/small only if large-v3 is
# too slow on CPU.
DEFAULT_LISTEN_MODELS = {
    "en": listening_en.DEFAULT_MODEL,
    "bn": listening_bn.DEFAULT_MODEL,
}
GPT_5_4_MINI_CONTEXT_LIMIT = 400_000
REPLY_OUTPUT_TOKEN_LIMIT = 50
CHAT_MESSAGE_TOKEN_OVERHEAD = 4
CHAT_REPLY_PRIMER_TOKENS = 2


class PromptTokenLimitError(ValueError):
    """Raised when a request would exceed the model context window."""


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE lines from assistant/.env into the environment.

    Mainly so an HF_TOKEN here raises the Hugging Face download rate limits used
    when faster-whisper and the Bengali TTS model are fetched. Variables already
    set in the real environment win, so the file is only a fallback.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def natural_timestamp(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now().astimezone()

    hour = now.hour % 12 or 12
    am_pm = "AM" if now.hour < 12 else "PM"
    return f"{now.strftime('%B')} {now.day}, {now.year} at {hour}:{now.minute:02d} {am_pm}"


def make_message(role: str, content: str) -> dict[str, str]:
    return {
        "role": role,
        "content": content,
        "timestamp": natural_timestamp(),
    }


def new_session(model: str, reasoning_effort: str) -> dict[str, Any]:
    return {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "messages": [
            make_message("system", DEFAULT_SYSTEM_PROMPT),
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
        if "timestamp" in item and not isinstance(item.get("timestamp"), str):
            return False
    return True


def add_missing_timestamps(messages: list[dict[str, str]], timestamp: str) -> None:
    for message in messages:
        message.setdefault("timestamp", timestamp)


def sync_system_prompt(messages: list[dict[str, str]]) -> None:
    for message in messages:
        if message.get("role") == "system" and not message.get("kind"):
            message["content"] = DEFAULT_SYSTEM_PROMPT
            message.setdefault("timestamp", natural_timestamp())
            return

    messages.insert(0, make_message("system", DEFAULT_SYSTEM_PROMPT))


def should_persist_message(message: dict[str, str]) -> bool:
    # Assistant replies stay in memory for the active run, but are not saved.
    return message.get("role") != "assistant"


def persisted_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [message for message in messages if should_persist_message(message)]


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

    session_timestamp = natural_timestamp(
        datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    )
    add_missing_timestamps(data["messages"], session_timestamp)
    data["messages"] = persisted_messages(data["messages"])
    sync_system_prompt(data["messages"])
    data["model"] = model
    data["reasoning_effort"] = reasoning_effort
    return data


def save_session(path: Path, session: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    data = dict(session)
    data["messages"] = persisted_messages(session["messages"])
    temp_path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def format_message_for_prompt(message: dict[str, str]) -> dict[str, str]:
    timestamp = message.get("timestamp")
    if not timestamp:
        return {"role": message["role"], "content": message["content"]}

    return {
        "role": message["role"],
        "content": f"Timestamp: {timestamp}\n\n{message['content']}",
    }


def format_messages_for_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [format_message_for_prompt(message) for message in messages]


def remove_message_by_identity(
    messages: list[dict[str, str]], target: dict[str, str]
) -> bool:
    for index, message in enumerate(messages):
        if message is target:
            del messages[index]
            return True
    return False


def encoding_for_model(model: str) -> tiktoken.Encoding:
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_text_tokens(text: str, model: str) -> int:
    return len(encoding_for_model(model).encode(text))


def count_message_tokens(messages: list[dict[str, str]], model: str) -> int:
    """Estimate chat prompt tokens before sending the request."""
    encoding = encoding_for_model(model)
    total = CHAT_REPLY_PRIMER_TOKENS

    for message in messages:
        total += CHAT_MESSAGE_TOKEN_OVERHEAD
        total += len(encoding.encode(message["role"]))
        total += len(encoding.encode(message["content"]))

    return total


def ensure_prompt_fits_gpt_5_4_mini(
    messages: list[dict[str, str]], model: str
) -> int:
    prompt_tokens = count_message_tokens(messages, model)
    if prompt_tokens > GPT_5_4_MINI_CONTEXT_LIMIT:
        raise PromptTokenLimitError(
            "Full prompt exceeds gpt-5.4-mini context limit: "
            f"{prompt_tokens:,} estimated tokens > "
            f"{GPT_5_4_MINI_CONTEXT_LIMIT:,} token limit."
        )
    return prompt_tokens


def request_reply(
    client: Any,
    model: str,
    reasoning_effort: str,
    messages: list[dict[str, str]],
) -> str:
    print("Waiting for reply from OpenAI...", end="", flush=True)
    response = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=messages,
        max_completion_tokens=REPLY_OUTPUT_TOKEN_LIMIT,
        max_tokens=REPLY_OUTPUT_TOKEN_LIMIT,
    )
    print("\r\033[K", end="", flush=True)
    choice = response.choices[0]
    content = choice.message.content
    return sanitize_llm_text(content) if content else ""


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
        "--lang",
        required=True,
        choices=LANGUAGES,
        help=(
            "Conversation language. 'en': speak and hear English. 'bn': speak "
            "Bengali; it is transcribed then translated to English for the "
            "assistant, and replies are translated back to Bengali (all local)."
        ),
    )
    parser.add_argument(
        "--listen-model",
        default=None,
        help=(
            "faster-whisper model used to listen/transcribe. Defaults to base.en "
            "for --lang en and large-v3 (multilingual) for --lang bn. Common: "
            "tiny.en, base.en, small.en, medium.en, large-v3. The plain names "
            "(e.g. small, medium, large-v3) are multilingual; .en variants are "
            "English-only. Bengali needs a large multilingual model for usable "
            "accuracy (small ones transliterate or garble it); use medium/small "
            "only if large-v3 is too slow on CPU."
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


def listening_module(lang: str) -> Any:
    return listening_bn if lang == "bn" else listening_en


def speaking_module(lang: str) -> Any:
    return speaking_bn if lang == "bn" else speaking_en


def read_user_text(listen_model: str, lang: str) -> str:
    """Read the next user turn by listening.

    For Bengali, transcribe the speech to Bengali text first, then translate it
    to English (both local); the English text is what the assistant receives.
    """
    listener = listening_module(lang)
    text = listener.listen(listen_model)
    if lang != "bn":
        print(f"you said: {text}")
        return text

    if not text:
        return ""
    english = translate_bn_to_en.translate_bangla_to_english(text)
    print(f"you said (bn): {text}")
    print(f"translated (en): {english}")
    return english


def main() -> int:
    load_env_file()
    args = parse_args()
    model = args.model
    reasoning_effort = args.effort
    lang = args.lang
    listen_model = args.listen_model or DEFAULT_LISTEN_MODELS[lang]
    listener = listening_module(lang)
    speaker_module = speaking_module(lang)

    client = build_client()
    session = load_session(SESSION_PATH, model, reasoning_effort)
    messages = session["messages"]
    count_session_text_tokens = lambda text: count_text_tokens(text, model)
    count_session_prompt_tokens = lambda prompt: count_message_tokens(
        prompt, model
    )

    try:
        summarized, _user_tokens = summary.summarize_history_if_needed(
            client,
            model,
            reasoning_effort,
            session,
            count_session_text_tokens,
            count_session_prompt_tokens,
            natural_timestamp,
        )
        if summarized:
            save_session(SESSION_PATH, session)
    except Exception as exc:
        print_api_error(exc)
        return 1

    if lang == "bn":
        print("Loading Bengali speech model (MMS-TTS)...", flush=True)
        speaker = speaker_module.Speaker()
    else:
        speaker = speaker_module.Speaker(args.speak_voice)

    print(f"model: {model}")
    print(f"reasoning effort: {reasoning_effort}")
    print(f"language: {lang}")
    print(f"session: {SESSION_PATH}")
    print(f"listen: on (faster-whisper {listen_model}, local)")
    print("Loading listening model...", flush=True)
    listener.warm_up(listen_model)
    if lang == "bn":
        print("Loading translation models (bn<->en)...", flush=True)
        translate_bn_to_en.warm_up_bangla_to_english()
        translate_en_to_bn.warm_up_english_to_bangla()
        print("speak: on (MMS-TTS Bengali, local)")
    else:
        print(f"speak: on (Piper {args.speak_voice}, local)")
    print("Listening starts automatically. Press Ctrl-C to exit.")

    try:
        while True:
            try:
                user_text = read_user_text(listen_model, lang)
            except KeyboardInterrupt:
                print()
                return 0

            if not user_text:
                continue

            user_message = make_message("user", user_text)
            messages.append(user_message)

            try:
                summarized, _user_tokens = summary.summarize_history_if_needed(
                    client,
                    model,
                    reasoning_effort,
                    session,
                    count_session_text_tokens,
                    count_session_prompt_tokens,
                    natural_timestamp,
                )
                if summarized:
                    save_session(SESSION_PATH, session)
            except Exception as exc:
                remove_message_by_identity(messages, user_message)
                print_api_error(exc)
                return 1

            prompt_messages = format_messages_for_prompt(messages)

            try:
                prompt_tokens = ensure_prompt_fits_gpt_5_4_mini(
                    prompt_messages, model
                )
            except PromptTokenLimitError as exc:
                remove_message_by_identity(messages, user_message)
                print(f"Token limit error: {exc}", file=sys.stderr)
                return 1

            print(
                f"reply prompt tokens: {prompt_tokens:,}/"
                f"{GPT_5_4_MINI_CONTEXT_LIMIT:,}"
            )

            try:
                assistant_text = request_reply(
                    client, model, reasoning_effort, prompt_messages
                )
                output_tokens = count_text_tokens(assistant_text, model)
                print(
                    f"reply output tokens: "
                    f"{output_tokens:,}/{REPLY_OUTPUT_TOKEN_LIMIT:,}"
                )
                if lang == "bn":
                    reply_text = translate_en_to_bn.translate_english_to_bangla(
                        assistant_text
                    )
                else:
                    reply_text = assistant_text
                print(reply_text)
                speaker_module.speak_text(reply_text, speaker)
                speaker.wait()
            except KeyboardInterrupt:
                speaker.stop()
                remove_message_by_identity(messages, user_message)
                print("\nInterrupted; response was not saved.", file=sys.stderr)
                continue
            except Exception as exc:
                remove_message_by_identity(messages, user_message)
                print_api_error(exc)
                return 1

            messages.append(make_message("assistant", assistant_text))

            try:
                save_session(SESSION_PATH, session)
            except OSError as exc:
                print(f"Could not save session: {exc}", file=sys.stderr)
                return 1
    finally:
        speaker.close()


if __name__ == "__main__":
    raise SystemExit(main())
