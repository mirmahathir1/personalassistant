"""Summarize and compact long user dialogue history."""

from __future__ import annotations

from typing import Any, Callable

from .sanitize import sanitize_llm_text


USER_DIALOGUE_SUMMARY_TOKEN_LIMIT = 500
SUMMARY_MESSAGE_KIND = "user_dialogue_summary"
SUMMARY_SYSTEM_PROMPT = (
    "You polish and compact long user dialogue history for a personal "
    "assistant. Summarize only what the user said. Preserve durable facts, "
    "preferences, goals, concerns, emotional themes, important dates, open "
    "questions, and commitments. Do not include assistant replies, do not "
    "answer the user, and do not invent details. Return only the polished "
    "summary."
)

TextTokenCounter = Callable[[str], int]
PromptTokenCounter = Callable[[list[dict[str, str]]], int]
TimestampFactory = Callable[[], str]


def make_summary_message(content: str, timestamp: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": "Polished summary of prior user dialogue:\n\n" + content.strip(),
        "timestamp": timestamp,
        "kind": SUMMARY_MESSAGE_KIND,
    }


def unsummarized_dialogue_start_index(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("kind") == SUMMARY_MESSAGE_KIND:
            return index + 1

    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1
    return index


def count_user_dialogue_tokens(
    messages: list[dict[str, str]], count_text_tokens: TextTokenCounter
) -> int:
    return sum(
        count_text_tokens(message["content"])
        for message in messages
        if message.get("role") == "user"
    )


def format_user_messages_for_summary(messages: list[dict[str, str]]) -> str:
    entries = []
    user_index = 1
    for message in messages:
        if message.get("role") != "user":
            continue

        timestamp = message.get("timestamp", "unknown time")
        entries.append(
            f"User message {user_index} ({timestamp}):\n{message['content']}"
        )
        user_index += 1

    return "\n\n---\n\n".join(entries)


def build_summary_prompt(user_messages_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "The unsummarized user dialogue has exceeded "
                f"{USER_DIALOGUE_SUMMARY_TOKEN_LIMIT:,} tokens. Create a "
                "polished, compact summary from these user messages only:\n\n"
                f"{user_messages_text}"
            ),
        },
    ]


def summarize_user_dialogue(
    client: Any,
    model: str,
    reasoning_effort: str,
    messages: list[dict[str, str]],
    count_text_tokens: TextTokenCounter,
    count_prompt_tokens: PromptTokenCounter,
) -> str:
    user_messages_text = format_user_messages_for_summary(messages)
    summary_prompt = build_summary_prompt(user_messages_text)
    print(
        f"summary prompt tokens: {count_prompt_tokens(summary_prompt):,}",
        flush=True,
    )
    response = client.chat.completions.create(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=summary_prompt,
    )
    summary = response.choices[0].message.content
    summary = sanitize_llm_text(summary) if summary else ""
    if not summary:
        raise ValueError("Summary pass returned an empty summary.")
    print(f"summary output tokens: {count_text_tokens(summary):,}", flush=True)
    return summary


def summarize_history_if_needed(
    client: Any,
    model: str,
    reasoning_effort: str,
    session: dict[str, Any],
    count_text_tokens: TextTokenCounter,
    count_prompt_tokens: PromptTokenCounter,
    timestamp_factory: TimestampFactory,
) -> tuple[bool, int]:
    messages = session["messages"]
    start_index = unsummarized_dialogue_start_index(messages)
    candidates = messages[start_index:]
    user_tokens = count_user_dialogue_tokens(candidates, count_text_tokens)

    if user_tokens <= USER_DIALOGUE_SUMMARY_TOKEN_LIMIT:
        print(
            "Summary not needed "
            f"({user_tokens:,}/{USER_DIALOGUE_SUMMARY_TOKEN_LIMIT:,} "
            "unsummarized user tokens).",
            flush=True,
        )
        return False, user_tokens

    print(
        "Doing summary for unsummarized user dialogue "
        f"({user_tokens:,}/{USER_DIALOGUE_SUMMARY_TOKEN_LIMIT:,} tokens)...",
        flush=True,
    )
    summary = summarize_user_dialogue(
        client,
        model,
        reasoning_effort,
        candidates,
        count_text_tokens,
        count_prompt_tokens,
    )
    messages[start_index:] = [make_summary_message(summary, timestamp_factory())]
    print("Summarization completed.", flush=True)
    return True, user_tokens
