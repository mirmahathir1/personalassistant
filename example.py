"""Example usage of the local Codex proxy client helpers."""

from codex_client import build_client


def main() -> None:
    client = build_client()
    response = client.chat.completions.create(
        model='gpt-5.5',
        reasoning_effort='medium',
        messages=[{"role": "user", "content": "How are you?"}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
