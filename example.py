"""Example usage of the local Codex proxy client helpers."""

from codex_client import MODEL, build_client


def main() -> None:
    client = build_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "How are you?"}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
