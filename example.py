"""Example usage of the local Codex proxy client helpers."""

from codex_client import chat, stream


def main() -> None:
    response = chat("Write a concise one-sentence explanation of recursion.")
    print(response)

    print("\n--- streaming ---")
    stream("List three practical uses for Python generators.")


if __name__ == "__main__":
    main()
