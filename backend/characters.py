"""Character store — per-character chat threads, persisted as namespaced JSON.

Each character is one chat thread with its own history, retrieval index, and
long-term facts, kept under ``data/<char_id>/``. The character index (name,
gender, voice, intelligence) lives in ``data/characters.json``.

A character's *intelligence* maps onto one of the three existing Ollama chat
models (no new downloads): low->1.7B, medium->3B, high->8B. Its *gender* gates
which Kokoro voices are offered (af_/bf_ = female, am_/bm_ = male).

Settings (name, gender, voice, intelligence) are chosen at creation time and are
immutable afterwards — there is no update method here on purpose.
"""

import json
import os
import re
import shutil
import uuid
from pathlib import Path

# Data root: backend/data/ (override with CHARACTERS_DATA_DIR). The per-character
# subdirs and the index file live here.
DATA_DIR = Path(
    os.environ.get("CHARACTERS_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
)
INDEX_FILE = DATA_DIR / "characters.json"

GENDERS = ("female", "male")
INTELLIGENCE_LEVELS = ("low", "medium", "high")

# Intelligence level -> existing chat provider id (see CHAT_MODELS in main.py).
INTELLIGENCE_PROVIDER = {
    "low": "ollama-1b",   # Qwen3 1.7B abliterated, fastest + most uncensored
    "medium": "ollama-3b",  # Llama 3B
    "high": "ollama",     # Lexi 8B, smartest
}


def voice_gender(voice_id: str) -> str | None:
    """Infer a Kokoro voice's gender from its id prefix, or None if unknown.

    Kokoro ids are namespaced "kokoro:<key>"; the key's second letter encodes
    gender: af_/bf_ = female, am_/bm_ = male.
    """
    _, _, key = voice_id.partition(":")
    key = key or voice_id
    if len(key) >= 2 and key[1] == "f":
        return "female"
    if len(key) >= 2 and key[1] == "m":
        return "male"
    return None


def _char_dir(char_id: str) -> Path:
    return DATA_DIR / char_id


def char_paths(char_id: str) -> dict:
    """The namespaced file paths backing one character's thread."""
    d = _char_dir(char_id)
    return {
        "dir": d,
        "history": d / "chat_history.json",
        "facts": d / "facts.json",
        "vectors": d / "chat_vectors.npy",  # .keys.json is derived by ConversationIndex
        "bio_vectors": d / "bio_vectors.npy",  # separate cache for the bio index
    }


class CharacterStore:
    """Loads/saves the character index and creates/deletes character data dirs."""

    def __init__(self, index_file: Path = INDEX_FILE):
        self.index_file = Path(index_file)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.characters: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.index_file.exists():
            try:
                data = json.loads(self.index_file.read_text())
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[characters] could not read {self.index_file}: {exc}; starting empty")
        return []

    def _save(self) -> None:
        tmp = self.index_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.characters, ensure_ascii=False, indent=2))
        tmp.replace(self.index_file)

    def exists(self) -> bool:
        """True once the index file has been written (used to gate migration)."""
        return self.index_file.exists()

    def get(self, char_id: str) -> dict | None:
        return next((c for c in self.characters if c.get("id") == char_id), None)

    def list(self) -> list[dict]:
        # Newest first.
        return sorted(self.characters, key=lambda c: c.get("created_at", ""), reverse=True)

    def create(
        self,
        name: str,
        gender: str,
        voice: str,
        intelligence: str,
        *,
        created_at: str,
        persona_core: str = "",
        bio: str = "",
        char_id: str | None = None,
    ) -> dict:
        """Validate + persist a new character and create its data dir.

        Raises ValueError on bad input. `created_at` is passed in (the caller
        stamps the time) since this module does no clock access.

        `persona_core` is a short identity blurb injected into the system prompt
        on every turn; `bio` is free-form long-form lore that the caller indexes
        for retrieval (surfaced only when relevant). Both are optional free text.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("Character name is required")
        if gender not in GENDERS:
            raise ValueError(f"gender must be one of {GENDERS}")
        if intelligence not in INTELLIGENCE_LEVELS:
            raise ValueError(f"intelligence must be one of {INTELLIGENCE_LEVELS}")
        vg = voice_gender(voice)
        if vg is not None and vg != gender:
            raise ValueError(f"voice {voice!r} does not match gender {gender!r}")

        cid = char_id or uuid.uuid4().hex[:12]
        char = {
            "id": cid,
            "name": name,
            "gender": gender,
            "voice": voice,
            "intelligence": intelligence,
            "persona_core": (persona_core or "").strip(),
            "bio": (bio or "").strip(),
            "created_at": created_at,
        }
        _char_dir(cid).mkdir(parents=True, exist_ok=True)
        self.characters.append(char)
        self._save()
        return char

    def delete(self, char_id: str) -> bool:
        """Remove a character from the index and delete its data dir."""
        char = self.get(char_id)
        if char is None:
            return False
        self.characters = [c for c in self.characters if c.get("id") != char_id]
        self._save()
        shutil.rmtree(_char_dir(char_id), ignore_errors=True)
        return True

    def provider_for(self, char_id: str) -> str | None:
        """The chat provider id implied by a character's intelligence level."""
        char = self.get(char_id)
        if char is None:
            return None
        return INTELLIGENCE_PROVIDER.get(char.get("intelligence", ""), None)
