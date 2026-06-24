"""LLM-driven memory helpers: query rewrite (B) and a fact store (D).

These complement the chunk retrieval in retrieval.py:

  B. rewrite_query  — turn a conversational message ("what did I say about it?")
     into a standalone, reference-resolved search query before retrieval, so the
     embedder/BM25 have real content to match. Cheap, big recall win.

  D. FactStore      — a small, durable store of facts about the user/conversation
     ("user's dog is named Rex", "prefers offline TTS"), extracted by the chat
     model after each exchange, deduped, and persisted to facts.json. This is the
     part that makes the assistant actually *remember* across a long thread: the
     store is tiny and high-signal, so its facts can be injected on every request
     regardless of how the raw transcript ages out of retrieval.

Both helpers take an OpenAI-compatible chat client + model and fail soft: any
error leaves behavior unchanged (rewrite returns the original query; extraction
is skipped). Nothing here is required for the app to run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FACTS_ENABLED = os.environ.get("MEMORY_FACTS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
REWRITE_ENABLED = os.environ.get("MEMORY_REWRITE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
# Cap the store so the always-injected block stays small.
MAX_FACTS = int(os.environ.get("MEMORY_MAX_FACTS", "60"))


def rewrite_query(client, model: str, recent_turns: list[dict], message: str) -> str:
    """Rewrite `message` into a standalone retrieval query using recent context.

    Returns the original message on any failure or when disabled.
    """
    if not REWRITE_ENABLED or client is None or not model:
        return message
    context = "\n".join(
        f"{m.get('role')}: {m.get('content')}" for m in recent_turns[-6:]
    )
    prompt = (
        "Rewrite the user's latest message into a single self-contained search "
        "query for retrieving relevant earlier conversation. Resolve pronouns and "
        "references using the recent turns. Output ONLY the rewritten query, no "
        "quotes or preamble.\n\n"
        f"Recent turns:\n{context}\n\nLatest message: {message}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip()
        # Keep both: the rewrite may drop keywords the original had.
        return f"{message}\n{out}" if out and out.lower() != message.lower() else message
    except Exception as exc:
        print(f"[memory] query rewrite failed ({exc}); using original")
        return message


class FactStore:
    """Durable, deduped facts about the user/conversation, persisted to JSON."""

    def __init__(self, path: Path, client=None, model: str = ""):
        self.path = Path(path)
        self.client = client
        self.model = model
        self.facts: list[str] = self._load()

    def _load(self) -> list[str]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, list):
                    return [str(f) for f in data]
            except Exception as exc:
                print(f"[memory] could not read {self.path}: {exc}")
        return []

    def _save(self) -> None:
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.facts, ensure_ascii=False, indent=2))
            tmp.replace(self.path)
        except Exception as exc:
            print(f"[memory] could not save facts: {exc}")

    def clear(self) -> None:
        self.facts = []
        self._save()

    def update(self, user_msg: str, assistant_msg: str) -> None:
        """Extract durable facts from the latest exchange and merge them in.

        The model is asked for facts worth remembering long-term and for any
        existing facts this exchange makes obsolete, so the store stays current
        (e.g. "moved to Berlin" supersedes "lives in Paris"). Fails soft.
        """
        if not FACTS_ENABLED or self.client is None or not self.model:
            return
        existing = "\n".join(f"- {f}" for f in self.facts) or "(none yet)"
        prompt = (
            "You maintain a long-term memory of durable facts about the user and "
            "their stable preferences, for a voice assistant. Given the latest "
            "exchange, decide what to remember.\n\n"
            "Return ONLY a JSON object with two arrays:\n"
            '  "add": new durable facts (short, third-person, e.g. "User\'s dog is named Rex"). '
            "Only stable facts/preferences — NOT one-off questions, chit-chat, or transient state.\n"
            '  "remove": existing facts (verbatim from the list) now obsolete or contradicted.\n'
            "Use [] for either if nothing applies.\n\n"
            f"Existing facts:\n{existing}\n\n"
            f"Latest exchange:\nUser: {user_msg}\nAssistant: {assistant_msg}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content or ""
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return
            obj = json.loads(text[start:end + 1])
        except Exception as exc:
            print(f"[memory] fact extraction failed ({exc}); skipping")
            return

        remove = {str(f).strip() for f in obj.get("remove", []) if str(f).strip()}
        if remove:
            self.facts = [f for f in self.facts if f not in remove]
        seen = {f.lower() for f in self.facts}
        for f in obj.get("add", []):
            f = str(f).strip()
            if f and f.lower() not in seen:
                self.facts.append(f)
                seen.add(f.lower())
        # Keep the store bounded; drop oldest first.
        if len(self.facts) > MAX_FACTS:
            self.facts = self.facts[-MAX_FACTS:]
        self._save()

    def prune_facts(self, deleted_texts: list[str]) -> list[str]:
        """Remove facts that were derived from now-deleted messages.

        Facts are decoupled from raw turns (a fact's text doesn't contain the
        message it came from), so we ask the model which stored facts the deleted
        content supports, and drop those. Returns the removed facts. Fails soft:
        on any error the store is left unchanged.
        """
        if not FACTS_ENABLED or self.client is None or not self.model:
            return []
        if not self.facts or not deleted_texts:
            return []
        existing = "\n".join(f"- {f}" for f in self.facts)
        deleted = "\n".join(f"- {t}" for t in deleted_texts if t and t.strip())
        if not deleted.strip():
            return []
        prompt = (
            "Some messages were deleted from a conversation and must be forgotten. "
            "From the stored long-term facts below, identify ONLY those that were "
            "derived from (supported by) the deleted messages and should now be "
            "forgotten. Do NOT remove facts that are still supported by other, "
            "non-deleted information.\n\n"
            "Return ONLY a JSON array of the fact strings to remove (verbatim from "
            "the list), or [] if none.\n\n"
            f"Stored facts:\n{existing}\n\nDeleted messages:\n{deleted}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content or ""
            start, end = text.find("["), text.rfind("]")
            remove = json.loads(text[start:end + 1]) if start != -1 and end != -1 else []
        except Exception as exc:
            print(f"[memory] fact pruning failed ({exc}); leaving facts unchanged")
            return []
        remove_set = {str(f).strip() for f in remove if str(f).strip()}
        removed = [f for f in self.facts if f in remove_set]
        if removed:
            self.facts = [f for f in self.facts if f not in remove_set]
            self._save()
        return removed

    def render_block(self) -> str:
        """Format the fact store for always-on prompt injection (empty if none)."""
        if not self.facts:
            return ""
        lines = "\n".join(f"- {f}" for f in self.facts)
        return "What you know about the user (long-term memory):\n" + lines
