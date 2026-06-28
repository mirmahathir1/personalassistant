"""Hybrid retrieval over the full conversation archive.

The conversation thread is never truncated — every turn stays in the archive
and on disk. To keep prompts inside the model's context window we retrieve only
the *relevant* older context for a given query. Quality comes from four layers
that compose (each degrades gracefully if its dependency is missing):

  A. Windowed chunks  — index sliding windows of several turns, not lone
     messages. A single turn ("yeah, do that") is too small to embed or to be
     useful when injected; a window carries enough context to mean something.
  Hybrid candidates   — BM25 (lexical) + embeddings (semantic via Ollama),
     fused with Reciprocal Rank Fusion, to gather a broad candidate set.
  C. LLM rerank       — an optional cross-encoder-style pass where the chat
     model scores each candidate's relevance to the (rewritten) query and we
     keep the best few. Fixes "topically near but not actually relevant".

Query rewrite (B) and extracted-fact memory (D) live in memory.py; the caller
combines them with this module.

Embeddings are cached on disk (chat_vectors.npy + a parallel key list) keyed by
window content, so only new/changed windows are embedded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None


# ---------------------------------------------------------------------------
# Config (env-overridable; defaults chosen for a single voice thread)
# ---------------------------------------------------------------------------

ENABLED = os.environ.get("RETRIEVAL_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
# Recent turns always sent verbatim by the caller; retrieval covers older context.
RECENT_N = int(os.environ.get("RETRIEVAL_RECENT_N", "8"))
# Final number of retrieved chunks to inject after reranking.
TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
# Candidate depth per method before fusion, and total fused candidates fed to rerank.
CANDIDATES = int(os.environ.get("RETRIEVAL_CANDIDATES", "30"))
RERANK_CANDIDATES = int(os.environ.get("RETRIEVAL_RERANK_CANDIDATES", "20"))
# Window size (turns per chunk) and stride (overlap = WINDOW - STRIDE).
WINDOW = int(os.environ.get("RETRIEVAL_WINDOW", "4"))
STRIDE = int(os.environ.get("RETRIEVAL_STRIDE", "2"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
# LLM rerank on by default; needs a chat client passed in. Falls back to fused order.
RERANK_ENABLED = os.environ.get("RETRIEVAL_RERANK", "1").strip().lower() not in {"0", "false", "no"}
# Minimum cosine similarity for an embedding candidate to be eligible. KNN always
# returns `CANDIDATES` neighbours even when nothing is actually relevant, so an
# unrelated chunk gets injected into *every* prompt — the single biggest source of
# hallucination/contradiction in a RAG-augmented chat. Filtering here, before the
# chunk can reach the rerank or the prompt, is open-webui's RELEVANCE_THRESHOLD
# (their #1 documented pitfall). 0 disables; ~0.5 is a sane start, tune per model.
RELEVANCE_THRESHOLD = float(os.environ.get("RETRIEVAL_RELEVANCE_THRESHOLD", "0.5"))
# When the LLM reranker omits everything (judges all candidates irrelevant), inject
# NOTHING rather than backfilling the top fused candidates. Trusts the reranker's
# "omit irrelevant" instruction instead of overriding it. Off => legacy backfill.
RERANK_STRICT = os.environ.get("RETRIEVAL_RERANK_STRICT", "1").strip().lower() not in {"0", "false", "no"}
_RRF_K = 60


def _parse_index_list(text: str) -> "list[int] | None":
    """Pull a list of integer indices out of a reranker reply, tolerantly.

    Small local models rarely emit one clean JSON array: they wrap it in ```fences,
    prepend reasoning, or print the picks on multiple lines ("[1]\\n[2,0]"). The old
    find("[")..rfind("]") slice spanned several arrays at once and json.loads then
    raised "Extra data: ...". Here we (1) try to json.loads the first *balanced*
    bracket span, and (2) fall back to scraping every integer in document order.
    Returns the indices, or None if the reply contained no integers at all (which
    the caller treats as a deliberate "nothing relevant").
    """
    # 1. First balanced [...] span — stops at the matching close bracket, so a
    #    second array on the next line is no longer swept into the parse.
    depth = 0
    span_start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                span_start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    try:
                        val = json.loads(text[span_start:i + 1])
                    except json.JSONDecodeError:
                        break  # malformed first span; fall through to integer scrape
                    if isinstance(val, list):
                        ints = [v for v in val if isinstance(v, int)]
                        if ints or val == []:
                            return ints
                    break
    # 2. Last resort: every integer mentioned, in order. Handles "1, 2, 0" with no
    #    brackets at all, or reasoning prose that names the indices.
    ints = [int(m) for m in re.findall(r"-?\d+", text)]
    return ints if ints else None


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _turn_text(msg: dict) -> str:
    return f"{msg.get('role', '')}: {msg.get('content', '')}".strip()


def _chunk_text(turns: list[dict]) -> str:
    """Render a window of turns into one searchable/displayable block."""
    return "\n".join(_turn_text(t) for t in turns)


def _chunk_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ConversationIndex:
    """Incremental hybrid index over windowed chunks of the conversation.

    Pass an OpenAI-compatible embed client (Ollama) for the semantic half and an
    OpenAI-compatible chat client + model for LLM reranking; either may be None.
    """

    def __init__(self, vectors_path: Path, embed_client=None,
                 rerank_client=None, rerank_model: str = ""):
        self.vectors_path = Path(vectors_path)
        self.keys_path = self.vectors_path.with_suffix(".keys.json")
        self.embed_client = embed_client
        self.rerank_client = rerank_client
        self.rerank_model = rerank_model
        self._vec_cache: dict[str, "np.ndarray"] = {}
        # Per-build state.
        self._chunks: list[list[dict]] = []   # each chunk is a list of turns
        self._chunk_end: list[int] = []        # index (in body order) of last turn per chunk
        self._chunk_texts: list[str] = []
        self._bm25 = None
        self._matrix = None
        self._embed_ok = embed_client is not None and np is not None
        # Window size / stride for chunking; module defaults, overridable per
        # instance (BioIndex sets both to 1 to index each paragraph alone).
        self.window = WINDOW
        self.stride = STRIDE
        self._load_vector_cache()

    # -- persistence --------------------------------------------------------

    def _load_vector_cache(self) -> None:
        if np is None or not self.vectors_path.exists() or not self.keys_path.exists():
            return
        try:
            mat = np.load(self.vectors_path)
            keys = json.loads(self.keys_path.read_text())
            if len(keys) == mat.shape[0]:
                self._vec_cache = {k: mat[i] for i, k in enumerate(keys)}
        except Exception as exc:
            print(f"[retrieval] could not load vector cache: {exc}; rebuilding")

    def _save_vector_cache(self) -> None:
        if np is None:
            return
        try:
            if not self._vec_cache:
                # Emptied (everything deleted): remove the cache files entirely.
                self.vectors_path.unlink(missing_ok=True)
                self.keys_path.unlink(missing_ok=True)
                return
            keys = list(self._vec_cache.keys())
            mat = np.vstack([self._vec_cache[k] for k in keys])
            np.save(self.vectors_path, mat)
            self.keys_path.write_text(json.dumps(keys))
        except Exception as exc:
            print(f"[retrieval] could not save vector cache: {exc}")

    # -- embedding ----------------------------------------------------------

    def _embed(self, texts: list[str]) -> "list[np.ndarray] | None":
        if not self._embed_ok or not texts:
            return None
        try:
            resp = self.embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
            out = []
            for item in resp.data:
                v = np.asarray(item.embedding, dtype=np.float32)
                n = np.linalg.norm(v)
                out.append(v / n if n else v)
            return out
        except Exception as exc:
            print(f"[retrieval] embeddings unavailable ({exc}); falling back to BM25-only")
            self._embed_ok = False
            return None

    # -- build --------------------------------------------------------------

    def rebuild(self, conversation: list[dict]) -> None:
        body = [m for m in conversation
                if m.get("role") != "system" and (m.get("content") or "").strip()]

        # A. Build overlapping windows of WINDOW turns, stepping by STRIDE.
        self._chunks = []
        self._chunk_end = []
        if body:
            win = max(1, self.window)
            step = max(1, self.stride)
            i = 0
            while i < len(body):
                window = body[i:i + win]
                if window:
                    self._chunks.append(window)
                    self._chunk_end.append(min(i + len(window) - 1, len(body) - 1))
                if i + win >= len(body):
                    break
                i += step
        self._chunk_texts = [_chunk_text(c) for c in self._chunks]

        # BM25 over chunks.
        if BM25Okapi is not None and self._chunk_texts:
            self._bm25 = BM25Okapi([_tokenize(t) for t in self._chunk_texts])
        else:
            self._bm25 = None

        # Embeddings over chunks (cache by chunk content).
        if self._embed_ok and self._chunk_texts:
            keys = [_chunk_key(t) for t in self._chunk_texts]
            missing = [(k, t) for k, t in zip(keys, self._chunk_texts) if k not in self._vec_cache]
            new_vecs = bool(missing)
            if missing:
                vecs = self._embed([t for _, t in missing])
                if vecs is not None:
                    for (k, _), v in zip(missing, vecs):
                        self._vec_cache[k] = v
            # Prune vectors for chunks that no longer exist (e.g. after a delete),
            # so removed content is truly erased and the cache can't grow forever.
            live = set(keys)
            pruned = [k for k in self._vec_cache if k not in live]
            for k in pruned:
                del self._vec_cache[k]
            if new_vecs or pruned:
                self._save_vector_cache()
            if self._embed_ok and all(k in self._vec_cache for k in keys):
                self._matrix = np.vstack([self._vec_cache[k] for k in keys])
            else:
                self._matrix = None
        else:
            # No chunks left: clear any cached vectors too.
            if self._vec_cache:
                self._vec_cache = {}
                self._save_vector_cache()
            self._matrix = None

    # -- query --------------------------------------------------------------

    def retrieve(self, query: str, recent_n: int = RECENT_N, top_k: int = TOP_K) -> list[dict]:
        """Return up to `top_k` relevant older windows (excludes the recent tail).

        Returns a list of chunks; each chunk is a list of turn dicts. The caller
        renders them via render_context_block().
        """
        if not self._chunks:
            print("[retrieval] retrieve: index empty; nothing to search")
            return []
        # A chunk is eligible only if it ends before the recent tail begins.
        n_turns = self._chunk_end[-1] + 1 if self._chunk_end else 0
        recent_start = max(0, n_turns - recent_n)
        eligible = [i for i, end in enumerate(self._chunk_end) if end < recent_start]
        if not eligible:
            print(f"[retrieval] retrieve: all {len(self._chunks)} chunks are in the recent tail; nothing older to retrieve")
            return []
        elig_set = set(eligible)

        bm25_ranks = [i for i in self._bm25_candidates(query) if i in elig_set]
        emb_ranks = [i for i in self._embedding_candidates(query) if i in elig_set]
        print(
            f"[retrieval] retrieve: {len(self._chunks)} chunks "
            f"({len(eligible)} eligible) -> bm25 {len(bm25_ranks)}, embed {len(emb_ranks)} candidates"
        )

        scores: dict[int, float] = {}
        for ranks in (bm25_ranks, emb_ranks):
            for rank, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)
        if not scores:
            print("[retrieval] retrieve: no candidates passed filtering; injecting none")
            return []

        fused = sorted(scores, key=lambda i: scores[i], reverse=True)[:RERANK_CANDIDATES]
        # C. LLM rerank the fused candidates down to top_k (falls back to fused order).
        chosen = self._rerank(query, fused, top_k)
        chosen.sort()  # chronological for the prompt
        print(f"[retrieval] retrieve: returning {len(chosen)} chunk(s) for injection")
        return [self._chunks[i] for i in chosen]

    def _bm25_candidates(self, query: str) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        idxs = [i for i in range(len(scores)) if scores[i] > 0]
        idxs.sort(key=lambda i: scores[i], reverse=True)
        return idxs[:CANDIDATES]

    def _embedding_candidates(self, query: str) -> list[int]:
        if self._matrix is None or not self._embed_ok:
            return []
        qv = self._embed([query])
        if not qv:
            return []
        sims = self._matrix @ qv[0]  # cosine: both sides are L2-normalised
        order = [int(i) for i in np.argsort(-sims)[:CANDIDATES]]
        # Drop neighbours below the relevance floor so an unrelated chunk can't be
        # injected just because KNN always returns *something* (see RELEVANCE_THRESHOLD).
        if RELEVANCE_THRESHOLD > 0:
            kept = [i for i in order if sims[i] >= RELEVANCE_THRESHOLD]
            top = float(sims[order[0]]) if order else 0.0
            print(
                f"[retrieval] embed: top sim {top:.3f}, "
                f"{len(kept)}/{len(order)} >= threshold {RELEVANCE_THRESHOLD}"
            )
            order = kept
        return order

    def _rerank(self, query: str, candidates: list[int], top_k: int) -> list[int]:
        """Score each candidate chunk's relevance with the chat model; keep top_k.

        Returns the candidate indices ranked best-first. Falls back to the fused
        order (already in `candidates`) on any failure or when disabled.
        """
        if (not RERANK_ENABLED or self.rerank_client is None
                or not self.rerank_model or len(candidates) <= top_k):
            return candidates[:top_k]
        numbered = "\n\n".join(
            f"[{n}]\n{self._chunk_texts[i]}" for n, i in enumerate(candidates)
        )
        prompt = (
            "You are ranking excerpts from an earlier conversation by how useful "
            "they are for answering the user's current message. Return ONLY a JSON "
            f"array of the excerpt numbers, most useful first, at most {top_k} of them. "
            "Omit excerpts that are irrelevant.\n\n"
            f"User's current message:\n{query}\n\nExcerpts:\n{numbered}"
        )
        try:
            resp = self.rerank_client.chat.completions.create(
                model=self.rerank_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,  # output is a short JSON array of indices; cap it
            )
            text = resp.choices[0].message.content or ""
            order = _parse_index_list(text)
            if order is None:
                # No integers anywhere in the reply — couldn't extract a ranking.
                print(f"[retrieval] rerank: unparseable reply {text[:120]!r}; using fused order")
                return candidates[:top_k]
            picked = [candidates[n] for n in order
                      if isinstance(n, int) and 0 <= n < len(candidates)]
            if picked:
                print(f"[retrieval] rerank: {len(candidates)} candidates -> kept {len(picked[:top_k])}")
                return picked[:top_k]
            # Reranker ran and returned a well-formed but empty selection: it judged
            # every candidate irrelevant. In strict mode honour that (inject nothing)
            # instead of overriding it with the top fused candidates — backfilling
            # here is what reintroduces the off-topic context the rerank just rejected.
            if RERANK_STRICT:
                print(f"[retrieval] rerank: all {len(candidates)} candidates judged irrelevant; injecting none")
                return []
            print(f"[retrieval] rerank: empty selection; backfilling top {top_k} fused")
        except Exception as exc:
            print(f"[retrieval] rerank failed ({exc}); using fused order")
        # Parse failure or rerank disabled: fall back to the fused order.
        return candidates[:top_k]


class BioIndex:
    """Hybrid retrieval over a character's static bio (long-form lore).

    The bio is free-form prose, so we split it into paragraph "turns" and feed
    them to a ConversationIndex. Unlike the conversation, the bio never grows and
    has no "recent tail" to exclude — `retrieve` searches the whole thing.

    Built once per character (the bio is immutable after creation). Reuses all of
    ConversationIndex's BM25 + embedding + rerank machinery and its on-disk vector
    cache, so bio chunks are embedded once and persist across restarts.
    """

    def __init__(self, vectors_path: Path, bio: str, embed_client=None,
                 rerank_client=None, rerank_model: str = ""):
        self._index = ConversationIndex(
            vectors_path, embed_client=embed_client,
            rerank_client=rerank_client, rerank_model=rerank_model,
        )
        # One chunk per non-blank paragraph, indexed individually. We override
        # WINDOW/STRIDE to 1 on this index instance so each paragraph is its own
        # retrievable unit: prose bios are usually a handful of paragraphs, and a
        # multi-paragraph window would collapse short bios into a single chunk —
        # which makes BM25's IDF degenerate (every term in 100% of docs scores 0).
        # role="bio" keeps these out of any conversation-shaped logic.
        self._index.window = 1
        self._index.stride = 1
        paras = [p.strip() for p in re.split(r"\n\s*\n", bio or "") if p.strip()]
        self._turns = [{"role": "bio", "content": p} for p in paras]
        self._index.rebuild(self._turns)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Up to `top_k` relevant bio chunks (no recent-tail exclusion)."""
        if not self._turns:
            return []
        # recent_n=0 makes the whole bio eligible (recent_start == n_turns).
        return self._index.retrieve(query, recent_n=0, top_k=top_k)


def render_bio_block(chunks: list[list[dict]]) -> str:
    """Format retrieved bio chunks as a system note about who the character is."""
    # Bio turns carry a "bio:" role prefix internally; drop it when rendering so
    # the prose reads naturally in the prompt.
    parts = ["\n".join(t.get("content", "") for t in chunk) for chunk in chunks]
    return (
        "Relevant background about you (your own history/personality — speak and "
        "act consistently with it):\n\n" + "\n---\n".join(parts)
    )


def render_context_block(chunks: list[list[dict]]) -> str:
    """Format retrieved chunks as a system-note string for prompt injection."""
    parts = []
    for chunk in chunks:
        parts.append(_chunk_text(chunk))
    return (
        "Relevant earlier context from this conversation (may help answer; "
        "ignore if irrelevant):\n\n" + "\n---\n".join(parts)
    )
