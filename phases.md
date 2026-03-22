# Implementation Phases

These phases turn the retrieval-memory plan into an implementation sequence that fits the current app. The main shift is architectural: SQLite becomes the canonical store for conversation history, Qdrant becomes the semantic lookup layer for user-authored memory sentences, and prompt context is assembled per request instead of being treated as a long-lived rolling `llama.cpp` transcript.

## Phase 1: Runtime, Dependencies, and Docker Foundation

The first phase lays down the runtime foundation for sentence-level memory retrieval. The app should use `sqlite3` as the source of truth for threads, messages, and sentence records, `qdrant-client[fastembed]` in the backend, and a Dockerized Qdrant service for semantic lookup. Retrieval stays sentence-level, but the injected memory block should expand any matched sentence back to the full original user message before reply generation. The long-term goal is to build prompt context from stored thread history plus retrieved memory rather than relying on the current rolling `llama.cpp` state as the long-term store.

The Docker layout should be fixed and repository-owned. The backend mounts `./data:/data`, stores SQLite at `/data/app.db`, uses `/data/fastembed` for the FastEmbed cache, and talks to Qdrant over the Compose network at `http://qdrant:6333`. Qdrant runs as a separate Compose service named `qdrant`, uses the official `qdrant/qdrant` image, publishes `6333:6333`, and stores its data at `/qdrant/storage` backed by `./data/qdrant:/qdrant/storage`.

Backend startup should initialize the Qdrant collection if it does not already exist, and health checks should fail clearly when Qdrant is unreachable. SQLite and the FastEmbed cache should remain mounted separately from Qdrant storage. Memory, retrieval, and deployment settings should stay hardcoded in the repository instead of being exposed as knobs. In practice that means `docker-compose.yml` owns the `./data:/data` mount, the `./data/qdrant:/qdrant/storage` mount, the published `6333:6333` port, and the `qdrant/qdrant` image, while `backend/config.py` owns `SQLITE_PATH`, `QDRANT_URL`, `QDRANT_COLLECTION`, `FASTEMBED_CACHE_PATH`, retrieval mode, result sizes, relevance filtering, and semantic dedupe behavior. Dependencies for this phase are `qdrant-client[fastembed]`, Python `sqlite3` without an ORM, one lightweight sentence splitter behind an adapter module, and the official Docker Compose Qdrant service. The implementation surface is `backend/config.py`, `docker-compose.yml`, and `.gitignore`.

## Phase 2: Canonical SQLite Conversation Model

The second phase defines the canonical conversation model in SQLite. Every conversation should have two paired threads: a finalized thread for the user-visible chat and a trace thread for foreground and background execution artifacts. This gives the app a stable chat history for normal use while preserving a separate diagnostic trail for retrieval, draft generation, analysis passes, and other internal work.

The `threads` table should store `id`, `conversation_id`, `thread_type`, optional `title`, `created_at`, and `updated_at`. The `messages` table should store `id`, `thread_id`, `role`, `kind`, `content`, `turn_index`, and `created_at`, with a foreign key back to `thread_id`. The `message_sentences` table should store `id`, `message_id`, `thread_id`, `role`, `sentence_index`, `text`, `created_at`, and nullable `qdrant_point_id`, with foreign keys to both `message_id` and `thread_id`.

The finalized thread should contain only user messages and final assistant responses. The paired trace thread should contain all foreground and background request and response artifacts, including user messages, retrieved memory, draft pass I/O, long-term-memory analysis I/O, and final synthesis I/O. User sentence rows may carry a `qdrant_point_id`, but assistant sentence rows stay in SQLite only and are never indexed into Qdrant. Trace-thread messages are also never indexed into Qdrant.

Persistence should follow the same model consistently. Each conversation gets one finalized thread and one trace thread. Finalized user messages and finalized assistant responses are written to the finalized thread, while all foreground and background artifacts are written to the trace thread. Finalized user and assistant messages also receive sentence rows in SQLite. If generation fails, the failed user message should not be persisted, and failed-attempt artifacts should not be written to either thread. Optional later additions include a `memory_events` table for retrieval debugging and `thread_summaries` for long-history compression. The implementation surface is `backend/db.py` and `backend/repositories.py`.

## Phase 3: English-Only Sentence Splitting

The third phase introduces a dedicated sentence splitter so finalized messages can be stored and indexed consistently. Add `backend/sentence_splitter.py` and keep the implementation English-only for now. The choice of splitter should stay behind an adapter so it can be swapped later, and it should stay lightweight rather than pulling in a large NLP stack.

The splitter should normalize whitespace before splitting, ignore empty fragments, and preserve sentence order through `sentence_index`. Finalized user messages and finalized assistant messages should both be split into ordered SQLite sentence rows. Finalized assistant sentences should not create Qdrant points, and trace-thread artifacts should never be turned into Qdrant points either. The implementation surface is `backend/sentence_splitter.py`.

## Phase 4: Incremental Qdrant Sentence Index

The fourth phase adds the incremental Qdrant sentence index. The system should maintain one Dockerized Qdrant collection, for example `conversation_sentences`, that stores semantic lookup points only for finalized user-authored sentences. Each point should include the embedding vector and a small payload with `sentence_id`, `thread_id`, `message_id`, `role` set to `user`, `sentence_index`, and `created_at`.

SQLite remains the canonical home of sentence text. Qdrant only needs enough payload to identify the matching sentence quickly. Assistant responses are not stored in Qdrant, and trace-thread messages are not stored there either. Indexing should happen incrementally during normal writes rather than through a rebuild path that the app depends on in steady-state operation.

After a successful final assistant response, the backend should persist the finalized user message, split it into sentences, persist those sentence rows, generate embeddings for each user sentence, and upsert those sentence points into Qdrant. It should then persist the finalized assistant reply, split it into sentences, and persist assistant sentence rows in SQLite without generating embeddings or Qdrant points for them. Trace-thread entries for the user message, retrieved memory, draft pass, long-term-memory analysis pass, and final synthesis pass should still be persisted, but that trace persistence should stay separate from Qdrant indexing. Add a store wrapper at `backend/memory_store.py` to initialize the collection on startup, provide batch upsert and filtered search methods, and fail clearly when Qdrant is unreachable. The implementation surface is `backend/memory_store.py`.

## Phase 5: Retrieval, Ranking, Deduplication, and Memory Block Construction

The fifth phase implements retrieval itself. The system should search prior persisted user-authored messages across stored chat history, exclude the current unsaved message, and treat `thread_id` as metadata rather than as a required retrieval filter. If thread-aware balancing is ever needed later, it should be introduced as a ranking or selection rule instead of as a separate retrieval mode.

On each incoming user message, the backend should accept the finalized `thread_id`, or create a new conversation with a finalized thread and a paired trace thread if this is the first message. It should split the incoming user message into sentences, run a Qdrant search for each sentence against prior user-authored sentence points, merge the hits by `message_id` using the best sentence score per message, deduplicate strictly by `message_id`, sort the candidates by their best similarity score, and then load the full original user message content from SQLite. Retrieval only ever uses user-authored memory rows from Qdrant.

Ranking should start simple. The primary sort is Qdrant similarity score. Sentence hits are aggregated to the message level by `message_id`, very low-score matches are discarded, and message age should not lower rank on its own. Old memories remain eligible if they are still semantically relevant.

Deduplication should happen after retrieval and after sentence hits have been expanded back to full user messages. Exact dedupe is by `message_id`. Semantic dedupe happens at the full-message level: keep the highest-ranked message first, then compare each later candidate with already selected messages and skip it if similarity crosses the internal dedupe threshold. Continue until the block reaches its max size or the candidate list is exhausted.

The final memory block should always be built from full user messages loaded from SQLite, not from isolated sentence fragments. Retrieval remains sentence-level, but injected memory units are full user messages. The block should stay small: retrieve up to 5 sentence hits per user sentence, include at most 4 full user messages, enforce a hard token budget, prefer concise factual messages over long narrative ones, avoid duplicate or near-duplicate memories, exclude very weak matches, and treat the resulting memory block as supporting context rather than guaranteed truth.

An example of the intended memory block shape is:

```text
Relevant memory:
- User message: I am allergic to peanuts, and I avoid snacks with peanut oil too.
- User message: I prefer short answers when we discuss implementation plans.
```

The implementation surface is `backend/memory_service.py`.

## Phase 6: Multi-Pass Generation, Prompt Assembly, and State Isolation

The sixth phase replaces long-lived rolling-state conversation storage with request-time prompt assembly and a required multi-pass generation flow. This refactor is necessary because the current backend saves a rolling `llama.cpp` state snapshot after every turn. If retrieved memory is injected into that rolling state, the retrieved text becomes part of the saved state, repeats across later turns, pollutes future context, and wastes context window space that should be reserved for actual conversation state.

To avoid that, SQLite should own thread history, Qdrant should own the user-memory sentence index, and each `/api/chat` request should assemble a fresh prompt from SQLite history plus retrieved memory. The current `llama.cpp` rolling state should no longer be treated as the long-term conversation store once retrieval exists. The target behavior is simple: thread history comes from SQLite, memory retrieval comes from Qdrant, and `llama.cpp` generates from prompt inputs assembled for that specific request.

The implementation should keep the current tokenizer and manual chat template logic, stop relying on `SingleSessionManager` as the sole conversation history mechanism, and treat any in-memory caching as a later optimization rather than as the canonical source of truth. All intermediate analysis branches should be read-only and ephemeral.

Generation should follow a fixed three-pass flow. The draft pass generates the main answer draft from the current thread context without retrieved memory. The long-term-memory analysis pass uses the current thread context plus retrieved long-term memory and returns only relevant facts, constraints, preferences, and prior decisions. The final synthesis pass then produces the final user-visible answer from the current thread context plus the extracted long-term-memory analysis and the draft answer. The analysis pass may read from the current thread state, but it must not write back to the live `LlamaState`, and it must not produce a full standalone answer.

The prompt assembly should look like this:

```text
Draft pass:
System prompt
Recent thread messages
Current user message

Memory analysis pass:
System prompt
Relevant memory
Recent thread messages
Current user message

Final synthesis pass:
System prompt
Memory analysis output
Draft answer
Recent thread messages
Current user message
```

If `LlamaState` snapshots are still involved during the transition, the backend should capture the original current-thread state before the long-term-memory analysis pass, load that original state for analysis, append retrieval-analysis instructions and retrieved memory only for the temporary branch, discard the derived analysis state after generation, reload the original current-thread state before final synthesis, and never call `store_turn()` with the long-term-memory analysis branch.

Once request-time prompt assembly is in place, analysis prompts remain ephemeral. Their outputs and other intermediate artifacts are stored only in the paired trace thread, and only the final user-visible assistant response is committed to the finalized thread history that normal chat uses. After a successful final assistant response, finalized user and assistant messages go to the finalized thread and foreground and background pass artifacts go to the trace thread. If generation fails, the failed user message and failed-attempt artifacts are not persisted anywhere. The implementation surface is `backend/llm_service.py`.

## Phase 7: API and Three-Column Frontend

The seventh phase exposes the finalized and trace conversation model through both the API and the product UI. Chat request and response payloads should carry `thread_id`, and the backend should add endpoints for thread creation, thread listing, and `delete-all-memory`.

The frontend should load both the finalized thread and its paired trace thread for the active conversation. The UI should move to a three-column layout: the first column shows the finalized chat with user messages and final assistant responses, the second column shows the paired trace thread with all foreground and background request and response activity, and the third column shows previous conversations together with the global memory controls.

That third column should include a `Delete all memory` action. When it is triggered, the app should physically delete all stored SQLite and Qdrant data. Previous conversations remain browseable until that hard delete happens.

## Phase 8: Operational Health and Fixed System Behavior

The final phase locks in the operational safeguards and non-negotiable behavior from the plan. Health checks should report SQLite and Qdrant status separately, and startup should include readiness checks for the Dockerized Qdrant service.

At that point the system behavior should be considered fixed. Retrieval always works from prior persisted user-authored chat messages, asks Qdrant for up to 5 sentence hits per user sentence, and builds a final memory block with up to 4 full user messages. Old memories remain eligible regardless of age. The app uses one shared Qdrant collection for all user-authored memory sentences, indexes those sentences incrementally during normal writes, applies internal relevance filtering and semantic dedupe during selection, and always runs against one Dockerized Qdrant service shared by the backend.

Conversation structure is also fixed at this stage. Every conversation has one finalized thread and one paired trace thread, and every user message goes through the multi-pass generation flow. The frontend keeps the three-column layout: finalized chat in the first column, full trace thread in the second, and previous conversations plus the global delete control in the third. `Delete all memory` always hard-deletes both SQLite and Qdrant data. Failed user messages are never stored if generation fails, and sentence splitting plus embeddings remain English-only.
