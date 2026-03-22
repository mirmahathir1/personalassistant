# Implementation Phases

## Phase 1: Runtime, Dependencies, and Docker Foundation

- Objective: add sentence-level memory retrieval to the app using `sqlite3` as the canonical store, `qdrant-client[fastembed]` in the backend, a Dockerized Qdrant service, and a compact retrieved memory block injected before reply generation.
- Core architecture: SQLite is the source of truth for threads, messages, and sentence records. Qdrant is the semantic lookup layer. Retrieval stays sentence-level, but prompt injection expands matched sentences back to full original user messages.
- Core architecture: maintain one Qdrant point per user-authored memory sentence.
- Core architecture: build prompt context per request from stored thread history plus retrieved memory rather than relying on the current rolling `llama.cpp` state as the long-term store.
- Docker-only layout: mount `./data:/data`.
- Docker-only layout: SQLite database path is `/data/app.db`.
- Docker-only layout: Qdrant storage path inside the Qdrant container is `/qdrant/storage`.
- Docker-only layout: backend Qdrant URL on the Compose network is `http://qdrant:6333`.
- Docker-only layout: FastEmbed cache path is `/data/fastembed`.
- Compose shape: service name `qdrant`.
- Compose shape: image `qdrant/qdrant`.
- Compose shape: mount `./data/qdrant:/qdrant/storage`.
- Compose shape: publish `6333:6333`.
- Backend behavior: initialize the Qdrant collection on startup if it does not already exist.
- Backend behavior: fail health checks clearly if the Qdrant service is unreachable.
- Backend behavior: keep SQLite and FastEmbed cache mounted separately from Qdrant storage.
- Hardcoded repository defaults: keep memory, retrieval, and deployment settings fixed in the repository instead of exposing them as external knobs.
- Hardcoded in `docker-compose.yml`: backend data mount `./data:/data`.
- Hardcoded in `docker-compose.yml`: Qdrant storage mount `./data/qdrant:/qdrant/storage`.
- Hardcoded in `docker-compose.yml`: published Qdrant port `6333:6333`.
- Hardcoded in `docker-compose.yml`: Qdrant Docker image and tag `qdrant/qdrant`.
- Hardcoded in `backend/config.py`: `SQLITE_PATH`, `QDRANT_URL`, `QDRANT_COLLECTION`, `FASTEMBED_CACHE_PATH`, retrieval mode, result sizes, relevance filtering, and semantic dedupe behavior.
- Dependencies: `qdrant-client[fastembed]`.
- Dependencies: no ORM; use Python `sqlite3`.
- Dependencies: one lightweight sentence splitter behind an adapter module.
- Dependencies: Docker Compose service using the official `qdrant/qdrant` image.
- Implementation surface: `backend/config.py`, `docker-compose.yml`, `.gitignore`.

## Phase 2: Canonical SQLite Conversation Model

- Objective: model every conversation in SQLite with a finalized thread for user-visible chat and a paired trace thread for foreground and background execution artifacts.
- SQLite table `threads`: `id` TEXT PRIMARY KEY.
- SQLite table `threads`: `conversation_id` TEXT NOT NULL.
- SQLite table `threads`: `thread_type` TEXT NOT NULL.
- SQLite table `threads`: `title` TEXT NULL.
- SQLite table `threads`: `created_at` TEXT NOT NULL.
- SQLite table `threads`: `updated_at` TEXT NOT NULL.
- SQLite table `messages`: `id` TEXT PRIMARY KEY.
- SQLite table `messages`: `thread_id` TEXT NOT NULL.
- SQLite table `messages`: `role` TEXT NOT NULL.
- SQLite table `messages`: `kind` TEXT NOT NULL.
- SQLite table `messages`: `content` TEXT NOT NULL.
- SQLite table `messages`: `turn_index` INTEGER NOT NULL.
- SQLite table `messages`: `created_at` TEXT NOT NULL.
- SQLite table `messages`: foreign key on `thread_id`.
- SQLite table `message_sentences`: `id` TEXT PRIMARY KEY.
- SQLite table `message_sentences`: `message_id` TEXT NOT NULL.
- SQLite table `message_sentences`: `thread_id` TEXT NOT NULL.
- SQLite table `message_sentences`: `role` TEXT NOT NULL.
- SQLite table `message_sentences`: `sentence_index` INTEGER NOT NULL.
- SQLite table `message_sentences`: `text` TEXT NOT NULL.
- SQLite table `message_sentences`: `created_at` TEXT NOT NULL.
- SQLite table `message_sentences`: `qdrant_point_id` TEXT NULL.
- SQLite table `message_sentences`: foreign key on `message_id`.
- SQLite table `message_sentences`: foreign key on `thread_id`.
- Conversation rule: each conversation has one `finalized` thread and one `trace` thread.
- Conversation rule: the finalized thread stores user messages and final assistant responses only.
- Conversation rule: the trace thread stores all foreground and background request/response artifacts, including user messages, retrieved memory, draft pass I/O, long-term-memory analysis I/O, and final synthesis I/O.
- Sentence-row rule: user sentence rows may have a `qdrant_point_id`.
- Sentence-row rule: assistant sentence rows stay in SQLite only and are not indexed into Qdrant.
- Sentence-row rule: trace-thread messages are not indexed into Qdrant.
- Persistence policy: create one finalized thread and one paired trace thread per conversation.
- Persistence policy: persist user messages and final assistant responses into the finalized thread.
- Persistence policy: persist all foreground and background request/response artifacts into the paired trace thread.
- Persistence policy: add sentence rows for finalized user and finalized assistant messages in SQLite.
- Persistence policy: if generation fails, do not persist the failed user message in SQLite.
- Persistence policy: if generation fails, do not write failed-attempt artifacts to the finalized thread or the trace thread.
- Optional later addition: `memory_events` for debugging retrieval decisions.
- Optional later addition: `thread_summaries` for long-history compression.
- Implementation surface: `backend/db.py`, `backend/repositories.py`.

## Phase 3: English-Only Sentence Splitting

- Objective: split finalized user and finalized assistant messages into ordered sentence rows that can be stored and indexed consistently.
- Sentence splitter: add `backend/sentence_splitter.py`.
- Sentence splitter: support English-only sentence splitting.
- Sentence splitter: keep the choice isolated behind an adapter so it can be swapped later.
- Sentence splitter: prefer a lightweight splitter over a large NLP stack.
- Sentence splitter: normalize whitespace before splitting.
- Sentence splitter: ignore empty fragments.
- Sentence splitter: preserve sentence order with `sentence_index`.
- Sentence splitter: write sentence rows to SQLite for finalized user messages.
- Sentence splitter: write sentence rows to SQLite for finalized assistant messages.
- Sentence splitter: do not create Qdrant points for finalized assistant sentences.
- Sentence splitter: do not create Qdrant points for trace-thread artifacts.
- Implementation surface: `backend/sentence_splitter.py`.

## Phase 4: Incremental Qdrant Sentence Index

- Objective: maintain a Dockerized Qdrant collection that stores semantic lookup points for finalized user-authored sentences only.
- Collection shape: use one collection, for example `conversation_sentences`.
- Point contents: embedding vector.
- Point contents: payload `sentence_id`.
- Point contents: payload `thread_id`.
- Point contents: payload `message_id`.
- Point contents: payload `role` with value `user`.
- Point contents: payload `sentence_index`.
- Point contents: payload `created_at`.
- Storage rule: keep the canonical sentence text in SQLite. Qdrant only needs enough payload to identify the matching sentence quickly.
- Storage rule: assistant responses are not stored in Qdrant.
- Storage rule: trace-thread messages are not stored in Qdrant.
- Indexing rule: index only finalized user-authored sentence rows in Qdrant.
- Indexing rule: add user-authored memory sentences to Qdrant incrementally during normal writes.
- Indexing rule: do not rely on any full rebuild path during normal operation.
- Write flow: after a successful final assistant response, persist the finalized user message, split it into sentences, persist sentence rows, generate embeddings for each user sentence, and upsert those sentence points into Qdrant.
- Write flow: after that same successful final assistant response, persist the finalized assistant reply, split it into sentences, persist assistant sentence rows, and do not generate embeddings or upsert Qdrant points for assistant sentences.
- Trace policy: persist trace-thread entries for the user message, retrieved memory, draft pass, long-term-memory analysis pass, and final synthesis pass, but keep that trace persistence separate from Qdrant indexing.
- Store wrapper: add `backend/memory_store.py`.
- Store wrapper: initialize the sentence collection on startup.
- Store wrapper: add methods for batch upsert and filtered search.
- Store wrapper: fail clearly when Qdrant is unreachable.
- Implementation surface: `backend/memory_store.py`.

## Phase 5: Retrieval, Ranking, Deduplication, and Memory Block Construction

- Objective: retrieve semantically relevant prior user-authored memories and convert them into a compact full-message memory block.
- Retrieval scope: search prior persisted user-authored messages across stored chat history.
- Retrieval scope: exclude the current unsaved message.
- Retrieval scope: keep `thread_id` as metadata, not as a required retrieval filter.
- Retrieval scope: if thread-aware balancing is ever needed later, add it as a ranking or selection rule rather than as a separate retrieval mode.
- Read flow: accept the finalized `thread_id`, or create a new conversation with a finalized thread and a paired trace thread if this is the first message.
- Read flow: split the incoming user message into sentences.
- Read flow: for each sentence, run a Qdrant search against prior user-authored sentence points.
- Read flow: merge sentence hits by `message_id` using the best matching sentence score per message.
- Read flow: deduplicate strictly by `message_id` so the same user message block is never added multiple times.
- Read flow: sort message candidates by their best sentence similarity score.
- Read flow: load the full original user message content from SQLite for the message candidates.
- Read flow: retrieve only user-authored memory rows from Qdrant.
- Ranking rule: primary sort is vector similarity score from Qdrant.
- Ranking rule: aggregate sentence hits up to the message level by `message_id`.
- Ranking rule: discard very low-score matches using a minimum threshold.
- Ranking rule: do not reduce rank based on message age alone.
- Eligibility rule: old memories remain eligible if they are still semantically relevant.
- Deduplication rule: apply deduplication after retrieval and after sentence hits are expanded to full user messages.
- Deduplication rule: exact dedupe by `message_id`.
- Deduplication rule: semantic dedupe at the full-message level.
- Deduplication rule: keep the highest-ranked message first.
- Deduplication rule: for each next candidate, compare it to already selected memory messages.
- Deduplication rule: if similarity to any selected memory exceeds the internal dedupe threshold, skip it.
- Deduplication rule: continue until the memory block reaches its max size or the candidate list is exhausted.
- Threshold rule: retrieval threshold decides whether a memory is relevant to the current user query.
- Threshold rule: dedupe threshold decides whether a candidate memory is too similar to a memory already selected.
- Memory block rule: build the retrieved context block from full user messages loaded from SQLite, not isolated sentence fragments.
- Memory block rule: embedding lookup is sentence-level, but injected memory units are full user messages.
- Memory block rule: keep the block small; the general target is a small set of full messages and the fixed cap is 4 full user messages.
- Memory block rule: retrieve up to 5 sentence hits per user sentence.
- Memory block rule: never include the same user message more than once in a single memory block.
- Memory block rule: do not include a message if it is semantically too similar to a message already selected for the same block.
- Memory block rule: prefer concise factual messages over long narrative ones.
- Memory block rule: avoid duplicate or near-duplicate messages.
- Memory block rule: exclude very weak matches.
- Memory block rule: keep a hard token budget for the memory block.
- Memory block rule: treat the memory block as supporting context, not guaranteed truth.
- Memory block example:

```text
Relevant memory:
- User message: I am allergic to peanuts, and I avoid snacks with peanut oil too.
- User message: I prefer short answers when we discuss implementation plans.
```

- Implementation surface: `backend/memory_service.py`.

## Phase 6: Multi-Pass Generation, Prompt Assembly, and State Isolation

- Objective: replace long-lived rolling-state conversation storage with request-time prompt assembly and a mandatory multi-pass generation flow.
- Current-app constraint: the backend currently keeps a rolling `llama.cpp` state snapshot and saves that state after every turn.
- Current-app constraint: if a retrieved memory block is injected into that rolling state, the retrieved text becomes part of the saved state and accumulates across turns.
- Current-app constraint: that causes retrieved memory to repeat and pollute future context.
- Current-app constraint: that also causes the context window to fill with old retrieval blocks instead of real conversation state.
- Required refactor: SQLite stores thread history.
- Required refactor: Qdrant stores the user-memory sentence index.
- Required refactor: each `/api/chat` request assembles fresh prompt context from SQLite history plus retrieved memory.
- Required refactor: do not treat the current `llama.cpp` rolling state as the long-term conversation store once retrieval is added.
- Target behavior: thread history comes from SQLite.
- Target behavior: memory retrieval comes from Qdrant.
- Target behavior: `llama.cpp` generates from assembled prompt inputs for that request.
- Implementation path: keep the current tokenizer and manual chat template logic.
- Implementation path: stop relying on `SingleSessionManager` as the sole conversation history mechanism.
- Implementation path: optionally keep in-memory caching later as a performance optimization only.
- Implementation path: treat all intermediate analysis branches as read-only and ephemeral.
- Canonical generation flow: draft pass generates the main answer draft from the current thread context without retrieved memory.
- Canonical generation flow: long-term-memory analysis pass uses the current thread context plus retrieved long-term memory and returns only relevant facts, constraints, preferences, and prior decisions.
- Canonical generation flow: final synthesis pass generates the final user-visible answer from the current thread context plus the extracted long-term-memory analysis and the draft answer.
- Important rule: the long-term-memory analysis pass may read from the current thread state, but it must not write back to the live `LlamaState`.
- Important rule: the long-term-memory analysis pass must not produce a full standalone answer.
- Prompt assembly shape:

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

- State handling when `LlamaState` snapshots are still used: capture the original current-thread state before the long-term-memory analysis pass.
- State handling when `LlamaState` snapshots are still used: load that original state for the analysis pass.
- State handling when `LlamaState` snapshots are still used: append retrieval-analysis instructions and retrieved memory only for that temporary pass.
- State handling when `LlamaState` snapshots are still used: discard the derived analysis state after generation.
- State handling when `LlamaState` snapshots are still used: reload the original current-thread state before the final synthesis pass.
- State handling when `LlamaState` snapshots are still used: never call `store_turn()` with the long-term-memory analysis branch.
- State handling when request-time prompt assembly is used: analysis prompts are ephemeral.
- State handling when request-time prompt assembly is used: analysis outputs and other intermediate artifacts are not stored as finalized thread messages.
- State handling when request-time prompt assembly is used: analysis outputs and other intermediate artifacts are stored only in the paired trace thread.
- State handling when request-time prompt assembly is used: only the final user-visible assistant response is committed to the finalized thread history used for normal chat context.
- Persistence rule: after a successful final assistant response, persist finalized user and assistant messages to the finalized thread and persist foreground/background pass artifacts to the paired trace thread.
- Failure rule: if generation fails, do not persist the failed user message and do not store failed-attempt artifacts in either thread.
- Implementation surface: `backend/llm_service.py`.

## Phase 7: API and Three-Column Frontend

- Objective: expose the finalized/trace conversation model in the product UI and wire the backend endpoints needed to browse conversations and clear all stored memory.
- API schema: update chat request and response payloads to carry `thread_id`.
- API surface: add endpoints for thread creation, thread listing, and `delete-all-memory`.
- Frontend data loading: load both the finalized thread and its paired trace thread for the active conversation.
- Frontend layout: use a three-column UI.
- Frontend layout: the first column shows the finalized chat with user messages and final assistant responses.
- Frontend layout: the second column shows the paired trace thread with all foreground and background request/response activity.
- Frontend layout: the third column shows previous conversations and the global memory controls.
- Frontend control: include a `Delete all memory` button.
- Delete-all behavior: when `Delete all memory` is triggered, physically delete all stored SQLite and Qdrant data.
- Product behavior: previous conversations remain browseable in the third column until `Delete all memory` is triggered.

## Phase 8: Operational Health and Fixed System Behavior

- Objective: lock in the operational safeguards and non-negotiable system behavior described by the plan.
- Operational health: add health checks that report SQLite and Qdrant status separately.
- Operational health: include a startup readiness check for the Dockerized Qdrant service.
- Fixed behavior: retrieve up to 5 sentence hits per user sentence.
- Fixed behavior: build a final memory block with up to 4 full user messages.
- Fixed behavior: retrieve from prior persisted user-authored chat messages.
- Fixed behavior: old memories remain eligible regardless of age.
- Fixed behavior: use one Qdrant collection for all user-authored memory sentences.
- Fixed behavior: index user-authored memory sentences incrementally on normal writes.
- Fixed behavior: use internal relevance filtering and internal semantic dedupe during selection.
- Fixed behavior: run against one Dockerized Qdrant service shared by the backend.
- Fixed behavior: use the multi-pass generation flow on every user message.
- Fixed behavior: keep one finalized thread and one paired trace thread per conversation.
- Fixed behavior: the first UI column shows the finalized chat.
- Fixed behavior: the second UI column shows the full trace thread for foreground/background request and response activity.
- Fixed behavior: the third UI column shows previous conversations and the global delete control.
- Fixed behavior: the frontend includes a `Delete all memory` action.
- Fixed behavior: `Delete all memory` hard-deletes all stored SQLite and Qdrant data.
- Fixed behavior: failed user messages are not stored if generation fails.
- Fixed behavior: sentence splitting and embedding support English only.
