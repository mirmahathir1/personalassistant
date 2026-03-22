# Retrieval Memory Plan

## Goal

Add sentence-level memory retrieval to the current app using:

- `sqlite3` as the source of truth for threads, messages, and sentence records
- `qdrant-client[fastembed]` in the backend, connected to a Qdrant service running in Docker
- a compact retrieved memory block injected on every user message before reply generation

## Recommendation

- Use SQLite for all canonical chat data.
- Use a Dockerized Qdrant service for semantic lookup.
- Maintain two SQLite-backed threads per conversation:
  a finalized thread for user messages and final assistant responses
  a paired trace thread for all foreground and background request/response activity
- Keep one Qdrant point per user-authored memory sentence.
- Build the prompt per request from stored thread history plus a retrieved memory block.
- Do not treat the current `llama.cpp` rolling state as the long-term conversation store once retrieval is added.
- Use a mandatory multi-pass generation flow with a read-only long-term-memory analysis pass instead of a separate long-term-memory-only answer pass.
- SQLite keeps assistant responses, but Qdrant does not store or index assistant responses.
- Retrieval must stay sentence-level, but prompt injection must expand a matched sentence back to the full original user message.

## Important Constraint In The Current App

The current backend keeps a rolling `llama.cpp` state snapshot and saves that state after every turn. If a retrieved memory block is injected into the prompt under that model, the retrieved text becomes part of the saved state and will accumulate across turns.

That causes two problems:

- retrieved memory repeats and pollutes future context
- the context window fills with old retrieval blocks instead of real conversation state

Because of that, the recommended implementation is:

- SQLite stores the thread history
- Qdrant stores the user-memory sentence index
- each `/api/chat` request assembles fresh prompt context from SQLite history plus retrieved memory

## Proposed Storage Layout

Docker-only layout:

- mount `./data:/data`
- SQLite database: `/data/app.db`
- Qdrant storage inside the Qdrant container: `/qdrant/storage`
- backend Qdrant URL on the Compose network: `http://qdrant:6333`
- FastEmbed cache: `/data/fastembed`

SQLite path and Qdrant URL should be hardcoded in the backend config file instead of exposed as environment knobs.
Recommended location: `backend/config.py`.

## Docker Qdrant Service

Use Qdrant as its own Docker Compose service instead of Qdrant local mode.

Recommended Compose shape:

- service name: `qdrant`
- image: `qdrant/qdrant`
- backend connects to `http://qdrant:6333`
- mount `./data/qdrant:/qdrant/storage`
- publish `6333:6333`

Recommended backend behavior:

- initialize the Qdrant collection on startup if it does not exist
- fail health checks clearly if the Qdrant service is unreachable
- keep SQLite and FastEmbed cache mounted separately from Qdrant storage
- hardcode the SQLite path and Qdrant URL in `backend/config.py`

## Hardcoded Defaults

Memory, retrieval, and deployment settings should stay fixed in the repository rather than exposed as external knobs.

Hardcoded in `docker-compose.yml`:

- backend data mount: `./data:/data`
- Qdrant storage mount: `./data/qdrant:/qdrant/storage`
- published Qdrant port: `6333:6333`
- Qdrant Docker image and tag: `qdrant/qdrant`

Hardcoded in `backend/config.py`:

- `SQLITE_PATH`
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `FASTEMBED_CACHE_PATH`
- retrieval mode, result sizes, relevance filtering, and semantic dedupe behavior

## Data Model

### SQLite

Use SQLite as the canonical store.

`threads`

- `id` TEXT PRIMARY KEY
- `conversation_id` TEXT NOT NULL
- `thread_type` TEXT NOT NULL
- `title` TEXT NULL
- `created_at` TEXT NOT NULL
- `updated_at` TEXT NOT NULL

`messages`

- `id` TEXT PRIMARY KEY
- `thread_id` TEXT NOT NULL
- `role` TEXT NOT NULL
- `kind` TEXT NOT NULL
- `content` TEXT NOT NULL
- `turn_index` INTEGER NOT NULL
- `created_at` TEXT NOT NULL
- foreign key on `thread_id`

`message_sentences`

- `id` TEXT PRIMARY KEY
- `message_id` TEXT NOT NULL
- `thread_id` TEXT NOT NULL
- `role` TEXT NOT NULL
- `sentence_index` INTEGER NOT NULL
- `text` TEXT NOT NULL
- `created_at` TEXT NOT NULL
- `qdrant_point_id` TEXT NULL
- foreign key on `message_id`
- foreign key on `thread_id`

Rule:

- each conversation has one `finalized` thread and one `trace` thread
- the finalized thread stores user messages and final assistant responses only
- the trace thread stores all foreground and background request/response artifacts, including user messages, retrieved memory, draft pass I/O, long-term-memory analysis I/O, and final synthesis I/O
- user sentence rows may have a `qdrant_point_id`
- assistant sentence rows stay in SQLite only and are not indexed into Qdrant
- trace-thread messages are not indexed into Qdrant

Optional later:

- `memory_events` for debugging retrieval decisions
- `thread_summaries` if you later need long-history compression

### Qdrant Service

Use one collection, for example `conversation_sentences`.

Each point should store:

- embedding vector
- payload `sentence_id`
- payload `thread_id`
- payload `message_id`
- payload `role` with value `user`
- payload `sentence_index`
- payload `created_at`

Keep the canonical sentence text in SQLite. Qdrant only needs enough payload to identify the matching sentence quickly.
Assistant responses are not stored in Qdrant.
Trace-thread messages are not stored in Qdrant.
The retrieved context block should be built from full user messages loaded from SQLite, not isolated sentence fragments.

## Recommended Dependencies

- `qdrant-client[fastembed]`
- no ORM; use Python `sqlite3`
- one lightweight sentence splitter behind an adapter module
- Docker Compose service using the official `qdrant/qdrant` image

For sentence splitting, keep the choice isolated in one file so it can be swapped later. A lightweight splitter is preferable to pulling in a large NLP stack for this app.

## Retrieval Behavior

### Write Path

After a successful final assistant response:

1. Persist the user message in SQLite.
   Persist it in the finalized thread.
2. Split the user message into individual sentences.
3. Persist user sentence rows in SQLite.
4. Generate embeddings for each user sentence.
5. Upsert user sentence points into Qdrant.
6. Persist the assistant reply in SQLite.
   Persist it in the finalized thread.
7. Split the assistant reply into sentences.
8. Persist assistant sentence rows in SQLite.
9. Do not generate embeddings for assistant sentences.
10. Do not upsert assistant sentences into Qdrant.

Additionally:

- persist trace-thread entries for the user message, retrieved memory, draft pass, long-term-memory analysis pass, and final synthesis pass
- keep trace-thread persistence separate from Qdrant indexing

If generation fails:

- do not persist the failed user message in SQLite
- do not write failed-attempt artifacts to the finalized thread or the trace thread

### Read Path

On every incoming user message:

1. Accept the finalized `thread_id`, or create a new conversation with a finalized thread and a paired trace thread if this is the first message.
2. Split the incoming user message into sentences.
3. For each sentence, run a Qdrant search against prior user-authored sentence points.
4. Exclude the current unsaved message from retrieval.
5. Apply the retrieval scope strategy.
6. Merge sentence hits by `message_id`, using the best matching sentence score per message.
7. Deduplicate strictly by `message_id` so the same user message block is never added multiple times.
8. Sort message candidates by their best sentence similarity score.
9. Load the full original user message content from SQLite for the message candidates.
10. Apply post-retrieval memory deduplication at the message level:
    exact dedupe by `message_id`
    semantic dedupe by message similarity against already selected memory blocks
11. Do not apply age-based downranking; old memories remain eligible if they are still semantically relevant.
12. Build a compact memory block from full user messages, not from isolated sentences.
13. Run the canonical multi-pass generation flow:
    draft from the recent thread transcript plus the current user message
    run a read-only long-term-memory analysis pass using the recent thread transcript, the current user message, and the retrieved memory block
    synthesize the final answer from the current thread context plus the extracted memory analysis and the draft answer
14. Persist finalized user and assistant messages to the finalized thread, and persist foreground/background pass artifacts to the paired trace thread.

## Retrieval Scope Strategy

Use one retrieval scope:

- search prior persisted user-authored messages across stored chat history
- exclude the current unsaved message
- keep `thread_id` as metadata, not as a required retrieval filter

If thread-aware balancing is needed later, add it as a ranking or selection rule rather than a separate retrieval mode.

## Memory Ranking

Start simple:

- primary sort: vector similarity score from Qdrant
- aggregate sentence hits up to the message level by `message_id`
- dedupe by `message_id`
- rank each message by its best matching sentence score
- discard very low-score matches using a minimum threshold
- after ranking, apply a greedy semantic dedupe pass so a newly considered memory is skipped if it is too similar to a message already selected
- do not reduce rank based on message age alone

## Memory Deduplication

Apply deduplication after retrieval and after sentence hits are expanded to full user messages.

Required behavior:

- exact dedupe by `message_id`
- semantic dedupe at the full-message level
- keep the highest-ranked message first
- for each next candidate, compare it to already selected memory messages
- if similarity to any selected memory exceeds the internal dedupe threshold, skip it
- continue until the memory block reaches its max size or candidate list is exhausted

Deduplication should operate on full user message text, not on individual sentence hits.

The dedupe threshold should be distinct from the retrieval relevance threshold:

- retrieval threshold decides whether a memory is relevant to the current user query
- dedupe threshold decides whether a candidate memory is too similar to a memory already selected

## Memory Block Format

The memory block should be short, explicit, and easy for the model to use.
Embedding lookup is sentence-level, but the injected memory units are full user messages.

Suggested structure:

```text
Relevant memory:
- User message: I am allergic to peanuts, and I avoid snacks with peanut oil too.
- User message: I prefer short answers when we discuss implementation plans.
```

Rules:

- keep the block small, for example top 3 to 6 full user messages
- retrieve at sentence granularity but inject the full original user message
- never include the same user message more than once in a single memory block
- do not include a message if it is semantically too similar to a message already selected for the same block
- prefer concise factual messages over long narrative ones
- avoid duplicate or near-duplicate messages
- exclude very weak matches
- keep a hard token budget for the memory block

## Prompt Assembly

Use the multi-pass section below as the canonical request flow.

Recommended pass inputs:

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

The memory block should be treated as supporting context, not guaranteed truth.

## Multi-Pass Mode

Use this multi-pass flow for every user message:

1. Draft pass: generate the main answer draft from the current thread context.
2. Long-term-memory analysis pass: use the current thread context plus retrieved long-term memory, but ask the model to return only:
   relevant facts
   constraints
   preferences
   prior decisions
3. Final synthesis pass: generate the final user-visible answer from the current thread context plus the extracted long-term-memory analysis and the draft answer.

Important rule:

- the long-term-memory analysis pass may read from the current thread state, but it must not write back to the live `LlamaState`

Do not have the long-term-memory pass produce a full standalone answer. It should return distilled evidence only.

### State Handling Rules For Multi-Pass Mode

If `LlamaState` snapshots are still used for short-term context:

- capture the original current-thread state before the long-term-memory analysis pass
- load that original state for the analysis pass
- append retrieval-analysis instructions and retrieved memory only for that temporary pass
- discard the derived analysis state after generation
- reload the original current-thread state before the final synthesis pass
- never call `store_turn()` with the long-term-memory analysis branch

If the app is refactored to request-time prompt assembly from SQLite instead of relying on rolling `LlamaState`, apply the same rule conceptually:

- analysis prompts are ephemeral
- analysis outputs and other intermediate artifacts are not stored as finalized thread messages
- analysis outputs and other intermediate artifacts are stored only in the paired trace thread
- only the final user-visible assistant response is committed to the finalized thread history used for normal chat context

## Recommended Refactor

Replace the current single rolling-state conversation model with request-time prompt assembly.

Target behavior:

- thread history comes from SQLite
- memory retrieval comes from Qdrant
- `llama.cpp` generates from the assembled prompt inputs for that request

Possible implementation path:

- keep the current tokenizer and manual chat template logic
- stop relying on `SingleSessionManager` as the sole conversation history mechanism
- optionally keep in-memory caching later as a performance optimization only
- treat all intermediate analysis branches as read-only and ephemeral

## File-Level Implementation Plan

### Phase 1: Persistence Foundation

- hardcode `SQLITE_PATH`, `QDRANT_URL`, and `QDRANT_COLLECTION` in `backend/config.py`
- keep `FASTEMBED_CACHE_PATH` in `backend/config.py` as a code default
- add a `backend/db.py` module for SQLite connection and schema creation
- add a `backend/repositories.py` module for thread, message, and sentence CRUD
- add `./data/` to `.gitignore`
- update Docker config to mount `./data:/data`
- add a `qdrant` service to `docker-compose.yml`
- mount `./data/qdrant:/qdrant/storage` for the Qdrant service
- connect the backend to `http://qdrant:6333` in Docker

### Phase 2: Thread And Message Model

- add `thread_id` support to the backend API
- support true multi-thread persistence
- create one finalized thread and one paired trace thread per conversation
- persist user messages and final assistant responses into the finalized thread
- persist all foreground/background request and response artifacts into the paired trace thread
- add sentence rows for finalized user and finalized assistant messages in SQLite
- index only finalized user-authored sentence rows in Qdrant

### Phase 3: Sentence Index

- add a `backend/memory_store.py` wrapper around the Qdrant HTTP service
- initialize the sentence collection on startup
- add methods for batch upsert and filtered search
- store only user sentence identifiers and search metadata in Qdrant payloads

### Phase 4: Sentence Splitting

- add `backend/sentence_splitter.py`
- support English-only sentence splitting
- normalize whitespace before splitting
- ignore empty fragments
- preserve sentence order with `sentence_index`

### Phase 5: Retrieval Service

- add `backend/memory_service.py`
- split incoming user text into sentences
- run one search per sentence
- merge hits up to the message level
- dedupe strictly by `message_id`
- apply greedy semantic dedupe at the full-message level
- apply score threshold and result caps
- load full original user message text from SQLite
- return a formatted memory block composed of full user messages
- retrieve only user-authored memory rows from Qdrant

### Phase 6: LLM Service Refactor

- refactor `backend/llm_service.py`
- replace rolling conversation dependence with request-time prompt assembly
- keep token budgeting explicit so memory does not crowd out the active turn
- add an ephemeral generation helper for intermediate passes that must not be committed to the live conversation state
- always:
  draft from current thread context without retrieved memory
  run a read-only long-term-memory analysis pass that uses current thread context plus retrieved memory
  synthesize the final answer without saving intermediate prompts or outputs as normal thread history

### Phase 7: API And Frontend Integration

- update chat request and response schemas to carry `thread_id`
- add endpoints for thread creation, thread listing, and delete-all-memory
- add frontend data loading for the finalized thread and its paired trace thread
- add a three-column frontend layout:
  first column shows the finalized chat with user messages and final assistant responses
  second column shows the paired trace thread with all foreground and background request/response activity
  third column shows previous conversations and the global memory controls
- add a frontend `Delete all memory` button
- when `Delete all memory` is triggered, physically delete all stored SQLite and Qdrant data

### Phase 8: Maintenance And Recovery

- add health checks that report SQLite and Qdrant status separately
- include a startup readiness check for the Dockerized Qdrant service

## Fixed Behavior

- retrieve up to 5 sentence hits per user sentence
- build a final memory block with up to 4 full user messages
- retrieve from prior persisted user-authored chat messages
- old memories remain eligible regardless of age
- use one Qdrant collection for all user-authored memory sentences
- index user-authored memory sentences incrementally on normal writes
- use internal relevance filtering and internal semantic dedupe during selection
- run against one Dockerized Qdrant service shared by the backend
- use the multi-pass generation flow on every user message
- keep one finalized thread and one paired trace thread per conversation
- the first UI column shows the finalized chat
- the second UI column shows the full trace thread for foreground/background request and response activity
- the third UI column shows previous conversations and the global delete control
- the frontend includes a `Delete all memory` action
- `Delete all memory` hard-deletes all stored SQLite and Qdrant data
- failed user messages are not stored if generation fails
- sentence splitting and embedding support English only

