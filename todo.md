# Openclaw Assistant — Improvement TODO

Roadmap of planned improvements. Current architecture: FastAPI backend
([backend/main.py](backend/main.py)) + Vue/Vite frontend
([frontend/src/App.vue](frontend/src/App.vue)), fully offline (Ollama chat, Piper
TTS, faster-whisper STT), started via [run.sh](run.sh).

---

## 1. More natural voice (TTS)

Current TTS is local Piper ([backend/main.py](backend/main.py) `_tts_piper`), which
sounds robotic. Goal: noticeably more natural/expressive speech while staying offline.

- [ ] Evaluate higher-quality offline neural TTS engines (e.g. Kokoro, XTTS-v2,
      Coqui, Parler-TTS, Piper's `*-high` quality models) for naturalness vs. CPU cost.
- [ ] Add an abstraction so the voice engine is pluggable (Piper today, a better
      engine tomorrow) — generalize `_synthesize` / voice-id namespacing
      (`piper:<id>` → `<engine>:<id>`).
- [ ] Support prosody / emotion / streaming synthesis if the chosen engine allows.
- [ ] Keep the existing `*asterisk*` segment splicing working with the new engine.
- [ ] Benchmark latency on a laptop CPU; ensure no perf regression vs. Piper.
- [ ] Update [run.sh](run.sh) voice download/setup for the new engine.

## 2. Move Llama models into a local folder

Today models live wherever Ollama stores them (`~/.ollama`); chat models are
defined in `CHAT_MODELS` ([backend/main.py:28](backend/main.py#L28)).

- [ ] Store/load all Llama (and embedding) models from a project-local `models/`
      folder so the app is self-contained and portable.
- [ ] Point Ollama at the local folder (`OLLAMA_MODELS` env var) or bundle GGUFs
      and load them directly.
- [ ] Update [run.sh](run.sh) and `.gitignore` to manage the local model dir
      (models are large — keep out of git, document the download step).
- [ ] Verify retrieval embeddings (`nomic-embed-text`) also resolve from the local dir.

## 3. Single-binary, cross-platform (Windows + Mac), no perf loss

Today requires Python venv + Node/Vite + Ollama installed separately via
[run.sh](run.sh). Goal: one runnable binary per platform; UI pops up showing only a QR code.

- [ ] Bundle backend (FastAPI/uvicorn) into a single executable (PyInstaller /
      Nuitka), embedding Piper, faster-whisper, retrieval deps.
- [ ] Build the frontend to static assets and serve them from the backend (drop the
      dev Vite server; backend already builds to [frontend/dist/](frontend/dist/)).
- [ ] Bundle or auto-provision Ollama (or replace with an embedded llama.cpp runtime)
      so the user needs no separate install.
- [ ] On launch: start services, then show a minimal window/tray displaying a **QR code**
      pointing at the LAN address (no full UI on the host).
- [ ] Package per-OS: `.app`/`.dmg` for macOS, `.exe` for Windows.
- [ ] Confirm GPU/accelerated inference still used where available (no perf loss vs.
      the dev setup); fall back to CPU gracefully.
- [ ] Handle first-run model/voice download with progress (reuse the existing SSE
      pull stream, `/api/model/pull`).

## 4. Mobile app over LAN via QR-code pairing

Goal: phone app connects to the laptop on the local network by scanning the QR
code; the chat UI runs on the phone.

- [ ] Build a mobile app (e.g. Capacitor/Expo/Flutter) wrapping the existing web UI
      or a native client talking to the same `/api/*` endpoints.
- [ ] QR code encodes host LAN address + port + a pairing token/credentials.
- [ ] Implement QR scanning + connection flow in the mobile app.
- [ ] Handle LAN discovery / changing IPs gracefully (re-pair via fresh QR).
- [ ] Make the backend bind to `0.0.0.0` and tighten CORS (currently
      `allow_origins=["*"]`, [backend/main.py:351](backend/main.py#L351)).
- [ ] Ensure mic capture / audio playback work in the mobile client (STT/TTS round-trip).

## 5. Authentication (username + password)

No auth today — any LAN device can hit the API.

- [ ] Add username/password login; backend session/token auth on all `/api/*` routes.
- [ ] Securely store credentials (hashed + salted, e.g. argon2/bcrypt) — not the
      plaintext `api_key.txt` currently in the repo root.
- [ ] Login screen in web + mobile UI; persist session on the device.
- [ ] Tie the QR pairing flow (item 4) to issuing a scoped auth token.
- [ ] Remove committed secrets (`api_key.txt`) from the repo and `.gitignore` them.

## 6. Encryption

- [ ] Serve over HTTPS/TLS (self-signed cert for LAN, with the QR pairing trusting it),
      so traffic between phone and laptop is encrypted in transit.
- [ ] Encrypt sensitive data at rest: chat history ([backend/chat_history.json](backend/chat_history.json)),
      facts ([backend/facts.json](backend/facts.json)), retrieval vectors, per-profile data.
- [ ] Define a key-management approach (derive from the user password / OS keystore).
- [ ] Ensure auth tokens and credentials are never transmitted/stored in plaintext.

## 7. Multiple chat threads

Today there is exactly **one** persisted thread (`conversation` global,
[backend/main.py:214](backend/main.py#L214); history in a single JSON file).
Retrieval index + facts are also global.

- [ ] Replace the single-thread model with a thread store (id, title, created/updated,
      messages) — likely move from flat JSON to SQLite.
- [ ] Per-thread retrieval index + vector cache + fact store (currently global
      singletons: `conv_index`, `fact_store`).
- [ ] API: create / list / rename / delete / switch threads; scope
      `/api/chat`, `/api/history`, `/api/reset`, `/api/delete`, `/api/tts` by thread id.
- [ ] Frontend: thread list / sidebar, new-chat button, switching, per-thread titles
      (auto-title from first message).
- [ ] Migrate the existing single thread into the new store on upgrade.

## 8. Minimalistic profiles (name + age integration)

- [ ] Add lightweight user profiles with at least **name** and **age**.
- [ ] Scope threads, history, facts, and settings per profile.
- [ ] Inject profile (name/age) into the system prompt / fact store so replies are
      personalized and age-appropriate.
- [ ] Profile selection/creation UI; tie profiles to auth (item 5) and encryption (item 6).
- [ ] Keep it minimal — small editable profile, no heavy account system.

---

## Cross-cutting / ordering notes

- Items **5 (auth)**, **6 (encryption)**, and **4 (mobile/QR)** are tightly coupled —
  the QR pairing should hand off an auth token over a TLS channel. Design them together.
- Item **7 (threads)** and **8 (profiles)** both push toward a real datastore (SQLite);
  do the storage refactor once and build both on it.
- Item **3 (single binary)** affects how everything ships — decide the packaging
  approach before deep work on the others so models, certs, and the DB land in
  predictable, bundle-friendly locations.
- Throughout: preserve the **fully-offline** guarantee and avoid latency regressions.
