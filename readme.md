# Minimal Setup

From a fresh clone:

```bash
git clone <repo-url>
cd personalassistant
mkdir -p api/auth
docker compose -f api/docker-compose.yml pull
docker compose -f api/docker-compose.yml run --rm --service-ports cliproxy /CLIProxyAPI/CLIProxyAPI --codex-login --no-browser
```

Open the printed OpenAI URL in your browser, sign in, and finish OAuth. Then
start the proxy:

```bash
docker compose -f api/docker-compose.yml up -d
```

The local OpenAI-compatible API will be available at:

```text
http://localhost:8317/v1
```

## API smoke test

`test/api` calls the running proxy and checks that the answer to a simple
arithmetic prompt contains the expected value:

```bash
python3 test/api/test_api.py
```

## Run the assistant

The assistant is a listen-and-speak CLI that talks to the proxy above, so make
sure the proxy is up first (`docker compose -f api/docker-compose.yml up -d`). It
listens through your microphone and speaks its replies, so run it on the host:

```bash
python3 -m venv assistant/.venv
source assistant/.venv/bin/activate
pip install -r api/requirements.txt
pip install -r assistant/requirements.txt
python -m piper.download_voices en_US-lessac-medium --download-dir .assistant/voices
python -m assistant.assistant --model gpt-5.4-mini --effort low --lang bn
```

`--lang` is required and selects the conversation language:

- `--lang en` — speak and hear English (uses the Piper voice above).
- `--lang bn` — speak Bengali. Your speech is transcribed to Bengali and then
  translated to English for the assistant; its English reply is translated back
  to Bengali and spoken. Transcription (faster-whisper, multilingual),
  translation (Argos Translate `bn<->en`) and Bengali speech (Facebook MMS-TTS,
  `facebook/mms-tts-ben`) all run locally — the model files download once on
  first use, then run offline. No Piper voice download is needed for `bn`.
  Bengali transcription defaults to the `large-v3` Whisper model (a ~3 GB
  download and slower CPU decode) because smaller models often transliterate or
  garble Bengali into another script. If `large-v3` is too slow, pass
  `--listen-model medium`, but expect lower accuracy. The first run downloads
  the chosen model once.

The assistant listens through your microphone automatically; replies stream to
the terminal and are spoken aloud.
