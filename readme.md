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

## Novelty of this project
The purpose of this project is to build an AI counselor that does the following unlike any other project out there.

1) It is an AI counselor that remembers you. Current OpenAI ChatGPT interface has an option to remember chat history to remember past context. But it does not make partitions on what you want it to remember for your counselling sessions or your coding sessions. Therefore, if you use the ChatGPT Plus subscription as your counsellor, your coding context will pollute the chats that you use for your counseling chat sessions. 

2) Your chat sessions will be kept private in your laptop. As a user, you won't find any traces of your conversation in the web (e.g. ChatGPT UI), therefore, you can use a ChatGPT account that is shared by multiple people as well. Your private conversations will stay inside your local machine. OpenAI may keep your chats internally in their servers. If you don't want that, this repo also supports uncensored Llama to completely keep your conversations private in your local machine- in this way, you won't get banned for asking sensitive questions. However, you would need a good machine to run the counselor.

3) This project can use a free ChatGPT account without violating any OpenAI terms. Therefore, it is completely free of cost. If you want more intelligence from the AI counselor, just simply upgrade your subscription.

4) Current ChatGPT subscription does not speak or listen to Bangla speech. This project supports Bangla language counselling.

## Current unsolved problems
1) Previous chat history may go beyond the limit of the context window of ChatGPT, therefore, we need to implement memory retrieval that are relevant.

2) Only user messages are saved now for context. It misses the interactions the user had in response to the questions of the counselor. We need to keep track of the counselors messages efficiently as well.

3) It needs an intuitive UI.

4) Bangla speech support is too weak now.

5) The counselor talks too much without respecting the output token count limit.

6) We can have a password protected encryption mechanism so that even the local data is protected from other users.

