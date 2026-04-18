import os

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)


INDEX_HTML = """<!doctype html>
<title>Assistant</title>
<input id=p><button onclick="send()">Send</button><pre id=r></pre>
<script>
async function send(){
  const r=await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p.value})});
  document.getElementById('r').textContent=(await r.json()).reply;
}
</script>"""


@app.get("/")
def index():
    return INDEX_HTML

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://llama:8080")
REQUEST_TIMEOUT = float(os.getenv("LLAMA_REQUEST_TIMEOUT_SECONDS", "180"))
LLAMA_TEMPERATURE = 0.2
LLAMA_MAX_TOKENS = 256


@app.post("/api/prompt")
def chat():
    body = request.get_json()
    prompt = body["prompt"]

    upstream_payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLAMA_TEMPERATURE,
        "max_tokens": LLAMA_MAX_TOKENS,
        "stream": False,
    }

    response = requests.post(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        json=upstream_payload,
        timeout=REQUEST_TIMEOUT,
    )

    data = response.json()

    return jsonify(
        {
            "reply": data["choices"][0]["message"]["content"]
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
