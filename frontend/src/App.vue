<script setup>
import { ref, onMounted, nextTick } from 'vue'

const messages = ref([])        // { role, content }
const input = ref('')
const sending = ref(false)
const status = ref('')
const statusError = ref(false)
const recording = ref(false)

const voices = ref([])          // { id, label }
const selectedVoice = ref('')   // current Piper voice id

const messagesEl = ref(null)
let mediaRecorder = null
let audioChunks = []
let currentAudio = null

function setStatus(text, isError = false) {
  status.value = text
  statusError.value = isError
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history')
    const data = await res.json()
    messages.value = data.messages || []
    scrollToBottom()
  } catch {
    setStatus('Could not reach backend on :8000', true)
  }
}

async function send() {
  const text = input.value.trim()
  if (!text || sending.value) return
  input.value = ''
  messages.value.push({ role: 'user', content: text })
  sending.value = true
  setStatus('')
  scrollToBottom()

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const data = await res.json()
    messages.value.push({ role: 'assistant', content: data.reply })
    scrollToBottom()
    speak(data.reply) // auto-speak the assistant's reply
  } catch (e) {
    setStatus(`Chat failed: ${e.message}`, true)
  } finally {
    sending.value = false
  }
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

async function reset() {
  await fetch('/api/reset', { method: 'POST' })
  messages.value = []
  setStatus('Conversation cleared.')
}

// ---- TTS ----
async function speak(text) {
  try {
    if (currentAudio) {
      currentAudio.pause()
      currentAudio = null
    }
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, voice: selectedVoice.value || undefined }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const blob = await res.blob()
    currentAudio = new Audio(URL.createObjectURL(blob))
    currentAudio.play()
  } catch (e) {
    setStatus(`TTS failed: ${e.message}`, true)
  }
}

// ---- STT (record mic, transcribe, drop into input) ----
async function toggleRecord() {
  if (recording.value) {
    mediaRecorder?.stop()
    return
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder = new MediaRecorder(stream)
    audioChunks = []
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data)
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop())
      recording.value = false
      await transcribe(new Blob(audioChunks, { type: 'audio/webm' }))
    }
    mediaRecorder.start()
    recording.value = true
    setStatus('Recording… tap the mic again to stop.')
  } catch (e) {
    setStatus(`Mic access failed: ${e.message}`, true)
  }
}

async function transcribe(blob) {
  setStatus('Transcribing…')
  try {
    const form = new FormData()
    form.append('file', blob, 'audio.webm')
    const res = await fetch('/api/stt', { method: 'POST', body: form })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const data = await res.json()
    input.value = (input.value ? input.value + ' ' : '') + data.text.trim()
    setStatus('')
  } catch (e) {
    setStatus(`STT failed: ${e.message}`, true)
  }
}

async function loadVoices() {
  try {
    const res = await fetch('/api/voices')
    const data = await res.json()
    voices.value = data.voices || []
    selectedVoice.value = data.default || (voices.value[0]?.id ?? '')
  } catch {
    // non-fatal: TTS just falls back to the server default voice
  }
}

onMounted(() => {
  loadHistory()
  loadVoices()
})
</script>

<template>
  <div class="app">
    <header class="header">
      <div>
        <h1>Openclaw Assistant</h1>
        <div class="sub">Groq Llama 70B · single thread · voice in/out</div>
      </div>
      <div class="header-actions">
        <select
          v-if="voices.length > 1"
          v-model="selectedVoice"
          class="voice-select"
          title="Voice"
        >
          <option v-for="v in voices" :key="v.id" :value="v.id">🎙 {{ v.label }}</option>
        </select>
        <button class="reset-btn" @click="reset">Clear</button>
      </div>
    </header>

    <div class="messages" ref="messagesEl">
      <div v-if="messages.length === 0" class="empty">
        Say hi or tap the mic to start talking.
      </div>
      <div
        v-for="(m, i) in messages"
        :key="i"
        class="msg"
        :class="m.role"
      >
        <div class="bubble">{{ m.content }}</div>
        <button
          v-if="m.role === 'assistant'"
          class="speak-btn"
          title="Play"
          @click="speak(m.content)"
        >🔊</button>
      </div>
      <div v-if="sending" class="msg assistant">
        <div class="bubble">
          <span class="typing"><span></span><span></span><span></span></span>
        </div>
      </div>
    </div>

    <div class="status" :class="{ error: statusError }">{{ status }}</div>

    <div class="composer">
      <button
        class="icon-btn"
        :class="{ recording }"
        :title="recording ? 'Stop recording' : 'Record'"
        @click="toggleRecord"
      >🎤</button>
      <textarea
        v-model="input"
        rows="1"
        placeholder="Type a message, or use the mic…"
        @keydown="onKeydown"
      ></textarea>
      <button
        class="icon-btn send"
        :disabled="sending || !input.trim()"
        title="Send"
        @click="send"
      >➤</button>
    </div>
  </div>
</template>
