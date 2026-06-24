<script setup>
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'

const messages = ref([])        // { role, content }
const input = ref('')
const sending = ref(false)
const status = ref('')
const statusError = ref(false)
const recording = ref(false)
const selected = ref(new Set()) // indices (into messages) ticked for deletion
const deleting = ref(false)
const selectMode = ref(false)   // when on, per-message deletion checkboxes are shown

const mode = ref('text')        // 'text' (no voice) | 'call' (voice-only)
const callState = ref('idle')   // 'idle' | 'recording' | 'thinking' | 'speaking'
const menuOpen = ref(false)     // three-dot overflow menu
const contactOpen = ref(false)  // WhatsApp-style "View Contact" side pane
const confirmNuke = ref(false)  // nuke-confirmation modal
const nuking = ref(false)

const voices = ref([])          // { id, label }
const selectedVoice = ref('')   // namespaced voice id, e.g. "piper:en_US-amy-medium"

const providers = ref([])       // { id, label, model }
const selectedProvider = ref('')// chat provider id, e.g. "ollama"

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
  clearSelection()  // appended messages shift nothing, but keep selection state clean
  sending.value = true
  setStatus('')
  scrollToBottom()

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, provider: selectedProvider.value || undefined }),
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
  if (nuking.value) return
  nuking.value = true
  try {
    await fetch('/api/reset', { method: 'POST' })
    messages.value = []
    selected.value = new Set()
    setStatus('Conversation nuked.')
  } finally {
    nuking.value = false
    confirmNuke.value = false
  }
}

// ---- per-message selection + delete (forget) ----
function toggleSelect(i) {
  const next = new Set(selected.value)
  next.has(i) ? next.delete(i) : next.add(i)
  selected.value = next
}

function clearSelection() {
  selected.value = new Set()
}

// Toggle the deletion checkboxes on/off. Turning them off clears any ticks.
function toggleSelectMode() {
  selectMode.value = !selectMode.value
  if (!selectMode.value) clearSelection()
}

async function deleteSelected() {
  if (selected.value.size === 0 || deleting.value) return
  const indices = [...selected.value]
  deleting.value = true
  setStatus('Forgetting selected messages…')
  try {
    const res = await fetch('/api/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ indices }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const data = await res.json()
    messages.value = data.messages || []
    selected.value = new Set()
    selectMode.value = false  // hide the checkboxes again after deleting
    const n = data.removed_facts?.length || 0
    setStatus(`Forgot ${data.deleted} message(s)${n ? `, pruned ${n} fact(s)` : ''}.`)
  } catch (e) {
    setStatus(`Delete failed: ${e.message}`, true)
  } finally {
    deleting.value = false
  }
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

// ---- Call mode: one button records, then auto-transcribe → chat → speak ----
async function toggleCall() {
  if (callState.value === 'recording') {
    mediaRecorder?.stop()
    return
  }
  if (callState.value !== 'idle') return // busy thinking/speaking
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder = new MediaRecorder(stream)
    audioChunks = []
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data)
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop())
      await runCallTurn(new Blob(audioChunks, { type: 'audio/webm' }))
    }
    mediaRecorder.start()
    callState.value = 'recording'
    setStatus('Listening… tap to stop.')
  } catch (e) {
    callState.value = 'idle'
    setStatus(`Mic access failed: ${e.message}`, true)
  }
}

// Full voice turn: transcribe the recording, send to chat, speak the reply.
async function runCallTurn(blob) {
  callState.value = 'thinking'
  setStatus('Thinking…')
  try {
    const form = new FormData()
    form.append('file', blob, 'audio.webm')
    const sttRes = await fetch('/api/stt', { method: 'POST', body: form })
    if (!sttRes.ok) {
      const err = await sttRes.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${sttRes.status}`)
    }
    const { text } = await sttRes.json()
    const spoken = text.trim()
    if (!spoken) {
      setStatus('Did not catch that — tap to try again.')
      callState.value = 'idle'
      return
    }
    messages.value.push({ role: 'user', content: spoken })
    scrollToBottom()

    const chatRes = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: spoken, provider: selectedProvider.value || undefined }),
    })
    if (!chatRes.ok) {
      const err = await chatRes.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${chatRes.status}`)
    }
    const { reply } = await chatRes.json()
    messages.value.push({ role: 'assistant', content: reply })
    scrollToBottom()

    callState.value = 'speaking'
    setStatus('')
    await speakAndWait(reply)
  } catch (e) {
    setStatus(`Call failed: ${e.message}`, true)
  } finally {
    callState.value = 'idle'
  }
}

// Like speak(), but resolves when playback finishes so call state can reset.
function speakAndWait(text) {
  return new Promise(async (resolve) => {
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
      currentAudio.onended = () => resolve()
      currentAudio.onerror = () => resolve()
      currentAudio.play()
    } catch (e) {
      setStatus(`TTS failed: ${e.message}`, true)
      resolve()
    }
  })
}

const callLabel = {
  idle: 'Tap to talk',
  recording: 'Listening… tap to stop',
  thinking: 'Thinking…',
  speaking: 'Speaking…',
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

async function loadProviders() {
  try {
    const res = await fetch('/api/providers')
    const data = await res.json()
    providers.value = data.providers || []
    selectedProvider.value = data.default || (providers.value[0]?.id ?? '')
  } catch {
    // non-fatal: chat just falls back to the server default provider
  }
}

// Whether we've finished loading saved settings; guards the auto-save watchers
// so applying loaded values doesn't immediately POST them back.
let settingsLoaded = false

async function loadSettings() {
  try {
    const res = await fetch('/api/settings')
    const s = await res.json()
    // Apply saved selections, but only if still valid against current lists.
    if (s.chatProvider && providers.value.some((p) => p.id === s.chatProvider)) {
      selectedProvider.value = s.chatProvider
    }
    if (s.voice && voices.value.some((v) => v.id === s.voice)) {
      selectedVoice.value = s.voice
    }
  } catch {
    // non-fatal: fall back to endpoint defaults already applied
  } finally {
    settingsLoaded = true
  }
}

async function saveSettings() {
  if (!settingsLoaded) return
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chatProvider: selectedProvider.value,
        voice: selectedVoice.value,
      }),
    })
  } catch {
    // non-fatal: selection still works for this session
  }
}

watch([selectedProvider, selectedVoice], saveSettings)

// Close the three-dot menu when clicking anywhere outside it.
function onDocClick() {
  menuOpen.value = false
}

onMounted(async () => {
  loadHistory()
  document.addEventListener('click', onDocClick)
  // Load the option lists first so saved selections can be validated against them.
  await Promise.all([loadVoices(), loadProviders()])
  await loadSettings()
})

onUnmounted(() => document.removeEventListener('click', onDocClick))
</script>

<template>
  <div class="app">
    <header class="header">
      <div></div>
      <div class="header-actions">
        <button
          class="wa-icon"
          :class="{ active: mode === 'text' }"
          title="Chat"
          @click="mode = 'text'"
        >
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <path
              fill="currentColor"
              d="M12 3C6.48 3 2 6.86 2 11.62c0 2.7 1.45 5.11 3.72 6.7-.16 1.2-.66 2.3-1.45 3.18-.18.2-.05.52.22.5 1.83-.13 3.5-.78 4.86-1.78.84.2 1.73.3 2.65.3 5.52 0 10-3.86 10-8.6S17.52 3 12 3zm-4 9.75a1.25 1.25 0 110-2.5 1.25 1.25 0 010 2.5zm4 0a1.25 1.25 0 110-2.5 1.25 1.25 0 010 2.5zm4 0a1.25 1.25 0 110-2.5 1.25 1.25 0 010 2.5z"
            />
          </svg>
        </button>
        <button
          class="wa-icon"
          :class="{ active: mode === 'call' }"
          title="Call"
          @click="mode = 'call'"
        >
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <path
              fill="currentColor"
              d="M6.54 5c.06.89.21 1.76.45 2.59l-1.2 1.2c-.41-1.2-.67-2.47-.76-3.79h1.51M16.4 17.02c.85.24 1.72.39 2.6.45v1.49c-1.32-.09-2.59-.35-3.8-.75l1.2-1.19M7.5 3H4c-.55 0-1 .45-1 1 0 9.39 7.61 17 17 17 .55 0 1-.45 1-1v-3.49c0-.55-.45-1-1-1-1.24 0-2.45-.2-3.57-.57a.84.84 0 00-.31-.05c-.26 0-.51.1-.71.29l-2.2 2.2a15.07 15.07 0 01-6.59-6.59l2.2-2.2c.28-.28.36-.67.25-1.02A11.36 11.36 0 018.5 4c0-.55-.45-1-1-1z"
            />
          </svg>
        </button>
        <div class="menu-wrap">
          <button
            class="menu-btn"
            title="More options"
            @click.stop="menuOpen = !menuOpen"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
              <circle cx="12" cy="5" r="1.6" fill="currentColor" />
              <circle cx="12" cy="12" r="1.6" fill="currentColor" />
              <circle cx="12" cy="19" r="1.6" fill="currentColor" />
            </svg>
          </button>
          <div v-if="menuOpen" class="menu-panel" @click.stop>
            <button class="menu-item" @click="menuOpen = false; contactOpen = true">
              View contact
            </button>
            <button class="menu-item" @click="menuOpen = false; toggleSelectMode()">
              {{ selectMode ? 'Hide deletion checkboxes' : 'Delete messages' }}
            </button>
          </div>
        </div>
      </div>
    </header>

    <div v-if="confirmNuke" class="modal-overlay" @click="confirmNuke = false">
      <div class="modal" @click.stop>
        <h2 class="modal-title">Nuke entire chat?</h2>
        <p class="modal-body">
          This permanently erases the whole conversation, the retrieval index,
          and all long-term memory facts. This cannot be undone.
        </p>
        <div class="modal-actions">
          <button class="modal-cancel" @click="confirmNuke = false">Cancel</button>
          <button class="modal-nuke" :disabled="nuking" @click="reset">
            {{ nuking ? 'Nuking…' : '💥 Nuke it' }}
          </button>
        </div>
      </div>
    </div>

    <!-- WhatsApp-style "View Contact" side pane -->
    <transition name="contact-slide">
      <aside v-if="contactOpen" class="contact-pane" @click.stop>
        <div class="contact-head">
          <button class="contact-close" title="Close" @click="contactOpen = false">
            <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
              <path
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                d="M19 12H5m0 0 6-6m-6 6 6 6"
              />
            </svg>
          </button>
          <span class="contact-head-title">Contact info</span>
        </div>
        <div class="contact-body">
          <div class="contact-id">
            <div class="contact-avatar">
              <svg viewBox="0 0 212 212" width="96" height="96" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M106 0C47.5 0 0 47.5 0 106s47.5 106 106 106 106-47.5 106-106S164.5 0 106 0zm0 53c16.6 0 30 13.4 30 30s-13.4 30-30 30-30-13.4-30-30 13.4-30 30-30zm0 138c-25 0-47.2-12.1-61-30.7 9.8-17.4 28.4-29.3 49.8-30.2a44 44 0 0 0 22.4 0c21.4.9 40 12.8 49.8 30.2-13.8 18.6-36 30.7-61 30.7z"
                />
              </svg>
            </div>
            <div class="contact-name">Assistant</div>
            <div class="contact-sub">Local chat · voice · memory</div>
          </div>

          <div v-if="providers.length > 1" class="contact-section">
            <label class="menu-label">Chat model</label>
            <select v-model="selectedProvider" class="voice-select menu-select" title="Chat model">
              <option v-for="p in providers" :key="p.id" :value="p.id">
                {{ p.label }}
              </option>
            </select>
          </div>

          <div class="contact-section">
            <label class="menu-label">Voice</label>
            <select v-model="selectedVoice" class="voice-select menu-select" title="Voice">
              <option v-for="v in voices" :key="v.id" :value="v.id">
                {{ v.label }}
              </option>
            </select>
          </div>

          <div class="contact-section">
            <button class="contact-nuke" @click="confirmNuke = true">
              <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
                <path
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1.8"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m2 0v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V7M10 11v6M14 11v6"
                />
              </svg>
              <span>Delete chat</span>
            </button>
          </div>
        </div>
      </aside>
    </transition>

    <div v-if="mode === 'text'" class="messages" ref="messagesEl">
      <div v-if="messages.length === 0" class="empty">
        Say hi or tap the mic to start talking.
      </div>
      <div
        v-for="(m, i) in messages"
        :key="i"
        class="msg"
        :class="[m.role, { picked: selected.has(i) }]"
      >
        <input
          v-if="selectMode"
          type="checkbox"
          class="msg-check"
          title="Select to forget"
          :checked="selected.has(i)"
          @change="toggleSelect(i)"
        />
        <div class="bubble">{{ m.content }}</div>
      </div>
      <div v-if="sending" class="msg assistant">
        <div class="bubble">
          <span class="typing"><span></span><span></span><span></span></span>
        </div>
      </div>
    </div>

    <div v-if="selectMode" class="delete-bar">
      <span>{{ selected.size }} selected</span>
      <button
        class="delete-btn"
        :disabled="deleting || selected.size === 0"
        @click="deleteSelected"
      >
        🗑 Forget selected
      </button>
      <button class="delete-cancel" @click="toggleSelectMode">Cancel</button>
    </div>

    <div class="status" :class="{ error: statusError }">{{ status }}</div>

    <div v-if="mode === 'text'" class="composer">
      <textarea
        v-model="input"
        rows="1"
        placeholder="Type a message…"
        @keydown="onKeydown"
      ></textarea>
      <button
        class="icon-btn send"
        :disabled="sending || !input.trim()"
        title="Send"
        @click="send"
      >➤</button>
    </div>

    <div v-else class="call-panel">
      <button
        class="call-btn"
        :class="callState"
        :disabled="callState === 'thinking' || callState === 'speaking'"
        @click="toggleCall"
      >
        <svg viewBox="0 0 24 24" width="34" height="34" aria-hidden="true">
          <path
            fill="currentColor"
            d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"
          />
        </svg>
      </button>
      <div class="call-hint">{{ callLabel[callState] }}</div>
    </div>
  </div>
</template>
