<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'

// ---- top-level navigation ----
const view = ref('home')          // 'home' (character list) | 'chat'
const characters = ref([])        // [{ id, name, gender, voice, intelligence, provider }]
const activeChar = ref(null)      // the character whose thread is open in chat view

// ---- chat state (scoped to activeChar) ----
const messages = ref([])          // { role, content }
const input = ref('')
const sending = ref(false)
const status = ref('')
const statusError = ref(false)
const recording = ref(false)
const selected = ref(new Set())   // indices (into messages) ticked for deletion
const deleting = ref(false)
const selectMode = ref(false)     // when on, per-message deletion checkboxes are shown

// Display-only cleanup for assistant replies: drop newlines, quote marks
// ("/'/curly) and asterisks, then collapse the whitespace they leave behind.
function cleanAssistantText(text) {
  return text
    .replace(/[\r\n]+/g, ' ')
    .replace(/["'“”‘’*]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

// Display-only expansion: assistant replies are shown as one bubble per
// sentence (split on full stops). The backend reply stays a single message;
// each bubble keeps its source index (`i`) so selection/delete still maps to
// the real `messages` entry.
const displayMessages = computed(() => {
  const out = []
  messages.value.forEach((m, i) => {
    if (m.role === 'assistant') {
      const cleaned = cleanAssistantText(m.content)
      const parts = cleaned
        .split('.')
        .map((s) => s.trim())
        .filter(Boolean)
      const bubbles = parts.length ? parts : [cleaned]
      bubbles.forEach((content, j) => {
        out.push({ role: m.role, content, i, first: j === 0 })
      })
    } else {
      out.push({ role: m.role, content: m.content, i, first: true })
    }
  })
  return out
})

const mode = ref('text')          // 'text' (no voice) | 'call' (voice-only)
const callState = ref('idle')     // 'idle' | 'recording' | 'thinking' | 'speaking'
const menuOpen = ref(false)       // three-dot overflow menu
const contactOpen = ref(false)    // WhatsApp-style "View Contact" side pane
const confirmDelete = ref(false)  // delete-character confirmation modal
const deletingChar = ref(false)

// ---- character creation flow ----
const createOpen = ref(false)
const creating = ref(false)
const createError = ref('')
const newName = ref('')
const newGender = ref('')         // '' until picked → gates the voice step
const newVoice = ref('')
const newIntelligence = ref(2)    // slider 0..2 → low/medium/high; default max (Smart)
const newPersonaCore = ref('')    // short identity blurb → always in the system prompt
const newBio = ref('')            // long-form lore → indexed, surfaced on demand
const genderVoices = ref([])      // voices for the picked gender

// Model download state, shown in the creation modal when the chosen
// intelligence's Ollama model isn't present yet.
const downloading = ref(false)
const dlText = ref('')            // human-readable progress line
const dlPercent = ref(null)       // 0..100, or null for an indeterminate bar

const INTELLIGENCE_LEVELS = ['low', 'medium', 'high']
const INTELLIGENCE_LABELS = ['Quick (1.7B)', 'Balanced (3B)', 'Smart (8B)']

const messagesEl = ref(null)
let mediaRecorder = null
let audioChunks = []
let currentAudio = null

// Pretty one-liner for a character row / contact pane subtitle.
function charSubtitle(c) {
  if (!c) return ''
  const lvl = { low: 'Quick', medium: 'Balanced', high: 'Smart' }[c.intelligence] || c.intelligence
  const v = voiceLabel(c.voice)
  return `${lvl}${v ? ' · ' + v : ''}`
}

const voicesById = ref({})        // id -> label, for resolving a character's voice name
function voiceLabel(id) {
  return voicesById.value[id] || id?.replace(/^kokoro:/, '') || ''
}

function setStatus(text, isError = false) {
  status.value = text
  statusError.value = isError
}

async function scrollToBottom() {
  await nextTick()
  if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight
}

// ---- characters ----
async function loadCharacters() {
  try {
    const res = await fetch('/api/characters')
    const data = await res.json()
    characters.value = data.characters || []
  } catch {
    setStatus('Could not reach backend on :8000', true)
  }
}

function openChat(c) {
  activeChar.value = c
  view.value = 'chat'
  mode.value = 'text'
  contactOpen.value = false
  menuOpen.value = false
  selectMode.value = false
  clearSelection()
  loadHistory()
}

function goHome() {
  view.value = 'home'
  activeChar.value = null
  contactOpen.value = false
  menuOpen.value = false
  if (currentAudio) { currentAudio.pause(); currentAudio = null }
}

// ---- creation flow ----
function openCreate() {
  newName.value = ''
  newGender.value = ''
  newVoice.value = ''
  newIntelligence.value = 2  // default to max (Smart / 8B)
  newPersonaCore.value = ''
  newBio.value = ''
  genderVoices.value = []
  createError.value = ''
  downloading.value = false
  dlText.value = ''
  dlPercent.value = null
  createOpen.value = true
}

// Maps the intelligence slider level → the backend chat provider id (mirrors
// INTELLIGENCE_PROVIDER in characters.py). The model/status + model/pull
// endpoints key off this provider.
const INTELLIGENCE_PROVIDER = { low: 'ollama-1b', medium: 'ollama-3b', high: 'ollama' }

// Stream the model pull for `provider`, updating the progress bar. Resolves when
// the model is fully present; rejects on error. Uses fetch + a streamed reader
// rather than EventSource since the pull endpoint is POST.
async function downloadModel(provider) {
  downloading.value = true
  dlText.value = 'Preparing download…'
  dlPercent.value = null
  try {
    const res = await fetch('/api/model/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chatProvider: provider }),
    })
    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      // SSE frames are separated by a blank line.
      const frames = buf.split('\n\n')
      buf = frames.pop() // keep the trailing partial frame
      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data:'))
        if (!line) continue
        let evt
        try {
          evt = JSON.parse(line.slice(5).trim())
        } catch {
          continue
        }
        if (evt.error) throw new Error(evt.error)
        if (evt.completed != null && evt.total) {
          dlPercent.value = Math.floor((evt.completed / evt.total) * 100)
          const mb = (n) => (n / 1e6).toFixed(0)
          dlText.value = `Downloading… ${mb(evt.completed)} / ${mb(evt.total)} MB`
        } else if (evt.status) {
          dlText.value = evt.status
          // Status-only lines (verifying, manifest, etc.) → indeterminate bar.
          if (evt.completed == null) dlPercent.value = null
        }
        if (evt.done) {
          dlPercent.value = 100
          dlText.value = 'Download complete'
        }
      }
    }
  } finally {
    downloading.value = false
  }
}

async function pickGender(g) {
  newGender.value = g
  newVoice.value = ''
  try {
    const res = await fetch(`/api/voices?gender=${encodeURIComponent(g)}`)
    const data = await res.json()
    genderVoices.value = data.voices || []
    // Default to the first matching voice so the field is never empty.
    newVoice.value = genderVoices.value[0]?.id || ''
  } catch {
    genderVoices.value = []
  }
}

const canCreate = computed(
  () => newName.value.trim() && newGender.value && newVoice.value,
)

async function submitCreate() {
  if (!canCreate.value || creating.value || downloading.value) return
  creating.value = true
  createError.value = ''
  try {
    // Ensure the model for the chosen intelligence is downloaded first; if not,
    // pull it with a progress bar before creating the character.
    const provider = INTELLIGENCE_PROVIDER[INTELLIGENCE_LEVELS[newIntelligence.value]]
    const statusRes = await fetch(`/api/model/status?provider=${encodeURIComponent(provider)}`)
    if (!statusRes.ok) {
      const err = await statusRes.json().catch(() => ({}))
      throw new Error(err.detail || `Could not check model (HTTP ${statusRes.status})`)
    }
    const { present } = await statusRes.json()
    if (!present) await downloadModel(provider)

    const res = await fetch('/api/characters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: newName.value.trim(),
        gender: newGender.value,
        voice: newVoice.value,
        intelligence: INTELLIGENCE_LEVELS[newIntelligence.value],
        persona_core: newPersonaCore.value.trim(),
        bio: newBio.value.trim(),
      }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const char = await res.json()
    createOpen.value = false
    await loadCharacters()
    openChat(char) // jump straight into the new thread
  } catch (e) {
    createError.value = e.message
  } finally {
    creating.value = false
  }
}

async function deleteCharacter() {
  if (!activeChar.value || deletingChar.value) return
  deletingChar.value = true
  try {
    await fetch(`/api/characters/${encodeURIComponent(activeChar.value.id)}`, {
      method: 'DELETE',
    })
    await loadCharacters()
    confirmDelete.value = false
    goHome()
  } finally {
    deletingChar.value = false
  }
}

// ---- chat (scoped to activeChar.id) ----
async function loadHistory() {
  if (!activeChar.value) return
  try {
    const res = await fetch(`/api/history?character=${encodeURIComponent(activeChar.value.id)}`)
    const data = await res.json()
    messages.value = data.messages || []
    scrollToBottom()
  } catch {
    setStatus('Could not load history', true)
  }
}

async function send() {
  const text = input.value.trim()
  if (!text || sending.value || !activeChar.value) return
  input.value = ''
  messages.value.push({ role: 'user', content: text })
  clearSelection()
  sending.value = true
  setStatus('')
  scrollToBottom()

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, character: activeChar.value.id }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const data = await res.json()
    messages.value.push({ role: 'assistant', content: data.reply })
    scrollToBottom()
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
  if (deletingChar.value || !activeChar.value) return
  try {
    await fetch(`/api/reset?character=${encodeURIComponent(activeChar.value.id)}`, {
      method: 'POST',
    })
    messages.value = []
    selected.value = new Set()
    setStatus('Conversation cleared.')
  } finally {
    confirmDelete.value = false
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

function toggleSelectMode() {
  selectMode.value = !selectMode.value
  if (!selectMode.value) clearSelection()
}

async function deleteSelected() {
  if (selected.value.size === 0 || deleting.value || !activeChar.value) return
  const indices = [...selected.value]
  deleting.value = true
  setStatus('Forgetting selected messages…')
  try {
    const res = await fetch('/api/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character: activeChar.value.id, indices }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const data = await res.json()
    messages.value = data.messages || []
    selected.value = new Set()
    selectMode.value = false
    const n = data.removed_facts?.length || 0
    setStatus(`Forgot ${data.deleted} message(s)${n ? `, pruned ${n} fact(s)` : ''}.`)
  } catch (e) {
    setStatus(`Delete failed: ${e.message}`, true)
  } finally {
    deleting.value = false
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
  if (callState.value !== 'idle') return
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

async function runCallTurn(blob) {
  if (!activeChar.value) return
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
      body: JSON.stringify({ message: spoken, character: activeChar.value.id }),
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
        body: JSON.stringify({ message: text, character: activeChar.value?.id }),
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

// Build the id->label voice map once (for showing a character's voice name).
async function loadVoiceLabels() {
  try {
    const res = await fetch('/api/voices')
    const data = await res.json()
    const map = {}
    for (const v of data.voices || []) map[v.id] = v.label
    voicesById.value = map
  } catch {
    // non-fatal: voice names just fall back to the raw id
  }
}

function onDocClick() {
  menuOpen.value = false
}

onMounted(async () => {
  document.addEventListener('click', onDocClick)
  await Promise.all([loadVoiceLabels(), loadCharacters()])
})

onUnmounted(() => document.removeEventListener('click', onDocClick))
</script>

<template>
  <div class="app">
    <!-- ================= HOME: character list ================= -->
    <template v-if="view === 'home'">
      <header class="header">
        <div class="brand">
          <span class="brand-main">WhatsApp</span>
          <span class="brand-sub">Uncensored</span>
        </div>
        <div class="header-actions">
          <button class="wa-icon" title="New character" @click="openCreate">
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <path fill="currentColor" d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
            </svg>
          </button>
        </div>
      </header>

      <div class="char-list">
        <div v-if="characters.length === 0" class="empty">
          No characters yet.<br />Tap ＋ to create one.
        </div>
        <button
          v-for="c in characters"
          :key="c.id"
          class="char-row"
          @click="openChat(c)"
        >
          <div class="char-avatar">
            <svg viewBox="0 0 212 212" width="48" height="48" aria-hidden="true">
              <path
                fill="currentColor"
                d="M106 0C47.5 0 0 47.5 0 106s47.5 106 106 106 106-47.5 106-106S164.5 0 106 0zm0 53c16.6 0 30 13.4 30 30s-13.4 30-30 30-30-13.4-30-30 13.4-30 30-30zm0 138c-25 0-47.2-12.1-61-30.7 9.8-17.4 28.4-29.3 49.8-30.2a44 44 0 0 0 22.4 0c21.4.9 40 12.8 49.8 30.2-13.8 18.6-36 30.7-61 30.7z"
              />
            </svg>
          </div>
          <div class="char-meta">
            <div class="char-name">{{ c.name }}</div>
            <div class="char-sub">{{ charSubtitle(c) }}</div>
          </div>
        </button>
      </div>
    </template>

    <!-- ================= CHAT: one character's thread ================= -->
    <template v-else>
      <header class="header">
        <div class="header-left">
          <button class="wa-icon" title="Back" @click="goHome">
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <path
                fill="none" stroke="currentColor" stroke-width="2"
                stroke-linecap="round" d="M19 12H5m0 0 6-6m-6 6 6 6"
              />
            </svg>
          </button>
          <button class="header-title" @click="contactOpen = true">
            {{ activeChar?.name }}
          </button>
        </div>
        <div class="header-actions">
          <button
            class="wa-icon" :class="{ active: mode === 'text' }"
            title="Chat" @click="mode = 'text'"
          >
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <path fill="currentColor" d="M12 3C6.48 3 2 6.86 2 11.62c0 2.7 1.45 5.11 3.72 6.7-.16 1.2-.66 2.3-1.45 3.18-.18.2-.05.52.22.5 1.83-.13 3.5-.78 4.86-1.78.84.2 1.73.3 2.65.3 5.52 0 10-3.86 10-8.6S17.52 3 12 3zm-4 9.75a1.25 1.25 0 110-2.5 1.25 1.25 0 010 2.5zm4 0a1.25 1.25 0 110-2.5 1.25 1.25 0 010 2.5zm4 0a1.25 1.25 0 110-2.5 1.25 1.25 0 010 2.5z" />
            </svg>
          </button>
          <button
            class="wa-icon" :class="{ active: mode === 'call' }"
            title="Call" @click="mode = 'call'"
          >
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <path fill="currentColor" d="M6.54 5c.06.89.21 1.76.45 2.59l-1.2 1.2c-.41-1.2-.67-2.47-.76-3.79h1.51M16.4 17.02c.85.24 1.72.39 2.6.45v1.49c-1.32-.09-2.59-.35-3.8-.75l1.2-1.19M7.5 3H4c-.55 0-1 .45-1 1 0 9.39 7.61 17 17 17 .55 0 1-.45 1-1v-3.49c0-.55-.45-1-1-1-1.24 0-2.45-.2-3.57-.57a.84.84 0 00-.31-.05c-.26 0-.51.1-.71.29l-2.2 2.2a15.07 15.07 0 01-6.59-6.59l2.2-2.2c.28-.28.36-.67.25-1.02A11.36 11.36 0 018.5 4c0-.55-.45-1-1-1z" />
            </svg>
          </button>
          <div class="menu-wrap">
            <button class="menu-btn" title="More options" @click.stop="menuOpen = !menuOpen">
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

      <!-- delete-character confirmation -->
      <div v-if="confirmDelete" class="modal-overlay" @click="confirmDelete = false">
        <div class="modal" @click.stop>
          <h2 class="modal-title">Delete {{ activeChar?.name }}?</h2>
          <p class="modal-body">
            This permanently erases this character, the whole conversation, the
            retrieval index, and all long-term memory facts. This cannot be undone.
          </p>
          <div class="modal-actions">
            <button class="modal-cancel" @click="confirmDelete = false">Cancel</button>
            <button class="modal-nuke" :disabled="deletingChar" @click="deleteCharacter">
              {{ deletingChar ? 'Deleting…' : '🗑 Delete' }}
            </button>
          </div>
        </div>
      </div>

      <!-- read-only contact pane -->
      <transition name="contact-slide">
        <aside v-if="contactOpen" class="contact-pane" @click.stop>
          <div class="contact-head">
            <button class="contact-close" title="Close" @click="contactOpen = false">
              <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
                <path fill="none" stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" d="M19 12H5m0 0 6-6m-6 6 6 6" />
              </svg>
            </button>
            <span class="contact-head-title">Contact info</span>
          </div>
          <div class="contact-body">
            <div class="contact-id">
              <div class="contact-avatar">
                <svg viewBox="0 0 212 212" width="96" height="96" aria-hidden="true">
                  <path fill="currentColor" d="M106 0C47.5 0 0 47.5 0 106s47.5 106 106 106 106-47.5 106-106S164.5 0 106 0zm0 53c16.6 0 30 13.4 30 30s-13.4 30-30 30-30-13.4-30-30 13.4-30 30-30zm0 138c-25 0-47.2-12.1-61-30.7 9.8-17.4 28.4-29.3 49.8-30.2a44 44 0 0 0 22.4 0c21.4.9 40 12.8 49.8 30.2-13.8 18.6-36 30.7-61 30.7z" />
                </svg>
              </div>
              <div class="contact-name">{{ activeChar?.name }}</div>
              <div class="contact-sub">Local chat · voice · memory</div>
            </div>

            <!-- character settings are fixed at creation: shown read-only -->
            <div class="contact-section">
              <div class="info-row"><span class="info-key">Gender</span><span class="info-val">{{ activeChar?.gender }}</span></div>
              <div class="info-row"><span class="info-key">Voice</span><span class="info-val">{{ voiceLabel(activeChar?.voice) }}</span></div>
              <div class="info-row"><span class="info-key">Intelligence</span><span class="info-val">{{ { low: 'Quick (1.7B)', medium: 'Balanced (3B)', high: 'Smart (8B)' }[activeChar?.intelligence] }}</span></div>
              <p class="info-note">Settings are locked after creation.</p>
            </div>

            <div v-if="activeChar?.persona_core || activeChar?.bio" class="contact-section">
              <div v-if="activeChar?.persona_core" class="info-block">
                <span class="info-key">Persona</span>
                <p class="info-text">{{ activeChar.persona_core }}</p>
              </div>
              <div v-if="activeChar?.bio" class="info-block">
                <span class="info-key">Bio</span>
                <p class="info-text bio-text">{{ activeChar.bio }}</p>
              </div>
            </div>

            <div class="contact-section">
              <button class="contact-nuke" @click="confirmDelete = true">
                <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
                  <path fill="none" stroke="currentColor" stroke-width="1.8"
                    stroke-linecap="round" stroke-linejoin="round"
                    d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m2 0v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V7M10 11v6M14 11v6" />
                </svg>
                <span>Delete character</span>
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
          v-for="(m, di) in displayMessages"
          :key="di"
          class="msg"
          :class="[m.role, { picked: selected.has(m.i) }]"
        >
          <input
            v-if="selectMode && m.first"
            type="checkbox"
            class="msg-check"
            title="Select to forget"
            :checked="selected.has(m.i)"
            @change="toggleSelect(m.i)"
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

      <div v-if="status" class="status" :class="{ error: statusError }">{{ status }}</div>

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
            <path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z" />
          </svg>
        </button>
        <div class="call-hint">{{ callLabel[callState] }}</div>
      </div>
    </template>

    <!-- ================= CREATE CHARACTER modal ================= -->
    <div v-if="createOpen" class="modal-overlay" @click="createOpen = false">
      <div class="modal create-modal" @click.stop>
        <h2 class="modal-title">New character</h2>

        <label class="field-label">Name</label>
        <input v-model="newName" class="text-input" placeholder="e.g. Luna" maxlength="40" />

        <label class="field-label">Gender</label>
        <div class="seg">
          <button
            class="seg-btn" :class="{ on: newGender === 'female' }"
            @click="pickGender('female')"
          >Female</button>
          <button
            class="seg-btn" :class="{ on: newGender === 'male' }"
            @click="pickGender('male')"
          >Male</button>
        </div>

        <!-- voice only appears once a gender is picked; options match the gender -->
        <template v-if="newGender">
          <label class="field-label">Voice</label>
          <select v-model="newVoice" class="voice-select menu-select">
            <option v-for="v in genderVoices" :key="v.id" :value="v.id">{{ v.label }}</option>
          </select>
        </template>

        <label class="field-label">Intelligence</label>
        <input
          type="range" min="0" max="2" step="1"
          v-model.number="newIntelligence" class="slider"
        />
        <div class="slider-ticks">
          <span :class="{ on: newIntelligence === 0 }">Quick</span>
          <span :class="{ on: newIntelligence === 1 }">Balanced</span>
          <span :class="{ on: newIntelligence === 2 }">Smart</span>
        </div>
        <div class="slider-current">{{ INTELLIGENCE_LABELS[newIntelligence] }}</div>

        <label class="field-label">Persona core</label>
        <input
          v-model="newPersonaCore" class="text-input"
          placeholder="One line: who they are (kept in mind every reply)"
          maxlength="280"
        />
        <p class="field-hint">A short identity blurb. Always present, so keep it tight.</p>

        <label class="field-label">Bio</label>
        <textarea
          v-model="newBio" class="text-input bio-input" rows="6"
          placeholder="Backstory, relationships, quirks, lore… Separate topics into paragraphs."
        ></textarea>
        <p class="field-hint">
          Everything else about them. Surfaced into the chat only when relevant,
          so length is cheap — write as much as you like.
        </p>

        <!-- Model download progress (only while pulling a missing model) -->
        <div v-if="downloading" class="model-dl">
          <div class="model-status">
            <span class="model-status-text">{{ dlText }}</span>
            <span v-if="dlPercent != null">{{ dlPercent }}%</span>
          </div>
          <div class="model-bar">
            <div
              class="model-bar-fill"
              :class="{ indeterminate: dlPercent == null }"
              :style="dlPercent != null ? { width: dlPercent + '%' } : {}"
            ></div>
          </div>
          <p class="field-hint">Downloading this character's model — this happens once per model.</p>
        </div>

        <p v-if="createError" class="create-error">{{ createError }}</p>

        <div class="modal-actions">
          <button class="modal-cancel" :disabled="creating || downloading" @click="createOpen = false">
            Cancel
          </button>
          <button
            class="modal-create"
            :disabled="!canCreate || creating || downloading"
            @click="submitCreate"
          >
            {{ downloading ? 'Downloading…' : creating ? 'Creating…' : 'Create' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
