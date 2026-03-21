<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

const initialAssistantMessage = {
  role: 'assistant',
  content:
    'Ask a question to the local Llama 3.2 backend. The backend keeps one live conversation in memory until you reset it.',
}

const messages = ref([initialAssistantMessage])
const draft = ref('')
const errorMessage = ref('')
const isLoading = ref(false)
const backendReady = ref(false)
const backendLabel = ref('Checking backend...')
const sessionLabel = ref('Conversation reset')
const messagesPanel = ref(null)
let healthTimer = null

const canSend = computed(() => draft.value.trim().length > 0 && !isLoading.value)

function describeSession(session) {
  if (!session?.initialized || session.turn_count === 0) {
    return 'Conversation reset'
  }

  const noun = session.turn_count === 1 ? 'turn' : 'turns'
  return `Memory active: ${session.turn_count} ${noun}`
}

async function readJson(response) {
  try {
    return await response.json()
  } catch (error) {
    return {}
  }
}

async function loadHealth() {
  try {
    const response = await fetch(`${apiBaseUrl}/api/health`)
    if (!response.ok) {
      throw new Error('The backend health check failed.')
    }

    const payload = await readJson(response)
    backendReady.value = payload.model_loaded
    backendLabel.value = payload.model_loaded
      ? `Model ready: ${payload.model}`
      : `Backend online: ${payload.model}`
  } catch (error) {
    backendReady.value = false
    backendLabel.value = 'Backend unavailable'
  }
}

async function loadSession() {
  try {
    const response = await fetch(`${apiBaseUrl}/api/session`)
    if (!response.ok) {
      throw new Error('The backend session check failed.')
    }

    const payload = await readJson(response)
    sessionLabel.value = describeSession(payload)

    if (payload.turn_count > 0 && messages.value.length === 1) {
      const noun = payload.turn_count === 1 ? 'turn' : 'turns'
      messages.value = [
        ...messages.value,
        {
          role: 'assistant',
          content: `The backend still remembers ${payload.turn_count} earlier ${noun}. Reset the conversation if you want a clean context.`,
        },
      ]
    }
  } catch (error) {
    sessionLabel.value = 'Session unavailable'
  }
}

async function resetConversation() {
  if (isLoading.value) {
    return
  }

  errorMessage.value = ''

  try {
    const response = await fetch(`${apiBaseUrl}/api/reset`, {
      method: 'POST',
    })
    const payload = await readJson(response)
    if (!response.ok) {
      throw new Error(payload.detail || 'The conversation reset failed.')
    }

    messages.value = [initialAssistantMessage]
    draft.value = ''
    sessionLabel.value = describeSession(payload)
  } catch (error) {
    errorMessage.value =
      error instanceof Error ? error.message : 'The conversation reset failed.'
  }

  await loadHealth()
  await nextTick()
  scrollToBottom()
}

function scrollToBottom() {
  if (!messagesPanel.value) {
    return
  }

  messagesPanel.value.scrollTop = messagesPanel.value.scrollHeight
}

function handleComposerKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void sendMessage()
  }
}

async function sendMessage() {
  const message = draft.value.trim()
  if (!message || isLoading.value) {
    return
  }

  messages.value = [...messages.value, { role: 'user', content: message }]
  draft.value = ''
  errorMessage.value = ''
  isLoading.value = true
  await nextTick()
  scrollToBottom()

  try {
    const response = await fetch(`${apiBaseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    })

    const payload = await readJson(response)
    if (!response.ok) {
      throw new Error(payload.detail || 'The assistant request failed.')
    }

    messages.value = [...messages.value, { role: 'assistant', content: payload.reply }]
    backendReady.value = true
    backendLabel.value = `Model ready: ${payload.model}`
    sessionLabel.value = describeSession(payload)
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'The assistant request failed.'
    errorMessage.value = message
    messages.value = [
      ...messages.value,
      { role: 'assistant', content: `I could not generate a reply.\n\n${message}` },
    ]
  } finally {
    isLoading.value = false
    await nextTick()
    scrollToBottom()
  }
}

onMounted(async () => {
  await Promise.all([loadHealth(), loadSession()])
  healthTimer = window.setInterval(() => {
    void loadHealth()
  }, 10000)
  scrollToBottom()
})

onUnmounted(() => {
  if (healthTimer !== null) {
    window.clearInterval(healthTimer)
  }
})
</script>

<template>
  <div class="shell">
    <header class="hero">
      <p class="eyebrow">Local GGUF Chat</p>
      <h1>Stateful Llama 3.2 Assistant</h1>
      <p class="lede">
        This interface keeps the visible transcript in the browser, while the backend keeps one
        live conversation in memory until you reset it.
      </p>
      <div class="hero__actions">
        <span class="status-pill" :data-ready="backendReady">
          {{ backendLabel }}
        </span>
        <span class="status-pill">
          {{ sessionLabel }}
        </span>
        <button class="secondary-button" type="button" @click="resetConversation">
          Reset conversation
        </button>
      </div>
    </header>

    <main class="chat-panel">
      <section ref="messagesPanel" class="messages" aria-live="polite">
        <article
          v-for="(message, index) in messages"
          :key="`${message.role}-${index}`"
          class="message"
          :data-role="message.role"
        >
          <p class="message__label">
            {{ message.role === 'user' ? 'You' : 'Assistant' }}
          </p>
          <p class="message__content">{{ message.content }}</p>
        </article>

        <article v-if="isLoading" class="message message--loading" data-role="assistant">
          <p class="message__label">Assistant</p>
          <p class="message__content">Generating a reply...</p>
        </article>
      </section>

      <form class="composer" @submit.prevent="sendMessage">
        <label class="composer__label" for="prompt">
          Continue the current conversation
        </label>
        <textarea
          id="prompt"
          v-model="draft"
          class="composer__input"
          rows="4"
          placeholder="Ask a question..."
          @keydown="handleComposerKeydown"
        />

        <div class="composer__footer">
          <p v-if="errorMessage" class="composer__error">{{ errorMessage }}</p>
          <p v-else class="composer__hint">
            Enter submits. Shift+Enter adds a new line.
          </p>
          <button class="primary-button" type="submit" :disabled="!canSend">
            {{ isLoading ? 'Working...' : 'Send' }}
          </button>
        </div>
      </form>
    </main>
  </div>
</template>
