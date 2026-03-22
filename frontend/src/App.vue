<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')
const timestampFormatter = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
})

const conversations = ref([])
const activeConversation = ref(null)
const activeThreadId = ref(null)
const draft = ref('')
const errorMessage = ref('')
const controlMessage = ref('')
const controlMessageTone = ref('muted')
const isLoadingConversation = ref(false)
const isCreatingConversation = ref(false)
const isDeletingAllMemory = ref(false)
const isSending = ref(false)
const backendReady = ref(false)
const backendLabel = ref('Checking backend...')
const memoryLabel = ref('No stored conversations')
const pendingUserMessage = ref('')
const finalizedPanel = ref(null)
const tracePanel = ref(null)
let healthTimer = null

const canSend = computed(() => {
  return draft.value.trim().length > 0 && !isSending.value && !isDeletingAllMemory.value
})

const activeConversationTitle = computed(() => {
  return activeConversation.value?.title || 'Untitled conversation'
})

const finalizedStreamMessages = computed(() => {
  const baseMessages = activeConversation.value?.finalized_messages
    ? [...activeConversation.value.finalized_messages]
    : []

  if (!pendingUserMessage.value) {
    return baseMessages
  }

  const pendingTimestamp = new Date().toISOString()
  baseMessages.push({
    id: 'pending-user',
    thread_id: activeConversation.value?.finalized_thread_id || 'pending',
    role: 'user',
    kind: 'pending_user_message',
    content: pendingUserMessage.value,
    turn_index: baseMessages.length,
    created_at: pendingTimestamp,
    pending: true,
  })
  baseMessages.push({
    id: 'pending-assistant',
    thread_id: activeConversation.value?.finalized_thread_id || 'pending',
    role: 'assistant',
    kind: 'pending_assistant_response',
    content: 'Running draft, memory analysis, and final synthesis...',
    turn_index: baseMessages.length,
    created_at: pendingTimestamp,
    pending: true,
  })
  return baseMessages
})

const traceStreamMessages = computed(() => {
  const baseMessages = activeConversation.value?.trace_messages
    ? [...activeConversation.value.trace_messages]
    : []

  if (!pendingUserMessage.value) {
    return baseMessages
  }

  const pendingTimestamp = new Date().toISOString()
  baseMessages.push(
    {
      id: 'pending-trace-user',
      thread_id: activeConversation.value?.trace_thread_id || 'pending',
      role: 'user',
      kind: 'foreground_user_message',
      content: pendingUserMessage.value,
      turn_index: baseMessages.length,
      created_at: pendingTimestamp,
      pending: true,
    },
    {
      id: 'pending-trace-memory',
      thread_id: activeConversation.value?.trace_thread_id || 'pending',
      role: 'system',
      kind: 'retrieved_memory_block',
      content: 'Retrieving relevant memory...',
      turn_index: baseMessages.length + 1,
      created_at: pendingTimestamp,
      pending: true,
    },
    {
      id: 'pending-trace-draft',
      thread_id: activeConversation.value?.trace_thread_id || 'pending',
      role: 'assistant',
      kind: 'draft_pass_output',
      content: 'Draft pass running...',
      turn_index: baseMessages.length + 2,
      created_at: pendingTimestamp,
      pending: true,
    },
    {
      id: 'pending-trace-analysis',
      thread_id: activeConversation.value?.trace_thread_id || 'pending',
      role: 'assistant',
      kind: 'memory_analysis_output',
      content: 'Memory analysis running...',
      turn_index: baseMessages.length + 3,
      created_at: pendingTimestamp,
      pending: true,
    },
    {
      id: 'pending-trace-final',
      thread_id: activeConversation.value?.trace_thread_id || 'pending',
      role: 'assistant',
      kind: 'final_synthesis_output',
      content: 'Final synthesis running...',
      turn_index: baseMessages.length + 4,
      created_at: pendingTimestamp,
      pending: true,
    }
  )

  return baseMessages
})

function setControlMessage(message, tone = 'muted') {
  controlMessage.value = message
  controlMessageTone.value = tone
}

function describeConversationCount(count) {
  if (count === 0) {
    return 'No stored conversations'
  }

  const noun = count === 1 ? 'conversation' : 'conversations'
  return `${count} stored ${noun}`
}

function describeMessageCount(count) {
  if (count === 0) {
    return 'No turns yet'
  }

  const noun = count === 1 ? 'message' : 'messages'
  return `${count} ${noun}`
}

function formatTimestamp(value) {
  if (!value) {
    return 'Pending'
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return timestampFormatter.format(date)
}

function formatTraceKind(kind) {
  return kind.replace(/_/g, ' ')
}

function formatRole(role) {
  if (role === 'user') {
    return 'User'
  }
  if (role === 'assistant') {
    return 'Assistant'
  }
  if (role === 'system') {
    return 'System'
  }
  return role
}

function scrollPanel(element) {
  if (!element) {
    return
  }

  element.scrollTop = element.scrollHeight
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
    const payload = await readJson(response)
    const hasHealthPayload =
      typeof payload?.status === 'string' &&
      typeof payload?.sqlite_ready === 'boolean' &&
      typeof payload?.qdrant_ready === 'boolean'

    if (!response.ok && !hasHealthPayload) {
      throw new Error(payload.detail || 'The backend health check failed.')
    }

    backendReady.value =
      Boolean(payload.model_loaded) &&
      Boolean(payload.sqlite_ready) &&
      Boolean(payload.qdrant_ready)

    const storageIssues = []
    if (!payload.sqlite_ready) {
      storageIssues.push(payload.sqlite_error || 'SQLite unavailable')
    }
    if (!payload.qdrant_ready) {
      storageIssues.push(payload.qdrant_error || 'Qdrant unavailable')
    }

    if (storageIssues.length > 0) {
      backendLabel.value = `Storage degraded: ${storageIssues.join(' | ')}`
      return
    }

    backendLabel.value = payload.model_loaded
      ? `Model ready: ${payload.model}`
      : `Backend online: ${payload.model}`
  } catch (error) {
    backendReady.value = false
    backendLabel.value = 'Backend unavailable'
  }
}

async function loadConversation(threadId) {
  if (!threadId) {
    activeConversation.value = null
    activeThreadId.value = null
    return
  }

  isLoadingConversation.value = true
  try {
    const response = await fetch(`${apiBaseUrl}/api/threads/${encodeURIComponent(threadId)}`)
    const payload = await readJson(response)
    if (!response.ok) {
      throw new Error(payload.detail || 'Failed to load the selected conversation.')
    }

    activeConversation.value = payload
    activeThreadId.value = payload.finalized_thread_id
  } catch (error) {
    activeConversation.value = null
    activeThreadId.value = null
    setControlMessage(
      error instanceof Error ? error.message : 'Failed to load the selected conversation.',
      'error'
    )
  } finally {
    isLoadingConversation.value = false
    await nextTick()
    scrollPanel(finalizedPanel.value)
    scrollPanel(tracePanel.value)
  }
}

async function loadConversations(options = {}) {
  const selectedThreadId = options.selectedThreadId ?? activeThreadId.value
  const loadSelection = options.loadSelection !== false

  try {
    const response = await fetch(`${apiBaseUrl}/api/threads`)
    const payload = await readJson(response)
    if (!response.ok) {
      throw new Error(payload.detail || 'Failed to load conversations.')
    }

    conversations.value = Array.isArray(payload) ? payload : []
    memoryLabel.value = describeConversationCount(conversations.value.length)

    if (!loadSelection) {
      if (
        selectedThreadId &&
        !conversations.value.some(
          (conversation) => conversation.finalized_thread_id === selectedThreadId
        )
      ) {
        activeConversation.value = null
        activeThreadId.value = null
      }
      return
    }

    if (
      selectedThreadId &&
      conversations.value.some(
        (conversation) => conversation.finalized_thread_id === selectedThreadId
      )
    ) {
      await loadConversation(selectedThreadId)
      return
    }

    if (conversations.value.length > 0) {
      await loadConversation(conversations.value[0].finalized_thread_id)
      return
    }

    activeConversation.value = null
    activeThreadId.value = null
  } catch (error) {
    setControlMessage(
      error instanceof Error ? error.message : 'Failed to load conversations.',
      'error'
    )
  }
}

async function createConversation() {
  if (isCreatingConversation.value || isSending.value || isDeletingAllMemory.value) {
    return
  }

  isCreatingConversation.value = true
  errorMessage.value = ''

  try {
    const response = await fetch(`${apiBaseUrl}/api/threads`, {
      method: 'POST',
    })
    const payload = await readJson(response)
    if (!response.ok) {
      throw new Error(payload.detail || 'Failed to create a new conversation.')
    }

    activeConversation.value = payload
    activeThreadId.value = payload.finalized_thread_id
    draft.value = ''
    setControlMessage('Created a new blank conversation.', 'success')
    await loadConversations({
      selectedThreadId: payload.finalized_thread_id,
      loadSelection: false,
    })
  } catch (error) {
    setControlMessage(
      error instanceof Error ? error.message : 'Failed to create a new conversation.',
      'error'
    )
  } finally {
    isCreatingConversation.value = false
    await nextTick()
    scrollPanel(finalizedPanel.value)
    scrollPanel(tracePanel.value)
  }
}

async function deleteAllMemory() {
  if (isDeletingAllMemory.value || isSending.value) {
    return
  }

  const confirmed = window.confirm(
    'Delete every stored conversation, trace thread, and Qdrant memory point? This cannot be undone.'
  )
  if (!confirmed) {
    return
  }

  isDeletingAllMemory.value = true
  errorMessage.value = ''

  try {
    const response = await fetch(`${apiBaseUrl}/api/delete-all-memory`, {
      method: 'POST',
    })
    const payload = await readJson(response)
    if (!response.ok) {
      throw new Error(payload.detail || 'Failed to delete all stored memory.')
    }

    conversations.value = []
    activeConversation.value = null
    activeThreadId.value = null
    draft.value = ''
    pendingUserMessage.value = ''
    memoryLabel.value = describeConversationCount(0)
    setControlMessage(
      `Deleted ${payload.deleted_conversations} conversations and ${payload.deleted_messages} messages.`,
      'success'
    )
    await loadHealth()
  } catch (error) {
    setControlMessage(
      error instanceof Error ? error.message : 'Failed to delete all stored memory.',
      'error'
    )
  } finally {
    isDeletingAllMemory.value = false
  }
}

function handleComposerKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void sendMessage()
  }
}

async function sendMessage() {
  const message = draft.value.trim()
  if (!message || isSending.value || isDeletingAllMemory.value) {
    return
  }

  draft.value = ''
  errorMessage.value = ''
  setControlMessage('', 'muted')
  isSending.value = true
  pendingUserMessage.value = message

  await nextTick()
  scrollPanel(finalizedPanel.value)
  scrollPanel(tracePanel.value)

  try {
    const response = await fetch(`${apiBaseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        thread_id: activeConversation.value?.finalized_thread_id || null,
      }),
    })

    const payload = await readJson(response)
    if (!response.ok) {
      throw new Error(payload.detail || 'The assistant request failed.')
    }

    await Promise.all([
      loadConversation(payload.thread_id),
      loadConversations({
        selectedThreadId: payload.thread_id,
        loadSelection: false,
      }),
      loadHealth(),
    ])
  } catch (error) {
    draft.value = message
    errorMessage.value =
      error instanceof Error ? error.message : 'The assistant request failed.'
  } finally {
    pendingUserMessage.value = ''
    isSending.value = false
    await nextTick()
    scrollPanel(finalizedPanel.value)
    scrollPanel(tracePanel.value)
  }
}

watch(
  finalizedStreamMessages,
  async () => {
    await nextTick()
    scrollPanel(finalizedPanel.value)
  },
  { deep: true }
)

watch(
  traceStreamMessages,
  async () => {
    await nextTick()
    scrollPanel(tracePanel.value)
  },
  { deep: true }
)

onMounted(async () => {
  await loadHealth()
  await loadConversations()
  healthTimer = window.setInterval(() => {
    void loadHealth()
  }, 10000)
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
      <p class="eyebrow">Local Retrieval Chat</p>
      <h1>Finalized Chat, Trace, and Memory</h1>
      <p class="lede">
        Every reply is rebuilt from stored chat history plus retrieved memory. The left column
        shows the clean finalized conversation, the middle column shows request-time trace
        activity, and the right column manages previous conversations and hard deletes.
      </p>
      <div class="hero__actions">
        <span class="status-pill" :data-ready="backendReady">
          {{ backendLabel }}
        </span>
        <span class="status-pill">
          {{ memoryLabel }}
        </span>
      </div>
    </header>

    <main class="workspace">
      <section class="panel panel--finalized">
        <header class="panel__header">
          <div>
            <p class="panel__eyebrow">Finalized thread</p>
            <h2>{{ activeConversationTitle }}</h2>
          </div>
          <p class="panel__meta">
            {{
              activeConversation
                ? `Updated ${formatTimestamp(activeConversation.updated_at)}`
                : 'No conversation selected'
            }}
          </p>
        </header>

        <section ref="finalizedPanel" class="panel__stream" aria-live="polite">
          <div
            v-if="!finalizedStreamMessages.length"
            class="empty-state"
          >
            <p class="empty-state__title">No finalized messages yet</p>
            <p class="empty-state__copy">
              Start a new conversation from the sidebar or send a message here.
            </p>
          </div>

          <article
            v-for="message in finalizedStreamMessages"
            :key="message.id"
            class="message-card"
            :data-role="message.role"
            :data-pending="message.pending ? 'true' : 'false'"
          >
            <div class="message-card__meta">
              <p class="message-card__label">
                {{ message.role === 'user' ? 'You' : 'Assistant' }}
              </p>
              <time class="message-card__time">
                {{ formatTimestamp(message.created_at) }}
              </time>
            </div>
            <p class="message-card__content">{{ message.content }}</p>
          </article>
        </section>

        <form class="composer" @submit.prevent="sendMessage">
          <label class="composer__label" for="prompt">
            {{ activeConversation ? 'Reply in the selected conversation' : 'Start a new conversation' }}
          </label>
          <textarea
            id="prompt"
            v-model="draft"
            class="composer__input"
            rows="5"
            placeholder="Ask a question..."
            @keydown="handleComposerKeydown"
          />

          <div class="composer__footer">
            <p v-if="errorMessage" class="composer__error">{{ errorMessage }}</p>
            <p v-else class="composer__hint">Enter submits. Shift+Enter adds a new line.</p>
            <button class="primary-button" type="submit" :disabled="!canSend">
              {{ isSending ? 'Running passes...' : 'Send' }}
            </button>
          </div>
        </form>
      </section>

      <section class="panel panel--trace">
        <header class="panel__header">
          <div>
            <p class="panel__eyebrow">Trace thread</p>
            <h2>Request activity</h2>
          </div>
          <p class="panel__meta">
            {{ isLoadingConversation ? 'Loading trace...' : 'Retrieved memory and pass outputs' }}
          </p>
        </header>

        <section ref="tracePanel" class="panel__stream panel__stream--trace" aria-live="polite">
          <div v-if="!traceStreamMessages.length" class="empty-state">
            <p class="empty-state__title">No trace activity yet</p>
            <p class="empty-state__copy">
              Retrieved memory, draft output, analysis output, and synthesis output appear here
              after a successful request.
            </p>
          </div>

          <article
            v-for="message in traceStreamMessages"
            :key="message.id"
            class="trace-card"
            :data-role="message.role"
            :data-pending="message.pending ? 'true' : 'false'"
          >
            <div class="trace-card__meta">
              <p class="trace-card__kind">{{ formatTraceKind(message.kind) }}</p>
              <time class="trace-card__time">{{ formatTimestamp(message.created_at) }}</time>
            </div>
            <p class="trace-card__role">{{ formatRole(message.role) }}</p>
            <p class="trace-card__content">{{ message.content }}</p>
          </article>
        </section>
      </section>

      <aside class="panel panel--sidebar">
        <header class="panel__header">
          <div>
            <p class="panel__eyebrow">Previous conversations</p>
            <h2>History</h2>
          </div>
          <p class="panel__meta">Select any finalized thread to reopen it.</p>
        </header>

        <div class="sidebar__controls">
          <button
            class="secondary-button"
            type="button"
            :disabled="isCreatingConversation || isSending || isDeletingAllMemory"
            @click="createConversation"
          >
            {{ isCreatingConversation ? 'Creating...' : 'New conversation' }}
          </button>
          <button
            class="secondary-button"
            type="button"
            :disabled="isSending || isDeletingAllMemory"
            @click="loadConversations()"
          >
            Refresh list
          </button>
        </div>

        <p
          v-if="controlMessage"
          class="sidebar__feedback"
          :data-tone="controlMessageTone"
        >
          {{ controlMessage }}
        </p>

        <div class="conversation-list">
          <button
            v-for="conversation in conversations"
            :key="conversation.finalized_thread_id"
            class="conversation-row"
            :data-active="conversation.finalized_thread_id === activeThreadId"
            :disabled="isSending || isDeletingAllMemory"
            type="button"
            @click="loadConversation(conversation.finalized_thread_id)"
          >
            <span class="conversation-row__title">{{ conversation.title }}</span>
            <span class="conversation-row__meta">
              {{ formatTimestamp(conversation.updated_at) }} ·
              {{ describeMessageCount(conversation.message_count) }}
            </span>
            <span class="conversation-row__preview">
              {{ conversation.last_message_preview || 'No finalized messages yet.' }}
            </span>
          </button>

          <div v-if="!conversations.length" class="empty-state empty-state--compact">
            <p class="empty-state__title">No stored conversations</p>
            <p class="empty-state__copy">
              Create a new conversation or send a message to start one.
            </p>
          </div>
        </div>

        <section class="memory-controls">
          <p class="memory-controls__eyebrow">Global memory</p>
          <p class="memory-controls__copy">
            Delete every stored finalized transcript, trace artifact, and Qdrant memory point.
          </p>
          <button
            class="danger-button"
            type="button"
            :disabled="isDeletingAllMemory || isSending"
            @click="deleteAllMemory"
          >
            {{ isDeletingAllMemory ? 'Deleting...' : 'Delete all memory' }}
          </button>
        </section>
      </aside>
    </main>
  </div>
</template>
