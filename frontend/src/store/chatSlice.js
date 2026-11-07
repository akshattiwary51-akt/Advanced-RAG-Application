import { createAsyncThunk, createSlice } from '@reduxjs/toolkit'
import { api, errorMessage, streamChat } from '../services/api'
import { pushToast } from './uiSlice'

let abortCurrent = null

export const fetchConversations = createAsyncThunk('chat/list', async (_, { rejectWithValue }) => {
  try { return await api.listConversations() } catch (e) { return rejectWithValue(errorMessage(e)) }
})

export const openConversation = createAsyncThunk('chat/open', async (id, { rejectWithValue }) => {
  try { return await api.getConversation(id) } catch (e) { return rejectWithValue(errorMessage(e)) }
})

export const removeConversation = createAsyncThunk('chat/delete', async (id, { dispatch, rejectWithValue }) => {
  try { await api.deleteConversation(id); return id } catch (e) {
    dispatch(pushToast(errorMessage(e), 'error'))
    return rejectWithValue(errorMessage(e))
  }
})

/** Send a message and stream the answer into the store token by token. */
export const sendMessage = createAsyncThunk('chat/send', async (text, { dispatch, getState }) => {
  const { settings, chat } = getState()
  dispatch(addUserMessage(text))
  const payload = {
    message: text,
    conversation_id: chat.activeId || undefined,
    top_k: settings.topK,
    similarity_threshold: settings.similarityThreshold,
    temperature: settings.temperature,
    max_tokens: settings.maxTokens,
    method: settings.method,
    use_mmr: settings.useMmr,
    mmr_lambda: settings.mmrLambda,
    rerank: settings.rerank,
    expand_query: settings.expandQuery,
    compress: settings.compress,
    rewrite_query: settings.rewriteQuery,
    model: settings.model || undefined,
    document_ids: chat.scopeDocumentIds.length ? chat.scopeDocumentIds : undefined,
  }
  await new Promise((resolve) => {
    abortCurrent = streamChat(payload, {
      onMeta: (ev) => dispatch(setActive(ev.conversation_id)),
      onSources: (ev) => dispatch(setStreamSources(ev.citations)),
      onToken: (t) => dispatch(appendToken(t)),
      onDone: (ev) => { dispatch(finishStream(ev)); dispatch(fetchConversations()); resolve() },
      onError: (msg, partial) => { dispatch(failStream({ message: msg, partial })); resolve() },
    })
  })
  abortCurrent = null
})

export const stopStreaming = () => (dispatch) => {
  abortCurrent?.()
  abortCurrent = null
  dispatch(abortStream())
}

const slice = createSlice({
  name: 'chat',
  initialState: {
    conversations: [], activeId: null, messages: [], streaming: false,
    scopeDocumentIds: [], loadingConversation: false,
  },
  reducers: {
    newConversation: (s) => { abortCurrent?.(); s.activeId = null; s.messages = []; s.streaming = false },
    setActive: (s, a) => { s.activeId = a.payload },
    setScope: (s, a) => { s.scopeDocumentIds = a.payload },
    addUserMessage: (s, a) => {
      s.streaming = true
      s.messages.push({ role: 'user', content: a.payload })
      s.messages.push({ role: 'assistant', content: '', citations: [], pending: true })
    },
    setStreamSources: (s, a) => { const m = s.messages[s.messages.length - 1]; if (m) m.citations = a.payload },
    appendToken: (s, a) => {
      const m = s.messages[s.messages.length - 1]
      if (m) { m.content += a.payload; m.pending = false }
    },
    finishStream: (s, a) => {
      const m = s.messages[s.messages.length - 1]
      if (m) {
        m.content = a.payload.answer
        m.citations = a.payload.citations
        m.pending = false
        m.grounded = a.payload.grounded
        m.warnings = a.payload.warnings
        m.timings = a.payload.timings_ms
        m.rewritten = a.payload.rewritten_query
      }
      s.streaming = false
    },
    failStream: (s, a) => {
      const m = s.messages[s.messages.length - 1]
      if (m) { m.pending = false; m.error = a.payload.message; m.content = a.payload.partial || m.content }
      s.streaming = false
    },
    abortStream: (s) => {
      const m = s.messages[s.messages.length - 1]
      if (m && s.streaming) { m.pending = false; m.stopped = true }
      s.streaming = false
    },
  },
  extraReducers: (b) => {
    b.addCase(fetchConversations.fulfilled, (s, a) => { s.conversations = a.payload })
      .addCase(openConversation.pending, (s) => { s.loadingConversation = true })
      .addCase(openConversation.fulfilled, (s, a) => {
        abortCurrent?.()
        s.loadingConversation = false; s.streaming = false
        s.activeId = a.payload.id
        s.messages = a.payload.messages.map((m) => ({ role: m.role, content: m.content, citations: m.citations || [] }))
      })
      .addCase(openConversation.rejected, (s) => { s.loadingConversation = false })
      .addCase(removeConversation.fulfilled, (s, a) => {
        s.conversations = s.conversations.filter((c) => c.id !== a.payload)
        if (s.activeId === a.payload) { s.activeId = null; s.messages = [] }
      })
  },
})

export const {
  newConversation, setActive, setScope, addUserMessage, setStreamSources,
  appendToken, finishStream, failStream, abortStream,
} = slice.actions
export default slice.reducer
