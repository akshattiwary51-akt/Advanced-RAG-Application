import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE_URL || '/api'

export const http = axios.create({ baseURL: BASE, timeout: 120000 })

/** Normalise any axios/network failure into a readable message. */
export function errorMessage(err) {
  const apiMsg = err?.response?.data?.error?.message
  if (apiMsg) return apiMsg
  if (err?.code === 'ECONNABORTED') return 'The request timed out.'
  if (err?.message === 'Network Error' || !err?.response) {
    return 'Cannot reach the backend. Is the Flask server running on port 5000?'
  }
  return `Request failed (HTTP ${err.response.status}).`
}

export const api = {
  health: () => http.get('/health').then((r) => r.data),
  stats: () => http.get('/stats').then((r) => r.data),
  defaults: () => http.get('/settings').then((r) => r.data),

  listDocuments: () => http.get('/documents').then((r) => r.data.documents),
  deleteDocument: (id) => http.delete(`/documents/${id}`).then((r) => r.data),
  reindexDocument: (id, body = {}) =>
    http.post(`/documents/${id}/reindex`, body).then((r) => r.data),

  /** Upload one file with progress. Returns the per-file result object. */
  async uploadFile(file, onProgress) {
    const form = new FormData()
    form.append('files', file)
    try {
      const { data } = await http.post('/documents/upload', form, {
        onUploadProgress: (e) => e.total && onProgress?.(Math.round((e.loaded / e.total) * 100)),
      })
      return data.results[0]
    } catch (err) {
      // Per-file failures (409/415/422...) carry a results array; surface the file's error.
      const r = err?.response?.data?.results?.[0]
      if (r) return r
      throw err
    }
  },

  listConversations: () => http.get('/conversations').then((r) => r.data.conversations),
  getConversation: (id) => http.get(`/conversations/${id}`).then((r) => r.data),
  deleteConversation: (id) => http.delete(`/conversations/${id}`).then((r) => r.data),

  debugRetrieval: (params) => http.get('/debug/retrieval', { params }).then((r) => r.data),
}

/**
 * Stream a chat answer over Server-Sent Events.
 * Axios cannot stream in the browser, so this uses fetch + ReadableStream.
 * Handlers: onMeta, onSources, onToken, onDone, onError. Returns an abort function.
 */
export function streamChat(payload, handlers) {
  const controller = new AbortController()
  ;(async () => {
    let finished = false
    try {
      const res = await fetch(`${BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })
      if (!res.ok) {
        let msg = `Request failed (HTTP ${res.status}).`
        try { msg = (await res.json())?.error?.message || msg } catch { /* not JSON */ }
        throw new Error(msg)
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          if (!frame.startsWith('data: ')) continue
          const ev = JSON.parse(frame.slice(6))
          if (ev.type === 'meta') handlers.onMeta?.(ev)
          else if (ev.type === 'sources') handlers.onSources?.(ev)
          else if (ev.type === 'token') handlers.onToken?.(ev.content)
          else if (ev.type === 'done') { finished = true; handlers.onDone?.(ev) }
          else if (ev.type === 'error') { finished = true; handlers.onError?.(ev.message, ev.partial) }
        }
      }
      if (!finished) handlers.onError?.('The response ended unexpectedly.')
    } catch (err) {
      if (err.name === 'AbortError') return
      const offline = err instanceof TypeError
      handlers.onError?.(offline ? 'Cannot reach the backend. Is the Flask server running?' : err.message)
    }
  })()
  return () => controller.abort()
}
