import { createAsyncThunk, createSlice } from '@reduxjs/toolkit'
import { api, errorMessage } from '../services/api'
import { pushToast } from './uiSlice'

export const fetchDocuments = createAsyncThunk('documents/fetch', async (_, { rejectWithValue }) => {
  try { return await api.listDocuments() } catch (e) { return rejectWithValue(errorMessage(e)) }
})

export const fetchStats = createAsyncThunk('documents/stats', async (_, { rejectWithValue }) => {
  try {
    const [stats, health] = await Promise.all([api.stats(), api.health()])
    return { stats, health }
  } catch (e) { return rejectWithValue(errorMessage(e)) }
})

/** Upload files sequentially so progress and errors are reported per file. */
export const uploadFiles = createAsyncThunk('documents/upload', async (files, { dispatch }) => {
  for (const { uid, file } of files) {
    dispatch(setUpload({ uid, status: 'uploading', progress: 0 }))
    try {
      const res = await api.uploadFile(file, (p) =>
        dispatch(setUpload({ uid, status: p >= 100 ? 'processing' : 'uploading', progress: p })))
      if (res.status === 'indexed') {
        dispatch(setUpload({ uid, status: 'done', progress: 100 }))
      } else {
        const msg = res.error?.message || 'Upload failed'
        dispatch(setUpload({ uid, status: res.status === 'duplicate' ? 'duplicate' : 'error', error: msg }))
      }
    } catch (e) {
      dispatch(setUpload({ uid, status: 'error', error: errorMessage(e) }))
    }
  }
  dispatch(fetchDocuments())
  dispatch(fetchStats())
})

export const removeDocument = createAsyncThunk('documents/delete', async (id, { dispatch, rejectWithValue }) => {
  try {
    await api.deleteDocument(id)
    dispatch(pushToast('Document deleted', 'success'))
    dispatch(fetchStats())
    return id
  } catch (e) {
    dispatch(pushToast(errorMessage(e), 'error'))
    return rejectWithValue(errorMessage(e))
  }
})

export const reindexDocument = createAsyncThunk('documents/reindex', async ({ id, chunkSize, chunkOverlap }, { dispatch, rejectWithValue }) => {
  try {
    const doc = await api.reindexDocument(id, { chunk_size: chunkSize, chunk_overlap: chunkOverlap })
    dispatch(pushToast(`Re-indexed into ${doc.chunk_count} chunks`, 'success'))
    dispatch(fetchStats())
    return doc
  } catch (e) {
    dispatch(pushToast(errorMessage(e), 'error'))
    dispatch(fetchDocuments())
    return rejectWithValue(errorMessage(e))
  }
})

const slice = createSlice({
  name: 'documents',
  initialState: { items: [], loading: false, error: null, uploads: {}, busy: {}, stats: null, health: null, statsError: null },
  reducers: {
    setUpload: (state, a) => { state.uploads[a.payload.uid] = { ...state.uploads[a.payload.uid], ...a.payload } },
    queueUploads: (state, a) => {
      a.payload.forEach(({ uid, name, error }) => {
        state.uploads[uid] = { uid, name, status: error ? 'error' : 'queued', progress: 0, error }
      })
    },
    clearFinishedUploads: (state) => {
      Object.keys(state.uploads).forEach((k) => {
        if (['done', 'error', 'duplicate'].includes(state.uploads[k].status)) delete state.uploads[k]
      })
    },
  },
  extraReducers: (b) => {
    b.addCase(fetchDocuments.pending, (s) => { s.loading = true })
      .addCase(fetchDocuments.fulfilled, (s, a) => { s.loading = false; s.error = null; s.items = a.payload })
      .addCase(fetchDocuments.rejected, (s, a) => { s.loading = false; s.error = a.payload })
      .addCase(fetchStats.fulfilled, (s, a) => { s.stats = a.payload.stats; s.health = a.payload.health; s.statsError = null })
      .addCase(fetchStats.rejected, (s, a) => { s.statsError = a.payload; s.health = null })
      .addCase(removeDocument.pending, (s, a) => { s.busy[a.meta.arg] = 'deleting' })
      .addCase(removeDocument.fulfilled, (s, a) => { delete s.busy[a.payload]; s.items = s.items.filter((d) => d.id !== a.payload) })
      .addCase(removeDocument.rejected, (s, a) => { delete s.busy[a.meta.arg] })
      .addCase(reindexDocument.pending, (s, a) => {
        s.busy[a.meta.arg.id] = 'reindexing'
        const d = s.items.find((x) => x.id === a.meta.arg.id)
        if (d) d.status = 'processing'
      })
      .addCase(reindexDocument.fulfilled, (s, a) => {
        delete s.busy[a.payload.id]
        s.items = s.items.map((d) => (d.id === a.payload.id ? a.payload : d))
      })
      .addCase(reindexDocument.rejected, (s, a) => { delete s.busy[a.meta.arg.id] })
  },
})

export const { setUpload, queueUploads, clearFinishedUploads } = slice.actions
export default slice.reducer
