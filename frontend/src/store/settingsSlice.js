import { createSlice } from '@reduxjs/toolkit'

const KEY = 'rag.settings.v1'

export const DEFAULT_SETTINGS = {
  topK: 5,
  similarityThreshold: 0.30,
  temperature: 0.2,
  maxTokens: 1024,
  model: '', // empty = use the server's configured model
  method: 'semantic',
  chunkSize: 800,
  chunkOverlap: 150,
  useMmr: false,
  mmrLambda: 0.7,
  rerank: false,
  expandQuery: false,
  compress: false,
  rewriteQuery: false,
}

function load() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(KEY) || '{}') }
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

const slice = createSlice({
  name: 'settings',
  initialState: load(),
  reducers: {
    updateSettings: (state, action) => ({ ...state, ...action.payload }),
    resetSettings: () => ({ ...DEFAULT_SETTINGS }),
  },
})

export const { updateSettings, resetSettings } = slice.actions
export const persistSettings = (settings) => {
  try { localStorage.setItem(KEY, JSON.stringify(settings)) } catch { /* private mode */ }
}
export default slice.reducer
