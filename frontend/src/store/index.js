import { configureStore } from '@reduxjs/toolkit'
import chat from './chatSlice'
import documents from './documentsSlice'
import settings, { persistSettings } from './settingsSlice'
import ui from './uiSlice'

export const store = configureStore({ reducer: { chat, documents, settings, ui } })

let last = store.getState().settings
store.subscribe(() => {
  const cur = store.getState().settings
  if (cur !== last) { last = cur; persistSettings(cur) }
})
