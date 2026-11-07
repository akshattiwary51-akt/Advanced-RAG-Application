import { createSlice, nanoid } from '@reduxjs/toolkit'

const slice = createSlice({
  name: 'ui',
  initialState: { page: 'dashboard', toasts: [] },
  reducers: {
    navigate: (state, a) => { state.page = a.payload },
    pushToast: {
      reducer: (state, a) => { state.toasts.push(a.payload) },
      prepare: (message, kind = 'info') => ({ payload: { id: nanoid(), message, kind } }),
    },
    dismissToast: (state, a) => { state.toasts = state.toasts.filter((t) => t.id !== a.payload) },
  },
})

export const { navigate, pushToast, dismissToast } = slice.actions
export default slice.reducer
