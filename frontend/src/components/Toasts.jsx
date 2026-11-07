import { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { dismissToast } from '../store/uiSlice'

const TONE = {
  success: 'bg-emerald-600',
  error: 'bg-rose-600',
  info: 'bg-slate-800',
}

function Toast({ toast }) {
  const dispatch = useDispatch()
  useEffect(() => {
    const t = setTimeout(() => dispatch(dismissToast(toast.id)), 5000)
    return () => clearTimeout(t)
  }, [dispatch, toast.id])
  return (
    <div role="status" className={`pointer-events-auto flex items-start gap-3 rounded-lg px-4 py-3 text-sm text-white shadow-lg ${TONE[toast.kind] || TONE.info}`}>
      <span className="flex-1">{toast.message}</span>
      <button aria-label="Dismiss" onClick={() => dispatch(dismissToast(toast.id))} className="opacity-70 hover:opacity-100">✕</button>
    </div>
  )
}

export default function Toasts() {
  const toasts = useSelector((s) => s.ui.toasts)
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2">
      {toasts.map((t) => <Toast key={t.id} toast={t} />)}
    </div>
  )
}
