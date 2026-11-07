import { useDispatch, useSelector } from 'react-redux'
import { navigate } from '../store/uiSlice'

const ITEMS = [
  { id: 'dashboard', label: 'Dashboard', d: 'M3 12l9-9 9 9M5 10v10h5v-6h4v6h5V10' },
  { id: 'documents', label: 'Documents', d: 'M7 3h7l5 5v13H7V3zM14 3v5h5' },
  { id: 'chat', label: 'Chat', d: 'M21 12a8 8 0 01-11.6 7.1L4 20l1-4.6A8 8 0 1121 12z' },
  { id: 'settings', label: 'Settings', d: 'M12 15a3 3 0 100-6 3 3 0 000 6zM19 12a7 7 0 00-.1-1.2l2-1.5-2-3.4-2.3.9a7 7 0 00-2-1.2L14.2 3h-4l-.4 2.6a7 7 0 00-2 1.2l-2.3-.9-2 3.4 2 1.5a7 7 0 000 2.4l-2 1.5 2 3.4 2.3-.9a7 7 0 002 1.2l.4 2.6h4l.4-2.6a7 7 0 002-1.2l2.3.9 2-3.4-2-1.5c.1-.4.1-.8.1-1.2z' },
]

export default function Sidebar() {
  const page = useSelector((s) => s.ui.page)
  const llm = useSelector((s) => s.documents.health?.llm?.status)
  const dispatch = useDispatch()
  const dot = llm === 'ok' ? 'bg-emerald-500' : llm ? 'bg-amber-500' : 'bg-slate-300'

  return (
    <aside className="flex w-full shrink-0 flex-row items-center gap-1 border-b border-slate-200 bg-white px-3 py-2 md:w-56 md:flex-col md:items-stretch md:border-b-0 md:border-r md:px-3 md:py-5">
      <div className="hidden px-3 pb-5 md:block">
        <div className="text-base font-semibold text-slate-900">RAG Assistant</div>
        <div className="text-xs text-slate-500">Chat with your documents</div>
      </div>
      <nav className="flex flex-1 gap-1 md:flex-col" aria-label="Main">
        {ITEMS.map((it) => (
          <button
            key={it.id}
            onClick={() => dispatch(navigate(it.id))}
            aria-current={page === it.id ? 'page' : undefined}
            aria-label={it.label}
            title={it.label}
            className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
              page === it.id ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            <svg className="h-5 w-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d={it.d} />
            </svg>
            <span className="hidden sm:inline">{it.label}</span>
          </button>
        ))}
      </nav>
      <div className="hidden items-center gap-2 px-3 pt-4 text-xs text-slate-500 md:flex" title="Ollama connection">
        <span className={`h-2 w-2 rounded-full ${dot}`} />
        LLM {llm === 'ok' ? 'connected' : llm === 'model_missing' ? 'model missing' : llm ? 'offline' : 'unknown'}
      </div>
    </aside>
  )
}
