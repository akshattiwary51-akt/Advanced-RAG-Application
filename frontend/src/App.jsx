import { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import Sidebar from './components/Sidebar'
import Toasts from './components/Toasts'
import Chat from './pages/Chat'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Settings from './pages/Settings'
import { fetchDocuments, fetchStats } from './store/documentsSlice'

const PAGES = { dashboard: Dashboard, documents: Documents, chat: Chat, settings: Settings }

export default function App() {
  const dispatch = useDispatch()
  const page = useSelector((s) => s.ui.page)
  const Page = PAGES[page] || Dashboard

  useEffect(() => {
    dispatch(fetchDocuments())
    dispatch(fetchStats())
    const t = setInterval(() => dispatch(fetchStats()), 15000) // keeps LLM status fresh
    return () => clearInterval(t)
  }, [dispatch])

  return (
    <div className="flex h-full flex-col md:flex-row">
      <Sidebar />
      <main className="min-w-0 flex-1 overflow-y-auto p-4 md:p-8">
        <Page />
      </main>
      <Toasts />
    </div>
  )
}
