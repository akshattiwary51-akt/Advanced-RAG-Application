import { useDispatch, useSelector } from 'react-redux'
import { fetchStats } from '../store/documentsSlice'
import { navigate } from '../store/uiSlice'
import { formatBytes } from '../utils/format'
import { Button, Card, ErrorBanner, StatCard } from '../components/ui'

const LLM_TEXT = {
  ok: ['Connected', 'green'],
  model_missing: ['Model not installed', 'amber'],
  unavailable: ['Offline', 'red'],
}

export default function Dashboard() {
  const dispatch = useDispatch()
  const { stats, health, statsError } = useSelector((s) => s.documents)
  const settings = useSelector((s) => s.settings)

  const llm = health?.llm
  const [llmText, llmTone] = LLM_TEXT[llm?.status] || ['Unknown', 'slate']
  const vs = stats?.vector_store || health?.vector_store

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Dashboard</h1>
          <p className="text-sm text-slate-500">System status and index overview</p>
        </div>
        <Button variant="secondary" onClick={() => dispatch(fetchStats())}>Refresh</Button>
      </div>

      {statsError && <ErrorBanner onRetry={() => dispatch(fetchStats())}>{statsError}</ErrorBanner>}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Documents" value={stats?.documents.total_documents ?? '–'} hint={stats ? formatBytes(stats.documents.total_bytes) : undefined} tone="indigo" />
        <StatCard label="Indexed" value={stats?.documents.indexed_documents ?? '–'} hint={stats?.documents.failed_documents ? `${stats.documents.failed_documents} failed` : 'all healthy'} tone={stats?.documents.failed_documents ? 'amber' : 'green'} />
        <StatCard label="Total chunks" value={stats?.documents.total_chunks?.toLocaleString() ?? '–'} tone="indigo" />
        <StatCard label="LLM" value={llmText} hint={llm?.model} tone={llmTone} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Embedding model">
          <dl className="space-y-2 text-sm">
            <Row k="Model" v={health?.embedding.model} />
            <Row k="Backend" v={health?.embedding.backend} />
            <Row k="Dimensions" v={health?.embedding.dimension} />
          </dl>
        </Card>
        <Card title="Vector storage">
          <dl className="space-y-2 text-sm">
            <Row k="Vectors stored" v={vs?.chunks?.toLocaleString()} />
            <Row k="Documents in index" v={vs?.documents} />
            <Row k="In-memory size" v={vs && formatBytes(vs.memory_bytes)} />
            <Row k="On disk" v={vs && formatBytes(vs.disk_bytes)} />
            <Row k="Index type" v="Exact cosine (NumPy, brute force)" />
          </dl>
        </Card>
      </div>

      {llm?.status !== 'ok' && health && (
        <ErrorBanner>
          {llm?.status === 'model_missing'
            ? <>Ollama is running but model <b>{llm.model}</b> is not installed. Run <code>ollama pull {llm.model}</code>.</>
            : <>Ollama is not reachable. Start it with <code>ollama serve</code> to enable answers. Documents can still be indexed and searched.</>}
        </ErrorBanner>
      )}

      <Card title="Current retrieval settings" action={<Button variant="ghost" onClick={() => dispatch(navigate('settings'))}>Edit</Button>}>
        <div className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
          <Row k="Top-K" v={settings.topK} stacked />
          <Row k="Threshold" v={settings.similarityThreshold} stacked />
          <Row k="Method" v={settings.method} stacked />
          <Row k="Chunk size / overlap" v={`${settings.chunkSize} / ${settings.chunkOverlap}`} stacked />
        </div>
      </Card>
    </div>
  )
}

function Row({ k, v, stacked }) {
  return (
    <div className={stacked ? '' : 'flex justify-between gap-4'}>
      <dt className="text-slate-500">{k}</dt>
      <dd className="font-medium text-slate-800">{v ?? '–'}</dd>
    </div>
  )
}
