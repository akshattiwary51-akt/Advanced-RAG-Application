import { useDispatch, useSelector } from 'react-redux'
import { DEFAULT_SETTINGS, resetSettings, updateSettings } from '../store/settingsSlice'
import { Button, Card } from '../components/ui'

function Slider({ label, hint, value, min, max, step, onChange, format = (v) => v }) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        <span className="text-sm font-semibold text-indigo-600">{format(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="mt-2 w-full accent-indigo-600" />
      {hint && <span className="text-xs text-slate-500">{hint}</span>}
    </label>
  )
}

function Toggle({ label, hint, checked, onChange }) {
  return (
    <label className="flex cursor-pointer items-start gap-3">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="mt-1 h-4 w-4 accent-indigo-600" />
      <span>
        <span className="block text-sm font-medium text-slate-700">{label}</span>
        <span className="text-xs text-slate-500">{hint}</span>
      </span>
    </label>
  )
}

function NumberField({ label, hint, value, min, max, onChange }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <input
        type="number" min={min} max={max} value={value}
        onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))}
        className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
      />
      {hint && <span className="text-xs text-slate-500">{hint}</span>}
    </label>
  )
}

export default function Settings() {
  const dispatch = useDispatch()
  const s = useSelector((st) => st.settings)
  const health = useSelector((st) => st.documents.health)
  const set = (patch) => dispatch(updateSettings(patch))
  const overlapInvalid = s.chunkOverlap >= s.chunkSize

  return (
    <div className="max-w-3xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Settings</h1>
          <p className="text-sm text-slate-500">Saved in this browser and applied to every question</p>
        </div>
        <Button variant="secondary" onClick={() => dispatch(resetSettings())}>Reset to defaults</Button>
      </div>

      <Card title="Retrieval">
        <div className="grid gap-6 sm:grid-cols-2">
          <Slider label="Top-K" value={s.topK} min={1} max={20} step={1} onChange={(v) => set({ topK: v })} hint="How many chunks are sent to the model as evidence." />
          <Slider label="Similarity threshold" value={s.similarityThreshold} min={0} max={0.9} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ similarityThreshold: v })} hint="Chunks scoring below this are ignored. Higher = stricter." />
          <label className="block sm:col-span-2">
            <span className="text-sm font-medium text-slate-700">Retrieval method</span>
            <select value={s.method} onChange={(e) => set({ method: e.target.value })} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm">
              <option value="semantic">Semantic (cosine similarity on embeddings)</option>
              <option value="bm25">Keyword (BM25)</option>
              <option value="hybrid">Hybrid (semantic + BM25, Reciprocal Rank Fusion)</option>
            </select>
            <span className="text-xs text-slate-500">Hybrid helps with exact names, codes and rare terms that embeddings blur.</span>
          </label>
          <div className="space-y-3 sm:col-span-2">
            <Toggle label="Rerank candidates" checked={s.rerank} onChange={(v) => set({ rerank: v })} hint="Re-scores the candidate list for better ordering. Adds latency." />
            <Toggle label="Diversify results (MMR)" checked={s.useMmr} onChange={(v) => set({ useMmr: v })} hint="Avoids sending near-duplicate chunks; may demote a second relevant one." />
            {s.useMmr && (
              <Slider label="MMR balance (λ)" value={s.mmrLambda} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ mmrLambda: v })} hint="1 = pure relevance, 0 = pure diversity." />
            )}
            <Toggle label="Expand query" checked={s.expandQuery} onChange={(v) => set({ expandQuery: v })} hint="Adds terms from the top results to the search. Can drift off-topic." />
            <Toggle label="Rewrite follow-up questions" checked={s.rewriteQuery} onChange={(v) => set({ rewriteQuery: v })} hint="Uses the LLM to make follow-ups standalone. One extra LLM call." />
            <Toggle label="Compress context" checked={s.compress} onChange={(v) => set({ compress: v })} hint="Sends only the relevant sentences of each chunk. May cut needed detail." />
          </div>
        </div>
      </Card>

      <Card title="Language model">
        <div className="grid gap-6 sm:grid-cols-2">
          <label className="block sm:col-span-2">
            <span className="text-sm font-medium text-slate-700">Ollama model</span>
            <input
              value={s.model} placeholder={health?.llm?.model ? `Server default: ${health.llm.model}` : 'Server default'}
              onChange={(e) => set({ model: e.target.value })}
              className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
            />
            <span className="text-xs text-slate-500">
              Leave empty to use the server's model.
              {health?.llm?.available_models?.length > 0 && <> Installed: {health.llm.available_models.join(', ')}.</>}
            </span>
          </label>
          <Slider label="Temperature" value={s.temperature} min={0} max={1.5} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ temperature: v })} hint="Low = factual and consistent. High = more varied." />
          <NumberField label="Max tokens" value={s.maxTokens} min={64} max={8192} onChange={(v) => set({ maxTokens: v })} hint="Upper bound on answer length." />
        </div>
      </Card>

      <Card title="Chunking">
        <div className="grid gap-6 sm:grid-cols-2">
          <NumberField label="Chunk size (characters)" value={s.chunkSize} min={100} max={5000} onChange={(v) => set({ chunkSize: v })} />
          <NumberField label="Chunk overlap (characters)" value={s.chunkOverlap} min={0} max={2000} onChange={(v) => set({ chunkOverlap: v })} />
        </div>
        {overlapInvalid && <p role="alert" className="mt-3 text-xs text-rose-600">Overlap must be smaller than chunk size.</p>}
        <p className="mt-3 text-xs text-slate-500">
          Chunking applies when a document is <b>re-indexed</b> (Documents → Re-index). Defaults: {DEFAULT_SETTINGS.chunkSize} / {DEFAULT_SETTINGS.chunkOverlap}.
        </p>
      </Card>

      <Card title="Embedding model (read-only)">
        <dl className="grid gap-2 text-sm sm:grid-cols-3">
          <div><dt className="text-slate-500">Model</dt><dd className="font-medium">{health?.embedding.model ?? '–'}</dd></div>
          <div><dt className="text-slate-500">Dimensions</dt><dd className="font-medium">{health?.embedding.dimension ?? '–'}</dd></div>
          <div><dt className="text-slate-500">Normalisation</dt><dd className="font-medium">L2 (cosine)</dd></div>
        </dl>
        <p className="mt-3 text-xs text-slate-500">Changing the embedding model requires setting <code>EMBEDDING_MODEL</code> on the server and re-indexing all documents.</p>
      </Card>
    </div>
  )
}
