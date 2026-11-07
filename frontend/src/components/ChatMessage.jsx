import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { linkifyCitations } from '../utils/format'
import { Badge, Spinner } from './ui'

/**
 * Model output derives from untrusted documents, so only allow known-safe link schemes.
 * Anything else (javascript:, data:, vbscript: ...) is dropped.
 */
const SAFE_URL = /^(https?:|mailto:|cite:\d{1,3}$|#)/i
export const safeUrl = (u) => (SAFE_URL.test(u) ? u : '')

function CodeBlock({ children, className }) {
  const [copied, setCopied] = useState(false)
  const lang = /language-(\w+)/.exec(className || '')?.[1]
  const code = String(children).replace(/\n$/, '')
  return (
    <div className="my-3 overflow-hidden rounded-lg bg-slate-900 text-slate-100">
      <div className="flex items-center justify-between bg-slate-800 px-3 py-1 text-xs text-slate-400">
        <span>{lang || 'code'}</span>
        <button
          onClick={() => navigator.clipboard?.writeText(code).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })}
          className="hover:text-white"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-sm"><code>{code}</code></pre>
    </div>
  )
}

function SourceCard({ c, expanded, onToggle }) {
  const pct = Math.max(0, Math.min(100, Math.round(c.score * 100)))
  return (
    <div id={`src-${c.id}`} className={`rounded-lg border bg-white ${c.cited_in_answer ? 'border-indigo-200' : 'border-slate-200'}`}>
      <button onClick={onToggle} aria-expanded={expanded} className="flex w-full items-center gap-3 px-3 py-2 text-left">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-indigo-600 text-xs font-semibold text-white">{c.id}</span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-slate-700">{c.filename}</span>
          <span className="text-xs text-slate-500">
            {c.page != null ? `Page ${c.page} · ` : ''}chunk {c.chunk_id.split(':').pop()}
            {!c.cited_in_answer && ' · retrieved, not cited'}
          </span>
        </span>
        <span className="shrink-0 text-right" title="Cosine similarity">
          <span className="block text-xs font-semibold text-slate-700">{c.score.toFixed(3)}</span>
          <span className="mt-0.5 block h-1 w-12 overflow-hidden rounded bg-slate-100">
            <span className="block h-full bg-emerald-500" style={{ width: `${pct}%` }} />
          </span>
        </span>
        <span className="text-slate-400">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <p className="whitespace-pre-wrap border-t border-slate-100 px-3 py-2 text-xs leading-relaxed text-slate-600">{c.text || c.snippet}</p>
      )}
    </div>
  )
}

export default function ChatMessage({ message, streaming }) {
  const [expanded, setExpanded] = useState({})
  const [showSources, setShowSources] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white">{message.content}</div>
      </div>
    )
  }

  const citations = message.citations || []
  const jump = (n) => {
    setShowSources(true)
    setExpanded((e) => ({ ...e, [n]: true }))
    setTimeout(() => document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50)
  }

  const components = {
    a: ({ href, children }) =>
      href?.startsWith('cite:') ? (
        <button onClick={() => jump(Number(href.slice(5)))} className="mx-0.5 inline-flex h-4 min-w-4 -translate-y-0.5 items-center justify-center rounded bg-indigo-100 px-1 text-[10px] font-semibold text-indigo-700 hover:bg-indigo-200" title="Show source">
          {children}
        </button>
      ) : (
        <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>
      ),
    code: ({ inline, className, children }) =>
      inline || !/\n/.test(String(children)) && !className ? <code>{children}</code> : <CodeBlock className={className}>{children}</CodeBlock>,
    pre: ({ children }) => <>{children}</>,
  }

  return (
    <div className="flex">
      <div className="max-w-[92%] min-w-0 flex-1">
        <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800 shadow-sm">
          {message.pending ? (
            <div className="flex items-center gap-2 text-slate-500"><Spinner /> Searching your documents…</div>
          ) : (
            <div className={`md ${streaming ? 'cursor' : ''}`}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={components}
                urlTransform={safeUrl}
              >
                {linkifyCitations(message.content || '')}
              </ReactMarkdown>
            </div>
          )}
          {message.stopped && <p className="mt-2 text-xs text-slate-400">Generation stopped.</p>}
          {message.error && (
            <p role="alert" className="mt-2 rounded bg-rose-50 px-2 py-1.5 text-xs text-rose-700">{message.error}</p>
          )}
          {message.warnings?.length > 0 && (
            <p className="mt-2 rounded bg-amber-50 px-2 py-1.5 text-xs text-amber-700">⚠ {message.warnings.join(' ')}</p>
          )}
        </div>

        {message.rewritten && (
          <p className="mt-1.5 text-xs text-slate-400">Searched for: <span className="italic">{message.rewritten}</span></p>
        )}

        {citations.length > 0 && (
          <div className="mt-2">
            <button onClick={() => setShowSources((v) => !v)} className="flex items-center gap-2 text-xs font-medium text-slate-500 hover:text-slate-800">
              {showSources ? '▾' : '▸'} {citations.length} source{citations.length > 1 ? 's' : ''}
              <Badge tone="indigo">{citations.filter((c) => c.cited_in_answer).length} cited</Badge>
              {message.timings && (
                <span className="font-normal text-slate-400">
                  retrieval {Math.round(message.timings.retrieval)} ms
                  {message.timings.generation != null && ` · generation ${(message.timings.generation / 1000).toFixed(1)} s`}
                </span>
              )}
            </button>
            {showSources && (
              <div className="mt-2 space-y-1.5">
                {citations.map((c) => (
                  <SourceCard key={c.id} c={c} expanded={!!expanded[c.id]} onToggle={() => setExpanded((e) => ({ ...e, [c.id]: !e[c.id] }))} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
