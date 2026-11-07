import { useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { reindexDocument, removeDocument } from '../store/documentsSlice'
import { formatBytes, formatDate } from '../utils/format'
import { Badge, Button, Spinner } from './ui'

const TONE = { indexed: 'green', processing: 'amber', failed: 'red' }

function Details({ doc }) {
  const rows = [
    ['File type', doc.file_type.toUpperCase()],
    ['Size', formatBytes(doc.size_bytes)],
    ['Pages', doc.page_count || '–'],
    ['Chunks', doc.chunk_count],
    ['Characters', doc.char_count?.toLocaleString()],
    ['Uploaded', formatDate(doc.created_at)],
    ['Indexed', formatDate(doc.indexed_at)],
    ['SHA-256', `${doc.sha256.slice(0, 16)}…`],
    ['Document ID', doc.id],
  ]
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 bg-slate-50 px-4 py-3 text-xs sm:grid-cols-3">
      {rows.map(([k, v]) => (
        <div key={k}>
          <dt className="text-slate-500">{k}</dt>
          <dd className="break-all font-medium text-slate-800">{v}</dd>
        </div>
      ))}
      {doc.error && (
        <div className="col-span-full text-rose-600">Error: {doc.error}</div>
      )}
    </dl>
  )
}

export default function DocumentTable({ documents }) {
  const dispatch = useDispatch()
  const busy = useSelector((s) => s.documents.busy)
  const { chunkSize, chunkOverlap } = useSelector((s) => s.settings)
  const [open, setOpen] = useState(null)
  const [confirm, setConfirm] = useState(null)

  return (
    <ul className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200 bg-white">
      {documents.map((d) => {
        const state = busy[d.id]
        return (
          <li key={d.id}>
            <div className="flex flex-wrap items-center gap-3 px-4 py-3">
              <button
                onClick={() => setOpen(open === d.id ? null : d.id)}
                aria-expanded={open === d.id}
                className="min-w-0 flex-1 text-left"
              >
                <div className="truncate text-sm font-medium text-slate-800" title={d.filename}>{d.filename}</div>
                <div className="text-xs text-slate-500">
                  {d.file_type.toUpperCase()} · {formatBytes(d.size_bytes)} · {d.chunk_count} chunks · {formatDate(d.created_at)}
                </div>
              </button>
              <Badge tone={TONE[d.status] || 'slate'}>
                {d.status === 'processing' && <Spinner className="h-3 w-3" />}
                {d.status}
              </Badge>
              <div className="flex gap-2">
                <Button
                  variant="secondary"
                  disabled={!!state}
                  title={`Re-chunk with size ${chunkSize} / overlap ${chunkOverlap} (from Settings)`}
                  onClick={() => dispatch(reindexDocument({ id: d.id, chunkSize, chunkOverlap }))}
                >
                  {state === 'reindexing' ? <Spinner /> : null} Re-index
                </Button>
                {confirm === d.id ? (
                  <>
                    <Button variant="danger" disabled={!!state} onClick={() => { setConfirm(null); dispatch(removeDocument(d.id)) }}>
                      {state === 'deleting' ? <Spinner /> : null} Confirm
                    </Button>
                    <Button variant="ghost" onClick={() => setConfirm(null)}>Cancel</Button>
                  </>
                ) : (
                  <Button variant="danger" disabled={!!state} onClick={() => setConfirm(d.id)}>Delete</Button>
                )}
              </div>
            </div>
            {open === d.id && <Details doc={d} />}
          </li>
        )
      })}
    </ul>
  )
}
