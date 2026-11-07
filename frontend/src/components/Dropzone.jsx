import { useRef, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { clearFinishedUploads, queueUploads, uploadFiles } from '../store/documentsSlice'
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB, checkFile } from '../utils/format'
import { Button, Spinner } from './ui'

let counter = 0

const STATUS_LABEL = {
  queued: 'Waiting…', uploading: 'Uploading', processing: 'Extracting & indexing…',
  done: 'Indexed', duplicate: 'Already uploaded', error: 'Failed',
}

export default function Dropzone() {
  const dispatch = useDispatch()
  const uploads = Object.values(useSelector((s) => s.documents.uploads))
  const [over, setOver] = useState(false)
  const inputRef = useRef(null)
  const active = uploads.some((u) => ['queued', 'uploading', 'processing'].includes(u.status))

  function handle(fileList) {
    const entries = Array.from(fileList).map((file) => ({ uid: `u${++counter}`, file, error: checkFile(file) }))
    if (!entries.length) return
    dispatch(queueUploads(entries.map(({ uid, file, error }) => ({ uid, name: file.name, error }))))
    const valid = entries.filter((e) => !e.error)
    if (valid.length) dispatch(uploadFiles(valid))
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload documents"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files) }}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          over ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 bg-white hover:border-indigo-400 hover:bg-slate-50'
        }`}
      >
        <svg className="h-9 w-9 text-indigo-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" />
        </svg>
        <p className="mt-3 text-sm font-medium text-slate-700">Drag &amp; drop files here, or click to browse</p>
        <p className="mt-1 text-xs text-slate-500">
          {ACCEPTED_EXTENSIONS.join(', ')} · up to {MAX_UPLOAD_MB} MB each · multiple files supported
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={(e) => { handle(e.target.files); e.target.value = '' }}
        />
      </div>

      {uploads.length > 0 && (
        <ul className="mt-4 space-y-2">
          {uploads.map((u) => (
            <li key={u.uid} className="rounded-lg border border-slate-200 bg-white px-4 py-2.5">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="truncate font-medium text-slate-700" title={u.name}>{u.name}</span>
                <span className={`flex shrink-0 items-center gap-1.5 text-xs ${
                  u.status === 'done' ? 'text-emerald-600' : u.status === 'error' ? 'text-rose-600' : u.status === 'duplicate' ? 'text-amber-600' : 'text-slate-500'
                }`}>
                  {['uploading', 'processing'].includes(u.status) && <Spinner className="h-3.5 w-3.5" />}
                  {STATUS_LABEL[u.status]}
                </span>
              </div>
              {['uploading', 'processing'].includes(u.status) && (
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full bg-indigo-500 transition-all ${u.status === 'processing' ? 'animate-pulse' : ''}`}
                    style={{ width: `${u.status === 'processing' ? 100 : u.progress}%` }}
                  />
                </div>
              )}
              {u.error && <p className="mt-1 text-xs text-rose-600">{u.error}</p>}
            </li>
          ))}
          {!active && (
            <div className="text-right">
              <Button variant="ghost" onClick={() => dispatch(clearFinishedUploads())}>Clear list</Button>
            </div>
          )}
        </ul>
      )}
    </div>
  )
}

