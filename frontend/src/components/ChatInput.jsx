import { useRef, useState } from 'react'
import { Button, Spinner } from './ui'

const MAX = 4000

export default function ChatInput({ onSend, onStop, streaming, disabled, placeholder }) {
  const [text, setText] = useState('')
  const ref = useRef(null)

  function submit() {
    const t = text.trim()
    if (!t || streaming || disabled) return
    onSend(t)
    setText('')
    if (ref.current) ref.current.style.height = 'auto'
  }

  return (
    <div className="border-t border-slate-200 bg-white p-3 sm:p-4">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <textarea
          ref={ref}
          value={text}
          rows={1}
          maxLength={MAX}
          disabled={disabled}
          placeholder={placeholder}
          aria-label="Message"
          onChange={(e) => {
            setText(e.target.value)
            e.target.style.height = 'auto'
            e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
          }}
          className="max-h-40 flex-1 resize-none rounded-xl border border-slate-300 px-3.5 py-2.5 text-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 disabled:bg-slate-50"
        />
        {streaming ? (
          <Button variant="secondary" onClick={onStop} className="h-10"><Spinner /> Stop</Button>
        ) : (
          <Button onClick={submit} disabled={!text.trim() || disabled} className="h-10 px-4">Send</Button>
        )}
      </div>
      <p className="mx-auto mt-1.5 max-w-3xl text-[11px] text-slate-400">Enter to send · Shift+Enter for a new line</p>
    </div>
  )
}
