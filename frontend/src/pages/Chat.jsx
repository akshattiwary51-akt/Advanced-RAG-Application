import { useEffect, useRef } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import {
  fetchConversations, newConversation, openConversation, removeConversation,
  sendMessage, setScope, stopStreaming,
} from '../store/chatSlice'
import { navigate } from '../store/uiSlice'
import ChatInput from '../components/ChatInput'
import ChatMessage from '../components/ChatMessage'
import { Button, Spinner } from '../components/ui'

const SUGGESTIONS = ['Summarize the main points of my documents', 'What are the key dates or numbers mentioned?', 'Which topics do these documents cover?']

export default function Chat() {
  const dispatch = useDispatch()
  const { conversations, activeId, messages, streaming, scopeDocumentIds, loadingConversation } = useSelector((s) => s.chat)
  const docs = useSelector((s) => s.documents.items).filter((d) => d.status === 'indexed')
  const bottom = useRef(null)

  useEffect(() => { dispatch(fetchConversations()) }, [dispatch])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages])

  const noDocs = docs.length === 0
  const toggleScope = (id) =>
    dispatch(setScope(scopeDocumentIds.includes(id) ? scopeDocumentIds.filter((x) => x !== id) : [...scopeDocumentIds, id]))

  return (
    <div className="-m-4 flex h-[calc(100vh-3.25rem)] md:-m-8 md:h-screen">
      {/* conversation list */}
      <aside className="hidden w-60 shrink-0 flex-col border-r border-slate-200 bg-white lg:flex">
        <div className="p-3">
          <Button className="w-full" onClick={() => dispatch(newConversation())}>+ New conversation</Button>
        </div>
        <ul className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
          {conversations.length === 0 && <li className="px-3 py-2 text-xs text-slate-400">No conversations yet</li>}
          {conversations.map((c) => (
            <li key={c.id} className={`group flex items-center rounded-lg ${c.id === activeId ? 'bg-indigo-50' : 'hover:bg-slate-100'}`}>
              <button onClick={() => dispatch(openConversation(c.id))} className="min-w-0 flex-1 truncate px-3 py-2 text-left text-sm text-slate-700" title={c.title}>
                {c.title}
              </button>
              <button
                aria-label={`Delete conversation ${c.title}`}
                onClick={() => dispatch(removeConversation(c.id))}
                className="px-2 text-slate-400 opacity-0 hover:text-rose-600 focus:opacity-100 group-hover:opacity-100"
              >✕</button>
            </li>
          ))}
        </ul>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* scope bar */}
        <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-white px-4 py-2 text-xs">
          <Button variant="secondary" className="lg:hidden" onClick={() => dispatch(newConversation())}>New</Button>
          <span className="text-slate-500">Search in:</span>
          <button
            onClick={() => dispatch(setScope([]))}
            className={`rounded-full px-2.5 py-1 ring-1 ring-inset ${scopeDocumentIds.length === 0 ? 'bg-indigo-600 text-white ring-indigo-600' : 'bg-white text-slate-600 ring-slate-300 hover:bg-slate-50'}`}
          >All documents</button>
          {docs.map((d) => (
            <button
              key={d.id}
              onClick={() => toggleScope(d.id)}
              title={d.filename}
              className={`max-w-[10rem] truncate rounded-full px-2.5 py-1 ring-1 ring-inset ${scopeDocumentIds.includes(d.id) ? 'bg-indigo-600 text-white ring-indigo-600' : 'bg-white text-slate-600 ring-slate-300 hover:bg-slate-50'}`}
            >{d.filename}</button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-6" aria-live="polite">
          <div className="mx-auto max-w-3xl space-y-5">
            {loadingConversation && <div className="flex justify-center text-slate-400"><Spinner className="h-5 w-5" /></div>}
            {!loadingConversation && messages.length === 0 && (
              <div className="py-16 text-center">
                <h1 className="text-xl font-semibold">Ask your documents anything</h1>
                {noDocs ? (
                  <p className="mt-2 text-sm text-slate-500">
                    You have no indexed documents yet.{' '}
                    <button className="font-medium text-indigo-600 underline" onClick={() => dispatch(navigate('documents'))}>Upload some</button> first.
                  </p>
                ) : (
                  <div className="mt-5 flex flex-wrap justify-center gap-2">
                    {SUGGESTIONS.map((s) => (
                      <button key={s} onClick={() => dispatch(sendMessage(s))} className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-600 hover:border-indigo-300 hover:text-indigo-700">{s}</button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {messages.map((m, i) => (
              <ChatMessage key={i} message={m} streaming={streaming && i === messages.length - 1 && !m.pending} />
            ))}
            <div ref={bottom} />
          </div>
        </div>

        <ChatInput
          streaming={streaming}
          disabled={noDocs}
          placeholder={noDocs ? 'Upload a document to start chatting' : 'Ask a question about your documents…'}
          onSend={(t) => dispatch(sendMessage(t))}
          onStop={() => dispatch(stopStreaming())}
        />
      </div>
    </div>
  )
}
