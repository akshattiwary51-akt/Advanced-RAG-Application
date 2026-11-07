import { useDispatch, useSelector } from 'react-redux'
import { fetchDocuments } from '../store/documentsSlice'
import Dropzone from '../components/Dropzone'
import DocumentTable from '../components/DocumentTable'
import { EmptyState, ErrorBanner, Spinner } from '../components/ui'

export default function Documents() {
  const dispatch = useDispatch()
  const { items, loading, error } = useSelector((s) => s.documents)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Documents</h1>
        <p className="text-sm text-slate-500">Upload files to index them for question answering</p>
      </div>

      <Dropzone />

      <div>
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-700">
          Your documents ({items.length}) {loading && <Spinner className="h-3.5 w-3.5 text-slate-400" />}
        </h2>
        {error && <ErrorBanner onRetry={() => dispatch(fetchDocuments())}>{error}</ErrorBanner>}
        {!error && items.length === 0 && !loading && (
          <EmptyState title="No documents yet">Upload a PDF, DOCX, TXT or CSV above to get started.</EmptyState>
        )}
        {items.length > 0 && <DocumentTable documents={items} />}
      </div>
    </div>
  )
}
