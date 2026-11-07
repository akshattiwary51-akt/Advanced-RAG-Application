export function formatBytes(n) {
  if (n == null) return '–'
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

export function formatDate(iso) {
  if (!iso) return '–'
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export const ACCEPTED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.csv']
export const MAX_UPLOAD_MB = 25

/** Client-side pre-check; the server remains the source of truth. */
export function checkFile(file) {
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
  if (!ACCEPTED_EXTENSIONS.includes(ext)) return `Unsupported type "${ext || 'none'}"`
  if (file.size === 0) return 'File is empty'
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `Larger than ${MAX_UPLOAD_MB} MB`
  return null
}

/** Turn [1] markers into links the markdown renderer turns into citation badges. */
export function linkifyCitations(text) {
  return text.replace(/\[(\d{1,3})\]/g, '[$1](cite:$1)')
}
