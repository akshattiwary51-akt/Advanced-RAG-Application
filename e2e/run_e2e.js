const { chromium } = require('playwright')
const path = require('path')
const os = require('os')

// Config via env so the script runs anywhere. DOCS_DIR must contain solar.txt and bread.txt
// (backend/eval/dataset/docs does). Screenshots go to SHOTS_DIR.
const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:5173'
const DOCS_DIR = process.env.DOCS_DIR || path.join(__dirname, '..', 'backend', 'eval', 'dataset', 'docs')
const SHOTS_DIR = process.env.SHOTS_DIR || os.tmpdir()

const results = []
const check = (name, ok, extra = '') => { results.push({ name, ok }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${extra}`) }

;(async () => {
  const browser = await chromium.launch()
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  const consoleErrors = []
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()) })
  page.on('pageerror', (e) => consoleErrors.push(String(e)))

  await page.goto(BASE_URL)
  await page.getByText('System status and index overview').waitFor()
  check('dashboard renders', true)
  await page.getByText('mock-hash-bow').first().waitFor()
  check('dashboard shows embedding model from API', true)

  // --- upload two files through the real file input
  await page.getByRole('button', { name: 'Documents' }).click()
  await page.getByText('No documents yet').waitFor()
  check('documents empty state', true)
  await page.setInputFiles('input[type=file]', [path.join(DOCS_DIR, 'solar.txt'), path.join(DOCS_DIR, 'bread.txt')])
  await page.getByText('Indexed', { exact: true }).first().waitFor({ timeout: 15000 })
  await page.getByText('Your documents (2)').waitFor({ timeout: 10000 })
  check('multi-file upload indexes both documents', true)

  // --- duplicate upload is reported
  await page.setInputFiles('input[type=file]', [path.join(DOCS_DIR, 'solar.txt')])
  await page.getByText('Already uploaded', { exact: true }).waitFor({ timeout: 10000 })
  check('duplicate upload shows "Already uploaded"', true)

  // --- unsupported type rejected client-side
  await page.setInputFiles('input[type=file]', [{ name: 'evil.exe', mimeType: 'application/octet-stream', buffer: Buffer.from('MZ') }])
  await page.getByText('Unsupported type ".exe"').waitFor()
  check('unsupported file rejected with message', true)

  // --- document details expand
  await page.locator('button[aria-expanded]').first().click()
  await page.getByText('SHA-256').waitFor()
  check('document details expand', true)
  await page.screenshot({ path: path.join(SHOTS_DIR, 'shot_documents.png') })

  // mock embeddings give low cosine scores; lower the UI threshold via the real Settings control
  await page.getByRole('button', { name: 'Settings' }).click()
  await page.getByText('Saved in this browser').waitFor()
  await page.locator('input[type=range]').nth(1).fill('0.05')

  // --- chat with streaming
  await page.getByRole('button', { name: 'Chat' }).click()
  await page.getByText('Ask your documents anything').waitFor()
  await page.getByLabel('Message').fill('How efficient are solar panels?')
  await page.getByLabel('Message').press('Enter')
  await page.getByText('According to the documents').waitFor({ timeout: 15000 })
  check('streamed answer appears', true)
  await page.getByText(/\d+ sources?/).first().waitFor()
  await page.getByText(/\d+ sources?/).first().click()
  await page.getByText('solar.txt').last().waitFor()
  check('source citation list shows filename', true)
  const score = await page.locator('[id^=src-] [title="Cosine similarity"] span').first().innerText()
  check('similarity score displayed', /^\d\.\d{3}$/.test(score), `(${score})`)
  await page.locator('[id^=src-] button').first().click()
  await page.locator('#src-1 p').waitFor()
  const chunkText = await page.locator('#src-1 p').innerText()
  if (chunkText.length < 40) throw new Error('expanded chunk text too short: ' + chunkText)
  check('retrieved chunk expands to full text', true)
  await page.screenshot({ path: path.join(SHOTS_DIR, 'shot_chat.png') })

  // --- conversation persisted & listed
  await page.getByRole('button', { name: /How efficient are solar/ }).first().waitFor()
  check('conversation appears in history', true)
  await page.getByRole('button', { name: '+ New conversation' }).click()
  await page.getByText('Ask your documents anything').waitFor()
  await page.getByRole('button', { name: /How efficient are solar/ }).first().click()
  await page.getByText('According to the documents').waitFor()
  check('reopening a past conversation restores messages', true)

  // --- ungrounded answer for irrelevant question (raise threshold in settings first)
  await page.getByRole('button', { name: 'Settings' }).click()
  await page.getByText('Saved in this browser').waitFor()
  await page.locator('input[type=range]').nth(1).fill('0.6')
  await page.getByRole('button', { name: 'Chat' }).click()
  await page.getByRole('button', { name: '+ New conversation' }).click()
  await page.getByLabel('Message').fill('quantum chromodynamics lattice gauge theory')
  await page.getByLabel('Message').press('Enter')
  await page.getByText("couldn't find relevant information").waitFor({ timeout: 15000 })
  check('irrelevant question gets insufficient-information answer', true)

  // --- settings persisted across reload
  await page.reload()
  await page.getByRole('button', { name: 'Settings' }).click()
  const v = await page.locator('input[type=range]').nth(1).inputValue()
  check('settings persist across reload', v === '0.6', `(threshold=${v})`)

  // --- delete document
  await page.getByRole('button', { name: 'Documents' }).click()
  await page.getByText('Your documents (2)').waitFor()
  await page.getByRole('button', { name: 'Delete' }).first().click()
  await page.getByRole('button', { name: 'Confirm' }).click()
  await page.getByText('Your documents (1)').waitFor({ timeout: 10000 })
  check('delete document (with confirm) updates list', true)

  await page.getByRole('button', { name: 'Chat' }).click()
  await page.screenshot({ path: path.join(SHOTS_DIR, 'shot_final.png') })
  await page.setViewportSize({ width: 390, height: 800 })
  await page.getByRole('button', { name: 'Dashboard' }).click()
  await page.getByText('System status and index overview').waitFor()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
  check('mobile viewport has no horizontal overflow', !overflow)
  await page.screenshot({ path: path.join(SHOTS_DIR, 'shot_mobile.png') })

  // the 409 is the deliberately-triggered duplicate upload (browsers log failed fetches)
  const unexpected = consoleErrors.filter((e) => !e.includes('409'))
  check('no unexpected browser console errors', unexpected.length === 0, unexpected.slice(0, 3).join(' | '))
  await browser.close()
  const failed = results.filter((r) => !r.ok).length
  console.log(`\n${results.length - failed}/${results.length} checks passed`)
  process.exit(failed ? 1 : 0)
})().catch((e) => { console.error('E2E CRASH:', e.message.split('\n').slice(0, 6).join('\n')); process.exit(2) })
