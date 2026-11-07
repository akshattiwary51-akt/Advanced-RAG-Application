# Browser end-to-end test

Drives the real UI (headless Chromium) against a live backend + Vite dev server.
The backend must run with the offline mock backends and an empty data directory:

```powershell
# terminal 1 (backend/)
$env:EMBEDDING_BACKEND="mock"; $env:LLM_BACKEND="mock"; $env:DATA_DIR="$env:TEMP\rag_e2e"; $env:DATABASE_URL="sqlite:///$env:TEMP/rag_e2e/rag.db"
python app.py
# terminal 2 (frontend/)
npm run dev
# terminal 3 (e2e/)
npm init -y; npm i -D playwright; npx playwright install chromium
node run_e2e.js
```

Re-create the empty data directory before every run (the test uploads files and expects no duplicates).
