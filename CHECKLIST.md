# Implementation Checklist

Resume-safe progress tracker. Update after every phase. `[x]` = implemented AND tests run.

## Environment notes (measured, not assumed)
- Python 3.13, Node 22, npm 10 available in the build sandbox.
- Ollama is NOT installed in the sandbox; Hugging Face is unreachable there, so
  all-MiniLM-L6-v2 weights cannot be downloaded during automated tests.
  Core tests therefore use `MockEmbedder` / `MockLLM`. Real-model behaviour must be
  verified on your machine (see README).

## Architecture (data flow)
```
upload -> validate -> SHA-256 dedupe -> save -> extract (pages) -> clean
      -> chunk (paragraph/sentence aware, overlap) -> embed (batch, normalised)
      -> VectorStore.add (NumPy matrix + JSON metadata, persisted) + SQLite metadata

query -> preprocess -> embed -> retrieve (cosine, threshold, filter, dedupe)
      -> [rerank] -> build context -> injection-safe prompt -> Ollama (stream)
      -> citations -> save conversation
```

## Phases
- [x] 1  Project init + architecture
- [x] 2  Loaders + text extraction
- [x] 3  Custom chunker
- [x] 4  Embeddings (batch, normalise, cache)
- [x] 5  NumPy vector store (persist, delete, rebuild, dedupe)
- [x] 6  Semantic retriever (top-K, threshold, filter, dedupe)
- [x] 7  Ollama LLM integration (stream, timeout, errors)
- [x] 8  Full RAG pipeline + citations
- [x] 9  Flask API + Swagger + SQLAlchemy
- [x] 10 React + Vite frontend
- [x] 11 End-to-end integration
- [x] 12 Advanced retrieval (BM25, RRF, expansion, rewrite, compression, rerank, MMR)
- [x] 13 Evaluation + performance (real measurements only)
- [x] 14 Docs + deployment instructions

## Verification log
- Backend: `python -m pytest` -> 253 passed (mock embeddings + mock LLM, offline).
- Frontend: `npm run build` OK; headless-Chromium E2E against live Flask+Vite: 18/18 checks
  (upload, duplicate, client-side rejection, streaming chat, citations, score display,
  chunk expansion, conversation restore, settings persistence, delete, mobile layout).
- NOT verified in the build sandbox (no network to Hugging Face, no Ollama): the real
  all-MiniLM-L6-v2 model and a real Ollama server. See README for local verification steps.
- Evaluation (mock embeddings): `python -m eval.run_eval` -> `backend/eval/reports/eval_mock_lexical.{md,json}`.
  Re-run with `--embedding-backend sentence-transformers` locally for real-model numbers.
- Resume notes: all phases implemented. Highest-value next steps: run real-model eval, add OCR, background ingestion.
