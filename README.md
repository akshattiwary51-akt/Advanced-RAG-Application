# Advanced RAG Document Assistant

Upload PDF / DOCX / TXT / CSV files, ask questions in natural language, and get answers that are
grounded in your documents, with numbered citations back to the exact source chunks.

The retrieval engine is implemented from scratch: chunking, embedding cache, a NumPy vector store,
cosine similarity, BM25, Reciprocal Rank Fusion, MMR, query expansion and compression are all plain
Python/NumPy in `backend/src/`. No LangChain, LlamaIndex, Chroma, FAISS or Pinecone.

- **Backend:** Python, Flask, Flask-CORS, Flasgger (Swagger), SQLAlchemy + SQLite
- **RAG:** Sentence Transformers (`all-MiniLM-L6-v2`), NumPy, Ollama (local LLM)
- **Frontend:** React 18, Vite, Tailwind CSS 4, Redux Toolkit, Axios

---

## 1. Quick start (Windows / PowerShell)

Prerequisites: **Python 3.10+**, **Node.js 18+**, and [Ollama](https://ollama.com/download) for answers.

```powershell
# --- Ollama (once) -------------------------------------------------------
ollama pull llama3.2          # any chat model works; set OLLAMA_MODEL to match
ollama serve                  # skip if the Ollama app is already running

# --- Backend (terminal 1) --------------------------------------------------
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
Copy-Item .env.example .env   # edit if needed (model name, ports, ...)
python app.py                 # http://127.0.0.1:5000   Swagger: /apidocs

# --- Frontend (terminal 2) -------------------------------------------------
cd frontend
npm install
npm run dev                   # http://localhost:5173  (proxies /api to Flask)
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face. The dashboard shows whether
Ollama is connected; documents can be uploaded and searched even if it is not.

**Offline / no model download:** set `EMBEDDING_BACKEND=mock` and `LLM_BACKEND=mock` in `.env`. This uses a
deterministic hashing embedder and a canned LLM, which is what the automated tests use. It is for testing only.

### Run the tests

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pytest              # 253 tests, ~3 s, no model download and no Ollama needed
```

### Production build of the frontend

```powershell
cd frontend; npm run build    # output in frontend/dist (serve with any static host, proxy /api to Flask)
```

---

## 2. Architecture

```
upload -> validate (ext, MIME, size, magic bytes) -> SHA-256 dedupe -> save under generated name
      -> extract per page -> clean -> chunk (paragraph + sentence aware, overlap)
      -> embed (batched, L2-normalised, cached) -> NumpyVectorStore (vectors.npy + records.json) + SQLite row

question -> preprocess -> [rewrite using chat history] -> retrieve
      -> semantic | bm25 | hybrid(RRF) -> [expand] -> dedupe -> [rerank] -> [MMR] -> top-K -> [compress]
      -> number + sanitise context -> injection-resistant prompt -> Ollama (streaming)
      -> validate citations -> save conversation
```

| Module (`backend/`) | Responsibility |
|---|---|
| `src/loaders.py` | validation, SHA-256, PDF/DOCX/TXT/CSV extraction, typed errors |
| `src/text_processor.py` | cleaning, paragraph/sentence splitting, tokenizer, query preprocessing |
| `src/chunker.py` | custom chunking algorithm + `Chunk` metadata |
| `src/embeddings.py` | Sentence Transformers + mock backend, batching, content-addressed cache |
| `src/vector_store.py` | NumPy vector store, cosine math, atomic persistence, rebuild |
| `src/retriever.py` | baseline semantic retriever, threshold, document filter, de-duplication |
| `src/bm25.py`, `src/advanced_retrieval.py` | BM25, hybrid/RRF, MMR, expansion, rewriting, compression, rerankers |
| `src/prompt_builder.py` | context budget, sanitising, prompt, citation extraction/validation |
| `src/llm.py` | Ollama client (stream, timeouts, errors) + mock |
| `src/rag_pipeline.py` | `process_document()` and `answer_query()` / `stream_answer()` orchestration |
| `src/document_manager.py`, `src/conversation_manager.py` | lifecycle + persistence, metadata/vector reconciliation |
| `routes/`, `utils/`, `app.py` | thin Flask layer, validation, central error mapping |
| `eval/` | metrics, dataset, runner and saved reports |

Business logic lives in `src/`; route handlers only parse requests and call it.

### How the main pieces work

- **Chunking** (`chunker.py`): split into paragraphs, then sentences (abbreviation/initial/decimal aware);
  pack sentences greedily up to `CHUNK_SIZE`; start the next chunk with the trailing whole sentences that fit in
  `CHUNK_OVERLAP`; a too-small chunk may grow to 1.25x instead of closing; a tiny last chunk is merged back.
  Chunks never cross page boundaries, so page numbers are exact.
- **Cosine similarity** (`vector_store.py`): `dot(A,B) / (norm(A) * norm(B))` with explicit norms, vectorised
  as one matrix-vector product; top-K by `argpartition`. Exact search, no approximation.
- **Persistence:** `vectors.npy` + `records.json`, written to temp files and swapped in with `os.replace`
  so a crash cannot leave a half-written index. Dimension/model mismatch on load raises a clear "re-index" error.
- **Embedding cache:** keyed by SHA-256 of the text and namespaced by model name; re-indexing unchanged text
  makes no model calls.
- **Citations:** the prompt numbers sources `[1]..[n]`; the answer's `[n]` markers are validated against the
  sources actually supplied, so the model cannot cite a document it never saw (a warning is returned).
- **No-evidence guard:** if nothing clears the similarity threshold the LLM is **not called**; the API returns a
  fixed "couldn't find relevant information" answer with `grounded: false`.

---

## 3. Configuration (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `FLASK_HOST` / `FLASK_PORT` | 127.0.0.1 / 5000 | bind address |
| `CORS_ORIGINS` | localhost:5173 | comma-separated allowed origins |
| `MAX_UPLOAD_MB` | 25 | per-file limit |
| `DATA_DIR`, `DATABASE_URL` | `data`, `sqlite:///data/rag.db` | storage (relative to `backend/`) |
| `EMBEDDING_MODEL` / `EMBEDDING_BACKEND` | all-MiniLM-L6-v2 / sentence-transformers | or `mock` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `MIN_CHUNK_SIZE` | 800 / 150 / 100 | characters |
| `TOP_K` / `SIMILARITY_THRESHOLD` / `RETRIEVAL_METHOD` | 5 / 0.25 / semantic | defaults; overridable per request |
| `RERANKER_BACKEND` / `RERANKER_MODEL` | lexical / ms-marco-MiniLM-L-6-v2 | `cross-encoder` downloads a model |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | localhost:11434, llama3.2 | LLM server and default model |
| `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT_SECONDS` | 0.2, 1024, 120 | generation |
| `LLM_BACKEND` | ollama | or `mock` |

There are no credentials anywhere; nothing secret is hardcoded or returned by the API.

---

## 4. API

Interactive docs: **http://127.0.0.1:5000/apidocs** (OpenAPI JSON at `/apispec_1.json`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/stats`, `/api/settings` | status, dashboard numbers, safe defaults |
| POST | `/api/documents/upload` | multipart field `files` (repeatable); per-file results |
| GET | `/api/documents`, `/api/documents/{id}` | list / details |
| DELETE | `/api/documents/{id}` | delete file, vectors and metadata |
| POST | `/api/documents/{id}/reindex` | re-chunk and re-embed (optional `chunk_size`, `chunk_overlap`) |
| POST | `/api/chat` | grounded answer + citations + timings |
| POST | `/api/chat/stream` | same, as Server-Sent Events |
| GET/DELETE | `/api/conversations`, `/api/conversations/{id}` | history |
| GET | `/api/debug/retrieval` | inspect chunks, scores and latency without calling the LLM |

Chat body: `message` (required), `conversation_id`, `document_ids`, `top_k` (1-20), `similarity_threshold`,
`temperature`, `max_tokens`, `model`, `method` (`semantic|bm25|hybrid`), and booleans `use_mmr`, `mmr_lambda`,
`rerank`, `expand_query`, `compress`, `rewrite_query`.

SSE events (`data: <json>`): `meta` -> `sources` -> many `token` -> `done` (or `error`).
Errors always use `{"error": {"code": "...", "message": "..."}}` with a proper status
(400 validation, 404, 409 duplicate, 413 too large, 415 type, 422 unreadable, 503 LLM down, 502 LLM error).

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
curl.exe -F "files=@C:\docs\report.pdf" http://127.0.0.1:5000/api/documents/upload
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/chat -ContentType application/json `
  -Body '{"message":"What does the report say about revenue?","method":"hybrid"}'
Invoke-RestMethod "http://127.0.0.1:5000/api/debug/retrieval?q=revenue&method=hybrid&top_k=5"
```

---

## 5. Advanced retrieval: benefit, implementation, trade-off

Every option is off by default (the baseline path is untouched) and individually testable.

| # | Technique | Benefit | Implementation | Trade-off |
|---|---|---|---|---|
| 1 | **Hybrid search** (semantic + BM25) | exact names, codes and rare terms that embeddings blur | `bm25.py` (Okapi BM25 from the formula, rebuilt when the store's version changes) | second index in RAM; keyword hits can be false friends, so lexical-only hits must match >= 2 query terms |
| 2 | **Reciprocal Rank Fusion** | fuses rankings with different score scales without normalising | `rrf_fuse`: sum of `1/(60 + rank)` | uses ranks only; ignores how confident each retriever was |
| 3 | **Query expansion** | recovers passages that use different vocabulary | pseudo-relevance feedback: append top `tf*idf` terms from the top-3 results | can drift off-topic when the first pass is wrong (it did not help in our test, see below) |
| 4 | **Query rewriting** | resolves "it / that / they" in follow-ups | one low-temperature LLM call over recent history, with strict output checks and fallback to the original | +1 LLM call latency; a wrong rewrite hurts retrieval |
| 5 | **Contextual compression** | smaller prompts, less noise | keep sentences whose cosine to the query is >= 0.6x the best sentence | may drop a sentence the answer needs |
| 6 | **Cross-encoder reranking** | most accurate ordering (reads query and passage together) | `CrossEncoderReranker` (`RERANKER_BACKEND=cross-encoder`); a model-free `LexicalReranker` is the default | one model pass per candidate; extra model download |
| 7 | **Similarity threshold tuning** | stops irrelevant context and enables "I don't know" | `eval.metrics.tune_threshold` sweeps thresholds on answerable vs unanswerable questions | scores are model-specific: re-tune when you change the embedding model |
| 8 | **MMR** | diverse, non-redundant context | `mmr_select`: `lambda*relevance - (1-lambda)*max_sim_to_selected` | may demote a second genuinely relevant chunk |
| 9 | **Context de-duplication** | no repeated passages in the prompt | exact-text and near-duplicate (cosine >= 0.97) removal, keeping the best copy | none significant |
| 10 | **Configurable top-K** | trade recall for prompt size | `top_k` 1-20, UI slider and API field | larger K adds noise and latency |

---

## 6. Evaluation (measured, reproducible)

```powershell
cd backend
python -m eval.run_eval                                                    # offline mock embeddings
python -m eval.run_eval --embedding-backend sentence-transformers          # real MiniLM (needs Hugging Face access)
python -m eval.run_eval --embedding-backend sentence-transformers --reranker cross-encoder --llm ollama
```

The dataset (`backend/eval/dataset`) has 6 short documents and 36 questions (30 answerable, 6 unanswerable;
factoid / keyword / paraphrase types). A chunk counts as relevant if it is from the right document and contains the
gold answer phrase, so scoring does not depend on chunk boundaries. Metrics: Precision@K, Recall@K, MRR,
Hit Rate@K, latency; generation proxies: faithfulness, answer relevance, context relevance, citation correctness.

### Results actually measured in the build sandbox

> **These used the offline `mock-hash-bow` embedder, not MiniLM**, because the sandbox could not reach Hugging Face
> (HTTP 403 from its proxy) and had no Ollama. The mock is a hashed bag-of-words model, so these numbers
> characterise lexical matching and **cannot show the benefit of semantic search on paraphrases**. The corpus is only
> 13 chunks and 30 answerable questions (one question = 3.3 points), so only large gaps are meaningful.

| Configuration | Hit@1 | Hit@3 | R@5 | MRR | mean latency |
|---|---|---|---|---|---|
| baseline: semantic | 53.3% | 80.0% | 83.3% | 0.646 | 0.42 ms |
| bm25 (keyword only) | 86.7% | 93.3% | 93.3% | 0.900 | 0.07 ms |
| hybrid (RRF) | 80.0% | 93.3% | 93.3% | 0.861 | 0.42 ms |
| semantic + rerank | 80.0% | 93.3% | 93.3% | 0.861 | 0.56 ms |
| hybrid + rerank | 83.3% | 93.3% | 93.3% | 0.878 | 0.57 ms |
| hybrid + MMR | 53.3% | 76.7% | 90.0% | 0.658 | 0.44 ms |
| hybrid + query expansion | 50.0% | 83.3% | 93.3% | 0.662 | 0.52 ms |
| hybrid + rerank + MMR | 83.3% | 96.7% | 96.7% | 0.894 | 0.62 ms |

What this does and does not tell you:

- Keyword/BM25, fusion and reranking clearly beat the (mock) semantic baseline; that is expected, because the mock
  embedder has no stemming (`panel` != `panels`) while BM25 normalises term statistics.
  Do not read it as "BM25 beats MiniLM".
- Plain MMR and query expansion did **not** help here (changes within noise). MMR trades relevance for diversity
  and the questions each have one answer chunk; expansion drifted on short factoid queries.
- Compression made the context **48.0% smaller** while keeping the answer phrase in **92.0%** of relevant chunks,
  which is a real quality cost to weigh.
- The tuned similarity threshold from this run (0.35) belongs to the mock embedder. Re-tune with MiniLM.
- Generation metrics were **not run** (mock LLM would make them meaningless); use `--llm ollama`.

Full tables, ingestion timings (cold vs cached), memory and threshold sweep:
`backend/eval/reports/eval_mock_lexical.md` (and `.json`). Generation metrics are cheap lexical/embedding proxies,
good for regression tracking and catching bad citations, not a substitute for an LLM judge or human review.

---

## 7. Security and reliability

- File type checks on extension, declared MIME and magic bytes; size limit enforced while streaming to disk.
- Uploaded files are stored under generated names (`<uuid>.<ext>`); user filenames are only display metadata,
  so path traversal is impossible by construction (tested with `../../etc/evil.txt`).
- SHA-256 duplicate detection; failed ingests leave no file, row or vectors behind.
- Startup **reconciliation** repairs drift between SQLite, vectors and files (orphans, interrupted ingests).
- **Prompt-injection resistance:** retrieved text is wrapped as untrusted data, role markers/closing tags inside
  documents are neutralised, the system prompt forbids following instructions inside sources, the user question
  sits outside the data block, and cited numbers are validated. This reduces risk; it cannot eliminate it, so keep
  a human in the loop for high-stakes use.
- Markdown from the model is rendered with a URL allow-list (`http(s)`, `mailto`, citation links); `javascript:` links are dropped.
- Central error handler returns generic 500s with no traces or server paths; CORS is limited to configured origins.
- **There is no authentication.** Run it on localhost or behind your own auth/reverse proxy before exposing it.

---

## 8. What is verified, and what is not

Verified here (automated):

- 253 backend tests pass (loaders, cleaning, chunking, embeddings, cosine, store persistence, retrieval, prompt/citations,
  pipeline, every endpoint including SSE, delete/re-index consistency, advanced retrieval, metrics, evaluation runner).
- Frontend production build succeeds; a headless-Chromium run against the live Flask + Vite stack passed 18/18 checks
  (multi-file upload, duplicate and bad-type handling, streaming chat, citations and scores, chunk expansion,
  conversation restore, settings persistence, deletion, no horizontal scroll at phone width). Script: `e2e/`.

**Not verified in the build sandbox** (no Hugging Face access, no Ollama): the real `all-MiniLM-L6-v2` model, the
cross-encoder model, and a live Ollama server. The code paths are written to the library/HTTP contracts and the
Ollama client is tested against mocked HTTP responses, but **please run the app and
`python -m eval.run_eval --embedding-backend sentence-transformers` once on your machine** and treat that as the first
real end-to-end check. Not covered: OCR for scanned PDFs, authentication, multi-user isolation.

---

## 9. Scaling recommendations

1. **Vector search:** brute force is exact and fast to ~10^5 chunks. Beyond that, swap `NumpyVectorStore` for
   pgvector/Qdrant/FAISS behind the same small interface (`add / search / delete_document / stats`).
2. **Ingestion:** move it to a background worker queue (Celery/RQ) with status polling; today it is synchronous.
3. **Database:** SQLite -> PostgreSQL for concurrent writers; add a `users` table and `user_id` on documents and conversations.
4. **Serving:** run behind gunicorn/uvicorn workers; load the embedding model once per worker or move it to a model server.
5. **Quality:** add a cross-encoder reranker, an LLM-judge evaluation, OCR, and per-document access control.
6. **Observability:** structured logs with request ids, retrieval/latency metrics, and the evaluation set in CI.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard shows "LLM offline" | start Ollama (`ollama serve`) |
| "model not installed" | `ollama pull <model>` and match `OLLAMA_MODEL` |
| Every answer is "couldn't find relevant information" | lower the similarity threshold (Settings); check `/api/debug/retrieval?q=...` scores |
| "Re-index required" error at start-up | you changed `EMBEDDING_MODEL`; delete `backend/data/vector_store` and re-index documents |
| Scanned PDF rejected | it has no text layer; OCR is not supported |
| `Activate.ps1` blocked | `Set-ExecutionPolicy -Scope Process Bypass` |
| CORS errors | use `npm run dev` (proxy) or add your origin to `CORS_ORIGINS` |
