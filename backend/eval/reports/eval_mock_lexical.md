# RAG Evaluation Report

> **IMPORTANT - mock embeddings.** This run used `mock-hash-bow`, a deterministic hashed
> bag-of-words embedder (no semantics, no stemming). It makes runs reproducible offline, but
> these numbers describe *lexical* matching. **They say nothing about all-MiniLM-L6-v2** and
> cannot show the benefit of semantic search on paraphrased questions. Re-run with
> `--embedding-backend sentence-transformers` on a machine with Hugging Face access.

## Setup

- Generated: 2026-11-07T08:43:12+00:00 | Python 3.13.18 | Linux-6.18.44-fc-v310-x86_64-with-glibc2.39
- Embedding model: `mock-hash-bow` (384-d) | Reranker: `lexical` | LLM: `mock-llm`
- Corpus: 6 documents -> 13 chunks (size 400, overlap 60)
- Questions: 36 (30 answerable, 6 unanswerable); k=5; 10 timing repeats per query
- Relevance: a chunk is relevant iff it comes from the expected document and contains the gold answer phrase.

**How to read these numbers.** The corpus is deliberately small, so every configuration's candidate pool
(>= 20) covers the *entire* 13-chunk index: reranking and MMR reorder the whole corpus, and Hit@5
saturates. With only 30 answerable questions, one question moves a metric by 3.3 points, so
differences of a few points are noise. Treat large, consistent gaps as signal and everything else as a tie.

## Retrieval: baseline vs optimized

| Configuration | P@3 | P@5 | R@5 | Hit@1 | Hit@3 | Hit@5 | MRR | dMRR vs base | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: semantic | 26.7% | 16.7% | 83.3% | 53.3% | 80.0% | 83.3% | 0.646 | - | 0.42 | 0.52 |
| bm25 (keyword only) | 31.1% | 18.7% | 93.3% | 86.7% | 93.3% | 93.3% | 0.900 | +39.4% | 0.07 | 0.13 |
| hybrid (RRF) | 31.1% | 18.7% | 93.3% | 80.0% | 93.3% | 93.3% | 0.861 | +33.4% | 0.42 | 0.51 |
| semantic + rerank | 31.1% | 18.7% | 93.3% | 80.0% | 93.3% | 93.3% | 0.861 | +33.4% | 0.56 | 0.66 |
| hybrid + rerank | 31.1% | 18.7% | 93.3% | 83.3% | 93.3% | 93.3% | 0.878 | +36.0% | 0.57 | 0.68 |
| hybrid + MMR | 25.6% | 18.0% | 90.0% | 53.3% | 76.7% | 90.0% | 0.658 | +1.9% | 0.44 | 0.52 |
| hybrid + query expansion | 27.8% | 18.7% | 93.3% | 50.0% | 83.3% | 93.3% | 0.662 | +2.6% | 0.52 | 0.58 |
| hybrid + rerank + MMR | 32.2% | 19.3% | 96.7% | 83.3% | 96.7% | 96.7% | 0.894 | +38.6% | 0.62 | 0.68 |

### MRR by question type

| Configuration | factoid (n=18) | keyword (n=4) | paraphrase (n=8) |
|---|---|---|---|
| baseline: semantic | 0.650 | 0.833 | 0.542 |
| bm25 (keyword only) | 0.861 | 1.000 | 0.938 |
| hybrid (RRF) | 0.796 | 1.000 | 0.938 |
| semantic + rerank | 0.796 | 1.000 | 0.938 |
| hybrid + rerank | 0.824 | 1.000 | 0.938 |
| hybrid + MMR | 0.661 | 0.812 | 0.573 |
| hybrid + query expansion | 0.611 | 0.875 | 0.671 |
| hybrid + rerank + MMR | 0.861 | 1.000 | 0.917 |

## Contextual compression

- Context size: 47,776 -> 24,838 characters (**48.0% smaller**)
- Answer phrase still present in 92.0% of 25 relevant retrieved chunks

## Similarity threshold tuning (semantic)

Best balanced accuracy at threshold **0.35** (answers 20.0% of answerable, rejects 100.0% of unanswerable).

**This threshold is specific to the mock embedder and must not be reused with MiniLM**; cosine score distributions differ between models. Re-tune with the real model.

| Threshold | Answer rate | Reject rate | Balanced acc. |
|---|---|---|---|
| 0.00 | 83.3% | 0.0% | 41.7% |
| 0.10 | 73.3% | 16.7% | 45.0% |
| 0.20 | 66.7% | 16.7% | 41.7% |
| 0.30 | 43.3% | 50.0% | 46.7% |
| 0.40 | 20.0% | 100.0% | 60.0% |
| 0.50 | 3.3% | 100.0% | 51.7% |
| 0.60 | 0.0% | 100.0% | 50.0% |
| 0.70 | 0.0% | 100.0% | 50.0% |
| 0.80 | 0.0% | 100.0% | 50.0% |
| 0.90 | 0.0% | 100.0% | 50.0% |

## Performance

| Stage | Cold (ms, all docs) | Warm re-index (ms) |
|---|---|---|
| extract | 0.4 | 0.3 |
| clean | 0.5 | 0.3 |
| chunk | 0.5 | 0.4 |
| embed | 1.3 | 0.2 |
| store | 3.9 | 6.5 |
| total | 6.6 | 7.7 |

- Embedding cache: warm re-index served 13 chunk embeddings from cache.
- Vector store: 13 vectors, 19.5 KiB in memory, 26.2 KiB on disk.
- Process RSS: 43 MB at start -> 44 MB after ingest -> 46 MB after retrieval.

## Generation metrics

Not run: the mock LLM returns canned answers, so faithfulness/relevance scores would be meaningless. Re-run with `--llm ollama` to measure them.
