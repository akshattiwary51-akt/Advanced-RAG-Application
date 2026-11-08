# RAG Evaluation Report

## Setup

- Generated: 2026-11-08T02:27:59+00:00 | Python 3.11.9 | Windows-10-10.0.26300-SP0
- Embedding model: `all-MiniLM-L6-v2` (384-d) | Reranker: `lexical` | LLM: `mock-llm`
- Corpus: 6 documents -> 13 chunks (size 400, overlap 60)
- Questions: 36 (30 answerable, 6 unanswerable); k=5; 5 timing repeats per query
- Relevance: a chunk is relevant iff it comes from the expected document and contains the gold answer phrase.

**How to read these numbers.** The corpus is deliberately small, so every configuration's candidate pool
(>= 20) covers the *entire* 13-chunk index: reranking and MMR reorder the whole corpus, and Hit@5
saturates. With only 30 answerable questions, one question moves a metric by 3.3 points, so
differences of a few points are noise. Treat large, consistent gaps as signal and everything else as a tie.

## Retrieval: baseline vs optimized

| Configuration | P@3 | P@5 | R@5 | Hit@1 | Hit@3 | Hit@5 | MRR | dMRR vs base | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: semantic | 33.3% | 20.0% | 100.0% | 96.7% | 100.0% | 100.0% | 0.983 | - | 16.10 | 24.85 |
| bm25 (keyword only) | 31.1% | 18.7% | 93.3% | 86.7% | 93.3% | 93.3% | 0.900 | -8.5% | 11.49 | 20.71 |
| hybrid (RRF) | 33.3% | 20.0% | 100.0% | 90.0% | 100.0% | 100.0% | 0.950 | -3.4% | 15.39 | 25.09 |
| semantic + rerank | 33.3% | 20.0% | 100.0% | 90.0% | 100.0% | 100.0% | 0.950 | -3.4% | 11.90 | 18.70 |
| hybrid + rerank | 33.3% | 20.0% | 100.0% | 90.0% | 100.0% | 100.0% | 0.950 | -3.4% | 11.13 | 17.46 |
| hybrid + MMR | 33.3% | 20.0% | 100.0% | 96.7% | 100.0% | 100.0% | 0.983 | +0.0% | 15.62 | 24.13 |
| hybrid + query expansion | 33.3% | 20.0% | 100.0% | 56.7% | 100.0% | 100.0% | 0.767 | -22.0% | 27.20 | 45.03 |
| hybrid + rerank + MMR | 31.1% | 18.7% | 93.3% | 90.0% | 93.3% | 93.3% | 0.917 | -6.8% | 16.79 | 25.47 |

### MRR by question type

| Configuration | factoid (n=18) | keyword (n=4) | paraphrase (n=8) |
|---|---|---|---|
| baseline: semantic | 0.972 | 1.000 | 1.000 |
| bm25 (keyword only) | 0.861 | 1.000 | 0.938 |
| hybrid (RRF) | 0.944 | 1.000 | 0.938 |
| semantic + rerank | 0.944 | 1.000 | 0.938 |
| hybrid + rerank | 0.944 | 1.000 | 0.938 |
| hybrid + MMR | 0.972 | 1.000 | 1.000 |
| hybrid + query expansion | 0.741 | 1.000 | 0.708 |
| hybrid + rerank + MMR | 0.889 | 1.000 | 0.938 |

## Contextual compression

- Context size: 47,516 -> 27,473 characters (**42.2% smaller**)
- Answer phrase still present in 100.0% of 30 relevant retrieved chunks

## Similarity threshold tuning (semantic)

Best balanced accuracy at threshold **0.30** (answers 96.7% of answerable, rejects 100.0% of unanswerable).

| Threshold | Answer rate | Reject rate | Balanced acc. |
|---|---|---|---|
| 0.00 | 100.0% | 0.0% | 50.0% |
| 0.10 | 100.0% | 33.3% | 66.7% |
| 0.20 | 100.0% | 83.3% | 91.7% |
| 0.30 | 96.7% | 100.0% | 98.3% |
| 0.40 | 83.3% | 100.0% | 91.7% |
| 0.50 | 73.3% | 100.0% | 86.7% |
| 0.60 | 46.7% | 100.0% | 73.3% |
| 0.70 | 10.0% | 100.0% | 55.0% |
| 0.80 | 0.0% | 100.0% | 50.0% |
| 0.90 | 0.0% | 100.0% | 50.0% |

## Performance

| Stage | Cold (ms, all docs) | Warm re-index (ms) |
|---|---|---|
| extract | 91.1 | 3.7 |
| clean | 1.8 | 1.2 |
| chunk | 2.1 | 1.5 |
| embed | 432.9 | 0.6 |
| store | 37.5 | 56.3 |
| total | 565.4 | 63.3 |

- Embedding cache: warm re-index served 13 chunk embeddings from cache.
- Vector store: 13 vectors, 19.5 KiB in memory, 26.2 KiB on disk.
- Process RSS: 41 MB at start -> 520 MB after ingest -> 516 MB after retrieval.

## Generation metrics

Not run: the mock LLM returns canned answers, so faithfulness/relevance scores would be meaningless. Re-run with `--llm ollama` to measure them.
