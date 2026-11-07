"""Reproducible RAG evaluation: retrieval quality, performance, and (optionally) generation.

Usage (from backend/):
    python -m eval.run_eval                                   # mock embeddings, offline
    python -m eval.run_eval --embedding-backend sentence-transformers --reranker cross-encoder
    python -m eval.run_eval --embedding-backend sentence-transformers --llm ollama

Everything reported is measured by this script on this machine; nothing is estimated.
With the mock embedder the numbers describe a *lexical* hashing model, not MiniLM, and the
report says so at the top.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import psutil

from config import get_settings
from src.advanced_retrieval import build_reranker
from src.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder, SentenceTransformerEmbedder
from src.llm import GenerationParams, MockLLM, OllamaClient
from src.rag_pipeline import RAGPipeline, RetrievalOptions
from src.vector_store import NumpyVectorStore
from .metrics import (
    answer_relevance, best_threshold, citation_correctness, context_relevance, faithfulness,
    hit_at_k, precision_at_k, recall_at_k, reciprocal_rank, relative_change, summarize_latency,
    tune_threshold,
)

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"

CONFIGS: list[tuple[str, dict]] = [
    ("baseline: semantic", dict(method="semantic")),
    ("bm25 (keyword only)", dict(method="bm25")),
    ("hybrid (RRF)", dict(method="hybrid")),
    ("semantic + rerank", dict(method="semantic", rerank=True)),
    ("hybrid + rerank", dict(method="hybrid", rerank=True)),
    ("hybrid + MMR", dict(method="hybrid", use_mmr=True)),
    ("hybrid + query expansion", dict(method="hybrid", expand_query=True)),
    ("hybrid + rerank + MMR", dict(method="hybrid", rerank=True, use_mmr=True)),
]


# ------------------------------------------------------------------- dataset
def load_dataset(dataset_dir: Path = DATASET) -> tuple[dict[str, Path], list[dict]]:
    docs = {p.stem: p for p in sorted((dataset_dir / "docs").glob("*.txt"))}
    questions = json.loads((dataset_dir / "questions.json").read_text(encoding="utf-8"))["questions"]
    return docs, questions


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def is_relevant(record_or_chunk_doc: str, text: str, q: dict) -> bool:
    """Gold relevance: right document AND contains the answer phrase (chunking-independent)."""
    return q["document"] is not None and record_or_chunk_doc == q["document"] \
        and norm(q["must_contain"]) in norm(text)


def count_relevant(store: NumpyVectorStore, q: dict) -> int:
    return sum(is_relevant(r["document_id"], r["text"], q) for r in store.records())


# ----------------------------------------------------------------- pipeline
def make_pipeline(tmp: Path, svc: EmbeddingService, reranker, llm, chunk_size: int, overlap: int) -> RAGPipeline:
    store = NumpyVectorStore(tmp / "vs", svc.dimension, svc.model_name)
    return RAGPipeline(svc, store, llm, chunk_size, overlap, min_chunk_size=min(100, chunk_size // 2),
                       reranker=reranker)


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def ingest(pipe: RAGPipeline, docs: dict[str, Path]) -> dict:
    agg = {k: 0.0 for k in ("extract", "clean", "chunk", "embed", "store", "total")}
    chunks = 0
    for name, path in docs.items():
        res = pipe.process_document(path, name, path.name, ".txt")
        chunks += res.chunk_count
        for k in agg:
            agg[k] += res.timings_ms[k]
    return {"documents": len(docs), "chunks": chunks, "timings_ms": agg}


# --------------------------------------------------------------- retrieval eval
def evaluate_config(pipe: RAGPipeline, questions: list[dict], opts: RetrievalOptions,
                    k: int, repeats: int) -> dict:
    per_q, lat = [], []
    for q in questions:
        times = []
        results = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            results = pipe.retrieve_relevant_chunks(q["question"], opts)
            times.append((time.perf_counter() - t0) * 1000.0)
        lat.extend(times)
        if q["document"] is None:
            continue
        rel = [is_relevant(c.document_id, c.original_text or c.text, q) for c in results]
        total = count_relevant(pipe.store, q)
        per_q.append({"type": q["type"], "rel": rel, "total": total})

    def mean(f):
        return float(np.mean([f(x) for x in per_q])) if per_q else 0.0

    out = {
        "answerable_queries": len(per_q),
        f"precision@{k}": mean(lambda x: precision_at_k(x["rel"], k)),
        "precision@3": mean(lambda x: precision_at_k(x["rel"], 3)),
        f"recall@{k}": mean(lambda x: recall_at_k(x["rel"], k, x["total"])),
        "hit@1": mean(lambda x: hit_at_k(x["rel"], 1)),
        "hit@3": mean(lambda x: hit_at_k(x["rel"], 3)),
        f"hit@{k}": mean(lambda x: hit_at_k(x["rel"], k)),
        "mrr": mean(lambda x: reciprocal_rank(x["rel"])),
        "latency_ms": summarize_latency(lat),
        "mrr_by_type": {},
    }
    for t in sorted({x["type"] for x in per_q}):
        sub = [x for x in per_q if x["type"] == t]
        out["mrr_by_type"][t] = {"n": len(sub),
                                 "mrr": float(np.mean([reciprocal_rank(x["rel"]) for x in sub]))}
    return out


def compression_study(pipe: RAGPipeline, questions: list[dict], k: int) -> dict:
    """Does contextual compression shrink context without losing the answer phrase?"""
    before = after = kept = relevant = 0
    for q in questions:
        if q["document"] is None:
            continue
        for c in pipe.advanced.retrieve(q["question"], k, 0.0, method="semantic", compress=True):
            orig = c.original_text or c.text
            before += len(orig)
            after += len(c.text)
            if is_relevant(c.document_id, orig, q):
                relevant += 1
                kept += is_relevant(c.document_id, c.text, q)
    return {"context_chars_before": before, "context_chars_after": after,
            "size_reduction_pct": (1 - after / before) * 100 if before else 0.0,
            "answer_phrase_retention": kept / relevant if relevant else 0.0,
            "relevant_chunks_checked": relevant}


def threshold_study(pipe: RAGPipeline, questions: list[dict], k: int) -> dict:
    per_query = []
    for q in questions:
        res = pipe.retriever.retrieve(q["question"], k, threshold=-1.0, dedupe=False)
        per_query.append({"answerable": q["document"] is not None,
                          "scored": [(c.score, is_relevant(c.document_id, c.text, q)) for c in res]})
    rows = tune_threshold(per_query, [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)])
    return {"rows": rows, "best": best_threshold(rows),
            "answerable": sum(p["answerable"] for p in per_query),
            "unanswerable": sum(not p["answerable"] for p in per_query)}


# -------------------------------------------------------------- generation eval
def generation_eval(pipe: RAGPipeline, questions: list[dict], opts: RetrievalOptions,
                    params: GenerationParams, embed) -> dict:
    rows, lat = [], []
    for q in questions:
        if q["document"] is None:
            continue
        t0 = time.perf_counter()
        res = pipe.answer_query(q["question"], None, opts, params)
        lat.append((time.perf_counter() - t0) * 1000.0)
        ctx = {s.id: s.text for s in res.sources}
        cc = citation_correctness(res.answer, ctx)
        rows.append({
            "faithfulness": faithfulness(res.answer, list(ctx.values())),
            "answer_relevance": answer_relevance(q["question"], res.answer, embed),
            "context_relevance": context_relevance(q["question"], list(ctx.values()), embed),
            "citation_precision": cc["precision"], "citation_valid_rate": cc["valid_rate"],
            "citation_coverage": cc["coverage"],
            "llm_generation_ms": res.timings_ms.get("generation", 0.0),
        })
    keys = rows[0].keys() if rows else []
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["questions"] = len(rows)
    out["end_to_end_latency_ms"] = summarize_latency(lat)
    return out


# ------------------------------------------------------------------------ run
def run_evaluation(svc: EmbeddingService, reranker, llm, *, k: int = 5, repeats: int = 5,
                   chunk_size: int = 400, overlap: int = 60, include_generation: bool = False,
                   dataset_dir: Path = DATASET) -> dict:
    docs, questions = load_dataset(dataset_dir)
    mem = {"start_mb": rss_mb()}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # cold ingestion (empty embedding cache), then warm re-index (cache hits)
        pipe = make_pipeline(tmp, svc, reranker, llm, chunk_size, overlap)
        t0 = time.perf_counter()
        cold = ingest(pipe, docs)
        cold["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        hits_before = svc.cache.hits if svc.cache else 0
        t0 = time.perf_counter()
        warm = ingest(pipe, docs)
        warm["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        warm["cache_hits"] = (svc.cache.hits - hits_before) if svc.cache else 0
        mem["after_ingest_mb"] = rss_mb()

        # retrieval timing must not be flattered by a query-embedding cache
        uncached = EmbeddingService(svc.embedder, None, svc.batch_size)
        rpipe = RAGPipeline(uncached, pipe.store, llm, chunk_size, overlap, reranker=reranker)

        results = {}
        for name, cfg in CONFIGS:
            opts = RetrievalOptions(top_k=k, threshold=0.0, **cfg)
            results[name] = evaluate_config(rpipe, questions, opts, k, repeats)
        mem["after_retrieval_mb"] = rss_mb()

        report = {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": platform.python_version(), "platform": platform.platform(),
                "embedding_model": svc.model_name, "embedding_dimension": svc.dimension,
                "reranker": getattr(reranker, "name", str(reranker)), "llm": llm.model,
                "k": k, "repeats": repeats, "chunk_size": chunk_size, "chunk_overlap": overlap,
                "documents": len(docs), "questions": len(questions),
                "answerable": sum(q["document"] is not None for q in questions),
                "unanswerable": sum(q["document"] is None for q in questions),
                "indexed_chunks": pipe.store.count(),
            },
            "ingestion_cold": cold, "ingestion_warm": warm,
            "vector_store": pipe.store.stats(), "process_memory": mem,
            "retrieval": results,
            "compression": compression_study(rpipe, questions, k),
            "threshold_tuning": threshold_study(rpipe, questions, k),
        }
        if include_generation:
            report["generation"] = {
                name: generation_eval(pipe, questions, RetrievalOptions(top_k=k, threshold=0.0, **cfg),
                                      GenerationParams(temperature=0.0, max_tokens=256), uncached.embed_texts)
                for name, cfg in (CONFIGS[0], CONFIGS[4])
            }
        pipe.store = None  # release before temp dir cleanup
    return report


# --------------------------------------------------------------------- report
def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(r: dict, generation_note: Optional[str] = None) -> str:
    m, k = r["meta"], r["meta"]["k"]
    mock = m["embedding_model"].startswith("mock")
    L = ["# RAG Evaluation Report", ""]
    if mock:
        L += ["> **IMPORTANT - mock embeddings.** This run used `mock-hash-bow`, a deterministic hashed",
              "> bag-of-words embedder (no semantics, no stemming). It makes runs reproducible offline, but",
              "> these numbers describe *lexical* matching. **They say nothing about all-MiniLM-L6-v2** and",
              "> cannot show the benefit of semantic search on paraphrased questions. Re-run with",
              "> `--embedding-backend sentence-transformers` on a machine with Hugging Face access.", ""]
    L += ["## Setup", "",
          f"- Generated: {m['generated_at']} | Python {m['python']} | {m['platform']}",
          f"- Embedding model: `{m['embedding_model']}` ({m['embedding_dimension']}-d) | Reranker: `{m['reranker']}` | LLM: `{m['llm']}`",
          f"- Corpus: {m['documents']} documents -> {m['indexed_chunks']} chunks (size {m['chunk_size']}, overlap {m['chunk_overlap']})",
          f"- Questions: {m['questions']} ({m['answerable']} answerable, {m['unanswerable']} unanswerable); k={k}; {m['repeats']} timing repeats per query",
          "- Relevance: a chunk is relevant iff it comes from the expected document and contains the gold answer phrase.",
          "",
          "**How to read these numbers.** The corpus is deliberately small, so every configuration's candidate pool",
          f"(>= 20) covers the *entire* {m['indexed_chunks']}-chunk index: reranking and MMR reorder the whole corpus, and Hit@{k}",
          f"saturates. With only {m['answerable']} answerable questions, one question moves a metric by {100 / max(m['answerable'], 1):.1f} points, so",
          "differences of a few points are noise. Treat large, consistent gaps as signal and everything else as a tie.", ""]
    base = r["retrieval"]["baseline: semantic"]
    L += ["## Retrieval: baseline vs optimized", "",
          f"| Configuration | P@3 | P@{k} | R@{k} | Hit@1 | Hit@3 | Hit@{k} | MRR | dMRR vs base | mean ms | p95 ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, x in r["retrieval"].items():
        d = relative_change(x["mrr"], base["mrr"])
        dtxt = "-" if name.startswith("baseline") else (f"{d:+.1f}%" if d is not None else "n/a")
        L.append(f"| {name} | {pct(x['precision@3'])} | {pct(x[f'precision@{k}'])} | {pct(x[f'recall@{k}'])} | "
                 f"{pct(x['hit@1'])} | {pct(x['hit@3'])} | {pct(x[f'hit@{k}'])} | {x['mrr']:.3f} | {dtxt} | "
                 f"{x['latency_ms']['mean']:.2f} | {x['latency_ms']['p95']:.2f} |")
    L += ["", "### MRR by question type", "", "| Configuration | " + " | ".join(
        f"{t} (n={base['mrr_by_type'][t]['n']})" for t in base["mrr_by_type"]) + " |",
          "|---|" + "---|" * len(base["mrr_by_type"])]
    for name, x in r["retrieval"].items():
        L.append(f"| {name} | " + " | ".join(f"{x['mrr_by_type'][t]['mrr']:.3f}" for t in base["mrr_by_type"]) + " |")
    c = r["compression"]
    L += ["", "## Contextual compression", "",
          f"- Context size: {c['context_chars_before']:,} -> {c['context_chars_after']:,} characters "
          f"(**{c['size_reduction_pct']:.1f}% smaller**)",
          f"- Answer phrase still present in {pct(c['answer_phrase_retention'])} of {c['relevant_chunks_checked']} relevant retrieved chunks", ""]
    t = r["threshold_tuning"]
    L += ["## Similarity threshold tuning (semantic)", "",
          f"Best balanced accuracy at threshold **{t['best']['threshold']:.2f}** "
          f"(answers {pct(t['best']['answer_rate'])} of answerable, rejects {pct(t['best']['reject_rate'])} of unanswerable).", "",
          *(["**This threshold is specific to the mock embedder and must not be reused with MiniLM**; cosine score "
             "distributions differ between models. Re-tune with the real model.", ""] if mock else []),
          "| Threshold | Answer rate | Reject rate | Balanced acc. |", "|---|---|---|---|"]
    for row in t["rows"][::2]:
        L.append(f"| {row['threshold']:.2f} | {pct(row['answer_rate'])} | {pct(row['reject_rate'])} | {pct(row['balanced_accuracy'])} |")
    ic, iw, vs, mem = r["ingestion_cold"], r["ingestion_warm"], r["vector_store"], r["process_memory"]
    L += ["", "## Performance", "",
          "| Stage | Cold (ms, all docs) | Warm re-index (ms) |", "|---|---|---|"]
    for stage in ("extract", "clean", "chunk", "embed", "store", "total"):
        L.append(f"| {stage} | {ic['timings_ms'][stage]:.1f} | {iw['timings_ms'][stage]:.1f} |")
    L += ["", f"- Embedding cache: warm re-index served {iw['cache_hits']} chunk embeddings from cache.",
          f"- Vector store: {vs['chunks']} vectors, {vs['memory_bytes'] / 1024:.1f} KiB in memory, {vs['disk_bytes'] / 1024:.1f} KiB on disk.",
          f"- Process RSS: {mem['start_mb']:.0f} MB at start -> {mem['after_ingest_mb']:.0f} MB after ingest -> {mem['after_retrieval_mb']:.0f} MB after retrieval.", ""]
    if "generation" in r:
        L += ["## Generation (heuristic proxies)", "",
              "| Config | Faithfulness | Answer rel. | Context rel. | Citation precision | Citation valid | LLM ms (mean) |", "|---|---|---|---|---|---|---|"]
        for name, g in r["generation"].items():
            L.append(f"| {name} | {g['faithfulness']:.3f} | {g['answer_relevance']:.3f} | {g['context_relevance']:.3f} | "
                     f"{g['citation_precision']:.3f} | {g['citation_valid_rate']:.3f} | {g['llm_generation_ms']:.0f} |")
        L.append("")
    else:
        L += ["## Generation metrics", "", generation_note or "Not run.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------ CLI
def main(argv: Optional[list[str]] = None) -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding-backend", choices=["mock", "sentence-transformers"], default="mock")
    ap.add_argument("--reranker", choices=["lexical", "cross-encoder"], default="lexical")
    ap.add_argument("--llm", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--chunk-overlap", type=int, default=60)
    ap.add_argument("--out", type=Path, default=HERE / "reports")
    a = ap.parse_args(argv)

    embedder = MockEmbedder() if a.embedding_backend == "mock" else SentenceTransformerEmbedder(s.embedding_model)
    svc = EmbeddingService(embedder, EmbeddingCache(embedder.model_name), s.embedding_batch_size)
    reranker = build_reranker(s.with_overrides(reranker_backend=a.reranker))
    llm = MockLLM() if a.llm == "mock" else OllamaClient(s.ollama_base_url, s.ollama_model, s.llm_timeout_seconds)
    if a.llm == "ollama" and llm.health()["status"] != "ok":
        print(f"Ollama not ready: {llm.health()}", file=sys.stderr)
        return 2

    real_llm = a.llm == "ollama"
    report = run_evaluation(svc, reranker, llm, k=a.k, repeats=a.repeats, chunk_size=a.chunk_size,
                            overlap=a.chunk_overlap, include_generation=real_llm)
    note = None if real_llm else ("Not run: the mock LLM returns canned answers, so faithfulness/relevance scores "
                                  "would be meaningless. Re-run with `--llm ollama` to measure them.")
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.embedding_backend}_{a.reranker}"
    (a.out / f"eval_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_markdown(report, note)
    (a.out / f"eval_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
