"""Semantic retriever: query -> embedding -> cosine top-K -> filtered, de-duplicated hits.

The retriever is the only place that knows about *both* the embedding service and the
vector store; the pipeline talks to the retriever and never to the store directly.
Advanced strategies (hybrid/BM25, RRF, MMR, reranking) are added in Phase 12 as
separate, opt-in functions so this baseline stays simple and testable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .embeddings import EmbeddingService
from .text_processor import preprocess_query
from .vector_store import NumpyVectorStore, cosine_similarity

NEAR_DUPLICATE_SIMILARITY = 0.97


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page: Optional[int]
    text: str
    score: float
    rank: int = 0
    scores: dict = field(default_factory=dict)  # per-method scores (semantic, bm25, ...)
    original_text: Optional[str] = None  # set when contextual compression shortened ``text``

    @classmethod
    def from_record(cls, record: dict, score: float, rank: int = 0) -> "RetrievedChunk":
        return cls(
            chunk_id=record["chunk_id"],
            document_id=record["document_id"],
            filename=record.get("filename", ""),
            chunk_index=record.get("chunk_index", 0),
            page=record.get("page"),
            text=record["text"],
            score=float(score),
            rank=rank,
            scores={"semantic": float(score)},
        )

    def to_dict(self, include_text: bool = True) -> dict:
        d = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "score": round(self.score, 4),
            "rank": self.rank,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }
        d["compressed"] = self.original_text is not None
        if include_text:
            d["text"] = self.text
        return d


def deduplicate(hits: list[RetrievedChunk], vectors: Optional[np.ndarray] = None,
                near_threshold: float = NEAR_DUPLICATE_SIMILARITY) -> list[RetrievedChunk]:
    """Drop exact-duplicate texts and (if vectors given) near-duplicates of higher-ranked hits.

    Input must already be sorted best-first so the best copy is the one kept.
    """
    kept: list[RetrievedChunk] = []
    kept_vecs: list[np.ndarray] = []
    seen_texts: set[str] = set()
    for i, h in enumerate(hits):
        norm = " ".join(h.text.lower().split())
        if norm in seen_texts:
            continue
        if vectors is not None and any(
            cosine_similarity(vectors[i], kv) >= near_threshold for kv in kept_vecs
        ):
            continue
        seen_texts.add(norm)
        kept.append(h)
        if vectors is not None:
            kept_vecs.append(vectors[i])
    return kept


class SemanticRetriever:
    def __init__(self, embeddings: EmbeddingService, store: NumpyVectorStore):
        self.embeddings = embeddings
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = 0.0,
        document_ids: Optional[Iterable[str]] = None,
        dedupe: bool = True,
        preprocessed: bool = False,
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, best first, with scores >= ``threshold``."""
        q = query if preprocessed else preprocess_query(query)
        qvec = self.embeddings.embed_query(q)
        return self.retrieve_by_vector(qvec, top_k, threshold, document_ids, dedupe)

    def retrieve_by_vector(
        self,
        qvec: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = 0.0,
        document_ids: Optional[Iterable[str]] = None,
        dedupe: bool = True,
    ) -> list[RetrievedChunk]:
        doc_filter = list(document_ids) if document_ids is not None else None
        if doc_filter is not None and not doc_filter:
            return []
        # over-fetch so that dropping duplicates still leaves top_k results
        fetch = top_k * 3 if dedupe else top_k
        hits = self.store.search(qvec, fetch, threshold, doc_filter)
        chunks = [RetrievedChunk.from_record(h.record, h.score) for h in hits]
        if dedupe and chunks:
            vecs = self.store.get_vectors([h.row for h in hits])
            chunks = deduplicate(chunks, vecs)
        chunks = chunks[:top_k]
        for i, c in enumerate(chunks, start=1):
            c.rank = i
        return chunks

    def timed_retrieve(self, *args, **kwargs) -> tuple[list[RetrievedChunk], float]:
        """Same as ``retrieve`` but also returns latency in milliseconds."""
        t0 = time.perf_counter()
        out = self.retrieve(*args, **kwargs)
        return out, (time.perf_counter() - t0) * 1000.0
