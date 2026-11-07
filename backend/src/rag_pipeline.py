"""End-to-end RAG pipeline: ingestion (process_document) and querying (answer_query).

Pure orchestration. It owns no HTTP or database concerns, so it can be tested with
mock embeddings and a mock LLM, and reused from scripts or the evaluation module.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .advanced_retrieval import AdvancedRetriever, Reranker, rewrite_query
from .chunker import Chunk, chunk_pages
from .embeddings import EmbeddingService
from .llm import LLM, GenerationParams, LLMError
from .loaders import PageText, load_document
from .prompt_builder import (
    INSUFFICIENT_INFO,
    ContextSource,
    attach_citations,
    build_messages,
    construct_context,
    invalid_citation_ids,
)
from .retriever import RetrievedChunk, SemanticRetriever
from .text_processor import clean_text, preprocess_query
from .vector_store import NumpyVectorStore

log = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    document_id: str
    page_count: int
    chunk_count: int
    char_count: int
    timings_ms: dict = field(default_factory=dict)


@dataclass
class RetrievalOptions:
    top_k: int = 5
    threshold: float = 0.25
    method: str = "semantic"
    document_ids: Optional[list[str]] = None
    max_context_chars: int = 6000
    # advanced options (all off = baseline semantic retrieval)
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    rerank: bool = False
    expand_query: bool = False
    compress: bool = False
    rewrite_query: bool = False

    @property
    def is_baseline(self) -> bool:
        return self.method == "semantic" and not (
            self.use_mmr or self.rerank or self.expand_query or self.compress)


@dataclass
class AnswerResult:
    answer: str
    citations: list
    sources: list[ContextSource]
    retrieved: list[RetrievedChunk]
    timings_ms: dict = field(default_factory=dict)
    grounded: bool = True
    warnings: list[str] = field(default_factory=list)
    rewritten_query: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "grounded": self.grounded,
            "warnings": self.warnings,
            "rewritten_query": self.rewritten_query,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class RAGPipeline:
    def __init__(self, embeddings: EmbeddingService, store: NumpyVectorStore,
                 llm: LLM, chunk_size: int = 800, chunk_overlap: int = 150, min_chunk_size: int = 100,
                 retriever: Optional[SemanticRetriever] = None,
                 reranker: Optional[Reranker] = None):
        self.embeddings = embeddings
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.retriever = retriever or SemanticRetriever(embeddings, store)
        self.advanced = AdvancedRetriever(embeddings, store, reranker)

    # ------------------------------------------------------------ ingestion
    def extract_text(self, path: Path | str, ext: Optional[str] = None) -> list[PageText]:
        return load_document(path, ext).pages

    @staticmethod
    def clean_pages(pages: Iterable[PageText]) -> list[PageText]:
        cleaned = [PageText(p.page, clean_text(p.text)) for p in pages]
        return [p for p in cleaned if p.text]

    def chunk_text(self, pages: list[PageText], document_id: str, filename: str,
                   chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> list[Chunk]:
        return chunk_pages(pages, document_id, filename, chunk_size or self.chunk_size,
                           self.chunk_overlap if overlap is None else overlap, self.min_chunk_size)

    def generate_embeddings(self, chunks: list[Chunk]):
        return self.embeddings.embed_texts([c.text for c in chunks], persist_cache=True)

    def store_vectors(self, chunks: list[Chunk], vectors) -> int:
        return self.store.add([c.to_dict() for c in chunks], vectors, replace=True)

    def process_document(self, path: Path | str, document_id: str, filename: str,
                         ext: Optional[str] = None, chunk_size: Optional[int] = None,
                         overlap: Optional[int] = None) -> ProcessResult:
        """extract -> clean -> chunk -> embed -> store. Re-processing replaces old vectors."""
        t_all = time.perf_counter()
        timings: dict[str, float] = {}
        t = time.perf_counter()
        pages = self.extract_text(path, ext)
        timings["extract"] = _ms(t)
        t = time.perf_counter()
        pages = self.clean_pages(pages)
        timings["clean"] = _ms(t)
        t = time.perf_counter()
        chunks = self.chunk_text(pages, document_id, filename, chunk_size, overlap)
        timings["chunk"] = _ms(t)
        if not chunks:
            from .loaders import EmptyDocumentError
            raise EmptyDocumentError("No usable text remained after cleaning.")
        t = time.perf_counter()
        vectors = self.generate_embeddings(chunks)
        timings["embed"] = _ms(t)
        t = time.perf_counter()
        self.store.delete_document(document_id)  # idempotent re-index
        self.store_vectors(chunks, vectors)
        timings["store"] = _ms(t)
        timings["total"] = _ms(t_all)
        return ProcessResult(document_id, len(pages), len(chunks),
                             sum(len(p.text) for p in pages), timings)

    def delete_document(self, document_id: str) -> int:
        return self.store.delete_document(document_id)

    # ------------------------------------------------------------ retrieval
    def retrieve_relevant_chunks(self, query: str, opts: RetrievalOptions) -> list[RetrievedChunk]:
        if opts.is_baseline:
            return self.retriever.retrieve(query, opts.top_k, opts.threshold, opts.document_ids,
                                           preprocessed=True)
        return self.advanced.retrieve(
            query, opts.top_k, opts.threshold, opts.document_ids, method=opts.method,
            use_mmr=opts.use_mmr, mmr_lambda=opts.mmr_lambda, rerank=opts.rerank,
            expand=opts.expand_query, compress=opts.compress, preprocessed=True)

    def rerank_chunks(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Hook for Phase 12 (cross-encoder / MMR). Baseline keeps similarity order."""
        return chunks

    # ----------------------------------------------------------- generation
    def _prepare(self, query: str, history: Optional[list[dict]], opts: RetrievalOptions):
        timings: dict[str, float] = {}
        q = preprocess_query(query)
        search_q = q
        if opts.rewrite_query and history:
            t = time.perf_counter()
            search_q = rewrite_query(self.llm, q, history)
            timings["rewrite"] = _ms(t)
        t = time.perf_counter()
        retrieved = self.retrieve_relevant_chunks(search_q, opts)
        retrieved = self.rerank_chunks(search_q, retrieved)
        timings["retrieval"] = _ms(t)
        sources = construct_context(retrieved, opts.max_context_chars)
        messages = build_messages(q, sources, history) if sources else []
        return q, retrieved, sources, messages, timings, (search_q if search_q != q else None)

    def _finish(self, answer: str, retrieved, sources, timings, grounded=True,
                rewritten: Optional[str] = None) -> AnswerResult:
        warnings = []
        bad = invalid_citation_ids(answer, sources)
        if bad:
            warnings.append(f"Answer cited unknown source numbers: {bad}")
        return AnswerResult(answer, attach_citations(answer, sources), sources, retrieved,
                            timings, grounded, warnings, rewritten)

    def answer_query(self, query: str, history: Optional[list[dict]] = None,
                     opts: Optional[RetrievalOptions] = None,
                     params: Optional[GenerationParams] = None) -> AnswerResult:
        opts = opts or RetrievalOptions()
        params = (params or GenerationParams()).validate()
        _, retrieved, sources, messages, timings, rewritten = self._prepare(query, history, opts)
        if not sources:  # nothing relevant: do not let the LLM improvise
            return AnswerResult(INSUFFICIENT_INFO, [], [], retrieved, timings, grounded=False,
                                rewritten_query=rewritten)
        t = time.perf_counter()
        answer = self.llm.chat(messages, params).strip()
        timings["generation"] = _ms(t)
        return self._finish(answer, retrieved, sources, timings, rewritten=rewritten)

    def stream_answer(self, query: str, history: Optional[list[dict]] = None,
                      opts: Optional[RetrievalOptions] = None,
                      params: Optional[GenerationParams] = None) -> Iterator[dict]:
        """Yield events: ``{"type": "sources"|"token"|"done"|"error", ...}``."""
        opts = opts or RetrievalOptions()
        params = (params or GenerationParams()).validate()
        _, retrieved, sources, messages, timings, rewritten = self._prepare(query, history, opts)
        if not sources:
            yield {"type": "token", "content": INSUFFICIENT_INFO}
            yield {"type": "done", "result": AnswerResult(INSUFFICIENT_INFO, [], [], retrieved,
                                                          timings, grounded=False,
                                                          rewritten_query=rewritten)}
            return
        yield {"type": "sources", "citations": [c.to_dict() for c in attach_citations("", sources)]}
        t = time.perf_counter()
        parts: list[str] = []
        try:
            for piece in self.llm.stream(messages, params):
                parts.append(piece)
                yield {"type": "token", "content": piece}
        except LLMError as exc:
            yield {"type": "error", "message": str(exc), "partial": "".join(parts)}
            return
        timings["generation"] = _ms(t)
        yield {"type": "done", "result": self._finish("".join(parts).strip(), retrieved, sources,
                                                      timings, rewritten=rewritten)}
