"""Advanced retrieval strategies, layered on top of the baseline retriever.

Every technique is an independent, individually testable function or class, and each is
opt-in so the baseline path (``SemanticRetriever``) stays untouched.

| Technique              | Benefit                                   | Cost / trade-off                                  |
|------------------------|-------------------------------------------|---------------------------------------------------|
| BM25 + hybrid + RRF    | exact terms, IDs, names; robust fusion    | extra index in RAM; can admit lexical false hits  |
| Query expansion (PRF)  | recovers vocabulary-mismatched passages   | can drift off-topic when first pass is wrong      |
| Query rewriting (LLM)  | resolves "it/they/that" in follow-ups     | one extra LLM call of latency; may mis-rewrite    |
| Cross-encoder rerank   | most accurate relevance ordering          | O(candidates) model calls per query               |
| MMR                    | diverse, non-redundant context            | may demote a second genuinely relevant chunk      |
| Contextual compression | smaller prompts, less noise               | may cut a sentence the answer needed              |
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

from .bm25 import STOPWORDS, BM25Index, keyword_tokens
from .embeddings import EmbeddingService
from .llm import LLM, GenerationParams
from .retriever import RetrievedChunk, SemanticRetriever, deduplicate
from .text_processor import preprocess_query, split_sentences
from .vector_store import NumpyVectorStore, cosine_similarity

log = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the original RRF paper (Cormack et al., 2009)


# --------------------------------------------------------------------- fusion
def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank from 1.

    Uses ranks, not raw scores, so cosine similarities and BM25 scores (different scales)
    can be fused without normalisation.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


# ------------------------------------------------------------------------ MMR
def mmr_select(relevance: np.ndarray, vectors: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance. Greedily pick the item maximising
    ``lam * relevance - (1 - lam) * max_similarity_to_already_selected``.

    ``lam=1`` is pure relevance order, ``lam=0`` is pure diversity. Vectors must be
    L2-normalised (dot product = cosine).
    """
    n = len(relevance)
    if n == 0 or k <= 0:
        return []
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be between 0 and 1.")
    sim = vectors @ vectors.T
    selected = [int(np.argmax(relevance))]
    remaining = set(range(n)) - set(selected)
    while remaining and len(selected) < k:
        best, best_score = -1, -np.inf
        for i in sorted(remaining):
            score = lam * relevance[i] - (1 - lam) * max(sim[i, j] for j in selected)
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        remaining.discard(best)
    return selected


# ------------------------------------------------------------ query expansion
def expand_query_prf(query: str, feedback_texts: Iterable[str], idf, n_terms: int = 4) -> str:
    """Pseudo-relevance feedback: append the most characteristic terms of the top results.

    Terms are ranked by ``tf * idf`` over the feedback passages; query words and
    stop-words are excluded. Deterministic and needs no LLM.
    """
    have = set(keyword_tokens(query))
    tf: Counter = Counter()
    for text in feedback_texts:
        tf.update(t for t in keyword_tokens(text) if t not in have and t not in STOPWORDS
                  and len(t) > 2 and not t.isdigit())
    if not tf:
        return query
    ranked = sorted(tf, key=lambda t: (-tf[t] * idf(t), t))[:n_terms]
    return f"{query} {' '.join(ranked)}"


# ------------------------------------------------------------ query rewriting
REWRITE_SYSTEM = (
    "Rewrite the user's latest question as a single standalone search query, resolving "
    "pronouns and references using the conversation. Output ONLY the rewritten query on "
    "one line. Do not answer it. If it is already standalone, repeat it unchanged."
)


def rewrite_query(llm: LLM, query: str, history: Optional[list[dict]],
                  max_turns: int = 4) -> str:
    """Make a follow-up question standalone. Falls back to ``query`` on any problem."""
    if not history:
        return query
    convo = "\n".join(f"{t['role']}: {str(t['content'])[:300]}"
                      for t in history[-max_turns * 2:] if t.get("content"))
    messages = [{"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest question: {query}"}]
    try:
        out = llm.chat(messages, GenerationParams(temperature=0.0, max_tokens=64)).strip()
    except Exception as exc:  # rewriting is an optimisation; never fail the request over it
        log.warning("Query rewrite failed (%s); using original query.", type(exc).__name__)
        return query
    out = out.splitlines()[0].strip().strip('"') if out else ""
    if not out or len(out) > 300 or re.search(r"\[\d+\]", out):
        return query
    return out


# -------------------------------------------------------- contextual compression
def compress_text(text: str, qvec: np.ndarray, embeddings: EmbeddingService,
                  ratio: float = 0.6, min_sentences: int = 1) -> str:
    """Keep only sentences relevant to the query, in their original order.

    A sentence survives if its cosine similarity to the query is at least ``ratio`` x the
    best sentence's similarity (relative, so it adapts to the score scale of the model).
    Chunks with two or fewer sentences are returned unchanged.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return text
    sims = embeddings.embed_texts(sentences) @ qvec
    top = float(sims.max())
    keep = [i for i, s in enumerate(sims) if s >= ratio * top and s > 0]
    if len(keep) < min_sentences:
        keep = [int(np.argmax(sims))]
    return " ".join(sentences[i] for i in sorted(keep))


# -------------------------------------------------------------------- rerankers
class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class LexicalReranker:
    """Model-free reranker: IDF-free query-term coverage plus a bigram-phrase bonus.

    A cheap stand-in used for tests and as the default when no cross-encoder is installed.
    It is NOT comparable in quality to a real cross-encoder.
    """

    name = "lexical"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = list(dict.fromkeys(keyword_tokens(query)))
        bigrams = {(a, b) for a, b in zip(q, q[1:])}
        out = []
        for text in texts:
            toks = keyword_tokens(text)
            tokset = set(toks)
            cover = sum(t in tokset for t in q) / len(q) if q else 0.0
            pairs = {(a, b) for a, b in zip(toks, toks[1:])}
            phrase = len(bigrams & pairs) / len(bigrams) if bigrams else 0.0
            out.append(cover + 0.5 * phrase)
        return out


class CrossEncoderReranker:
    """Sentence-Transformers ``CrossEncoder`` (reads query and passage *together*).

    Much more accurate than comparing two independent embeddings, but it runs one model
    forward pass per candidate, so only apply it to a short candidate list.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model=None):
        self.name = f"cross-encoder:{model_name}"
        self.model_name = model_name
        self._model = model  # injectable for tests
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        preds = self._load().predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(p) for p in preds]


def build_reranker(settings) -> Reranker:
    backend = settings.reranker_backend.lower()
    if backend == "lexical":
        return LexicalReranker()
    if backend in ("cross-encoder", "cross_encoder"):
        return CrossEncoderReranker(settings.reranker_model)
    raise ValueError(f"Unknown RERANKER_BACKEND '{settings.reranker_backend}'.")


# ---------------------------------------------------------------- orchestration
def _minmax(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return a
    lo, hi = a.min(), a.max()
    return np.full_like(a, 1.0) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


class AdvancedRetriever:
    """Semantic / BM25 / hybrid retrieval with optional expansion, rerank, MMR, compression."""

    METHODS = ("semantic", "bm25", "hybrid")

    def __init__(self, embeddings: EmbeddingService, store: NumpyVectorStore,
                 reranker: Optional[Reranker] = None):
        self.embeddings = embeddings
        self.store = store
        self.reranker = reranker or LexicalReranker()
        self.base = SemanticRetriever(embeddings, store)
        self._bm25 = BM25Index()
        self._bm25_lock = threading.Lock()

    def bm25_index(self) -> BM25Index:
        """Index over the current store contents, rebuilt if the store changed."""
        with self._bm25_lock:
            if self._bm25.built_version != self.store.version:
                self._bm25.build(self.store.records(), self.store.version)
            return self._bm25

    def retrieve(
        self, query: str, top_k: int = 5, threshold: Optional[float] = 0.0,
        document_ids: Optional[Iterable[str]] = None, method: str = "semantic",
        use_mmr: bool = False, mmr_lambda: float = 0.7, rerank: bool = False,
        expand: bool = False, compress: bool = False, preprocessed: bool = False,
    ) -> list[RetrievedChunk]:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {list(self.METHODS)}.")
        q = query if preprocessed else preprocess_query(query)
        docs = list(document_ids) if document_ids is not None else None
        if (docs is not None and not docs) or self.store.count() == 0:
            return []
        threshold = 0.0 if threshold is None else threshold
        pool = max(top_k * 4, 20)
        index = self.bm25_index()

        q_text = q
        if expand:
            first = self.base.retrieve(q, 3, threshold, docs, dedupe=False, preprocessed=True)
            q_text = expand_query_prf(q, [c.text for c in first], index.idf)
        qvec = self.embeddings.embed_query(q_text)

        cands: dict[str, dict] = {}
        sem_rank: list[str] = []
        lex_rank: list[str] = []
        if method in ("semantic", "hybrid"):
            for h in self.store.search(qvec, pool, threshold, docs):
                cid = h.record["chunk_id"]
                cands[cid] = {"rec": h.record, "cos": h.score}
                sem_rank.append(cid)
        if method in ("bm25", "hybrid"):
            n_terms = len(set(keyword_tokens(q_text)))
            need = min(2, n_terms)
            for h in index.search(q_text, pool, docs):
                # hybrid: a lexical-only hit must match >=2 distinct query terms, so a lone
                # common word cannot smuggle an irrelevant chunk past the similarity gate
                if method == "hybrid" and h.record["chunk_id"] not in cands and h.matched_terms < need:
                    continue
                cid = h.record["chunk_id"]
                cands.setdefault(cid, {"rec": h.record, "cos": None})["bm25"] = h.score
                lex_rank.append(cid)
        if not cands:
            return []

        vecs = self.store.vectors_for(list(cands))
        for cid, c in cands.items():
            if c["cos"] is None:
                c["cos"] = cosine_similarity(qvec, vecs[cid])

        if method == "semantic":
            order = sem_rank
        elif method == "bm25":
            order = lex_rank
        else:
            fused = rrf_fuse([sem_rank, lex_rank])
            for cid, s in fused.items():
                cands[cid]["rrf"] = s
            order = sorted(fused, key=lambda c: (-fused[c], c))

        chunks: list[RetrievedChunk] = []
        for cid in order:
            c = cands[cid]
            rc = RetrievedChunk.from_record(c["rec"], c["cos"])
            rc.scores = {"semantic": c["cos"], **{k: c[k] for k in ("bm25", "rrf") if k in c}}
            chunks.append(rc)
        chunks = deduplicate(chunks, np.stack([vecs[c.chunk_id] for c in chunks]))

        relevance = np.array([c.score for c in chunks], dtype=np.float64)
        if rerank and chunks:
            rr = self.reranker.score(q, [c.text for c in chunks])
            for c, s in zip(chunks, rr):
                c.scores["rerank"] = float(s)
            idx = sorted(range(len(chunks)), key=lambda i: (-rr[i], i))
            chunks = [chunks[i] for i in idx]
            relevance = _minmax([rr[i] for i in idx])

        if use_mmr and len(chunks) > 1:
            cv = np.stack([vecs[c.chunk_id] for c in chunks])
            chunks = [chunks[i] for i in mmr_select(relevance, cv, top_k, mmr_lambda)]
        chunks = chunks[:top_k]

        if compress:
            for c in chunks:
                short = compress_text(c.text, qvec, self.embeddings)
                if short != c.text:
                    c.original_text, c.text = c.text, short
        for i, c in enumerate(chunks, start=1):
            c.rank = i
        return chunks
