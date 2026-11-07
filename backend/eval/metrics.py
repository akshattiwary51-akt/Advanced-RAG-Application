"""Evaluation metrics (pure functions, no I/O).

Retrieval metrics are exact, standard definitions over a ranked list of booleans
(``relevance[i]`` = whether the result at rank i+1 is relevant).

Generation metrics are **cheap, deterministic proxies** (lexical support and embedding
similarity), not human or LLM-judge scores. They are useful for regression tracking and
for catching hallucinated citations; for absolute quality claims use an LLM judge or
human review.
"""
from __future__ import annotations

import re
import statistics
from typing import Optional, Sequence

import numpy as np

from src.bm25 import keyword_tokens
from src.text_processor import split_sentences

# ------------------------------------------------------------------ retrieval
def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Relevant results among the top k, divided by k (missing results count as misses)."""
    if k <= 0:
        raise ValueError("k must be positive.")
    return sum(bool(r) for r in relevance[:k]) / k


def recall_at_k(relevance: Sequence[bool], k: int, total_relevant: int) -> float:
    """Relevant results found in the top k, divided by all relevant items in the corpus."""
    if total_relevant <= 0:
        raise ValueError("total_relevant must be positive for an answerable query.")
    return min(sum(bool(r) for r in relevance[:k]), total_relevant) / total_relevant


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for i, r in enumerate(relevance, start=1):
        if r:
            return 1.0 / i
    return 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summarize_latency(values_ms: Sequence[float]) -> dict:
    if not values_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": statistics.fmean(values_ms), "p50": percentile(values_ms, 50),
            "p95": percentile(values_ms, 95), "max": max(values_ms)}


# ------------------------------------------------------------ threshold tuning
def tune_threshold(per_query: Sequence[dict], thresholds: Sequence[float]) -> list[dict]:
    """Sweep similarity thresholds over precomputed results.

    ``per_query`` items: ``{"answerable": bool, "scored": [(cosine, is_relevant), ...]}``
    (retrieved at threshold 0). For each threshold t keep results with cosine >= t:
      * answerable query -> *answered correctly* if any kept result is relevant
      * unanswerable query -> *rejected correctly* if nothing is kept
    Balanced accuracy = mean(answer_rate, reject_rate), so the optimum is not skewed by
    the answerable/unanswerable ratio. Returns one row per threshold.
    """
    ans = [q for q in per_query if q["answerable"]]
    una = [q for q in per_query if not q["answerable"]]
    rows = []
    for t in thresholds:
        answer_rate = (sum(any(rel and s >= t for s, rel in q["scored"]) for q in ans) / len(ans)) if ans else 0.0
        reject_rate = (sum(not any(s >= t for s, _ in q["scored"]) for q in una) / len(una)) if una else 0.0
        parts = ([answer_rate] if ans else []) + ([reject_rate] if una else [])
        rows.append({"threshold": round(float(t), 4), "answer_rate": answer_rate,
                     "reject_rate": reject_rate,
                     "balanced_accuracy": sum(parts) / len(parts) if parts else 0.0})
    return rows


def best_threshold(rows: Sequence[dict]) -> dict:
    """Highest balanced accuracy; ties resolved toward the *lowest* threshold (keeps recall)."""
    return max(rows, key=lambda r: (round(r["balanced_accuracy"], 9), -r["threshold"]))


# ----------------------------------------------------------------- generation
_CITE = re.compile(r"\s*\[(\d{1,3})\]")


def _sentences(answer: str) -> list[str]:
    return [s for s in split_sentences(_CITE.sub("", answer)) if keyword_tokens(s)]


def _support(sentence: str, context_tokens: set[str]) -> float:
    toks = [t for t in keyword_tokens(sentence)]
    return sum(t in context_tokens for t in toks) / len(toks) if toks else 0.0


def faithfulness(answer: str, contexts: Sequence[str], min_support: float = 0.6) -> float:
    """Fraction of answer sentences whose content words are mostly present in the contexts.

    Proxy for "is every claim grounded in the retrieved text". 1.0 = fully supported.
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0
    ctx = {t for c in contexts for t in keyword_tokens(c)}
    return sum(_support(s, ctx) >= min_support for s in sents) / len(sents)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / d if d > 1e-12 else 0.0


def answer_relevance(question: str, answer: str, embed) -> float:
    """Cosine similarity between question and answer embeddings (``embed(list[str]) -> ndarray``)."""
    q, a = embed([question, _CITE.sub("", answer)])
    return _cos(q, a)


def context_relevance(question: str, contexts: Sequence[str], embed) -> float:
    """Mean cosine similarity between the question and each retrieved context."""
    if not contexts:
        return 0.0
    vecs = embed([question, *contexts])
    return float(np.mean([_cos(vecs[0], v) for v in vecs[1:]]))


def citation_correctness(answer: str, sources: dict[int, str], min_support: float = 0.3) -> dict:
    """Check every ``[n]`` in the answer.

    * ``valid``     : n refers to a source that was actually supplied
    * ``supported`` : the sentence carrying the citation shares >= ``min_support`` of its content
                      words with that source
    ``precision`` = supported citations / all citations; ``coverage`` = fraction of answer
    sentences that carry at least one citation.
    """
    total = valid = supported = 0
    sents = split_sentences(answer)
    covered = 0
    for sent in sents:
        ids = [int(m) for m in _CITE.findall(" " + sent)]
        if ids:
            covered += 1
        toks_sentence = sent
        for n in ids:
            total += 1
            if n in sources:
                valid += 1
                if _support(_CITE.sub("", toks_sentence), set(keyword_tokens(sources[n]))) >= min_support:
                    supported += 1
    return {
        "citations": total,
        "valid_rate": valid / total if total else 0.0,
        "precision": supported / total if total else 0.0,
        "coverage": covered / len(sents) if sents else 0.0,
    }


def relative_change(new: float, base: float) -> Optional[float]:
    """Percent change vs a baseline; None when the baseline is zero."""
    return None if abs(base) < 1e-12 else (new - base) / base * 100.0
