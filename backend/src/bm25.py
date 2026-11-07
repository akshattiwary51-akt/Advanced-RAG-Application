"""Okapi BM25 keyword index, implemented from the formula.

score(q, d) = sum over query terms t of
    idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)  = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

Why it complements embeddings: dense vectors blur exact strings (part numbers, names,
rare jargon); BM25 rewards them. The index is rebuilt lazily whenever the vector
store's ``version`` changes, so it can never serve deleted chunks.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from .text_processor import tokenize

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in into is it its of on or "
    "that the their then there these this to was were what when where which who why will "
    "with you your do does did can could should would about than so not no".split()
)


def keyword_tokens(text: str) -> list[str]:
    """Tokens with stop-words removed (falls back to all tokens if nothing remains)."""
    toks = tokenize(text)
    kept = [t for t in toks if t not in STOPWORDS]
    return kept or toks


@dataclass
class BM25Hit:
    record: dict
    score: float
    matched_terms: int


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._records: list[dict] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._postings: dict[str, list[int]] = {}
        self._avgdl = 0.0
        self.built_version: Optional[int] = None

    def build(self, records: Iterable[dict], version: Optional[int] = None) -> "BM25Index":
        self._records = list(records)
        self._tf, self._len, self._df, self._postings = [], [], Counter(), {}
        for i, r in enumerate(self._records):
            toks = keyword_tokens(r["text"])
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            for t in tf:
                self._df[t] += 1
                self._postings.setdefault(t, []).append(i)
        n = len(self._records)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        self.built_version = version
        return self

    def __len__(self) -> int:
        return len(self._records)

    def idf(self, term: str) -> float:
        n, df = len(self._records), self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10,
               document_ids: Optional[Iterable[str]] = None) -> list[BM25Hit]:
        terms = list(dict.fromkeys(keyword_tokens(query)))  # unique, order kept
        if not terms or not self._records:
            return []
        allowed = set(document_ids) if document_ids is not None else None
        scores: dict[int, float] = {}
        matched: Counter = Counter()
        for t in terms:
            post = self._postings.get(t)
            if not post:
                continue
            idf = self.idf(t)
            for i in post:
                if allowed is not None and self._records[i]["document_id"] not in allowed:
                    continue
                f = self._tf[i][t]
                norm = f + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avgdl or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / norm
                matched[i] += 1
        ranked = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        return [BM25Hit(self._records[i], scores[i], matched[i]) for i in ranked]
