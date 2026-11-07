"""Custom NumPy vector store with persistence.

Storage layout (inside ``directory``)::

    vectors.npy   float32 matrix (N, D), row i belongs to records[i]
    records.json  {"model": ..., "dimension": D, "records": [chunk metadata + text, ...]}

Design notes
------------
* Search is exact (brute force): one matrix-vector product over all rows. For the
  document counts this app targets (up to ~10^5 chunks) that is milliseconds; the
  ``VectorStore`` interface is deliberately tiny so FAISS/pgvector/Qdrant can replace
  it later without touching the retriever.
* Cosine similarity is implemented from its definition,
  ``dot(A, B) / (norm(A) * norm(B))``, with explicit norms, so it stays correct even
  if someone inserts un-normalised vectors.
* Writes are atomic (temp file + ``os.replace``) and guarded by a lock, so a crash
  never leaves a half-written index.
* Duplicate prevention: a ``chunk_id`` can exist only once. ``add`` raises on a
  duplicate unless ``replace=True``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)


class VectorStoreError(Exception):
    pass


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """cosine(A, B) = dot(A, B) / (||A|| * ||B||) for two 1-D vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_similarity_matrix(matrix: np.ndarray, query: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Vectorised cosine similarity of every row of ``matrix`` against ``query``."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    dots = matrix @ query
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.where(denom < eps, 0.0, dots / np.maximum(denom, eps)).astype(np.float32)


@dataclass
class SearchHit:
    record: dict
    score: float
    row: int


class VectorStore(Protocol):
    def add(self, records: Sequence[dict], vectors: np.ndarray, replace: bool = False) -> int: ...
    def search(self, query: np.ndarray, top_k: int, threshold: Optional[float] = None,
               document_ids: Optional[Iterable[str]] = None) -> list[SearchHit]: ...
    def delete_document(self, document_id: str) -> int: ...
    def stats(self) -> dict: ...


class NumpyVectorStore:
    """Brute-force cosine vector store persisted as ``.npy`` + ``.json``."""

    def __init__(self, directory: Path | str, dimension: int, model_name: str = "unknown",
                 autosave: bool = True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.model_name = model_name
        self.autosave = autosave
        self._lock = threading.RLock()
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._records: list[dict] = []
        self._ids: dict[str, int] = {}
        self.version = 0  # bumped on every mutation; lets derived indexes (BM25) detect staleness
        self._load()

    # ---------------------------------------------------------- persistence
    @property
    def _vec_path(self) -> Path:
        return self.directory / "vectors.npy"

    @property
    def _rec_path(self) -> Path:
        return self.directory / "records.json"

    def _load(self) -> None:
        if not (self._vec_path.exists() and self._rec_path.exists()):
            return
        try:
            vecs = np.load(self._vec_path)
            meta = json.loads(self._rec_path.read_text(encoding="utf-8"))
            records = meta["records"]
            if len(records) != len(vecs):
                raise VectorStoreError("vectors and records are out of sync")
            if meta.get("dimension") != self.dimension or (
                vecs.size and vecs.shape[1] != self.dimension
            ):
                raise VectorStoreError(
                    f"stored dimension {meta.get('dimension')} != expected {self.dimension} "
                    f"(model was '{meta.get('model')}', now '{self.model_name}'). Re-index required."
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Vector index is corrupted: {exc}") from exc
        self._vectors = vecs.astype(np.float32, copy=False)
        self._records = records
        self.stored_model = meta.get("model")
        self._reindex_ids()

    def save(self) -> None:
        with self._lock:
            tmp_v = self._vec_path.with_name("vectors.npy.tmp")
            tmp_r = self._rec_path.with_name("records.json.tmp")
            with open(tmp_v, "wb") as fh:
                np.save(fh, self._vectors)
            tmp_r.write_text(
                json.dumps({"model": self.model_name, "dimension": self.dimension,
                            "records": self._records}),
                encoding="utf-8",
            )
            os.replace(tmp_v, self._vec_path)
            os.replace(tmp_r, self._rec_path)

    def _reindex_ids(self) -> None:
        self._ids = {r["chunk_id"]: i for i, r in enumerate(self._records)}
        self.version += 1

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    # ------------------------------------------------------------- mutation
    def add(self, records: Sequence[dict], vectors: np.ndarray, replace: bool = False) -> int:
        """Insert records with their vectors. Returns the number inserted."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Expected vectors of shape (n, {self.dimension}), got {vectors.shape}."
            )
        if len(records) != len(vectors):
            raise VectorStoreError("records and vectors must have the same length.")
        if not len(records):
            return 0
        with self._lock:
            seen: set[str] = set()
            for r in records:
                cid = r.get("chunk_id")
                if not cid:
                    raise VectorStoreError("Every record needs a 'chunk_id'.")
                if cid in seen or (cid in self._ids and not replace):
                    raise VectorStoreError(f"Duplicate chunk_id '{cid}'.")
                seen.add(cid)
            if replace:
                stale = [self._ids[c] for c in seen if c in self._ids]
                if stale:
                    self._remove_rows(stale)
            self._vectors = np.vstack([self._vectors, vectors])
            self._records.extend(dict(r) for r in records)
            self._reindex_ids()
            self._maybe_save()
        return len(records)

    def _remove_rows(self, rows: Sequence[int]) -> None:
        drop = set(rows)
        keep = [i for i in range(len(self._records)) if i not in drop]
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dimension), np.float32)
        self._records = [self._records[i] for i in keep]
        self._reindex_ids()

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            rows = [i for i, r in enumerate(self._records) if r.get("document_id") == document_id]
            if rows:
                self._remove_rows(rows)
                self._maybe_save()
            return len(rows)

    def clear(self) -> None:
        with self._lock:
            self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
            self._records, self._ids = [], {}
            self._maybe_save()

    def rebuild(self) -> dict:
        """Drop corrupt/orphan rows (non-finite vectors, missing ids), re-save compactly."""
        with self._lock:
            before = len(self._records)
            bad = [
                i for i, (r, v) in enumerate(zip(self._records, self._vectors))
                if not r.get("chunk_id") or not np.isfinite(v).all()
            ]
            if bad:
                self._remove_rows(bad)
            self.save()
            return {"before": before, "after": len(self._records), "removed": len(bad)}

    # ------------------------------------------------------------ retrieval
    def search(self, query: np.ndarray, top_k: int = 5, threshold: Optional[float] = None,
               document_ids: Optional[Iterable[str]] = None) -> list[SearchHit]:
        """Top-k rows by cosine similarity (descending), optionally filtered."""
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise VectorStoreError(
                f"Query has dimension {query.shape[0]}, index has {self.dimension}."
            )
        if top_k < 1:
            return []
        with self._lock:
            n = len(self._records)
            if n == 0:
                return []
            scores = cosine_similarity_matrix(self._vectors, query)
            if document_ids is not None:
                allowed = set(document_ids)
                mask = np.fromiter((r["document_id"] in allowed for r in self._records),
                                   dtype=bool, count=n)
                scores = np.where(mask, scores, -np.inf)
            k = min(top_k, n)
            part = np.argpartition(-scores, k - 1)[:k]  # O(n) selection, then sort k
            order = part[np.argsort(-scores[part], kind="stable")]
            hits = []
            for row in order:
                s = float(scores[row])
                if not np.isfinite(s) or (threshold is not None and s < threshold):
                    continue
                hits.append(SearchHit(self._records[row], s, int(row)))
            return hits

    # --------------------------------------------------------------- access
    def get_vectors(self, rows: Sequence[int]) -> np.ndarray:
        with self._lock:
            return self._vectors[list(rows)]

    def vectors_for(self, chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Stored vectors for the given chunk ids (unknown ids are omitted)."""
        with self._lock:
            return {c: self._vectors[self._ids[c]] for c in chunk_ids if c in self._ids}

    def records(self, document_id: Optional[str] = None) -> list[dict]:
        with self._lock:
            if document_id is None:
                return list(self._records)
            return [r for r in self._records if r.get("document_id") == document_id]

    def document_ids(self) -> set[str]:
        with self._lock:
            return {r["document_id"] for r in self._records}

    def count(self, document_id: Optional[str] = None) -> int:
        with self._lock:
            if document_id is None:
                return len(self._records)
            return sum(1 for r in self._records if r.get("document_id") == document_id)

    def stats(self) -> dict:
        with self._lock:
            size = sum(p.stat().st_size for p in (self._vec_path, self._rec_path) if p.exists())
            return {
                "chunks": len(self._records),
                "documents": len({r["document_id"] for r in self._records}),
                "dimension": self.dimension,
                "model": self.model_name,
                "memory_bytes": int(self._vectors.nbytes),
                "disk_bytes": int(size),
            }
