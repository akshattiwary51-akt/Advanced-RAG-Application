"""Embedding generation: real (Sentence Transformers) and mock (offline) backends.

Both backends return **L2-normalised float32** matrices of shape ``(n, dim)`` so that
cosine similarity reduces to a dot product, and both go through the same
``EmbeddingCache`` so unchanged text is never embedded twice.

``MockEmbedder`` is a deterministic hashed bag-of-words model. It is NOT semantic
(it only knows shared words), but it is stable, fast and offline, which is what unit
tests need. Never use it to judge retrieval quality; see the evaluation module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from .text_processor import tokenize

log = logging.getLogger(__name__)


def l2_normalize(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation; zero vectors stay zero instead of becoming NaN."""
    m = np.asarray(m, dtype=np.float32)
    single = m.ndim == 1
    m2 = m[None, :] if single else m
    norms = np.linalg.norm(m2, axis=1, keepdims=True)
    out = m2 / np.maximum(norms, eps)
    return out[0] if single else out


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


# ------------------------------------------------------------------- mock
class MockEmbedder:
    """Deterministic hashed bag-of-words embedder (signed feature hashing)."""

    def __init__(self, dimension: int = 384, model_name: str = "mock-hash-bow"):
        self.dimension = dimension
        self.model_name = model_name

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[h % self.dimension] += sign
        return v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))


# ------------------------------------------------------- sentence-transformers
class SentenceTransformerEmbedder:
    """Thin wrapper over ``sentence_transformers.SentenceTransformer`` (lazy-loaded)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "sentence-transformers is not installed. Run "
                        "`pip install sentence-transformers`."
                    ) from exc
                log.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                # renamed in sentence-transformers 5.x; support both to avoid a FutureWarning
                get_dim = getattr(self._model, "get_embedding_dimension", None) or \
                    self._model.get_sentence_embedding_dimension
                self._dimension = int(get_dim())
        return self._model

    @property
    def dimension(self) -> int:  # type: ignore[override]
        if self._dimension is None:
            self._load()
        return int(self._dimension)  # type: ignore[arg-type]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        if not len(texts):
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise ourselves, in one visible place
            show_progress_bar=False,
        )
        return l2_normalize(vecs)


# -------------------------------------------------------------------- cache
class EmbeddingCache:
    """Content-addressed embedding cache (sha256(text) -> vector), optionally on disk.

    The cache is namespaced by model name, so switching models can never return stale
    vectors. Disk format: ``<model>.npy`` (matrix) + ``<model>.json`` (ordered keys).
    """

    def __init__(self, model_name: str, directory: Optional[Path] = None, max_items: int = 50_000):
        self.model_name = model_name
        self.max_items = max_items
        self._vecs: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._path: Optional[Path] = None
        self.hits = 0
        self.misses = 0
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
            self._path = directory / f"emb_cache_{safe}"
            self._load()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        assert self._path is not None
        npy, js = self._path.with_suffix(".npy"), self._path.with_suffix(".json")
        if not (npy.exists() and js.exists()):
            return
        try:
            mat = np.load(npy)
            keys = json.loads(js.read_text(encoding="utf-8"))
            if len(keys) == len(mat):
                self._vecs = {k: mat[i] for i, k in enumerate(keys)}
        except Exception:  # a corrupt cache must never break the app
            log.warning("Embedding cache unreadable; starting empty.")
            self._vecs = {}

    def save(self) -> None:
        if self._path is None:
            return
        with self._lock:
            if not self._vecs:
                return
            keys = list(self._vecs)
            mat = np.stack([self._vecs[k] for k in keys]).astype(np.float32)
            npy, js = self._path.with_suffix(".npy"), self._path.with_suffix(".json")
            tmp_npy, tmp_js = npy.with_name(npy.name + ".tmp"), js.with_name(js.name + ".tmp")
            with open(tmp_npy, "wb") as fh:
                np.save(fh, mat)
            tmp_js.write_text(json.dumps(keys), encoding="utf-8")
            os.replace(tmp_npy, npy)
            os.replace(tmp_js, js)

    def get(self, text: str) -> Optional[np.ndarray]:
        with self._lock:
            v = self._vecs.get(self.key(text))
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, text: str, vec: np.ndarray) -> None:
        with self._lock:
            if len(self._vecs) >= self.max_items:
                self._vecs.pop(next(iter(self._vecs)))  # evict oldest insertion
            self._vecs[self.key(text)] = np.asarray(vec, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._vecs)

    def clear(self) -> None:
        with self._lock:
            self._vecs.clear()
        if self._path is not None:
            for suffix in (".npy", ".json"):
                self._path.with_suffix(suffix).unlink(missing_ok=True)


# ------------------------------------------------------------------ service
class EmbeddingService:
    """Batching + caching front-end used by the pipeline and retriever."""

    def __init__(self, embedder: Embedder, cache: Optional[EmbeddingCache] = None, batch_size: int = 32):
        self.embedder = embedder
        self.cache = cache
        self.batch_size = max(1, batch_size)

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    def embed_texts(self, texts: Sequence[str], persist_cache: bool = False) -> np.ndarray:
        """Embed many texts; cached texts are skipped, the rest are batch-encoded."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out: list[Optional[np.ndarray]] = [None] * n
        todo: dict[str, list[int]] = {}  # dedupe identical texts inside one call
        for i, t in enumerate(texts):
            hit = self.cache.get(t) if self.cache is not None else None
            if hit is not None:
                out[i] = hit
            else:
                todo.setdefault(t, []).append(i)
        uniq = list(todo)
        for s in range(0, len(uniq), self.batch_size):
            batch = uniq[s : s + self.batch_size]
            vecs = self.embedder.embed(batch)
            for text, vec in zip(batch, vecs):
                if self.cache is not None:
                    self.cache.put(text, vec)
                for i in todo[text]:
                    out[i] = vec
        if persist_cache and self.cache is not None and uniq:
            self.cache.save()
        return np.stack(out).astype(np.float32)  # type: ignore[arg-type]

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


def build_embedding_service(settings, directory: Optional[Path] = None) -> EmbeddingService:
    """Factory driven by ``Settings.embedding_backend``."""
    backend = settings.embedding_backend.lower()
    if backend == "mock":
        embedder: Embedder = MockEmbedder()
    elif backend in ("sentence-transformers", "sentence_transformers", "st"):
        embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND '{settings.embedding_backend}'.")
    cache = EmbeddingCache(embedder.model_name, directory or settings.vector_dir.parent / "processed")
    return EmbeddingService(embedder, cache, settings.embedding_batch_size)