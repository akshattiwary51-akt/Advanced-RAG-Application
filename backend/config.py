"""Centralised, environment-driven configuration.

All tunables live here so no other module reads ``os.environ`` directly.
Relative paths are resolved against the backend directory, and no secret
or absolute server path is ever returned from an API response.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


@dataclass(frozen=True)
class Settings:
    # Flask
    flask_host: str = field(default_factory=lambda: _env_str("FLASK_HOST", "127.0.0.1"))
    flask_port: int = field(default_factory=lambda: _env_int("FLASK_PORT", 5000))
    cors_origins: tuple = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in _env_str(
                "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if o.strip()
        )
    )
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 25))

    # Storage
    data_dir: Path = field(default_factory=lambda: _resolve(_env_str("DATA_DIR", "data")))
    database_url: str = field(
        default_factory=lambda: _env_str("DATABASE_URL", "sqlite:///data/rag.db")
    )

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_backend: str = field(
        default_factory=lambda: _env_str("EMBEDDING_BACKEND", "sentence-transformers")
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32)
    )

    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    min_chunk_size: int = field(default_factory=lambda: _env_int("MIN_CHUNK_SIZE", 100))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.25)
    )
    retrieval_method: str = field(
        default_factory=lambda: _env_str("RETRIEVAL_METHOD", "semantic")
    )

    # Advanced retrieval
    reranker_backend: str = field(default_factory=lambda: _env_str("RERANKER_BACKEND", "lexical"))
    reranker_model: str = field(
        default_factory=lambda: _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # LLM
    ollama_base_url: str = field(
        default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: _env_str("OLLAMA_MODEL", "llama3.2"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))
    llm_timeout_seconds: int = field(
        default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 120)
    )
    llm_backend: str = field(default_factory=lambda: _env_str("LLM_BACKEND", "ollama"))

    # ---- derived paths -------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def sqlalchemy_url(self) -> str:
        """Resolve relative sqlite paths against backend/ so cwd does not matter."""
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix) and not self.database_url.startswith(
            "sqlite:////"
        ):
            rel = self.database_url[len(prefix):]
            if rel != ":memory:":
                return f"{prefix}{_resolve(rel)}"
        return self.database_url

    def ensure_dirs(self) -> None:
        for d in (self.uploads_dir, self.processed_dir, self.vector_dir):
            d.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **kwargs) -> "Settings":
        """Return a copy with fields replaced (used heavily by tests)."""
        return replace(self, **kwargs)

    def public_dict(self) -> dict:
        """Safe-to-expose settings: no paths, no URLs with credentials."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedding_backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_method": self.retrieval_method,
            "llm_model": self.ollama_model,
            "llm_temperature": self.llm_temperature,
            "llm_backend": self.llm_backend,
            "max_upload_mb": self.max_upload_mb,
        }


def get_settings() -> Settings:
    return Settings()
