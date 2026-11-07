"""Dependency container: builds every long-lived object once and wires them together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import Settings
from database.db import Database
from src.conversation_manager import ConversationManager
from src.advanced_retrieval import build_reranker
from src.document_manager import DocumentManager
from src.embeddings import EmbeddingService, build_embedding_service
from src.llm import LLM, build_llm
from src.rag_pipeline import RAGPipeline
from src.vector_store import NumpyVectorStore


@dataclass
class Services:
    settings: Settings
    db: Database
    embeddings: EmbeddingService
    store: NumpyVectorStore
    llm: LLM
    pipeline: RAGPipeline
    documents: DocumentManager
    conversations: ConversationManager


def build_services(settings: Settings, embeddings: Optional[EmbeddingService] = None,
                   llm: Optional[LLM] = None) -> Services:
    """Create all services. ``embeddings``/``llm`` may be injected (tests, evaluation)."""
    settings.ensure_dirs()
    db = Database(settings.sqlalchemy_url())
    db.create_all()
    embeddings = embeddings or build_embedding_service(settings, settings.processed_dir)
    llm = llm or build_llm(settings)
    store = NumpyVectorStore(settings.vector_dir, embeddings.dimension, embeddings.model_name)
    pipeline = RAGPipeline(embeddings, store, llm, settings.chunk_size,
                           settings.chunk_overlap, settings.min_chunk_size,
                           reranker=build_reranker(settings))
    return Services(settings, db, embeddings, store, llm, pipeline,
                    DocumentManager(settings, db, pipeline), ConversationManager(db))
