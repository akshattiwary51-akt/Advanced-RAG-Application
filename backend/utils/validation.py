"""Request validation helpers (no Flask imports: easy to unit test)."""
from __future__ import annotations

from typing import Any, Optional

from config import Settings
from src.llm import GenerationParams
from src.rag_pipeline import RetrievalOptions
from .errors import AppError

ALLOWED_METHODS = ("semantic", "bm25", "hybrid")
BOOL_FLAGS = {"use_mmr": "use_mmr", "rerank": "rerank", "expand_query": "expand_query",
              "compress": "compress", "rewrite_query": "rewrite_query"}
MAX_MESSAGE_CHARS = 4000
MAX_TOP_K = 20


def _num(body: dict, key: str, default, lo: float, hi: float, cast=float):
    val = body.get(key, default)
    if val is None:
        val = default
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise AppError(f"'{key}' must be a number.", 400, "invalid_request")
    val = cast(val)
    if not lo <= val <= hi:
        raise AppError(f"'{key}' must be between {lo} and {hi}.", 400, "invalid_request")
    return val


def parse_retrieval(body: dict, settings: Settings) -> RetrievalOptions:
    method = body.get("method", settings.retrieval_method)
    if method not in ALLOWED_METHODS:
        raise AppError(f"'method' must be one of {list(ALLOWED_METHODS)}.", 400, "invalid_request")
    doc_ids = body.get("document_ids")
    if doc_ids is not None:
        if not isinstance(doc_ids, list) or not all(isinstance(d, str) and len(d) <= 64 for d in doc_ids):
            raise AppError("'document_ids' must be a list of strings.", 400, "invalid_request")
    flags = {}
    for key in BOOL_FLAGS:
        val = body.get(key, False)
        if not isinstance(val, bool):
            raise AppError(f"'{key}' must be true or false.", 400, "invalid_request")
        flags[key] = val
    return RetrievalOptions(
        top_k=_num(body, "top_k", settings.top_k, 1, MAX_TOP_K, int),
        threshold=_num(body, "similarity_threshold", settings.similarity_threshold, -1.0, 1.0),
        method=method,
        document_ids=doc_ids,
        mmr_lambda=_num(body, "mmr_lambda", 0.7, 0.0, 1.0),
        **flags,
    )


def parse_generation(body: dict, settings: Settings) -> GenerationParams:
    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip() or len(model) > 100):
        raise AppError("'model' must be a non-empty string.", 400, "invalid_request")
    return GenerationParams(
        temperature=_num(body, "temperature", settings.llm_temperature, 0.0, 2.0),
        max_tokens=_num(body, "max_tokens", settings.llm_max_tokens, 1, 8192, int),
        model=model.strip() if model else None,
    ).validate()


def parse_chat_request(body: Any, settings: Settings) -> dict:
    if not isinstance(body, dict):
        raise AppError("Request body must be a JSON object.", 400, "invalid_request")
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise AppError("'message' is required and must be a non-empty string.", 400, "invalid_request")
    if len(message) > MAX_MESSAGE_CHARS:
        raise AppError(f"'message' exceeds {MAX_MESSAGE_CHARS} characters.", 400, "invalid_request")
    conv: Optional[str] = body.get("conversation_id")
    if conv is not None and (not isinstance(conv, str) or len(conv) > 64):
        raise AppError("'conversation_id' must be a string.", 400, "invalid_request")
    return {
        "message": message.strip(),
        "conversation_id": conv,
        "opts": parse_retrieval(body, settings),
        "params": parse_generation(body, settings),
    }
