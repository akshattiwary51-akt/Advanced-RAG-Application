"""Centralised exception -> HTTP response mapping.

Responses never include tracebacks, file-system paths or secrets: unexpected errors are
logged server-side and the client gets a generic 500 with a request-independent message.
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from src.conversation_manager import ConversationNotFoundError
from src.document_manager import DocumentNotFoundError, DuplicateDocumentError
from src.llm import LLMError, LLMUnavailableError
from src.loaders import (
    CorruptedDocumentError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from src.vector_store import VectorStoreError

log = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "bad_request", **extra):
        super().__init__(message)
        self.message, self.status, self.code, self.extra = message, status, code, extra


def error_body(message: str, code: str, **extra) -> dict:
    return {"error": {"code": code, "message": message, **extra}}


# (exception type, status, code) -- first isinstance match wins, so order matters.
_MAPPING = [
    (DuplicateDocumentError, 409, "duplicate_document"),
    (FileTooLargeError, 413, "file_too_large"),
    (UnsupportedFileTypeError, 415, "unsupported_file_type"),
    (EmptyDocumentError, 422, "empty_document"),
    (CorruptedDocumentError, 422, "corrupted_document"),
    (DocumentNotFoundError, 404, "document_not_found"),
    (ConversationNotFoundError, 404, "conversation_not_found"),
    (LLMUnavailableError, 503, "llm_unavailable"),
    (LLMError, 502, "llm_error"),
    (ValueError, 400, "invalid_request"),
]


def describe(exc: Exception) -> tuple[int, dict]:
    """Return (status, body) for any exception. Used by handlers and the SSE stream."""
    if isinstance(exc, AppError):
        return exc.status, error_body(exc.message, exc.code, **exc.extra)
    for etype, status, code in _MAPPING:
        if isinstance(exc, etype):
            extra = {"existing_document": exc.existing} if isinstance(exc, DuplicateDocumentError) else {}
            msg = str(exc) if not isinstance(exc, (DocumentNotFoundError, ConversationNotFoundError)) \
                else "Resource not found."
            return status, error_body(msg, code, **extra)
    if isinstance(exc, VectorStoreError):
        log.error("Vector store error: %s", exc)
        return 500, error_body("The search index is unavailable. Try re-indexing.", "index_error")
    log.exception("Unhandled error", exc_info=exc)
    return 500, error_body("Internal server error.", "internal_error")


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        code = {413: "file_too_large", 404: "not_found", 405: "method_not_allowed"}.get(
            exc.code or 500, "http_error")
        msg = "Request body too large." if exc.code == 413 else (exc.description or exc.name)
        return jsonify(error_body(msg, code)), exc.code or 500

    @app.errorhandler(Exception)
    def _any(exc: Exception):
        status, body = describe(exc)
        return jsonify(body), status
