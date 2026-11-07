import logging

from flask import Blueprint, current_app, jsonify, request

from src.document_manager import DuplicateDocumentError
from src.loaders import DocumentLoadError
from utils.errors import AppError, describe

log = logging.getLogger(__name__)
bp = Blueprint("upload", __name__, url_prefix="/api/documents")


@bp.post("/upload")
def upload():
    """Upload and index one or more documents (PDF, DOCX, TXT, CSV).
    ---
    tags: [Documents]
    consumes: [multipart/form-data]
    parameters:
      - name: files
        in: formData
        type: file
        required: true
        description: One or more files (repeat the field). A single `file` field is also accepted.
    responses:
      201: {description: At least one file was indexed; see per-file `results`}
      400: {description: No files supplied}
      409: {description: Every file was a duplicate}
      413: {description: File too large}
      415: {description: Unsupported file type}
      422: {description: Empty or corrupted document}
    """
    files = request.files.getlist("files") or request.files.getlist("file")
    files = [f for f in files if f and f.filename]
    if not files:
        raise AppError("No files uploaded. Send multipart field 'files'.", 400, "no_files")
    if len(files) > 20:
        raise AppError("At most 20 files per request.", 400, "too_many_files")

    docs = current_app.extensions["rag"].documents
    results, first_error = [], None
    for f in files:
        try:
            doc = docs.upload(f.stream, f.filename, f.mimetype)
            results.append({"filename": f.filename, "status": "indexed", "document": doc})
        except DocumentLoadError as exc:
            status, body = describe(exc)
            first_error = first_error or (status, body)
            kind = "duplicate" if isinstance(exc, DuplicateDocumentError) else "error"
            results.append({"filename": f.filename, "status": kind, **body})
        except Exception as exc:  # one bad file must not abort the rest of the batch
            status, body = describe(exc)
            first_error = first_error or (status, body)
            results.append({"filename": f.filename, "status": "error", **body})

    if any(r["status"] == "indexed" for r in results):
        return jsonify({"results": results}), 201
    status, _ = first_error
    return jsonify({"results": results, **first_error[1]}), status
