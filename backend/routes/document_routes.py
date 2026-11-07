from flask import Blueprint, current_app, jsonify, request

from utils.errors import AppError

bp = Blueprint("documents", __name__, url_prefix="/api")


def _svc():
    return current_app.extensions["rag"]


@bp.get("/documents")
def list_documents():
    """List all documents with indexing status.
    ---
    tags: [Documents]
    responses:
      200:
        description: Documents, newest first
    """
    return jsonify({"documents": _svc().documents.list()})


@bp.get("/documents/<document_id>")
def get_document(document_id: str):
    """Get one document's metadata.
    ---
    tags: [Documents]
    parameters:
      - {name: document_id, in: path, type: string, required: true}
    responses:
      200: {description: Document metadata}
      404: {description: Not found}
    """
    return jsonify(_svc().documents.get(document_id))


@bp.delete("/documents/<document_id>")
def delete_document(document_id: str):
    """Delete a document, its file and its vectors.
    ---
    tags: [Documents]
    parameters:
      - {name: document_id, in: path, type: string, required: true}
    responses:
      200: {description: Deleted}
      404: {description: Not found}
    """
    return jsonify(_svc().documents.delete(document_id))


@bp.post("/documents/<document_id>/reindex")
def reindex_document(document_id: str):
    """Re-extract, re-chunk and re-embed a stored document.
    ---
    tags: [Documents]
    parameters:
      - {name: document_id, in: path, type: string, required: true}
      - name: body
        in: body
        required: false
        schema:
          type: object
          properties:
            chunk_size: {type: integer, example: 800}
            chunk_overlap: {type: integer, example: 150}
    responses:
      200: {description: Re-indexed document}
      404: {description: Not found}
      422: {description: Document can no longer be read}
    """
    body = request.get_json(silent=True) or {}
    size, overlap = body.get("chunk_size"), body.get("chunk_overlap")
    for name, v in (("chunk_size", size), ("chunk_overlap", overlap)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            raise AppError(f"'{name}' must be an integer.", 400, "invalid_request")
    if size is not None and not 50 <= size <= 5000:
        raise AppError("'chunk_size' must be between 50 and 5000.", 400, "invalid_request")
    if overlap is not None and overlap < 0:
        raise AppError("'chunk_overlap' must be >= 0.", 400, "invalid_request")
    return jsonify(_svc().documents.reindex(document_id, size, overlap))
