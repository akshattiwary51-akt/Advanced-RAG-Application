from flask import Blueprint, current_app, jsonify

bp = Blueprint("health", __name__, url_prefix="/api")


def _svc():
    return current_app.extensions["rag"]


@bp.get("/health")
def health():
    """Service health, LLM connectivity, embedding model and index statistics.
    ---
    tags: [System]
    responses:
      200:
        description: Health report (always 200; check `llm.status` for Ollama state)
        schema:
          type: object
          properties:
            status: {type: string, example: ok}
            llm:
              type: object
              properties:
                status: {type: string, enum: [ok, model_missing, unavailable]}
                model: {type: string}
            embedding:
              type: object
              properties:
                model: {type: string}
                dimension: {type: integer}
            vector_store: {type: object}
    """
    s = _svc()
    return jsonify({
        "status": "ok",
        "llm": s.llm.health(),
        "embedding": {"model": s.embeddings.model_name, "dimension": s.embeddings.dimension,
                      "backend": s.settings.embedding_backend},
        "vector_store": s.store.stats(),
    })


@bp.get("/stats")
def stats():
    """Dashboard statistics: documents, chunks, vector storage.
    ---
    tags: [System]
    responses:
      200:
        description: Aggregate statistics
    """
    s = _svc()
    return jsonify({"documents": s.documents.stats(), "vector_store": s.store.stats(),
                    "embedding_model": s.embeddings.model_name,
                    "conversations": len(s.conversations.list(1000))})


@bp.get("/settings")
def get_settings():
    """Server-side defaults (safe subset; no paths or secrets).
    ---
    tags: [System]
    responses:
      200:
        description: Default retrieval, chunking and LLM settings
    """
    return jsonify(_svc().settings.public_dict())
