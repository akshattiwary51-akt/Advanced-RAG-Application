import json
import logging
import time

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from utils.errors import AppError, describe
from src.text_processor import preprocess_query
from utils.validation import BOOL_FLAGS, parse_chat_request, parse_retrieval

log = logging.getLogger(__name__)
bp = Blueprint("chat", __name__, url_prefix="/api")

def _svc():
    return current_app.extensions["rag"]


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@bp.post("/chat")
def chat():
    """Ask a question and receive a grounded answer with citations.
    ---
    tags: [Chat]
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required: [message]
          properties:
            message: {type: string, example: "What does the report say about revenue?"}
            conversation_id: {type: string, description: Omit to start a new conversation}
            document_ids: {type: array, items: {type: string}}
            top_k: {type: integer, example: 5}
            similarity_threshold: {type: number, example: 0.25}
            temperature: {type: number, example: 0.2}
            max_tokens: {type: integer, example: 1024}
            model: {type: string}
            method: {type: string, enum: [semantic]}
    responses:
      200:
        description: Answer, citations and timings
      400: {description: Invalid request}
      404: {description: Unknown conversation}
      503: {description: LLM unavailable or model not installed}
    """
    s = _svc()
    req = parse_chat_request(request.get_json(silent=True), s.settings)
    conv_id = s.conversations.ensure(req["conversation_id"], req["message"])
    history = s.conversations.history(conv_id)
    result = s.pipeline.answer_query(req["message"], history, req["opts"], req["params"])
    s.conversations.add_message(conv_id, "user", req["message"])
    msg = s.conversations.add_message(conv_id, "assistant", result.answer,
                                      [c.to_dict() for c in result.citations])
    return jsonify({"conversation_id": conv_id, "message": msg, **result.to_dict()})


@bp.post("/chat/stream")
def chat_stream():
    """Streaming variant of /chat using Server-Sent Events.

    Events (`data: <json>`): `meta` {conversation_id}, `sources` {citations},
    `token` {content}, `done` {message, citations, timings_ms, ...}, `error` {message}.
    ---
    tags: [Chat]
    produces: [text/event-stream]
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required: [message]
          properties:
            message: {type: string}
            conversation_id: {type: string}
            document_ids: {type: array, items: {type: string}}
            top_k: {type: integer}
            similarity_threshold: {type: number}
            temperature: {type: number}
            max_tokens: {type: integer}
            model: {type: string}
    responses:
      200: {description: SSE stream}
      400: {description: Invalid request (returned before the stream starts)}
    """
    s = _svc()
    req = parse_chat_request(request.get_json(silent=True), s.settings)
    conv_id = s.conversations.ensure(req["conversation_id"], req["message"])
    history = s.conversations.history(conv_id)

    def generate():
        yield _sse({"type": "meta", "conversation_id": conv_id})
        try:
            for ev in s.pipeline.stream_answer(req["message"], history, req["opts"], req["params"]):
                if ev["type"] == "done":
                    result = ev["result"]
                    s.conversations.add_message(conv_id, "user", req["message"])
                    msg = s.conversations.add_message(conv_id, "assistant", result.answer,
                                                      [c.to_dict() for c in result.citations])
                    yield _sse({"type": "done", "conversation_id": conv_id, "message": msg,
                                **result.to_dict()})
                else:
                    yield _sse(ev)
        except Exception as exc:  # headers are already sent; report in-band
            _, body = describe(exc)
            yield _sse({"type": "error", "message": body["error"]["message"]})

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.get("/conversations")
def list_conversations():
    """List conversations, most recently active first.
    ---
    tags: [Chat]
    responses:
      200: {description: Conversation summaries}
    """
    return jsonify({"conversations": _svc().conversations.list()})


@bp.get("/conversations/<conversation_id>")
def get_conversation(conversation_id: str):
    """Get a conversation with all messages and their citations.
    ---
    tags: [Chat]
    parameters:
      - {name: conversation_id, in: path, type: string, required: true}
    responses:
      200: {description: Conversation with messages}
      404: {description: Not found}
    """
    return jsonify(_svc().conversations.get(conversation_id))


@bp.delete("/conversations/<conversation_id>")
def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages.
    ---
    tags: [Chat]
    parameters:
      - {name: conversation_id, in: path, type: string, required: true}
    responses:
      200: {description: Deleted}
      404: {description: Not found}
    """
    _svc().conversations.delete(conversation_id)
    return jsonify({"id": conversation_id, "deleted": True})


@bp.get("/debug/retrieval")
def debug_retrieval():
    """Inspect retrieval without calling the LLM: chunks, scores and latency.
    ---
    tags: [Debug]
    parameters:
      - {name: q, in: query, type: string, required: true, description: Query text}
      - {name: top_k, in: query, type: integer, default: 5}
      - {name: similarity_threshold, in: query, type: number, default: 0.0}
      - {name: document_ids, in: query, type: string, description: Comma-separated ids}
      - {name: method, in: query, type: string, enum: [semantic, bm25, hybrid], default: semantic}
      - {name: use_mmr, in: query, type: boolean}
      - {name: mmr_lambda, in: query, type: number}
      - {name: rerank, in: query, type: boolean}
      - {name: expand_query, in: query, type: boolean}
      - {name: compress, in: query, type: boolean}
    responses:
      200: {description: Retrieved chunks with scores}
      400: {description: Missing or invalid query}
    """
    s = _svc()
    q = request.args.get("q", "")
    if not q.strip():
        raise AppError("Query parameter 'q' is required.", 400, "invalid_request")
    raw = request.args
    body: dict = {}
    try:
        if "top_k" in raw:
            body["top_k"] = int(raw["top_k"])
        body["similarity_threshold"] = float(raw.get("similarity_threshold", 0.0))
        if "mmr_lambda" in raw:
            body["mmr_lambda"] = float(raw["mmr_lambda"])
    except ValueError as exc:
        raise AppError("'top_k', 'similarity_threshold' and 'mmr_lambda' must be numbers.",
                       400, "invalid_request") from exc
    if "method" in raw:
        body["method"] = raw["method"]
    for flag in BOOL_FLAGS:
        if flag in raw:
            body[flag] = raw[flag].lower() in ("1", "true", "yes", "on")
    if raw.get("document_ids"):
        body["document_ids"] = [d for d in raw["document_ids"].split(",") if d]
    opts = parse_retrieval(body, s.settings)
    t0 = time.perf_counter()
    chunks = s.pipeline.retrieve_relevant_chunks(preprocess_query(q), opts)  # same path as chat
    ms = (time.perf_counter() - t0) * 1000.0
    return jsonify({
        "query": q.strip()[:200],
        "method": opts.method,
        "options": {"use_mmr": opts.use_mmr, "mmr_lambda": opts.mmr_lambda, "rerank": opts.rerank,
                    "expand_query": opts.expand_query, "compress": opts.compress},
        "top_k": opts.top_k,
        "similarity_threshold": opts.threshold,
        "latency_ms": round(ms, 2),
        "total_chunks_indexed": s.store.count(),
        "results": [c.to_dict() for c in chunks],
    })
