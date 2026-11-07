"""Flask application factory.

Run:  python app.py        (development)
Docs: http://127.0.0.1:5000/apidocs
"""
from __future__ import annotations

import logging
from typing import Optional

from flasgger import Swagger
from flask import Flask
from flask_cors import CORS

from config import Settings, get_settings
from routes import chat_routes, document_routes, health_routes, upload_routes
from services.container import Services, build_services
from utils.errors import register_error_handlers

log = logging.getLogger("rag")

SWAGGER_TEMPLATE = {
    "info": {
        "title": "Advanced RAG API",
        "description": "Upload documents, index them with a custom NumPy vector store, "
                       "and chat with cited, context-grounded answers.",
        "version": "1.0.0",
    },
    "tags": [{"name": t} for t in ("System", "Documents", "Chat", "Debug")],
}


def create_app(settings: Optional[Settings] = None, services: Optional[Services] = None) -> Flask:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__)
    # Per-file size is enforced in the document manager; this caps the whole request body.
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes * 10 + 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    CORS(app, resources={r"/api/*": {"origins": list(settings.cors_origins)}})
    Swagger(app, template=SWAGGER_TEMPLATE)
    register_error_handlers(app)

    services = services or build_services(settings)
    app.extensions["rag"] = services
    report = services.documents.reconcile()
    if any(report.values()):
        log.warning("Startup reconciliation repaired: %s", report)

    for module in (health_routes, upload_routes, document_routes, chat_routes):
        app.register_blueprint(module.bp)
    return app


if __name__ == "__main__":
    cfg = get_settings()
    create_app(cfg).run(host=cfg.flask_host, port=cfg.flask_port, threaded=True)
