"""Shared pytest configuration: make backend/ importable and keep tests offline."""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Core tests must never need a model download or an Ollama server.
os.environ.setdefault("EMBEDDING_BACKEND", "mock")
os.environ.setdefault("LLM_BACKEND", "mock")
