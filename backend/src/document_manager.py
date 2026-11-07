"""Document lifecycle: upload, dedupe, index, re-index, delete, reconcile.

This is the only module that touches SQLite document rows, the uploads folder and the
vector store together, so it is where metadata/vector consistency is enforced:

* Files are stored under a *generated* name (``<uuid>.<ext>``): user-supplied names never
  reach the filesystem, which removes path traversal by construction.
* A failed ingest leaves nothing behind (no row, no file, no vectors).
* ``delete`` removes vectors first, then the file, then the row; ``reconcile`` repairs
  any state a crash could leave (orphan vectors, rows without vectors, orphan files).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, Optional

from sqlalchemy import select

from config import Settings
from database.db import Database
from database.models import Document, new_id, utcnow
from .loaders import (
    CorruptedDocumentError,
    DocumentLoadError,
    FileTooLargeError,
    sanitize_display_name,
    sha256_file,
    validate_upload,
    verify_signature,
)
from .rag_pipeline import RAGPipeline

log = logging.getLogger(__name__)


class DuplicateDocumentError(DocumentLoadError):
    def __init__(self, existing: dict):
        super().__init__(f"This file was already uploaded as '{existing['filename']}'.")
        self.existing = existing


class DocumentNotFoundError(Exception):
    pass


class DocumentManager:
    def __init__(self, settings: Settings, db: Database, pipeline: RAGPipeline):
        self.settings = settings
        self.db = db
        self.pipeline = pipeline
        settings.ensure_dirs()

    # ----------------------------------------------------------------- paths
    def _path_for(self, doc: Document) -> Path:
        base = self.settings.uploads_dir.resolve()
        p = (base / doc.stored_name).resolve()
        if base not in p.parents:  # defence in depth; stored_name is generated anyway
            raise CorruptedDocumentError("Invalid stored path.")
        return p

    # ---------------------------------------------------------------- upload
    def upload(self, stream: BinaryIO, filename: str, mimetype: Optional[str] = None) -> dict:
        """Validate, save, dedupe and index one file. Raises ``DocumentLoadError`` subclasses."""
        display = sanitize_display_name(filename)
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(0)
        ext = validate_upload(display, mimetype, size, self.settings.max_upload_bytes)
        verify_signature(ext, stream.read(2048))
        stream.seek(0)

        doc_id = new_id()
        stored_name = f"{doc_id}{ext}"
        dest = self.settings.uploads_dir / stored_name
        written = 0
        try:
            with open(dest, "wb") as out:
                while chunk := stream.read(1 << 20):
                    written += len(chunk)
                    if written > self.settings.max_upload_bytes:  # do not trust declared size alone
                        raise FileTooLargeError("File exceeds the maximum upload size.")
                    out.write(chunk)
            digest = sha256_file(dest)
            with self.db.session() as s:
                existing = s.scalar(select(Document).where(Document.sha256 == digest))
                if existing is not None:
                    raise DuplicateDocumentError(existing.to_dict())
                doc = Document(id=doc_id, filename=display, stored_name=stored_name,
                               file_type=ext.lstrip("."), size_bytes=written, sha256=digest,
                               status="processing")
                s.add(doc)
            try:
                result = self.pipeline.process_document(dest, doc_id, display, ext)
            except Exception:
                self.pipeline.delete_document(doc_id)
                with self.db.session() as s:
                    row = s.get(Document, doc_id)
                    if row is not None:
                        s.delete(row)
                raise
            with self.db.session() as s:
                doc = s.get(Document, doc_id)
                doc.status, doc.error = "indexed", None
                doc.page_count, doc.chunk_count = result.page_count, result.chunk_count
                doc.char_count, doc.indexed_at = result.char_count, utcnow()
                out_dict = doc.to_dict()
            out_dict["timings_ms"] = {k: round(v, 2) for k, v in result.timings_ms.items()}
            return out_dict
        except BaseException:
            dest.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ read
    def list(self) -> list[dict]:
        with self.db.session() as s:
            rows = s.scalars(select(Document).order_by(Document.created_at.desc())).all()
            return [d.to_dict() for d in rows]

    def get(self, document_id: str) -> dict:
        with self.db.session() as s:
            d = s.get(Document, document_id)
            if d is None:
                raise DocumentNotFoundError(document_id)
            return d.to_dict()

    def indexed_ids(self) -> list[str]:
        with self.db.session() as s:
            return list(s.scalars(select(Document.id).where(Document.status == "indexed")))

    # --------------------------------------------------------------- re-index
    def reindex(self, document_id: str, chunk_size: Optional[int] = None,
                overlap: Optional[int] = None) -> dict:
        """Re-extract and re-embed from the stored file (e.g. after changing chunk settings)."""
        with self.db.session() as s:
            d = s.get(Document, document_id)
            if d is None:
                raise DocumentNotFoundError(document_id)
            path, ext, filename = self._path_for(d), "." + d.file_type, d.filename
            d.status = "processing"
        try:
            result = self.pipeline.process_document(path, document_id, filename, ext, chunk_size, overlap)
        except Exception as exc:
            self.pipeline.delete_document(document_id)
            with self.db.session() as s:
                d = s.get(Document, document_id)
                d.status, d.error, d.chunk_count = "failed", str(exc)[:500], 0
            raise
        with self.db.session() as s:
            d = s.get(Document, document_id)
            d.status, d.error = "indexed", None
            d.page_count, d.chunk_count = result.page_count, result.chunk_count
            d.char_count, d.indexed_at = result.char_count, utcnow()
            out = d.to_dict()
        out["timings_ms"] = {k: round(v, 2) for k, v in result.timings_ms.items()}
        return out

    # ----------------------------------------------------------------- delete
    def delete(self, document_id: str) -> dict:
        with self.db.session() as s:
            d = s.get(Document, document_id)
            if d is None:
                raise DocumentNotFoundError(document_id)
            path = self._path_for(d)
        removed = self.pipeline.delete_document(document_id)  # vectors first
        path.unlink(missing_ok=True)
        with self.db.session() as s:
            row = s.get(Document, document_id)
            if row is not None:
                s.delete(row)
        return {"id": document_id, "chunks_removed": removed}

    # -------------------------------------------------------------- reconcile
    def reconcile(self) -> dict:
        """Repair metadata/vector/file drift. Safe to run at every start-up."""
        store = self.pipeline.store
        report = {"orphan_vector_docs": 0, "marked_failed": 0, "orphan_files": 0}
        with self.db.session() as s:
            rows = s.scalars(select(Document)).all()
            known = {d.id for d in rows}
            for doc_id in store.document_ids() - known:
                store.delete_document(doc_id)
                report["orphan_vector_docs"] += 1
            for d in rows:
                n = store.count(d.id)
                if d.status == "indexed" and n == 0:
                    d.status, d.error = "failed", "Index missing; re-index required."
                    report["marked_failed"] += 1
                elif d.status == "processing":  # interrupted ingest
                    d.status, d.error = "failed", "Interrupted; re-index required."
                    report["marked_failed"] += 1
                elif d.status == "indexed" and n != d.chunk_count:
                    d.chunk_count = n
            stored = {d.stored_name for d in rows}
        for f in self.settings.uploads_dir.iterdir():
            if f.is_file() and f.name not in stored and f.name != ".gitkeep":
                f.unlink(missing_ok=True)
                report["orphan_files"] += 1
        return report

    def stats(self) -> dict:
        docs = self.list()
        return {
            "total_documents": len(docs),
            "indexed_documents": sum(d["status"] == "indexed" for d in docs),
            "failed_documents": sum(d["status"] == "failed" for d in docs),
            "total_chunks": sum(d["chunk_count"] for d in docs),
            "total_bytes": sum(d["size_bytes"] for d in docs),
        }
