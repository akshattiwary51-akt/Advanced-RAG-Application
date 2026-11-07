"""Document ingestion: validation, hashing and text extraction.

Responsibilities
----------------
* Validate an upload (extension, declared MIME type, size, magic bytes).
* Compute a SHA-256 digest used for duplicate detection.
* Extract text per page for PDF / DOCX / TXT / CSV.
* Fail *gracefully*: every problem becomes a ``DocumentLoadError`` subclass that
  the API layer maps to a clean HTTP status. Nothing here touches the network
  or the database.

Stored file names are never user-controlled (the document manager stores files
under a generated id), which is what actually prevents path traversal. The
display-name sanitiser below is for presentation and metadata only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional

ALLOWED_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
    ".txt": {"text/plain"},
    ".csv": {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"},
}
# Browsers/clients often send these for anything; the extension + magic bytes decide.
_GENERIC_MIMES = {"", "application/octet-stream", "binary/octet-stream"}

_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


# ---------------------------------------------------------------- exceptions
class DocumentLoadError(Exception):
    """Base class for all ingestion problems (maps to HTTP 4xx)."""


class UnsupportedFileTypeError(DocumentLoadError):
    pass


class FileTooLargeError(DocumentLoadError):
    pass


class EmptyDocumentError(DocumentLoadError):
    pass


class CorruptedDocumentError(DocumentLoadError):
    pass


# ------------------------------------------------------------------- models
@dataclass
class PageText:
    """Text of one logical page. ``page`` is None for formats without pages."""

    page: Optional[int]
    text: str


@dataclass
class LoadedDocument:
    file_type: str  # extension without dot: pdf, docx, txt, csv
    pages: list[PageText] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


# --------------------------------------------------------------- validation
def sanitize_display_name(filename: str, max_len: int = 150) -> str:
    """Return a safe, human-readable file name (no directories, no control chars)."""
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = _CONTROL.sub("", name).strip().strip(".")
    if len(name) > max_len:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: max_len - len(ext) - 1] + dot + ext) if dot else name[:max_len]
    return name or "unnamed"


def get_extension(filename: str) -> str:
    return Path(sanitize_display_name(filename)).suffix.lower()


def validate_upload(
    filename: str, mimetype: Optional[str], size_bytes: int, max_bytes: int
) -> str:
    """Validate metadata of an upload. Returns the lower-case extension (with dot)."""
    ext = get_extension(filename)
    if ext not in ALLOWED_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: "
            + ", ".join(sorted(ALLOWED_TYPES))
        )
    mime = (mimetype or "").split(";")[0].strip().lower()
    if mime not in _GENERIC_MIMES and mime not in ALLOWED_TYPES[ext]:
        raise UnsupportedFileTypeError(
            f"MIME type '{mime}' does not match the '{ext}' extension."
        )
    if size_bytes <= 0:
        raise EmptyDocumentError("The uploaded file is empty.")
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"File is {size_bytes / 1024 / 1024:.1f} MB; the limit is "
            f"{max_bytes / 1024 / 1024:.0f} MB."
        )
    return ext


def verify_signature(ext: str, head: bytes) -> None:
    """Check magic bytes so a renamed binary cannot masquerade as a document."""
    if ext == ".pdf" and not head.lstrip().startswith(b"%PDF"):
        raise CorruptedDocumentError("File content is not a valid PDF.")
    if ext == ".docx" and not head.startswith(b"PK"):
        raise CorruptedDocumentError("File content is not a valid DOCX archive.")
    if ext in (".txt", ".csv") and b"\x00" in head and not head.startswith(
        (b"\xff\xfe", b"\xfe\xff")
    ):
        raise CorruptedDocumentError("File looks binary, not text.")


# ------------------------------------------------------------------ hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_stream(stream: BinaryIO, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(block), b""):
        h.update(chunk)
    stream.seek(0)
    return h.hexdigest()


# ---------------------------------------------------------------- extractors
def _decode_text(raw: bytes) -> str:
    for enc in _TEXT_ENCODINGS:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _load_txt(path: Path) -> list[PageText]:
    return [PageText(None, _decode_text(path.read_bytes()))]


def _load_csv(path: Path) -> list[PageText]:
    """Render each row as ``col: value; col: value`` and separate rows by a blank line.

    Row-per-paragraph lets the chunker pack whole rows together instead of cutting
    a record in half, and gives the embedder readable key/value text.
    """
    text = _decode_text(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    except csv.Error as exc:
        raise CorruptedDocumentError(f"Could not parse CSV: {exc}") from exc
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:  # header only: treat as plain text
        return [PageText(None, " | ".join(header))]
    lines = []
    for row in body:
        pairs = [
            f"{(header[i] if i < len(header) else f'col{i + 1}').strip()}: {v.strip()}"
            for i, v in enumerate(row)
            if v.strip()
        ]
        if pairs:
            lines.append("; ".join(pairs) + ".")
    return [PageText(None, "\n\n".join(lines))]


def _load_docx(path: Path) -> list[PageText]:
    if not zipfile.is_zipfile(path):
        raise CorruptedDocumentError("DOCX file is not a valid archive.")
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(str(path))
        blocks: list[str] = []
        for child in doc.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                t = Paragraph(child, doc).text.strip()
                if t:
                    blocks.append(t)
            elif tag == "tbl":
                for row in Table(child, doc).rows:
                    cells = []
                    for c in row.cells:
                        v = c.text.strip()
                        if v and (not cells or cells[-1] != v):  # merged cells repeat
                            cells.append(v)
                    if cells:
                        blocks.append(" | ".join(cells))
    except DocumentLoadError:
        raise
    except Exception as exc:  # python-docx raises many unrelated types
        raise CorruptedDocumentError(f"Could not read DOCX: {type(exc).__name__}") from exc
    return [PageText(None, "\n\n".join(blocks))]


def _load_pdf(path: Path) -> list[PageText]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise CorruptedDocumentError("PDF is password-protected.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # one bad page must not sink the whole document
                text = ""
            pages.append(PageText(i, text))
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError(f"Could not read PDF: {type(exc).__name__}") from exc
    return pages


_LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt, ".csv": _load_csv}


def load_document(path: Path | str, ext: Optional[str] = None) -> LoadedDocument:
    """Extract text pages from ``path``.

    Raises ``UnsupportedFileTypeError``, ``CorruptedDocumentError`` or
    ``EmptyDocumentError`` (no extractable text, e.g. a scanned PDF).
    """
    path = Path(path)
    ext = (ext or path.suffix).lower()
    if ext not in _LOADERS:
        raise UnsupportedFileTypeError(f"Unsupported file type '{ext}'.")
    if not path.is_file():
        raise CorruptedDocumentError("File not found on disk.")
    with open(path, "rb") as fh:
        verify_signature(ext, fh.read(2048))

    pages = _LOADERS[ext](path)
    pages = [p for p in pages if p.text and p.text.strip()]
    if not pages:
        hint = " (it may be a scanned image; OCR is not supported)" if ext == ".pdf" else ""
        raise EmptyDocumentError(f"No extractable text found{hint}.")
    return LoadedDocument(file_type=ext.lstrip("."), pages=pages)
