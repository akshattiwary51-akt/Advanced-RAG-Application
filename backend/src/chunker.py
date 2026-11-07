"""Custom paragraph- and sentence-aware chunker with overlap.

Algorithm
---------
1. Split each page into paragraphs, each paragraph into sentences. A sentence longer
   than ``chunk_size`` is split on word boundaries (and hard-sliced as a last resort).
2. Greedily pack sentences into a chunk until adding the next one would exceed
   ``chunk_size``.
3. Overlap: the next chunk starts with the trailing *whole sentences* of the previous
   chunk whose total length fits in ``overlap`` characters.
4. Size guard: a chunk shorter than ``min_size`` may grow up to ``max_size``
   (1.25 x chunk_size) instead of being closed, and a tiny final chunk is merged into
   the previous one when it fits. This avoids slivers without ever exceeding
   ``max_size``.

Chunks never span pages, so the page number on every chunk is exact.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from .loaders import PageText
from .text_processor import split_paragraphs, split_sentences

MAX_SIZE_FACTOR = 1.25


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    page: Optional[int] = None
    text_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class _Unit:
    text: str
    new_paragraph: bool


def _split_oversized(sentence: str, max_len: int) -> list[str]:
    if len(sentence) <= max_len:
        return [sentence]
    pieces, cur = [], ""
    for word in sentence.split(" "):
        while len(word) > max_len:  # single token longer than the limit
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(word[:max_len])
            word = word[max_len:]
        if cur and len(cur) + 1 + len(word) > max_len:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        pieces.append(cur)
    return pieces


def _to_units(text: str, chunk_size: int) -> list[_Unit]:
    units: list[_Unit] = []
    for para in split_paragraphs(text):
        first = True
        for sentence in split_sentences(para):
            for piece in _split_oversized(sentence, chunk_size):
                units.append(_Unit(piece, first))
                first = False
    return units


def _sep(unit: _Unit, has_prev: bool) -> int:
    return (2 if unit.new_paragraph else 1) if has_prev else 0


def _render(units: list[_Unit]) -> str:
    out = []
    for i, u in enumerate(units):
        if i:
            out.append("\n\n" if u.new_paragraph else " ")
        out.append(u.text)
    return "".join(out)


def _length(units: list[_Unit]) -> int:
    return sum(len(u.text) + _sep(u, i > 0) for i, u in enumerate(units))


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    tail: list[_Unit] = []
    total = 0
    for u in reversed(units):
        cost = len(u.text) + (1 if tail else 0)
        if total + cost > overlap:
            break
        tail.insert(0, u)
        total += cost
    if len(tail) == len(units):  # never carry the entire chunk (guarantees progress)
        tail = tail[1:]
    return tail


def chunk_text(
    text: str, chunk_size: int = 800, overlap: int = 150, min_size: int = 100
) -> list[str]:
    """Split ``text`` into overlapping, sentence-aligned chunks."""
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")
    if min_size < 0 or min_size >= chunk_size:
        raise ValueError("min_size must be >= 0 and smaller than chunk_size.")

    units = _to_units(text, chunk_size)
    if not units:
        return []
    max_size = int(chunk_size * MAX_SIZE_FACTOR)

    # each entry: (units_in_chunk, index where *new* content starts)
    chunks: list[tuple[list[_Unit], int]] = []
    cur: list[_Unit] = []
    new_from = 0
    cur_len = 0

    for u in units:
        add = len(u.text) + _sep(u, bool(cur))
        overflow = cur and cur_len + add > chunk_size
        if overflow and cur_len < min_size and cur_len + add <= max_size:
            overflow = False  # grow a too-small chunk instead of closing it
        if overflow:
            chunks.append((cur, new_from))
            cur = _overlap_tail(cur, overlap)
            while cur and _length(cur) + len(u.text) + _sep(u, True) > chunk_size:
                cur = cur[1:]
            new_from = len(cur)
            cur_len = _length(cur)
            add = len(u.text) + _sep(u, bool(cur))
        cur.append(u)
        cur_len += add
    if len(cur) > new_from:
        chunks.append((cur, new_from))

    # merge a tiny final chunk's new content into its predecessor when it fits
    if len(chunks) > 1 and _length(chunks[-1][0]) < min_size:
        last_units, last_new = chunks[-1]
        prev_units, prev_new = chunks[-2]
        merged = prev_units + last_units[last_new:]
        if _length(merged) <= max_size:
            chunks[-2:] = [(merged, prev_new)]

    return [_render(u) for u, _ in chunks]


def _alnum(text: str) -> bool:
    return any(c.isalnum() for c in text)


def chunk_pages(
    pages: Iterable[PageText],
    document_id: str,
    filename: str,
    chunk_size: int = 800,
    overlap: int = 150,
    min_size: int = 100,
) -> list[Chunk]:
    """Chunk every page and attach document metadata. Indices are document-wide."""
    out: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text, chunk_size, overlap, min_size):
            if not _alnum(text):
                continue
            idx = len(out)
            out.append(
                Chunk(
                    chunk_id=f"{document_id}:{idx}",
                    document_id=document_id,
                    filename=filename,
                    chunk_index=idx,
                    text=text,
                    page=page.page,
                    text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
    return out
