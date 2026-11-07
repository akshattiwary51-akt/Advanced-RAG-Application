"""Text cleaning, paragraph splitting and sentence splitting.

Pure functions, no I/O. The sentence splitter is deliberately rule-based (no NLTK)
so its behaviour is fully visible: it breaks after ``. ! ?`` when the next token
looks like the start of a sentence, except after common abbreviations, initials
and decimals.
"""
from __future__ import annotations

import re
import unicodedata

MAX_QUERY_CHARS = 1000

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")
_INLINE_SPACE = re.compile(r"[ \t\f\v]+")
_SENT_END = re.compile(r"""([.!?]+["')\]]*)(\s+)""")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "fig", "figs", "no", "nos", "vol", "pp", "p", "approx", "inc", "ltd", "co",
    "u.s", "u.k", "cf", "al", "eq", "sec", "ch", "dept", "est", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}


def clean_text(text: str, unwrap_lines: bool = True) -> str:
    """Normalise raw extracted text.

    * Unicode NFKC (fixes ligatures such as 'ﬁ', non-breaking spaces, full-width forms)
    * Unifies newlines, strips control characters and soft hyphens
    * Re-joins words hyphenated across line breaks ("exam-\\nple" -> "example")
    * Collapses runs of spaces and 3+ newlines (paragraph breaks are kept)
    * ``unwrap_lines``: turn single hard line-wraps (typical of PDFs) into spaces
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _INLINE_SPACE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    if unwrap_lines:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; drop empties."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _is_boundary(before: str, after: str) -> bool:
    """Decide whether the punctuation run ending ``before`` really ends a sentence."""
    if not after:
        return False
    nxt = after[0]
    if not (nxt.isupper() or nxt.isdigit() or nxt in "\"'([“‘"):
        return False
    last_token = re.split(r"\s+", before.rstrip())[-1] if before.strip() else ""
    word = last_token.rstrip(".!?\"')]").lower()
    if before.rstrip().endswith("."):
        if word in _ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha():  # initials: "J. Smith"
            return False
    return True


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (whitespace inside is collapsed)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        if _is_boundary(text[start:end], text[m.end():]):
            sentences.append(text[start:end].strip())
            start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens (used by keyword/BM25 search and metrics)."""
    return _WORD.findall(text.lower())


def preprocess_query(query: str) -> str:
    """Normalise and bound a user query. Raises ``ValueError`` if it is empty."""
    q = clean_text(query or "", unwrap_lines=True)
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        raise ValueError("Query must not be empty.")
    return q[:MAX_QUERY_CHARS]
