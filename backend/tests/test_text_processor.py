import pytest

from src.text_processor import (
    clean_text,
    preprocess_query,
    split_paragraphs,
    split_sentences,
    tokenize,
)


def test_clean_text_basics():
    raw = "Hello  world\r\n\r\n\r\n\r\nNext\x00 para­graph"
    out = clean_text(raw)
    assert out == "Hello world\n\nNext paragraph"


def test_clean_text_ligatures_and_hyphenation():
    assert clean_text("eﬀective exam-\nple") == "effective example"


def test_clean_text_unwrap_toggle():
    assert clean_text("line one\nline two") == "line one line two"
    assert clean_text("line one\nline two", unwrap_lines=False) == "line one\nline two"


def test_clean_text_keeps_real_hyphens_before_capital():
    assert clean_text("well-\nKnown", unwrap_lines=False) == "well-\nKnown"


def test_clean_text_empty():
    assert clean_text("") == "" and clean_text(None) == ""


def test_split_paragraphs():
    assert split_paragraphs("a\n\n\nb\n \nc") == ["a", "b", "c"]


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]


def test_split_sentences_respects_abbreviations_initials_decimals():
    s = split_sentences("Dr. Smith met J. Doe. They paid 3.14 dollars. Done.")
    assert s == ["Dr. Smith met J. Doe.", "They paid 3.14 dollars.", "Done."]


def test_split_sentences_lowercase_continuation_not_split():
    assert split_sentences("See e.g. the table. then more") == ["See e.g. the table. then more"]


def test_split_sentences_collapses_internal_newlines():
    assert split_sentences("First line\nstill first. Second.") == ["First line still first.", "Second."]


def test_tokenize():
    assert tokenize("Hello, World! It's 2026.") == ["hello", "world", "it's", "2026"]


def test_preprocess_query():
    assert preprocess_query("  what   is\nRAG? ") == "what is RAG?"
    assert len(preprocess_query("x" * 5000)) == 1000
    with pytest.raises(ValueError):
        preprocess_query("   \n ")
