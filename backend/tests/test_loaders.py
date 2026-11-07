import hashlib

import pytest

from src.loaders import (
    CorruptedDocumentError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    load_document,
    sanitize_display_name,
    sha256_bytes,
    sha256_file,
    validate_upload,
)
from tests.helpers import make_docx, make_pdf

MB = 1024 * 1024


# ----------------------------------------------------------------- validation
def test_validate_accepts_known_types():
    assert validate_upload("a.PDF", "application/pdf", 10, MB) == ".pdf"
    assert validate_upload("a.txt", "text/plain; charset=utf-8", 10, MB) == ".txt"
    assert validate_upload("a.csv", "application/octet-stream", 10, MB) == ".csv"


@pytest.mark.parametrize("name", ["evil.exe", "noext", "archive.zip", "a.pdf.exe"])
def test_validate_rejects_bad_extension(name):
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(name, "text/plain", 10, MB)


def test_validate_rejects_mime_mismatch():
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload("a.pdf", "text/html", 10, MB)


def test_validate_size_limits():
    with pytest.raises(FileTooLargeError):
        validate_upload("a.txt", "text/plain", 2 * MB, MB)
    with pytest.raises(EmptyDocumentError):
        validate_upload("a.txt", "text/plain", 0, MB)


def test_sanitize_display_name_strips_paths_and_control_chars():
    assert sanitize_display_name("../../etc/passwd") == "passwd"
    assert sanitize_display_name("C:\\Users\\x\\report.pdf") == "report.pdf"
    assert sanitize_display_name("a\x00b\n.txt") == "ab.txt"
    assert sanitize_display_name("") == "unnamed"
    assert len(sanitize_display_name("x" * 500 + ".pdf")) <= 150


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_file(p) == expected == sha256_bytes(b"hello world")


# --------------------------------------------------------------------- loaders
def test_load_txt_utf8(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Héllo wörld. Second sentence.", encoding="utf-8")
    doc = load_document(p)
    assert doc.file_type == "txt"
    assert "Héllo" in doc.pages[0].text and doc.pages[0].page is None


def test_load_txt_cp1252_fallback(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("café résumé".encode("cp1252"))
    assert "café" in load_document(p).pages[0].text


def test_empty_and_whitespace_files_raise(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        load_document(p)


def test_binary_masquerading_as_txt(tmp_path):
    p = tmp_path / "b.txt"
    p.write_bytes(b"\x00\x01\x02\x03binary")
    with pytest.raises(CorruptedDocumentError):
        load_document(p)


def test_load_csv_renders_rows_as_key_value(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("name,city\nAda,London\nGrace,New York\n")
    text = load_document(p).pages[0].text
    assert "name: Ada; city: London." in text
    assert "name: Grace; city: New York." in text
    assert "\n\n" in text  # one paragraph per row


def test_load_csv_semicolon_delimiter(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a;b\n1;2\n3;4\n")
    assert "a: 1; b: 2." in load_document(p).pages[0].text


def test_load_docx_paragraphs_and_tables(tmp_path):
    p = tmp_path / "d.docx"
    make_docx(p, ["First paragraph.", "Second paragraph."], [["Item", "Qty"], ["Bolt", "10"]])
    text = load_document(p).pages[0].text
    assert "First paragraph." in text and "Second paragraph." in text
    assert "Item | Qty" in text and "Bolt | 10" in text


def test_docx_not_a_zip(tmp_path):
    p = tmp_path / "bad.docx"
    p.write_bytes(b"PK this is not really a zip")
    with pytest.raises(CorruptedDocumentError):
        load_document(p)


def test_load_pdf_extracts_pages_with_numbers(tmp_path):
    p = tmp_path / "d.pdf"
    p.write_bytes(make_pdf(["Alpha page one text", "Bravo page two text"]))
    doc = load_document(p)
    assert [pg.page for pg in doc.pages] == [1, 2]
    assert "Alpha" in doc.pages[0].text and "Bravo" in doc.pages[1].text
    assert doc.page_count == 2


def test_pdf_blank_pages_are_dropped_but_numbers_kept(tmp_path):
    p = tmp_path / "d.pdf"
    p.write_bytes(make_pdf(["", "Only content here"]))
    doc = load_document(p)
    assert [pg.page for pg in doc.pages] == [2]


def test_pdf_with_no_text_raises_empty(tmp_path):
    p = tmp_path / "d.pdf"
    p.write_bytes(make_pdf(["", ""]))
    with pytest.raises(EmptyDocumentError):
        load_document(p)


def test_corrupted_pdf_raises(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is garbage, not a pdf body")
    with pytest.raises((CorruptedDocumentError, EmptyDocumentError)):
        load_document(p)


def test_pdf_signature_mismatch(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"just some text pretending to be a pdf")
    with pytest.raises(CorruptedDocumentError):
        load_document(p)


def test_unsupported_extension_and_missing_file(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"MZ")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(p)
    with pytest.raises(CorruptedDocumentError):
        load_document(tmp_path / "missing.txt")
