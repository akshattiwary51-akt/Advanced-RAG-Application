import pytest

from src.chunker import MAX_SIZE_FACTOR, chunk_pages, chunk_text
from src.loaders import PageText
from src.text_processor import split_sentences

SENT = "This is sentence number {i} about topic {i} with some padding words."


def make_text(n: int, para_every: int = 0) -> str:
    parts = []
    for i in range(n):
        parts.append(SENT.format(i=i))
        if para_every and (i + 1) % para_every == 0:
            parts.append("\n\n")
        else:
            parts.append(" ")
    return "".join(parts).strip()


def test_single_small_text_is_one_chunk():
    assert chunk_text("Just one short sentence.", 200, 20, 10) == ["Just one short sentence."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("", 200, 20, 10) == []
    assert chunk_text("  \n\n ", 200, 20, 10) == []


def test_chunks_respect_max_size():
    chunks = chunk_text(make_text(80), chunk_size=300, overlap=60, min_size=50)
    assert len(chunks) > 3
    assert all(len(c) <= 300 * MAX_SIZE_FACTOR for c in chunks)


def test_chunks_break_on_sentence_boundaries():
    for c in chunk_text(make_text(40), 300, 60, 50):
        assert c.startswith("This is sentence")
        assert c.endswith("padding words.")


def test_overlap_repeats_trailing_sentences():
    chunks = chunk_text(make_text(40), chunk_size=300, overlap=150, min_size=50)
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(a)[-1]
        if last_sentence in b:
            shared += 1
    assert shared >= len(chunks) - 2  # essentially every boundary carries overlap


def test_zero_overlap_has_no_repetition():
    chunks = chunk_text(make_text(40), chunk_size=300, overlap=0, min_size=50)
    seen = [s for c in chunks for s in split_sentences(c)]
    assert len(seen) == len(set(seen)) == 40


def test_no_content_is_lost():
    text = make_text(60, para_every=5)
    chunks = chunk_text(text, 300, 60, 50)
    joined = " ".join(chunks)
    for i in range(60):
        assert SENT.format(i=i) in joined


def test_paragraph_breaks_preserved_inside_chunk():
    text = "Alpha one. Alpha two.\n\nBeta one. Beta two."
    (chunk,) = chunk_text(text, 200, 20, 10)
    assert chunk == "Alpha one. Alpha two.\n\nBeta one. Beta two."


def test_oversized_sentence_is_split_on_words():
    long_sentence = " ".join(f"word{i}" for i in range(400)) + "."
    chunks = chunk_text(long_sentence, 200, 20, 20)
    assert len(chunks) > 5
    assert all(len(c) <= 250 for c in chunks)


def test_token_longer_than_chunk_is_hard_sliced():
    chunks = chunk_text("x" * 1000, 100, 10, 10)
    assert all(len(c) <= 125 for c in chunks)
    assert "".join(chunks).count("x") >= 1000


def test_tiny_final_chunk_is_merged_into_previous():
    body = make_text(10)
    chunks = chunk_text(body + " Tail.", chunk_size=300, overlap=50, min_size=100)
    assert all(len(c) >= 20 for c in chunks)
    assert chunks[-1].endswith("Tail.")


def test_small_chunk_grows_rather_than_closing():
    text = "Tiny one. " + ("B" * 280) + "."
    chunks = chunk_text(text, chunk_size=300, overlap=20, min_size=50)
    assert len(chunks) == 1  # 10 + 281 fits within 1.25 x 300


def test_determinism():
    t = make_text(50, para_every=7)
    assert chunk_text(t, 300, 60, 50) == chunk_text(t, 300, 60, 50)


@pytest.mark.parametrize(
    "size,overlap,min_size",
    [(10, 0, 0), (200, 200, 10), (200, -1, 10), (200, 20, 200)],
)
def test_invalid_parameters(size, overlap, min_size):
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap, min_size)


def test_chunk_pages_metadata_and_indices():
    pages = [PageText(1, make_text(15)), PageText(2, make_text(15)), PageText(5, "Short page.")]
    chunks = chunk_pages(pages, "doc123", "file.pdf", 300, 60, 50)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == "doc123" and c.filename == "file.pdf" for c in chunks)
    assert chunks[0].chunk_id == "doc123:0"
    assert {c.page for c in chunks} == {1, 2, 5}
    pages_in_order = [c.page for c in chunks]
    assert pages_in_order == sorted(pages_in_order)  # never crosses a page
    assert all(len(c.text_hash) == 64 for c in chunks)


def test_chunk_pages_skips_non_alphanumeric_and_roundtrips_dict():
    chunks = chunk_pages([PageText(None, "— — —"), PageText(None, "Real text here.")], "d", "f.txt")
    assert len(chunks) == 1
    from src.chunker import Chunk

    assert Chunk.from_dict(chunks[0].to_dict()) == chunks[0]
