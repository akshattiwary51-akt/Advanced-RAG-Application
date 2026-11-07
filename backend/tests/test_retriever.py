import numpy as np
import pytest

from src.chunker import chunk_pages
from src.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder
from src.loaders import PageText
from src.retriever import SemanticRetriever, deduplicate, RetrievedChunk
from src.vector_store import NumpyVectorStore

DOCS = {
    "solar": "Solar panels convert sunlight into electricity using photovoltaic cells. "
             "Panel efficiency depends on cell material and temperature.",
    "bread": "Sourdough bread needs flour, water, salt and a live starter. "
             "Bake the loaf in a hot oven until the crust is golden.",
    "space": "The Mars rover explores the red planet and collects rock samples. "
             "Rover wheels were redesigned for sandy terrain.",
}


@pytest.fixture
def retriever(tmp_path):
    svc = EmbeddingService(MockEmbedder(384), EmbeddingCache("mock"))
    store = NumpyVectorStore(tmp_path / "vs", svc.dimension, svc.model_name)
    for name, text in DOCS.items():
        chunks = chunk_pages([PageText(1, text)], name, f"{name}.txt", 400, 50, 20)
        store.add([c.to_dict() for c in chunks], svc.embed_texts([c.text for c in chunks]))
    return SemanticRetriever(svc, store)


def test_retrieves_most_relevant_document_first(retriever):
    hits = retriever.retrieve("How efficient are solar panels?", top_k=3, threshold=0.0)
    assert hits[0].document_id == "solar"
    assert hits[0].score >= hits[-1].score
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_result_has_metadata(retriever):
    h = retriever.retrieve("sourdough starter", top_k=1)[0]
    assert (h.filename, h.page, h.document_id) == ("bread.txt", 1, "bread")
    d = h.to_dict()
    assert {"score", "rank", "text", "chunk_id", "page"} <= d.keys()
    assert "text" not in h.to_dict(include_text=False)


def test_threshold_excludes_irrelevant(retriever):
    assert retriever.retrieve("quantum chromodynamics lattice", top_k=3, threshold=0.5) == []


def test_document_filter(retriever):
    hits = retriever.retrieve("solar panels", top_k=5, threshold=0.0, document_ids=["bread"])
    assert {h.document_id for h in hits} == {"bread"}
    assert retriever.retrieve("solar", document_ids=[]) == []


def test_top_k_limit(retriever):
    assert len(retriever.retrieve("the", top_k=1, threshold=-1)) <= 1


def test_empty_query_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_empty_store_returns_nothing(tmp_path):
    svc = EmbeddingService(MockEmbedder(384))
    r = SemanticRetriever(svc, NumpyVectorStore(tmp_path / "e", 384, "m"))
    assert r.retrieve("anything at all") == []


def test_deduplicate_exact_and_near():
    def c(i, text):
        return RetrievedChunk(f"c{i}", "d", "f", i, None, text, 1.0 - i / 10)

    hits = [c(0, "Same text here"), c(1, "same   TEXT here"), c(2, "Other words"), c(3, "Other words!")]
    assert [h.chunk_id for h in deduplicate(hits)] == ["c0", "c2", "c3"]
    vecs = np.array([[1, 0], [1, 0], [0, 1], [0.001, 1]], dtype=np.float32)
    assert [h.chunk_id for h in deduplicate(hits, vecs)] == ["c0", "c2"]


def test_retriever_dedupes_identical_chunks(tmp_path):
    svc = EmbeddingService(MockEmbedder(384))
    store = NumpyVectorStore(tmp_path / "vs", 384, "m")
    text = "Identical paragraph about hydrogen fuel cells."
    for doc in ("d1", "d2"):
        store.add([{"chunk_id": f"{doc}:0", "document_id": doc, "filename": f"{doc}.txt",
                    "chunk_index": 0, "text": text}], svc.embed_texts([text]))
    r = SemanticRetriever(svc, store)
    assert len(r.retrieve("hydrogen fuel cells", top_k=5, dedupe=True)) == 1
    assert len(r.retrieve("hydrogen fuel cells", top_k=5, dedupe=False)) == 2


def test_timed_retrieve_reports_latency(retriever):
    hits, ms = retriever.timed_retrieve("solar panels")
    assert hits and ms >= 0
