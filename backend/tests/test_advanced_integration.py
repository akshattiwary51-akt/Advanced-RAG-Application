import io

import pytest

from app import create_app
from config import get_settings
from src.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder
from src.llm import MockLLM
from src.rag_pipeline import RAGPipeline, RetrievalOptions
from src.vector_store import NumpyVectorStore

TURBINE = b"The turbine model XJ9000 requires annual inspection of the rotor blades. Inspect twice if offshore."
SOLAR = b"Solar panels convert sunlight into electricity. Panel efficiency is about twenty percent."


@pytest.fixture
def pipe(tmp_path):
    svc = EmbeddingService(MockEmbedder(384), EmbeddingCache("mock"))
    store = NumpyVectorStore(tmp_path / "vs", 384, svc.model_name)
    p = RAGPipeline(svc, store, MockLLM(), chunk_size=300, chunk_overlap=50, min_chunk_size=50)
    for name, data in (("turbine", TURBINE), ("solar", SOLAR)):
        f = tmp_path / f"{name}.txt"
        f.write_bytes(data)
        p.process_document(f, name, f.name)
    return p


def test_options_baseline_flag():
    assert RetrievalOptions().is_baseline
    assert not RetrievalOptions(method="hybrid").is_baseline
    assert not RetrievalOptions(rerank=True).is_baseline
    assert RetrievalOptions(rewrite_query=True).is_baseline  # rewriting alone keeps baseline retrieval


@pytest.mark.parametrize("opts", [
    RetrievalOptions(method="bm25", threshold=0.0),
    RetrievalOptions(method="hybrid", threshold=0.0),
    RetrievalOptions(method="hybrid", threshold=0.0, rerank=True, use_mmr=True, compress=True, expand_query=True),
])
def test_every_method_answers_with_citations(pipe, opts):
    res = pipe.answer_query("XJ9000 annual inspection rotor", opts=opts)
    assert res.grounded and res.citations[0].filename == "turbine.txt" and "[1]" in res.answer


def test_bm25_finds_identifier_baseline_threshold_would_miss(pipe):
    base = pipe.answer_query("XJ9000", opts=RetrievalOptions(threshold=0.9))
    kw = pipe.answer_query("XJ9000", opts=RetrievalOptions(method="bm25", threshold=0.9))
    assert not base.grounded and kw.grounded


def test_rewrite_uses_history_for_retrieval_but_not_for_prompt(pipe):
    class Rewriter(MockLLM):
        def chat(self, messages, params):
            if messages[0]["content"].startswith("Rewrite"):
                return "How often does the XJ9000 turbine need inspection?"
            return super().chat(messages, params)

    pipe.llm = Rewriter()
    history = [{"role": "user", "content": "Tell me about the XJ9000 turbine"},
               {"role": "assistant", "content": "It is a turbine."}]
    res = pipe.answer_query("How often does it need that?", history,
                            RetrievalOptions(threshold=0.0, rewrite_query=True, method="bm25"))
    assert res.rewritten_query.startswith("How often does the XJ9000")
    assert res.citations[0].filename == "turbine.txt"
    assert "rewrite" in res.timings_ms and res.to_dict()["rewritten_query"] == res.rewritten_query


def test_rewrite_skipped_without_history(pipe):
    res = pipe.answer_query("XJ9000", [], RetrievalOptions(threshold=0.0, rewrite_query=True))
    assert res.rewritten_query is None and "rewrite" not in res.timings_ms


def test_stream_carries_rewritten_query(pipe):
    ev = list(pipe.stream_answer("annual inspection rotor", [{"role": "user", "content": "hi"}],
                                 RetrievalOptions(threshold=0.0, rewrite_query=True, method="hybrid")))
    assert ev[-1]["type"] == "done" and ev[-1]["result"].grounded


# ----------------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path):
    s = get_settings().with_overrides(
        data_dir=tmp_path / "data", database_url="sqlite:///:memory:", embedding_backend="mock",
        llm_backend="mock", similarity_threshold=0.05, chunk_size=300, chunk_overlap=50, min_chunk_size=50)
    c = create_app(s).test_client()
    for n, d in (("turbine.txt", TURBINE), ("solar.txt", SOLAR)):
        assert c.post("/api/documents/upload", data={"files": (io.BytesIO(d), n, "text/plain")},
                      content_type="multipart/form-data").status_code == 201
    return c


@pytest.mark.parametrize("method", ["semantic", "bm25", "hybrid"])
def test_api_chat_accepts_all_methods(client, method):
    r = client.post("/api/chat", json={"message": "XJ9000 annual inspection", "method": method,
                                       "similarity_threshold": 0.0})
    assert r.status_code == 200 and r.get_json()["citations"][0]["filename"] == "turbine.txt"


def test_api_chat_with_all_advanced_flags(client):
    r = client.post("/api/chat", json={"message": "annual inspection of rotor blades", "method": "hybrid",
                                       "use_mmr": True, "mmr_lambda": 0.5, "rerank": True,
                                       "expand_query": True, "compress": True, "rewrite_query": True,
                                       "similarity_threshold": 0.0})
    j = r.get_json()
    assert r.status_code == 200 and j["grounded"] and "rewritten_query" in j


@pytest.mark.parametrize("body", [
    {"use_mmr": "yes"}, {"rerank": 1}, {"compress": None}, {"mmr_lambda": 2}, {"mmr_lambda": -0.1},
    {"method": "dense"},
])
def test_api_rejects_bad_advanced_params(client, body):
    assert client.post("/api/chat", json={"message": "q", **body}).status_code == 400


def test_debug_endpoint_supports_method_and_flags(client):
    r = client.get("/api/debug/retrieval", query_string={
        "q": "XJ9000", "method": "bm25", "similarity_threshold": 0, "rerank": "true", "compress": "true"})
    j = r.get_json()
    assert r.status_code == 200 and j["method"] == "bm25"
    assert j["results"][0]["filename"] == "turbine.txt"
    assert {"semantic", "bm25", "rerank"} <= j["results"][0]["scores"].keys()
