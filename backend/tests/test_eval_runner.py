import json

import pytest

from eval.run_eval import (
    CONFIGS, count_relevant, is_relevant, load_dataset, main, render_markdown, run_evaluation,
)
from src.advanced_retrieval import LexicalReranker
from src.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder
from src.llm import MockLLM


@pytest.fixture(scope="module")
def report():
    svc = EmbeddingService(MockEmbedder(), EmbeddingCache("mock"))
    return run_evaluation(svc, LexicalReranker(), MockLLM(), repeats=1, include_generation=True)


def test_dataset_is_consistent():
    docs, qs = load_dataset()
    assert len(docs) == 6 and len(qs) == 36
    for q in qs:
        if q["document"]:
            text = " ".join(docs[q["document"]].read_text().lower().split())
            assert q["must_contain"].lower() in text, q
    assert sum(q["document"] is None for q in qs) == 6


def test_is_relevant_requires_document_and_phrase():
    q = {"document": "solar", "must_contain": "Twenty  Percent"}
    assert is_relevant("solar", "efficiency of twenty percent.", q)
    assert not is_relevant("bread", "efficiency of twenty percent.", q)
    assert not is_relevant("solar", "something else", q)
    assert not is_relevant("solar", "twenty percent", {"document": None, "must_contain": None})


def test_report_structure_and_all_configs(report):
    assert set(report["retrieval"]) == {n for n, _ in CONFIGS}
    for r in report["retrieval"].values():
        for key in ("precision@5", "recall@5", "hit@5", "mrr", "latency_ms"):
            assert key in r
        assert 0.0 <= r["mrr"] <= 1.0 and r["latency_ms"]["mean"] > 0
    assert report["meta"]["answerable"] == 30
    json.dumps(report)  # fully serialisable


def test_sanity_of_measured_values(report):
    r = report["retrieval"]
    assert r["bm25 (keyword only)"]["hit@5"] > 0.8  # answer phrases are literal text
    assert all(x["hit@5"] >= x["hit@1"] for x in r.values())
    assert report["ingestion_warm"]["cache_hits"] == report["meta"]["indexed_chunks"]
    assert report["threshold_tuning"]["best"]["balanced_accuracy"] >= 0.5
    assert 0 < report["compression"]["size_reduction_pct"] < 100


def test_generation_path_runs(report):
    g = report["generation"]["baseline: semantic"]
    assert g["questions"] == 30 and 0.0 <= g["faithfulness"] <= 1.0 and "citation_precision" in g


def test_markdown_flags_mock_and_has_tables(report):
    md = render_markdown(report)
    assert "IMPORTANT - mock embeddings" in md and "| baseline: semantic |" in md
    assert "must not be reused with MiniLM" in md


def test_cli_writes_reports(tmp_path):
    assert main(["--repeats", "1", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "eval_mock_lexical.json").exists() and (tmp_path / "eval_mock_lexical.md").exists()
