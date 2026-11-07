import numpy as np
import pytest

from eval.metrics import (
    answer_relevance,
    best_threshold,
    citation_correctness,
    context_relevance,
    faithfulness,
    hit_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relative_change,
    summarize_latency,
    tune_threshold,
)


# ---------------------------------------------------------- hand-computed values
def test_precision_at_k():
    rel = [True, False, True, False, False]
    assert precision_at_k(rel, 1) == 1.0
    assert precision_at_k(rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(rel, 5) == pytest.approx(0.4)
    assert precision_at_k([True], 5) == pytest.approx(0.2)  # short result list is penalised
    with pytest.raises(ValueError):
        precision_at_k(rel, 0)


def test_recall_at_k():
    rel = [True, False, True]
    assert recall_at_k(rel, 3, total_relevant=4) == pytest.approx(0.5)
    assert recall_at_k(rel, 1, total_relevant=2) == pytest.approx(0.5)
    assert recall_at_k([], 5, 2) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(rel, 3, 0)


def test_hit_rate_and_mrr():
    assert hit_at_k([False, False, True], 2) == 0.0 and hit_at_k([False, False, True], 3) == 1.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True]) == 1.0 and reciprocal_rank([False, False]) == 0.0 and reciprocal_rank([]) == 0.0


def test_latency_summary():
    s = summarize_latency([1, 2, 3, 4, 100])
    assert s["mean"] == pytest.approx(22.0) and s["p50"] == 3 and s["max"] == 100
    assert s["p95"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))
    assert summarize_latency([])["mean"] == 0.0 and percentile([], 50) == 0.0


def test_relative_change():
    assert relative_change(1.5, 1.0) == pytest.approx(50.0)
    assert relative_change(0.5, 1.0) == pytest.approx(-50.0)
    assert relative_change(1.0, 0.0) is None


# ------------------------------------------------------------ threshold tuning
def make_queries():
    return [
        {"answerable": True, "scored": [(0.8, True), (0.3, False)]},
        {"answerable": True, "scored": [(0.4, True)]},
        {"answerable": False, "scored": [(0.2, False)]},
        {"answerable": False, "scored": [(0.35, False)]},
    ]


def test_tune_threshold_rows_hand_checked():
    rows = {r["threshold"]: r for r in tune_threshold(make_queries(), [0.0, 0.3, 0.5, 0.9])}
    assert rows[0.0]["answer_rate"] == 1.0 and rows[0.0]["reject_rate"] == 0.0
    assert rows[0.3]["answer_rate"] == 1.0 and rows[0.3]["reject_rate"] == 0.5
    assert rows[0.5]["answer_rate"] == 0.5 and rows[0.5]["reject_rate"] == 1.0
    assert rows[0.9]["answer_rate"] == 0.0 and rows[0.9]["reject_rate"] == 1.0
    assert rows[0.3]["balanced_accuracy"] == pytest.approx(0.75)


def test_best_threshold_prefers_balance_then_lowest():
    rows = tune_threshold(make_queries(), [0.0, 0.3, 0.4, 0.5, 0.9])
    best = best_threshold(rows)
    assert best["threshold"] == 0.4 and best["balanced_accuracy"] == pytest.approx(1.0)  # 0.4 answers both, rejects both
    tie = best_threshold([{"threshold": 0.1, "balanced_accuracy": 0.8}, {"threshold": 0.5, "balanced_accuracy": 0.8}])
    assert tie["threshold"] == 0.1


def test_tune_threshold_handles_one_sided_sets():
    only_ans = tune_threshold([{"answerable": True, "scored": [(0.5, True)]}], [0.0, 0.9])
    assert [r["balanced_accuracy"] for r in only_ans] == [1.0, 0.0]
    assert tune_threshold([], [0.5])[0]["balanced_accuracy"] == 0.0


# ----------------------------------------------------------------- generation
CTX = ["Solar panels reach an efficiency of roughly twenty percent. Inverters last about twelve years."]


def test_faithfulness_grounded_vs_invented():
    assert faithfulness("Panels reach an efficiency of roughly twenty percent [1].", CTX) == 1.0
    assert faithfulness("Quantum entanglement powers the grid using magic.", CTX) == 0.0
    mixed = faithfulness("Inverters last about twelve years. Bananas orbit Jupiter nightly.", CTX)
    assert mixed == pytest.approx(0.5)
    assert faithfulness("", CTX) == 0.0


def test_citation_correctness_detects_invalid_and_unsupported():
    srcs = {1: CTX[0], 2: "Sourdough needs flour water and salt."}
    good = citation_correctness("Panels reach roughly twenty percent efficiency [1].", srcs)
    assert good == {"citations": 1, "valid_rate": 1.0, "precision": 1.0, "coverage": 1.0}
    wrong_source = citation_correctness("Panels reach roughly twenty percent efficiency [2].", srcs)
    assert wrong_source["valid_rate"] == 1.0 and wrong_source["precision"] == 0.0
    ghost = citation_correctness("Panels reach roughly twenty percent efficiency [9].", srcs)
    assert ghost["valid_rate"] == 0.0 and ghost["precision"] == 0.0
    none = citation_correctness("Panels are nice. They are good.", srcs)
    assert none["citations"] == 0 and none["coverage"] == 0.0 and none["precision"] == 0.0


def embed_fake(texts):
    table = {"q": [1, 0], "same": [1, 0], "ortho": [0, 1], "half": [1, 1]}
    return np.array([table[t] for t in texts], dtype=np.float32)


def test_answer_and_context_relevance():
    assert answer_relevance("q", "same [1]", embed_fake) == pytest.approx(1.0)
    assert answer_relevance("q", "ortho", embed_fake) == pytest.approx(0.0)
    assert context_relevance("q", ["same", "ortho"], embed_fake) == pytest.approx(0.5)
    assert context_relevance("q", [], embed_fake) == 0.0
