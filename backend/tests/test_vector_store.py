import numpy as np
import pytest

from src.vector_store import (
    NumpyVectorStore,
    VectorStoreError,
    cosine_similarity,
    cosine_similarity_matrix,
)


def rec(cid, doc="d1", text="t"):
    return {"chunk_id": cid, "document_id": doc, "filename": "f.txt", "chunk_index": 0, "text": text}


# ---------------------------------------------------------------- cosine math
def test_cosine_known_values():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([1, 1], [1, 0]) == pytest.approx(2 ** -0.5)


def test_cosine_is_scale_invariant_and_zero_safe():
    assert cosine_similarity([2, 4], [1, 2]) == pytest.approx(1.0)
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_matrix_version_matches_scalar():
    rng = np.random.default_rng(0)
    m, q = rng.normal(size=(20, 8)).astype(np.float32), rng.normal(size=8).astype(np.float32)
    expected = [cosine_similarity(r, q) for r in m]
    assert np.allclose(cosine_similarity_matrix(m, q), expected, atol=1e-5)
    assert cosine_similarity_matrix(np.zeros((0, 8), np.float32), q).shape == (0,)


# ------------------------------------------------------------------ store ops
@pytest.fixture
def store(tmp_path):
    return NumpyVectorStore(tmp_path / "vs", dimension=3, model_name="test")


def test_add_and_search_ranking(store):
    store.add([rec("a"), rec("b"), rec("c")], np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]]))
    hits = store.search(np.array([1, 0, 0]), top_k=2)
    assert [h.record["chunk_id"] for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0) and hits[0].score > hits[1].score


def test_search_works_with_unnormalised_vectors(store):
    store.add([rec("a"), rec("b")], np.array([[100, 0, 0], [0, 0.001, 0]]))
    hits = store.search(np.array([5, 0, 0]), top_k=2)
    assert hits[0].record["chunk_id"] == "a" and hits[0].score == pytest.approx(1.0)


def test_threshold_filters(store):
    store.add([rec("a"), rec("b")], np.array([[1, 0, 0], [0, 1, 0]]))
    hits = store.search(np.array([1, 0, 0]), top_k=5, threshold=0.5)
    assert [h.record["chunk_id"] for h in hits] == ["a"]


def test_document_filter(store):
    store.add([rec("a", "d1"), rec("b", "d2")], np.array([[1, 0, 0], [1, 0, 0]]))
    assert [h.record["chunk_id"] for h in store.search(np.array([1, 0, 0]), 5, document_ids=["d2"])] == ["b"]
    assert store.search(np.array([1, 0, 0]), 5, document_ids=[]) == []


def test_top_k_larger_than_store_and_empty_store(store):
    assert store.search(np.array([1, 0, 0]), 5) == []
    store.add([rec("a")], np.array([[1, 0, 0]]))
    assert len(store.search(np.array([1, 0, 0]), 50)) == 1
    assert store.search(np.array([1, 0, 0]), 0) == []


def test_duplicate_chunk_id_rejected_and_replace(store):
    store.add([rec("a", text="old")], np.array([[1, 0, 0]]))
    with pytest.raises(VectorStoreError):
        store.add([rec("a")], np.array([[0, 1, 0]]))
    with pytest.raises(VectorStoreError):
        store.add([rec("x"), rec("x")], np.array([[1, 0, 0], [1, 0, 0]]))
    store.add([rec("a", text="new")], np.array([[0, 1, 0]]), replace=True)
    assert store.count() == 1 and store.records()[0]["text"] == "new"


def test_dimension_and_length_validation(store):
    with pytest.raises(VectorStoreError):
        store.add([rec("a")], np.array([[1, 0]]))
    with pytest.raises(VectorStoreError):
        store.add([rec("a"), rec("b")], np.array([[1, 0, 0]]))
    with pytest.raises(VectorStoreError):
        store.search(np.array([1, 0]), 1)
    with pytest.raises(VectorStoreError):
        store.add([{"document_id": "d"}], np.array([[1, 0, 0]]))


def test_delete_document(store):
    store.add([rec("a", "d1"), rec("b", "d1"), rec("c", "d2")], np.eye(3))
    assert store.delete_document("d1") == 2
    assert store.count() == 1 and store.document_ids() == {"d2"}
    assert store.search(np.array([0, 0, 1]), 3)[0].record["chunk_id"] == "c"
    assert store.delete_document("missing") == 0


def test_persistence_roundtrip(tmp_path):
    s1 = NumpyVectorStore(tmp_path / "vs", 3, "test")
    s1.add([rec("a"), rec("b", "d2")], np.array([[1, 0, 0], [0, 1, 0]]))
    s2 = NumpyVectorStore(tmp_path / "vs", 3, "test")
    assert s2.count() == 2 and s2.document_ids() == {"d1", "d2"}
    assert s2.search(np.array([0, 1, 0]), 1)[0].record["chunk_id"] == "b"
    s2.delete_document("d2")
    assert NumpyVectorStore(tmp_path / "vs", 3, "test").count() == 1


def test_persistence_after_empty_state(tmp_path):
    s = NumpyVectorStore(tmp_path / "vs", 3, "test")
    s.add([rec("a")], np.array([[1, 0, 0]]))
    s.delete_document("d1")
    assert NumpyVectorStore(tmp_path / "vs", 3, "test").count() == 0


def test_dimension_mismatch_on_load_demands_reindex(tmp_path):
    NumpyVectorStore(tmp_path / "vs", 3, "old").add([rec("a")], np.array([[1, 0, 0]]))
    with pytest.raises(VectorStoreError, match="Re-index"):
        NumpyVectorStore(tmp_path / "vs", 4, "new")


def test_corrupted_index_raises_clear_error(tmp_path):
    s = NumpyVectorStore(tmp_path / "vs", 3, "t")
    s.add([rec("a")], np.array([[1, 0, 0]]))
    (tmp_path / "vs" / "records.json").write_text("{broken")
    with pytest.raises(VectorStoreError, match="corrupted"):
        NumpyVectorStore(tmp_path / "vs", 3, "t")


def test_no_temp_files_left_after_save(store):
    store.add([rec("a")], np.array([[1, 0, 0]]))
    assert not list(store.directory.glob("*.tmp"))


def test_rebuild_removes_nonfinite_rows(store):
    store.add([rec("a"), rec("b")], np.array([[1, 0, 0], [0, 1, 0]]))
    store._vectors[1, 0] = np.nan
    out = store.rebuild()
    assert out == {"before": 2, "after": 1, "removed": 1}
    assert store.count() == 1


def test_stats_and_clear(store):
    store.add([rec("a", "d1"), rec("b", "d2")], np.eye(3)[:2])
    st = store.stats()
    assert st["chunks"] == 2 and st["documents"] == 2 and st["dimension"] == 3
    assert st["disk_bytes"] > 0 and st["memory_bytes"] == 2 * 3 * 4
    store.clear()
    assert store.count() == 0
