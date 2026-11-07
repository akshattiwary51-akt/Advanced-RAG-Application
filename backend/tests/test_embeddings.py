import numpy as np
import pytest

from src.embeddings import (
    EmbeddingCache,
    EmbeddingService,
    MockEmbedder,
    build_embedding_service,
    l2_normalize,
)
from config import get_settings


def test_l2_normalize_rows_and_zero_vector():
    m = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(m[0], [0.6, 0.8])
    assert np.all(np.isfinite(m)) and np.allclose(m[1], 0)
    assert np.allclose(np.linalg.norm(l2_normalize(np.array([1.0, 1.0]))), 1.0)


def test_mock_embedder_shape_norm_determinism():
    e = MockEmbedder(64)
    a = e.embed(["the cat sat", "dogs bark"])
    assert a.shape == (2, 64) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, MockEmbedder(64).embed(["the cat sat", "dogs bark"]))
    assert e.embed([]).shape == (0, 64)


def test_mock_similarity_reflects_shared_words():
    e = MockEmbedder(384)
    q, near, far = e.embed(["solar panel efficiency", "efficiency of solar panel cells", "banana bread recipe"])
    assert q @ near > q @ far


class CountingEmbedder(MockEmbedder):
    def __init__(self):
        super().__init__(32)
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)


def test_service_batches_and_dedupes_within_call():
    emb = CountingEmbedder()
    svc = EmbeddingService(emb, EmbeddingCache("m"), batch_size=2)
    out = svc.embed_texts(["a x", "b y", "a x", "c z", "d w"])
    assert out.shape == (5, 32)
    assert np.array_equal(out[0], out[2])
    assert [len(c) for c in emb.calls] == [2, 2]  # 4 unique texts in batches of 2


def test_cache_prevents_recomputation():
    emb = CountingEmbedder()
    svc = EmbeddingService(emb, EmbeddingCache("m"), batch_size=8)
    svc.embed_texts(["hello world", "foo bar"])
    svc.embed_texts(["hello world", "foo bar", "new text"])
    assert emb.calls[1] == ["new text"]
    assert svc.cache.hits == 2


def test_cache_persists_to_disk_and_is_model_namespaced(tmp_path):
    c1 = EmbeddingCache("model/a", tmp_path)
    c1.put("hi there", np.ones(4, dtype=np.float32))
    c1.save()
    assert c1.__len__() == 1
    c2 = EmbeddingCache("model/a", tmp_path)
    assert np.array_equal(c2.get("hi there"), np.ones(4))
    assert EmbeddingCache("model/b", tmp_path).get("hi there") is None


def test_cache_survives_corruption(tmp_path):
    c = EmbeddingCache("m", tmp_path)
    c.put("x y", np.ones(3, dtype=np.float32))
    c.save()
    (tmp_path / "emb_cache_m.json").write_text("not json")
    assert len(EmbeddingCache("m", tmp_path)) == 0


def test_cache_eviction_and_clear(tmp_path):
    c = EmbeddingCache("m", tmp_path, max_items=2)
    for t in ("a1", "b2", "c3"):
        c.put(t, np.ones(2, dtype=np.float32))
    assert len(c) == 2 and c.get("a1") is None
    c.save()
    c.clear()
    assert len(c) == 0 and not list(tmp_path.glob("emb_cache_*"))


def test_factory_mock_and_unknown(tmp_path):
    s = get_settings().with_overrides(embedding_backend="mock", data_dir=tmp_path / "d")
    svc = build_embedding_service(s, tmp_path)
    assert svc.dimension == 384 and svc.model_name == "mock-hash-bow"
    with pytest.raises(ValueError):
        build_embedding_service(s.with_overrides(embedding_backend="nope"), tmp_path)
