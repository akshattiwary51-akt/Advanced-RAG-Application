import math

import numpy as np
import pytest

from src.advanced_retrieval import (
    AdvancedRetriever,
    CrossEncoderReranker,
    LexicalReranker,
    build_reranker,
    compress_text,
    expand_query_prf,
    mmr_select,
    rewrite_query,
    rrf_fuse,
)
from src.bm25 import BM25Index, keyword_tokens
from src.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder
from src.llm import MockLLM
from src.vector_store import NumpyVectorStore
from config import get_settings


def rec(cid, doc, text):
    return {"chunk_id": cid, "document_id": doc, "filename": f"{doc}.txt", "chunk_index": 0, "text": text}


# ------------------------------------------------------------------------ RRF
def test_rrf_formula_and_consensus():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63) and fused["d"] == pytest.approx(1 / 63)
    assert min(fused["a"], fused["b"]) > max(fused["c"], fused["d"])  # agreement beats single votes


def test_rrf_ignores_score_scale_and_handles_empty():
    assert rrf_fuse([]) == {} and rrf_fuse([[], []]) == {}
    assert rrf_fuse([["x"]])["x"] == pytest.approx(1 / 61)


# ------------------------------------------------------------------------ MMR
def test_mmr_lambda_one_is_pure_relevance():
    vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert mmr_select(np.array([1.0, 0.99, 0.5]), vecs, 3, lam=1.0) == [0, 1, 2]


def test_mmr_prefers_diverse_item_over_near_duplicate():
    vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert mmr_select(np.array([1.0, 0.99, 0.5]), vecs, 2, lam=0.5) == [0, 2]


def test_mmr_edge_cases():
    assert mmr_select(np.array([]), np.zeros((0, 2)), 3) == []
    assert mmr_select(np.array([1.0]), np.array([[1.0, 0]]), 0) == []
    assert len(mmr_select(np.array([1.0, 0.5]), np.eye(2), 10)) == 2
    with pytest.raises(ValueError):
        mmr_select(np.array([1.0]), np.eye(1), 1, lam=1.5)


# ----------------------------------------------------------------------- BM25
@pytest.fixture
def corpus():
    return [
        rec("a:0", "a", "The turbine model XJ9000 requires annual inspection of the rotor blades."),
        rec("b:0", "b", "Annual reports describe revenue growth and market trends across regions."),
        rec("c:0", "c", "Rotor " + "filler words about nothing in particular " * 30),
    ]


def test_bm25_rare_term_ranks_first_and_scores_positive(corpus):
    idx = BM25Index().build(corpus)
    hits = idx.search("xj9000 turbine", top_k=3)
    assert hits[0].record["chunk_id"] == "a:0" and hits[0].score > 0 and hits[0].matched_terms == 2


def test_bm25_idf_matches_formula(corpus):
    idx = BM25Index().build(corpus)
    n, df = 3, 2  # "annual" appears in a and b
    assert idx.idf("annual") == pytest.approx(math.log(1 + (n - df + 0.5) / (df + 0.5)))
    assert idx.idf("xj9000") > idx.idf("annual")


def test_bm25_length_normalisation(corpus):
    idx = BM25Index().build(corpus)
    ids = [h.record["chunk_id"] for h in idx.search("rotor")]
    assert ids.index("a:0") < ids.index("c:0")  # same term, much shorter doc wins


def test_bm25_filters_stopwords_and_unknown_terms(corpus):
    idx = BM25Index().build(corpus)
    assert idx.search("zzzunknown") == []
    assert keyword_tokens("what is the") == ["what", "is", "the"]  # all-stopword query kept, not emptied
    assert "the" not in keyword_tokens("the turbine")


def test_bm25_document_filter_and_empty_index(corpus):
    idx = BM25Index().build(corpus)
    assert {h.record["document_id"] for h in idx.search("annual", document_ids=["b"])} == {"b"}
    assert BM25Index().build([]).search("anything") == []


# ------------------------------------------------------------- query expansion
def test_prf_adds_characteristic_new_terms_only():
    idx = BM25Index().build([rec("1", "d", "alpha beta"), rec("2", "d", "gamma delta")])
    out = expand_query_prf("solar panels", ["photovoltaic cells convert sunlight; solar panels use photovoltaic silicon"],
                           idx.idf, n_terms=2)
    assert out.startswith("solar panels ") and "photovoltaic" in out
    assert out.count("solar") == 1  # existing query words are not repeated
    assert expand_query_prf("x", [], idx.idf) == "x"


# ------------------------------------------------------------- query rewriting
class ScriptedLLM(MockLLM):
    def __init__(self, reply=None, boom=False):
        self.reply, self.boom, self.seen = reply, boom, None

    def chat(self, messages, params):
        self.seen = messages
        if self.boom:
            raise RuntimeError("down")
        return self.reply


HIST = [{"role": "user", "content": "Tell me about the XJ9000 turbine"},
        {"role": "assistant", "content": "It is a wind turbine."}]


def test_rewrite_uses_history_and_returns_clean_line():
    llm = ScriptedLLM('"How often does the XJ9000 turbine need inspection?"\nextra line')
    out = rewrite_query(llm, "How often does it need inspection?", HIST)
    assert out == "How often does the XJ9000 turbine need inspection?"
    assert "XJ9000" in llm.seen[-1]["content"] and llm.seen[0]["role"] == "system"


def test_rewrite_fallbacks():
    q = "and its cost?"
    assert rewrite_query(ScriptedLLM("x"), q, []) == q  # no history -> no call needed
    assert rewrite_query(ScriptedLLM(boom=True), q, HIST) == q  # LLM failure
    assert rewrite_query(ScriptedLLM(""), q, HIST) == q  # empty
    assert rewrite_query(ScriptedLLM("The answer is 5 [1]"), q, HIST) == q  # answered instead of rewriting
    assert rewrite_query(ScriptedLLM("x" * 500), q, HIST) == q  # runaway output


# ---------------------------------------------------------------- compression
@pytest.fixture
def svc():
    return EmbeddingService(MockEmbedder(384), EmbeddingCache("mock"))


def test_compress_keeps_relevant_sentences_in_order(svc):
    text = ("Bananas are yellow fruit. Solar panels convert sunlight into electricity. "
            "The weather was mild yesterday. Solar panels reach efficiency of twenty percent.")
    q = svc.embed_query("solar panels efficiency")
    out = compress_text(text, q, svc)
    assert "Solar panels convert" in out and "reach efficiency of twenty" in out
    assert "Bananas" not in out and "weather" not in out
    assert out.index("convert") < out.index("efficiency")


def test_compress_short_chunks_unchanged(svc):
    q = svc.embed_query("anything")
    assert compress_text("One. Two.", q, svc) == "One. Two."


# ------------------------------------------------------------------ rerankers
def test_lexical_reranker_prefers_full_coverage_and_phrase():
    r = LexicalReranker()
    s = r.score("solar panel efficiency", ["panel of solar cells", "solar panel efficiency is high", "bread"])
    assert s[1] > s[0] > s[2]
    assert r.score("x", []) == []


def test_cross_encoder_wrapper_pairs_query_with_each_text():
    class Fake:
        def predict(self, pairs, show_progress_bar=False):
            self.pairs = pairs
            return np.array([0.1, 0.9])

    fake = Fake()
    ce = CrossEncoderReranker(model=fake)
    assert ce.score("q", ["a", "b"]) == [pytest.approx(0.1), pytest.approx(0.9)]
    assert fake.pairs == [("q", "a"), ("q", "b")] and ce.score("q", []) == []


def test_build_reranker_factory():
    s = get_settings()
    assert isinstance(build_reranker(s.with_overrides(reranker_backend="lexical")), LexicalReranker)
    assert isinstance(build_reranker(s.with_overrides(reranker_backend="cross-encoder")), CrossEncoderReranker)
    with pytest.raises(ValueError):
        build_reranker(s.with_overrides(reranker_backend="nope"))


# --------------------------------------------------------- AdvancedRetriever
TEXTS = {
    "turbine": "The turbine model XJ9000 requires annual inspection of the rotor blades.",
    "reports": "Annual reports describe revenue growth and market trends across regions.",
    "solar1": "Solar panels convert sunlight into electricity using photovoltaic cells.",
    "solar2": "Solar panels convert sunlight into electricity using photovoltaic cells and silicon.",
    "bread": "Sourdough bread needs flour, water and salt, then bake until golden.",
}


@pytest.fixture
def adv(tmp_path, svc):
    store = NumpyVectorStore(tmp_path / "vs", 384, "mock")
    for name, text in TEXTS.items():
        store.add([rec(f"{name}:0", name, text)], svc.embed_texts([text]))
    return AdvancedRetriever(svc, store)


def ids(chunks):
    return [c.document_id for c in chunks]


def test_semantic_method_matches_baseline_order(adv):
    got = adv.retrieve("solar panel electricity", 3, 0.0, method="semantic")
    base = adv.base.retrieve("solar panel electricity", 3, 0.0)
    assert ids(got) == ids(base)
    assert got[0].scores["semantic"] == pytest.approx(got[0].score)


def test_bm25_method_finds_exact_identifier(adv):
    got = adv.retrieve("XJ9000", 3, 0.0, method="bm25")
    assert ids(got) == ["turbine"] and got[0].scores["bm25"] > 0
    assert got[0].score == pytest.approx(got[0].scores["semantic"])  # `score` is always cosine


def test_hybrid_adds_rrf_and_both_signals(adv):
    got = adv.retrieve("annual inspection turbine rotor", 3, 0.0, method="hybrid")
    assert got[0].document_id == "turbine"
    assert {"semantic", "bm25", "rrf"} <= got[0].scores.keys()


def test_hybrid_surfaces_lexical_hit_below_semantic_threshold(adv):
    sem = adv.retrieve("annual inspection rotor blades", 5, 0.95, method="semantic")
    hyb = adv.retrieve("annual inspection rotor blades", 5, 0.95, method="hybrid")
    assert sem == [] and "turbine" in ids(hyb)  # >=2 matched terms clears the lexical gate


def test_hybrid_single_common_word_does_not_bypass_threshold(adv):
    assert adv.retrieve("annual quantum chromodynamics", 5, 0.95, method="hybrid") == []


def test_bm25_index_refreshes_after_store_changes(adv, svc):
    assert ids(adv.retrieve("sourdough", 3, 0.0, method="bm25")) == ["bread"]
    adv.store.delete_document("bread")
    assert adv.retrieve("sourdough", 3, 0.0, method="bm25") == []
    t = "Sourdough starter maintenance guide."
    adv.store.add([rec("new:0", "new", t)], svc.embed_texts([t]))
    assert ids(adv.retrieve("sourdough", 3, 0.0, method="bm25")) == ["new"]


def test_document_filter_applies_to_every_method(adv):
    for m in ("semantic", "bm25", "hybrid"):
        got = adv.retrieve("annual solar", 5, 0.0, document_ids=["reports"], method=m)
        assert set(ids(got)) <= {"reports"}, m
    assert adv.retrieve("solar", document_ids=[]) == []


def test_mmr_replaces_near_duplicate_with_diverse_chunk(adv):
    plain = ids(adv.retrieve("solar panels photovoltaic electricity", 2, 0.0))
    div = ids(adv.retrieve("solar panels photovoltaic electricity", 2, 0.0, use_mmr=True, mmr_lambda=0.3))
    assert set(plain) == {"solar1", "solar2"}
    assert len(set(div) & {"solar1", "solar2"}) == 1  # only one of the twins survives


def test_rerank_reorders_and_records_score(adv):
    adv.reranker = type("R", (), {"name": "fake",
                                  "score": lambda self, q, texts: [float(t.startswith("Sourdough")) for t in texts]})()
    got = adv.retrieve("solar panels", 3, -1.0, rerank=True)
    assert got[0].document_id == "bread" and got[0].scores["rerank"] == 1.0


def test_compress_shortens_text_and_keeps_original(adv, svc):
    long = ("Bananas are yellow fruit. Solar panels convert sunlight into electricity. "
            "The weather was mild yesterday. Solar panel efficiency is about twenty percent.")
    adv.store.add([rec("long:0", "long", long)], svc.embed_texts([long]))
    got = adv.retrieve("solar panel efficiency", 1, 0.0, compress=True)
    assert got[0].document_id == "long" and got[0].original_text == long
    assert "Bananas" not in got[0].text and got[0].to_dict()["compressed"] is True


def test_query_expansion_runs_and_returns_results(adv):
    assert adv.retrieve("solar panels", 3, 0.0, expand=True, method="hybrid")


def test_invalid_method_and_empty_store(adv, tmp_path, svc):
    with pytest.raises(ValueError):
        adv.retrieve("x", method="magic")
    empty = AdvancedRetriever(svc, NumpyVectorStore(tmp_path / "e", 384, "m"))
    assert empty.retrieve("anything") == []


def test_ranks_are_sequential_and_scores_descending_for_semantic(adv):
    got = adv.retrieve("solar panels", 4, -1.0)
    assert [c.rank for c in got] == list(range(1, len(got) + 1))
    assert [c.score for c in got] == sorted((c.score for c in got), reverse=True)
