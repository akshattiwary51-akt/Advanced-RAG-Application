import io
import json

import pytest

from app import create_app
from config import get_settings
from src.llm import LLMUnavailableError, MockLLM
from tests.helpers import make_docx, make_pdf

SOLAR = (b"Solar panels convert sunlight into electricity. Panel efficiency is about 20 percent. "
         b"Cells are made of silicon.\n\nInverters change DC power into AC power for the grid.")
BREAD = (b"Sourdough bread needs flour, water and salt. The starter ferments the dough for hours. "
         b"Bake the loaf until the crust is golden.")


@pytest.fixture
def app(tmp_path):
    s = get_settings().with_overrides(
        data_dir=tmp_path / "data", database_url="sqlite:///:memory:",
        embedding_backend="mock", llm_backend="mock", similarity_threshold=0.05,
        chunk_size=300, chunk_overlap=50, min_chunk_size=50, max_upload_mb=1)
    return create_app(s)


@pytest.fixture
def client(app):
    return app.test_client()


def up(client, *files):
    data = {"files": [(io.BytesIO(c), n, m) if m else (io.BytesIO(c), n) for n, c, m in files]}
    return client.post("/api/documents/upload", data=data, content_type="multipart/form-data")


def sse_events(resp):
    return [json.loads(l[6:]) for l in resp.get_data(as_text=True).split("\n\n") if l.startswith("data: ")]


# ---------------------------------------------------------------- system
def test_health_and_stats(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok" and j["llm"]["status"] == "ok" and j["embedding"]["dimension"] == 384
    assert client.get("/api/stats").get_json()["documents"]["total_documents"] == 0


def test_settings_endpoint_hides_paths(client):
    j = client.get("/api/settings").get_json()
    assert j["chunk_size"] == 300 and "data_dir" not in j and "ollama_base_url" not in j


def test_swagger_spec_lists_endpoints(client):
    spec = client.get("/apispec_1.json").get_json()
    for path in ("/api/health", "/api/documents/upload", "/api/chat", "/api/chat/stream",
                 "/api/debug/retrieval", "/api/documents/{document_id}/reindex",
                 "/api/conversations/{conversation_id}"):
        assert path in spec["paths"], path
    assert client.get("/apidocs/").status_code == 200


def test_cors_allows_configured_origin_only(client):
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    bad = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert "Access-Control-Allow-Origin" not in bad.headers


# ---------------------------------------------------------------- upload
def test_upload_multiple_files_and_list(client):
    r = up(client, ("solar.txt", SOLAR, "text/plain"), ("bread.txt", BREAD, "text/plain"))
    assert r.status_code == 201
    res = r.get_json()["results"]
    assert [x["status"] for x in res] == ["indexed", "indexed"]
    doc = res[0]["document"]
    assert doc["status"] == "indexed" and doc["chunk_count"] >= 1 and "stored_name" not in doc
    docs = client.get("/api/documents").get_json()["documents"]
    assert len(docs) == 2
    assert client.get(f"/api/documents/{doc['id']}").get_json()["filename"] == "solar.txt"
    assert client.get("/api/stats").get_json()["documents"]["indexed_documents"] == 2


def test_upload_pdf_and_docx(client, tmp_path):
    dx = tmp_path / "t.docx"
    make_docx(dx, ["Quarterly revenue grew twelve percent."])
    r = up(client, ("r.pdf", make_pdf(["Revenue figures for the year"]), "application/pdf"),
           ("t.docx", dx.read_bytes(), None))
    assert [x["status"] for x in r.get_json()["results"]] == ["indexed", "indexed"]
    assert r.get_json()["results"][0]["document"]["page_count"] == 1


def test_duplicate_upload_returns_409_with_existing(client):
    up(client, ("a.txt", SOLAR, "text/plain"))
    r = up(client, ("renamed.txt", SOLAR, "text/plain"))
    assert r.status_code == 409
    j = r.get_json()
    assert j["error"]["code"] == "duplicate_document"
    assert j["results"][0]["status"] == "duplicate"
    assert len(client.get("/api/documents").get_json()["documents"]) == 1


def test_mixed_batch_reports_each_file(client):
    r = up(client, ("good.txt", SOLAR, "text/plain"), ("bad.exe", b"MZ..", "application/octet-stream"),
           ("empty.txt", b"   ", "text/plain"))
    assert r.status_code == 201
    st = {x["filename"]: x["status"] for x in r.get_json()["results"]}
    assert st == {"good.txt": "indexed", "bad.exe": "error", "empty.txt": "error"}


@pytest.mark.parametrize("name,content,mime,status,code", [
    ("x.exe", b"MZ", "application/octet-stream", 415, "unsupported_file_type"),
    ("x.pdf", b"not a pdf", "application/pdf", 422, "corrupted_document"),
    ("x.txt", b"   \n ", "text/plain", 422, "empty_document"),
    ("x.txt", b"\x00\x01\x02bin", "text/plain", 422, "corrupted_document"),
    ("x.pdf", b"%PDF-1.4 hi", "text/html", 415, "unsupported_file_type"),
])
def test_upload_rejections(client, name, content, mime, status, code):
    r = up(client, (name, content, mime))
    assert r.status_code == status and r.get_json()["error"]["code"] == code
    assert client.get("/api/documents").get_json()["documents"] == []


def test_upload_too_large(client):
    r = up(client, ("big.txt", b"a " * (700 * 1024), "text/plain"))  # ~1.4 MB > 1 MB limit
    assert r.status_code == 413


def test_upload_no_files(client):
    r = client.post("/api/documents/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "no_files"


def test_path_traversal_filename_is_neutralised(client, app):
    r = up(client, ("../../../etc/evil.txt", SOLAR, "text/plain"))
    assert r.status_code == 201
    assert r.get_json()["results"][0]["document"]["filename"] == "evil.txt"
    files = [p.name for p in app.extensions["rag"].settings.uploads_dir.iterdir()]
    assert len(files) == 1 and files[0].endswith(".txt") and "evil" not in files[0]
    assert not (app.extensions["rag"].settings.data_dir.parent / "etc").exists()


def test_failed_upload_leaves_no_files_or_vectors(client, app):
    up(client, ("x.pdf", b"%PDF-1.4 garbage", "application/pdf"))
    svc = app.extensions["rag"]
    assert list(svc.settings.uploads_dir.iterdir()) == [] and svc.store.count() == 0


# ------------------------------------------------- delete / reindex / consistency
def test_delete_removes_metadata_file_and_vectors(client, app):
    doc = up(client, ("a.txt", SOLAR, "text/plain")).get_json()["results"][0]["document"]
    svc = app.extensions["rag"]
    assert svc.store.count(doc["id"]) > 0
    r = client.delete(f"/api/documents/{doc['id']}")
    assert r.status_code == 200 and r.get_json()["chunks_removed"] > 0
    assert svc.store.count() == 0 and list(svc.settings.uploads_dir.iterdir()) == []
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404
    # deleted content can be uploaded again (hash no longer blocks it)
    assert up(client, ("a.txt", SOLAR, "text/plain")).status_code == 201


def test_reindex_with_new_chunk_size(client, app):
    doc = up(client, ("a.txt", SOLAR, "text/plain")).get_json()["results"][0]["document"]
    r = client.post(f"/api/documents/{doc['id']}/reindex", json={"chunk_size": 100, "chunk_overlap": 20})
    assert r.status_code == 200
    out = r.get_json()
    assert out["chunk_count"] > doc["chunk_count"] and out["status"] == "indexed"
    assert app.extensions["rag"].store.count(doc["id"]) == out["chunk_count"]


def test_reindex_validation_and_404(client):
    assert client.post("/api/documents/nope/reindex").status_code == 404
    doc = up(client, ("a.txt", SOLAR, "text/plain")).get_json()["results"][0]["document"]
    assert client.post(f"/api/documents/{doc['id']}/reindex", json={"chunk_size": "x"}).status_code == 400
    assert client.post(f"/api/documents/{doc['id']}/reindex", json={"chunk_size": 10}).status_code == 400


def test_reindex_missing_file_marks_failed(client, app):
    doc = up(client, ("a.txt", SOLAR, "text/plain")).get_json()["results"][0]["document"]
    svc = app.extensions["rag"]
    for f in svc.settings.uploads_dir.iterdir():
        f.unlink()
    assert client.post(f"/api/documents/{doc['id']}/reindex").status_code == 422
    assert client.get(f"/api/documents/{doc['id']}").get_json()["status"] == "failed"
    assert svc.store.count(doc["id"]) == 0


def test_reconcile_repairs_drift(client, app):
    import numpy as np
    svc = app.extensions["rag"]
    d1 = up(client, ("a.txt", SOLAR, "text/plain")).get_json()["results"][0]["document"]
    svc.store.add([{"chunk_id": "ghost:0", "document_id": "ghost", "text": "x"}],
                  np.ones((1, svc.embeddings.dimension)))
    svc.store.delete_document(d1["id"])
    (svc.settings.uploads_dir / "stray.txt").write_text("orphan")
    report = svc.documents.reconcile()
    assert report == {"orphan_vector_docs": 1, "marked_failed": 1, "orphan_files": 1}
    assert client.get(f"/api/documents/{d1['id']}").get_json()["status"] == "failed"


# ------------------------------------------------------------------ chat
@pytest.fixture
def loaded(client):
    up(client, ("solar.txt", SOLAR, "text/plain"), ("bread.txt", BREAD, "text/plain"))
    return client


def test_chat_returns_cited_answer_and_persists(loaded):
    r = loaded.post("/api/chat", json={"message": "How efficient are solar panels?"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["grounded"] and "[1]" in j["answer"] and j["citations"][0]["filename"] == "solar.txt"
    assert j["citations"][0]["cited_in_answer"] and "retrieval" in j["timings_ms"]
    conv = loaded.get(f"/api/conversations/{j['conversation_id']}").get_json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["citations"][0]["filename"] == "solar.txt"
    assert conv["title"].startswith("How efficient")


def test_chat_follow_up_reuses_conversation(loaded):
    cid = loaded.post("/api/chat", json={"message": "solar panels"}).get_json()["conversation_id"]
    r = loaded.post("/api/chat", json={"message": "sourdough bread", "conversation_id": cid})
    assert r.get_json()["conversation_id"] == cid
    assert len(loaded.get(f"/api/conversations/{cid}").get_json()["messages"]) == 4
    assert [c["id"] for c in loaded.get("/api/conversations").get_json()["conversations"]] == [cid]


def test_chat_document_filter(loaded):
    docs = {d["filename"]: d["id"] for d in loaded.get("/api/documents").get_json()["documents"]}
    j = loaded.post("/api/chat", json={"message": "sourdough bread flour",
                                       "document_ids": [docs["bread.txt"]]}).get_json()
    assert {c["filename"] for c in j["citations"]} == {"bread.txt"}


def test_chat_no_relevant_context_is_ungrounded(loaded):
    j = loaded.post("/api/chat", json={"message": "quantum chromodynamics lattice",
                                       "similarity_threshold": 0.6}).get_json()
    assert j["grounded"] is False and j["citations"] == [] and "couldn't find" in j["answer"]


@pytest.mark.parametrize("body", [
    None, [], {}, {"message": ""}, {"message": 5}, {"message": "x" * 5000},
    {"message": "q", "top_k": 0}, {"message": "q", "top_k": 999}, {"message": "q", "top_k": "3"},
    {"message": "q", "temperature": 5}, {"message": "q", "similarity_threshold": 2},
    {"message": "q", "method": "magic"}, {"message": "q", "document_ids": "abc"},
    {"message": "q", "max_tokens": 0}, {"message": "q", "model": ""},
])
def test_chat_validation_400(loaded, body):
    r = loaded.post("/api/chat", json=body) if body is not None else loaded.post("/api/chat", data="notjson")
    assert r.status_code == 400 and "error" in r.get_json()


def test_chat_unknown_conversation_404(loaded):
    r = loaded.post("/api/chat", json={"message": "hi", "conversation_id": "nope"})
    assert r.status_code == 404


def test_chat_llm_unavailable_maps_to_503_without_persisting(loaded, app):
    class Down(MockLLM):
        def chat(self, *a, **k):
            raise LLMUnavailableError("Cannot reach the Ollama server.")
    app.extensions["rag"].pipeline.llm = Down()
    r = loaded.post("/api/chat", json={"message": "solar panels"})
    assert r.status_code == 503 and r.get_json()["error"]["code"] == "llm_unavailable"


def test_chat_stream_sse(loaded):
    r = loaded.post("/api/chat/stream", json={"message": "How efficient are solar panels?"})
    assert r.status_code == 200 and r.mimetype == "text/event-stream"
    ev = sse_events(r)
    types = [e["type"] for e in ev]
    assert types[0] == "meta" and types[1] == "sources" and types[-1] == "done"
    assert "token" in types
    done = ev[-1]
    assert "".join(e["content"] for e in ev if e["type"] == "token").strip() == done["answer"]
    assert done["citations"][0]["filename"] == "solar.txt"
    conv = loaded.get(f"/api/conversations/{done['conversation_id']}").get_json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]


def test_chat_stream_validation_happens_before_stream(loaded):
    assert loaded.post("/api/chat/stream", json={"message": ""}).status_code == 400


def test_chat_stream_llm_error_event(loaded, app):
    class Down(MockLLM):
        def stream(self, *a, **k):
            raise LLMUnavailableError("Cannot reach the Ollama server.")
            yield
    app.extensions["rag"].pipeline.llm = Down()
    ev = sse_events(loaded.post("/api/chat/stream", json={"message": "solar panels"}))
    assert ev[-1]["type"] == "error" and "Ollama" in ev[-1]["message"]


def test_delete_conversation(loaded):
    cid = loaded.post("/api/chat", json={"message": "solar"}).get_json()["conversation_id"]
    assert loaded.delete(f"/api/conversations/{cid}").status_code == 200
    assert loaded.get(f"/api/conversations/{cid}").status_code == 404


# ----------------------------------------------------------------- debug
def test_debug_retrieval(loaded):
    j = loaded.get("/api/debug/retrieval", query_string={"q": "solar panel efficiency", "top_k": 2}).get_json()
    assert j["results"][0]["filename"] == "solar.txt" and "text" in j["results"][0]
    assert j["latency_ms"] >= 0 and j["total_chunks_indexed"] > 0
    assert j["results"][0]["score"] >= j["results"][-1]["score"]


def test_debug_retrieval_validation(loaded):
    assert loaded.get("/api/debug/retrieval").status_code == 400
    assert loaded.get("/api/debug/retrieval?q=x&top_k=abc").status_code == 400


# ----------------------------------------------------------------- errors
def test_unknown_route_json_404_and_method_405(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["error"]["code"] == "not_found"
    assert client.put("/api/health").status_code == 405


def test_no_server_paths_leak_in_any_response(loaded, app):
    root = str(app.extensions["rag"].settings.data_dir)
    bodies = [loaded.get(p).get_data(as_text=True) for p in
              ("/api/health", "/api/stats", "/api/settings", "/api/documents")]
    bodies.append(up(loaded, ("x.pdf", b"%PDF garbage", "application/pdf")).get_data(as_text=True))
    assert all(root not in b and "/home/" not in b and "stored_name" not in b for b in bodies)


def test_unexpected_exception_is_generic_500(client, app):
    def boom():
        raise RuntimeError("secret /etc/passwd details")
    app.extensions["rag"].documents.list = boom
    r = client.get("/api/documents")
    assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)
    assert r.get_json()["error"]["code"] == "internal_error"
