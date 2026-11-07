import json

import pytest
import requests

from src import llm as llm_mod
from src.llm import GenerationParams, LLMError, LLMUnavailableError, MockLLM, OllamaClient, build_llm
from config import get_settings

MSGS = [{"role": "user", "content": "hi"}]


class FakeResp:
    def __init__(self, status=200, body=None, lines=None):
        self.status_code, self._body, self._lines = status, body, lines or []
        self.closed = False

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def iter_lines(self):
        yield from self._lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError()

    def close(self):
        self.closed = True


def patch_post(monkeypatch, resp=None, exc=None, capture=None):
    def fake(url, json=None, stream=False, timeout=None):
        if capture is not None:
            capture.update(url=url, json=json, timeout=timeout, stream=stream)
        if exc:
            raise exc
        return resp
    monkeypatch.setattr(llm_mod.requests, "post", fake)


def test_chat_success_and_payload(monkeypatch):
    cap = {}
    patch_post(monkeypatch, FakeResp(body={"message": {"content": "hello"}}), capture=cap)
    c = OllamaClient("http://x:1/", "m1", timeout=30)
    assert c.chat(MSGS, GenerationParams(temperature=0.5, max_tokens=50)) == "hello"
    assert cap["url"] == "http://x:1/api/chat" and cap["json"]["model"] == "m1"
    assert cap["json"]["options"] == {"temperature": 0.5, "num_predict": 50, "top_p": 0.9}
    assert cap["timeout"] == (5, 30) and cap["json"]["stream"] is False


def test_per_request_model_override(monkeypatch):
    cap = {}
    patch_post(monkeypatch, FakeResp(body={"message": {"content": "x"}}), capture=cap)
    OllamaClient("http://x", "m1").chat(MSGS, GenerationParams(model="other"))
    assert cap["json"]["model"] == "other"


@pytest.mark.parametrize("exc", [requests.ConnectionError(), requests.Timeout()])
def test_unreachable_and_timeout_are_unavailable(monkeypatch, exc):
    patch_post(monkeypatch, exc=exc)
    with pytest.raises(LLMUnavailableError):
        OllamaClient("http://x", "m").chat(MSGS, GenerationParams())


def test_missing_model_404(monkeypatch):
    patch_post(monkeypatch, FakeResp(status=404))
    with pytest.raises(LLMUnavailableError, match="ollama pull"):
        OllamaClient("http://x", "ghost").chat(MSGS, GenerationParams())


def test_server_error_and_bad_format(monkeypatch):
    patch_post(monkeypatch, FakeResp(status=500))
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").chat(MSGS, GenerationParams())
    patch_post(monkeypatch, FakeResp(body={"nope": 1}))
    with pytest.raises(LLMError, match="Unexpected"):
        OllamaClient("http://x", "m").chat(MSGS, GenerationParams())


def test_stream_yields_tokens_and_closes(monkeypatch):
    lines = [json.dumps({"message": {"content": t}, "done": False}).encode() for t in ("Hel", "lo")]
    lines += [b"", json.dumps({"message": {"content": ""}, "done": True}).encode()]
    resp = FakeResp(lines=lines)
    patch_post(monkeypatch, resp)
    assert list(OllamaClient("http://x", "m").stream(MSGS, GenerationParams())) == ["Hel", "lo"]
    assert resp.closed


def test_stream_error_line_and_malformed(monkeypatch):
    patch_post(monkeypatch, FakeResp(lines=[json.dumps({"error": "boom"}).encode()]))
    with pytest.raises(LLMError, match="boom"):
        list(OllamaClient("http://x", "m").stream(MSGS, GenerationParams()))
    patch_post(monkeypatch, FakeResp(lines=[b"{not json"]))
    with pytest.raises(LLMError, match="Malformed"):
        list(OllamaClient("http://x", "m").stream(MSGS, GenerationParams()))


def test_health_states(monkeypatch):
    def get_ok(url, timeout=None):
        return FakeResp(body={"models": [{"name": "llama3.2:latest"}]})
    monkeypatch.setattr(llm_mod.requests, "get", get_ok)
    assert OllamaClient("http://x", "llama3.2").health()["status"] == "ok"
    assert OllamaClient("http://x", "mistral").health()["status"] == "model_missing"

    def get_fail(url, timeout=None):
        raise requests.ConnectionError()
    monkeypatch.setattr(llm_mod.requests, "get", get_fail)
    assert OllamaClient("http://x", "m").health()["status"] == "unavailable"


@pytest.mark.parametrize("kw", [{"temperature": 3}, {"max_tokens": 0}, {"top_p": 0}])
def test_params_validation(kw):
    with pytest.raises(ValueError):
        GenerationParams(**kw).validate()
    assert GenerationParams().validate().temperature == 0.2


def test_mock_llm_cites_first_source_and_streams():
    user = '<sources>\n<source id="1" file="a.txt">\nCats purr. They sleep a lot.\n</source>\n</sources>\n\nQuestion: q'
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": user}]
    m = MockLLM()
    assert m.chat(msgs, GenerationParams()) == "According to the documents: Cats purr. [1]"
    assert "".join(m.stream(msgs, GenerationParams())) == m.chat(msgs, GenerationParams())
    assert "could not find" in m.chat([{"role": "user", "content": "no sources"}], GenerationParams())


def test_build_llm_factory():
    s = get_settings()
    assert isinstance(build_llm(s.with_overrides(llm_backend="mock")), MockLLM)
    assert isinstance(build_llm(s.with_overrides(llm_backend="ollama")), OllamaClient)
    with pytest.raises(ValueError):
        build_llm(s.with_overrides(llm_backend="x"))
