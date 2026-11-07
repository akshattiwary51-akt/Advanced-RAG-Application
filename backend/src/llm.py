"""LLM access: Ollama client (streaming, timeouts, clear errors) and an offline mock.

Generation is kept strictly separate from retrieval: this module only turns a list of
chat messages into text.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterator, Optional, Protocol

import requests

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Generic generation failure (maps to HTTP 502)."""


class LLMUnavailableError(LLMError):
    """Server unreachable, timed out, or model not installed (maps to HTTP 503)."""


@dataclass
class GenerationParams:
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 0.9
    model: Optional[str] = None  # per-request override

    def validate(self) -> "GenerationParams":
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2.")
        if not 1 <= self.max_tokens <= 8192:
            raise ValueError("max_tokens must be between 1 and 8192.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        return self


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], params: GenerationParams) -> str: ...
    def stream(self, messages: list[dict], params: GenerationParams) -> Iterator[str]: ...
    def health(self) -> dict: ...


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120, connect_timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.connect_timeout = connect_timeout

    def _payload(self, messages, params: GenerationParams, stream: bool) -> dict:
        return {
            "model": params.model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": params.temperature,
                "num_predict": params.max_tokens,
                "top_p": params.top_p,
            },
        }

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
                timeout=(self.connect_timeout, self.timeout),
            )
        except requests.ConnectionError as exc:
            raise LLMUnavailableError(
                "Cannot reach the Ollama server. Is it running (`ollama serve`)?"
            ) from exc
        except requests.Timeout as exc:
            raise LLMUnavailableError("The LLM server timed out.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{payload['model']}' is not installed. Run `ollama pull {payload['model']}`."
            )
        if resp.status_code >= 400:
            raise LLMError(f"LLM server returned HTTP {resp.status_code}.")
        return resp

    def chat(self, messages: list[dict], params: GenerationParams) -> str:
        resp = self._post(self._payload(messages, params, stream=False), stream=False)
        try:
            return resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Unexpected response format from the LLM server.") from exc

    def stream(self, messages: list[dict], params: GenerationParams) -> Iterator[str]:
        resp = self._post(self._payload(messages, params, stream=True), stream=True)
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError as exc:
                    raise LLMError("Malformed streaming response.") from exc
                if "error" in data:
                    raise LLMError(str(data["error"])[:200])
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break
        except requests.RequestException as exc:
            raise LLMUnavailableError("Connection to the LLM was lost mid-response.") from exc
        finally:
            resp.close()

    def health(self) -> dict:
        """Reachability and whether the configured model is installed."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=(self.connect_timeout, 5))
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return {"status": "unavailable", "model": self.model, "model_installed": False}
        installed = any(n == self.model or n.split(":")[0] == self.model.split(":")[0]
                        and (":" not in self.model) for n in names)
        return {
            "status": "ok" if installed else "model_missing",
            "model": self.model,
            "model_installed": installed,
            "available_models": names[:20],
        }


class MockLLM:
    """Deterministic offline LLM: quotes the first context block and cites it."""

    model = "mock-llm"

    def _answer(self, messages: list[dict]) -> str:
        user = messages[-1]["content"]
        if messages[0]["content"].startswith("Rewrite the user's latest question"):
            return user.rsplit("Latest question:", 1)[-1].strip()  # identity rewrite
        m = re.search(r'<source id="(\d+)"[^>]*>\s*(.*?)\s*</source>', user, re.S)
        if not m:
            return "I could not find enough information in the provided documents to answer."
        first_sentence = re.split(r"(?<=[.!?])\s", m.group(2).strip())[0]
        return f"According to the documents: {first_sentence} [{m.group(1)}]"

    def chat(self, messages: list[dict], params: GenerationParams) -> str:
        return self._answer(messages)

    def stream(self, messages: list[dict], params: GenerationParams) -> Iterator[str]:
        for token in re.findall(r"\S+\s*", self._answer(messages)):
            yield token

    def health(self) -> dict:
        return {"status": "ok", "model": self.model, "model_installed": True}


def build_llm(settings) -> LLM:
    backend = settings.llm_backend.lower()
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        return OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds)
    raise ValueError(f"Unknown LLM_BACKEND '{settings.llm_backend}'.")
