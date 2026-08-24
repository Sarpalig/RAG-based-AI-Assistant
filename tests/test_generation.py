import pytest
import requests

from src.generation import (
    LLMError,
    OllamaError,
    OpenRouterError,
    query_llm,
    query_ollama,
    query_openrouter,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self.payload = payload or {
            "choices": [{"message": {"content": "Merhaba"}}]
        }
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self.payload


def test_query_openrouter_returns_message_content(monkeypatch):
    def fake_post(url, headers, json, timeout):
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "test-model"
        assert json["messages"][0]["content"] == "Selam"
        assert timeout == 30
        return FakeResponse(payload={"choices": [{"message": {"content": "Cevap"}}]})

    monkeypatch.setattr("src.generation.requests.post", fake_post)

    answer = query_openrouter("test-key", "test-model", "Selam")

    assert answer == "Cevap"


def test_query_openrouter_requires_api_key():
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        query_openrouter("", "test-model", "Selam")


def test_query_openrouter_raises_readable_http_error(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse(status_code=401, text="invalid key")

    monkeypatch.setattr("src.generation.requests.post", fake_post)

    with pytest.raises(OpenRouterError, match="OpenRouter HTTP 401"):
        query_openrouter("bad-key", "test-model", "Selam")


def test_query_openrouter_does_not_retry_rate_limit(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(url)
        return FakeResponse(
            status_code=429,
            text="rate limit",
            headers={"Retry-After": "60"},
        )

    monkeypatch.setattr("src.generation.requests.post", fake_post)
    monkeypatch.setattr("src.generation.time.sleep", lambda delay: None)

    with pytest.raises(OpenRouterError, match="rate limit"):
        query_openrouter(
            "test-key",
            "test-model",
            "Selam",
            retries=2,
            retry_delay=0,
        )

    assert len(calls) == 1


def test_query_openrouter_retries_server_error(monkeypatch):
    responses = [
        FakeResponse(status_code=503, text="temporary error"),
        FakeResponse(payload={"choices": [{"message": {"content": "Son cevap"}}]}),
    ]

    def fake_post(url, headers, json, timeout):
        return responses.pop(0)

    monkeypatch.setattr("src.generation.requests.post", fake_post)
    monkeypatch.setattr("src.generation.time.sleep", lambda delay: None)

    answer = query_openrouter(
        "test-key",
        "test-model",
        "Selam",
        retries=1,
        retry_delay=0,
    )

    assert answer == "Son cevap"


def test_query_ollama_returns_message_content(monkeypatch):
    def fake_post(url, json, timeout):
        assert url == "http://localhost:11434/api/chat"
        assert json["model"] == "qwen3.5:9b"
        assert json["messages"][0]["content"] == "Selam"
        assert json["stream"] is False
        assert json["options"]["temperature"] == 0.2
        assert json["options"]["num_predict"] == 256
        assert timeout == 120
        return FakeResponse(payload={"message": {"content": "Yerel cevap"}})

    monkeypatch.setattr("src.generation.requests.post", fake_post)

    answer = query_ollama(
        "http://localhost:11434",
        "qwen3.5:9b",
        "Selam",
        max_tokens=256,
    )

    assert answer == "Yerel cevap"


def test_query_ollama_requires_model():
    with pytest.raises(OllamaError, match="OLLAMA_MODEL"):
        query_ollama("http://localhost:11434", "", "Selam")


def test_query_llm_routes_to_ollama_by_default(monkeypatch):
    def fake_query_ollama(base_url, model, prompt, max_tokens=512):
        assert base_url == "http://localhost:11434"
        assert model == "qwen3.5:9b"
        assert prompt == "Selam"
        assert max_tokens == 512
        return "Ollama cevap"

    monkeypatch.setattr("src.generation.query_ollama", fake_query_ollama)

    answer = query_llm(provider=None, prompt="Selam")

    assert answer == "Ollama cevap"


def test_query_llm_falls_back_when_openrouter_primary_is_rate_limited(monkeypatch):
    seen_models = []

    def fake_query_openrouter(
        api_key,
        model,
        prompt,
        max_tokens=512,
        retries=2,
        retry_delay=1,
    ):
        seen_models.append((model, retries))
        if model == "primary-model":
            from src.generation import OpenRouterRateLimitError

            raise OpenRouterRateLimitError("primary rate limited")
        return "Fallback cevap"

    monkeypatch.setattr("src.generation.query_openrouter", fake_query_openrouter)

    answer = query_llm(
        provider="openrouter",
        prompt="Selam",
        openrouter_api_key="test-key",
        openrouter_model="primary-model",
        openrouter_fallback_model="openrouter/free",
    )

    assert answer == "Fallback cevap"
    assert seen_models == [("primary-model", 2), ("openrouter/free", 0)]


def test_query_llm_rejects_unknown_provider():
    with pytest.raises(LLMError, match="Unsupported LLM_PROVIDER"):
        query_llm(provider="unknown", prompt="Selam")
