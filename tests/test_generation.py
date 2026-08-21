import requests
import pytest

from src.generation import OpenRouterError, query_openrouter


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
