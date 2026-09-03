import time

import requests


class LLMError(Exception):
    """Raised when an LLM provider cannot return a usable answer."""


class OpenRouterError(LLMError):
    """Raised when an OpenRouter request cannot return a usable answer."""


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when OpenRouter or an upstream provider returns HTTP 429."""


class OllamaError(LLMError):
    """Raised when Ollama cannot return a usable answer."""


def query_llm(
    provider,
    prompt,
    max_tokens=512,
    openrouter_api_key=None,
    openrouter_model=None,
    openrouter_fallback_model=None,
    ollama_base_url="http://localhost:11434",
    ollama_model="qwen3.5:9b",
    ollama_keep_alive="10m",
):
    provider = (provider or "ollama").lower()

    if provider == "ollama":
        return query_ollama(
            base_url=ollama_base_url,
            model=ollama_model,
            prompt=prompt,
            max_tokens=max_tokens,
            keep_alive=ollama_keep_alive,
        )

    if provider == "openrouter":
        try:
            return query_openrouter(
                api_key=openrouter_api_key,
                model=openrouter_model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except OpenRouterRateLimitError:
            if (
                not openrouter_fallback_model
                or openrouter_fallback_model == openrouter_model
            ):
                raise

            return query_openrouter(
                api_key=openrouter_api_key,
                model=openrouter_fallback_model,
                prompt=prompt,
                max_tokens=max_tokens,
                retries=0,
            )

    raise LLMError(f"Unsupported LLM_PROVIDER: {provider}")


def query_ollama(
    base_url,
    model,
    prompt,
    max_tokens=512,
    timeout=120,
    max_continuations=1,
    keep_alive="10m",
):
    if not base_url:
        raise OllamaError("OLLAMA_BASE_URL is missing.")
    if not model:
        raise OllamaError("OLLAMA_MODEL is missing.")
    if not prompt or not prompt.strip():
        raise OllamaError("Prompt cannot be empty.")

    url = f"{base_url.rstrip('/')}/api/chat"
    messages = [{"role": "user", "content": prompt}]
    answer_parts = []

    for _ in range(max_continuations + 1):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,
            },
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]
            answer_parts.append(content)
            if data.get("done_reason") != "length":
                return _join_answer_parts(answer_parts)

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Yanıt token sınırında yarıda kesildi. Önceki metni tekrar "
                        "etmeden sadece kaldığın yerden devam et ve yanıtı tamamla."
                    ),
                }
            )
        except requests.HTTPError as exc:
            raise OllamaError(_format_ollama_http_error(exc)) from exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise OllamaError(
                f"Ollama request failed. Is Ollama running at {base_url}? {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise OllamaError("Ollama returned an unexpected response format.") from exc

    return _join_answer_parts(answer_parts)


def query_openrouter(
    api_key,
    model,
    prompt,
    max_tokens=512,
    timeout=30,
    retries=2,
    retry_delay=1,
    max_continuations=1,
):
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is missing.")
    if not model:
        raise OpenRouterError("OPENROUTER_MODEL is missing.")
    if not prompt or not prompt.strip():
        raise OpenRouterError("Prompt cannot be empty.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "user", "content": prompt}]
    answer_parts = []

    for _ in range(max_continuations + 1):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        last_error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                answer_parts.append(content)
                if choice.get("finish_reason") != "length":
                    return _join_answer_parts(answer_parts)

                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Yanıt token sınırında yarıda kesildi. Önceki metni tekrar "
                            "etmeden sadece kaldığın yerden devam et ve yanıtı tamamla."
                        ),
                    }
                )
                break
            except requests.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code == 429:
                    raise OpenRouterRateLimitError(_format_rate_limit_error(exc)) from exc
                if status_code not in {500, 502, 503, 504}:
                    raise OpenRouterError(_format_http_error(exc)) from exc
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            except requests.RequestException as exc:
                raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise OpenRouterError("OpenRouter returned an unexpected response format.") from exc

            if attempt < retries:
                time.sleep(retry_delay)
        else:
            raise OpenRouterError(f"OpenRouter request failed after retries: {last_error}") from last_error

    return _join_answer_parts(answer_parts)


def _join_answer_parts(parts):
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _format_http_error(error):
    response = error.response
    if response is None:
        return f"OpenRouter HTTP error: {error}"

    response_text = response.text[:500] if response.text else ""
    return f"OpenRouter HTTP {response.status_code}: {response_text}"


def _format_rate_limit_error(error):
    response = error.response
    if response is None:
        return "OpenRouter rate limit reached. Please wait before asking again."

    retry_after = getattr(response, "headers", {}).get("Retry-After")
    wait_hint = f" Retry after {retry_after} seconds." if retry_after else ""
    response_text = response.text[:500] if response.text else ""
    details = f" Details: {response_text}" if response_text else ""
    return f"OpenRouter rate limit reached (HTTP 429).{wait_hint}{details}"


def _format_ollama_http_error(error):
    response = error.response
    if response is None:
        return f"Ollama HTTP error: {error}"

    response_text = response.text[:500] if response.text else ""
    return f"Ollama HTTP {response.status_code}: {response_text}"
