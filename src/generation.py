import time

import requests


class OpenRouterError(Exception):
    """Raised when an OpenRouter request cannot return a usable answer."""


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when OpenRouter or an upstream provider returns HTTP 429."""


def query_openrouter(
    api_key,
    model,
    prompt,
    max_tokens=512,
    timeout=30,
    retries=2,
    retry_delay=1,
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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
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

    raise OpenRouterError(f"OpenRouter request failed after retries: {last_error}") from last_error


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
