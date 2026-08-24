import os
import sys
from pathlib import Path

from dotenv import load_dotenv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import LLMError, query_llm


load_dotenv(PROJECT_ROOT / ".env")

provider = os.getenv("LLM_PROVIDER", "ollama")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_model = os.getenv("OPENROUTER_MODEL")
openrouter_fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

print("Manual LLM testi basliyor.")
print(f"Provider: {provider}")
if provider == "ollama":
    print(f"Ollama URL: {ollama_base_url}")
    print(f"Model: {ollama_model}")
else:
    print(f"Model: {openrouter_model or 'YOK'}")
    print(f"API key: {'var' if openrouter_api_key else 'YOK'}")

prompt = input("\nModele gonderecegin kisa prompt: ").strip()

try:
    print("\nLLM istegi gonderiliyor...")
    answer = query_llm(
        provider=provider,
        prompt=prompt,
        max_tokens=200,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        openrouter_fallback_model=openrouter_fallback_model,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
    )
    print("\nTEST OK - LLM cevap verdi.")
    print("\nCevap:\n")
    print(answer)
except LLMError as exc:
    print("\nTEST FAILED - LLM cevabi alinamadi.")
    print(f"Hata: {exc}")
