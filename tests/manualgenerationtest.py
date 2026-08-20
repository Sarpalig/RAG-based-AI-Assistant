import os
import sys
from pathlib import Path

from dotenv import load_dotenv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import OpenRouterError, query_openrouter


load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")

print("Manual OpenRouter testi basliyor.")
print(f"Model: {model or 'YOK'}")
print(f"API key: {'var' if api_key else 'YOK'}")

prompt = input("\nModele gonderecegin kisa prompt: ").strip()

try:
    print("\nOpenRouter istegi gonderiliyor...")
    answer = query_openrouter(
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_tokens=200,
        retries=0,
    )
    print("\nTEST OK - OpenRouter cevap verdi.")
    print("\nCevap:\n")
    print(answer)
except OpenRouterError as exc:
    print("\nTEST FAILED - OpenRouter cevabi alinamadi.")
    print(f"Hata: {exc}")
