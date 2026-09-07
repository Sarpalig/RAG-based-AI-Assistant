import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROFILES = [
    ("Genel", None),
    ("Intern Backend Developer", {"role": "Backend Developer", "seniority": "Intern"}),
    ("Junior Backend Developer", {"role": "Backend Developer", "seniority": "Junior"}),
    ("Senior Backend Developer", {"role": "Backend Developer", "seniority": "Senior"}),
    (
        "Senior Project Manager",
        {"role": "Project Manager", "seniority": "Senior"},
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ayni soruyu farkli rol/kidem profilleriyle calistirip cevap farklarini "
            "terminalden gosterir."
        )
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Sorulacak soru. Verilmezse terminalden istenir.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="RAG aramasinda kullanilacak kaynak sayisi. Varsayilan: 5",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="LLM cagirmadan profillere gore uretilen promptlari gosterir.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Bu test icin cevap token limitini override eder.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["general", "intern", "junior", "senior", "pm"],
        default=["general", "intern", "junior", "senior", "pm"],
        help="Calistirilacak profil seti.",
    )
    parser.add_argument(
        "--index-samples",
        action="store_true",
        help="Uygulamadaki sample dokumanlari gecici bir Chroma indeksine ekler.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="Var olan Chroma dizini. Verilmezse .env CHROMA_DIR/default kullanilir.",
    )
    parser.add_argument(
        "--collection",
        default="rag_documents",
        help="Chroma collection adi. Varsayilan: rag_documents",
    )
    return parser.parse_args()


def selected_profiles(profile_keys):
    profile_map = {
        "general": DEFAULT_PROFILES[0],
        "intern": DEFAULT_PROFILES[1],
        "junior": DEFAULT_PROFILES[2],
        "senior": DEFAULT_PROFILES[3],
        "pm": DEFAULT_PROFILES[4],
    }
    return [profile_map[key] for key in profile_keys]


def print_sources(search_results):
    print("\nBulunan kaynaklar:")
    for index, result in enumerate(search_results, start=1):
        location = ""
        if result.get("page_number") is not None:
            location = f", sayfa {result['page_number']}"
        elif result.get("paragraph_number") is not None:
            location = f", paragraf {result['paragraph_number']}"
        print(f"{index}. {result.get('file_name')}{location}")


def print_separator(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def index_sample_documents(rag):
    sample_dir = PROJECT_ROOT / "data" / "sample_documents"
    files = [
        open(path, "rb")
        for path in sorted(sample_dir.iterdir())
        if path.suffix.lower() in {".pdf", ".docx", ".md"}
    ]

    try:
        print("\nSample dokumanlar indeksleniyor...")
        results = rag.index_files(files)
    finally:
        for file in files:
            file.close()

    for result in results:
        status = "zaten vardi" if result["skipped"] else "eklendi"
        print(f"- {result['file_name']}: {status}, chunk={result['chunk_count']}")


def main():
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    from src.generation import LLMError
    from src.rag_pipeline import RagPipeline

    question = args.question or input("Sorunu yaz: ").strip()
    if not question:
        raise SystemExit("Soru bos olamaz.")

    chroma_dir = args.chroma_dir
    if args.index_samples and chroma_dir is None:
        chroma_dir = str(PROJECT_ROOT / ".tmp" / "manual_profile_query_chroma")

    rag = RagPipeline(chroma_dir=chroma_dir, collection_name=args.collection)
    if args.max_tokens:
        rag.answer_max_tokens = args.max_tokens

    if args.index_samples:
        index_sample_documents(rag)

    print(f"\nSoru: {question}")
    print(f"Provider: {rag.llm_provider}")
    print(f"Model: {rag.ollama_model if rag.llm_provider == 'ollama' else rag.openrouter_model}")
    print(f"Top-k: {args.top_k}")
    print(f"Max tokens: {rag.answer_max_tokens}")

    retrieval_started = time.perf_counter()
    search_results = rag.search(question, top_k=args.top_k)
    retrieval_elapsed = time.perf_counter() - retrieval_started

    if not search_results:
        print("\nBu soru icin kaynak bulunamadi. Once belge indeksini kontrol et.")
        return

    print_sources(search_results)
    print(f"Retrieval suresi: {retrieval_elapsed:.2f}s")

    for label, profile in selected_profiles(args.profiles):
        print_separator(label)
        prompt = rag.build_rag_prompt(
            question,
            search_results,
            user_profile=profile,
            chat_history=None,
        )

        if args.prompt_only:
            print(prompt)
            continue

        started = time.perf_counter()
        try:
            answer = rag._query_llm(prompt)
        except LLMError as exc:
            print(f"LLM hatasi: {exc}")
            continue

        elapsed = time.perf_counter() - started
        answer = rag.normalize_citation_format(answer)
        answer = rag.filter_invalid_citations(answer, len(search_results))

        print(answer)
        print(f"\nSure: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
