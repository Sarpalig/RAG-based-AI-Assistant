import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RagPipeline


sample_dir = PROJECT_ROOT / "data" / "sample_documents"
chroma_dir = PROJECT_ROOT / ".tmp" / "manual_retrieval_chroma"

rag = RagPipeline(
    chroma_dir=str(chroma_dir),
    collection_name="manual_retrieval_test",
)

files = [
    open(path, "rb")
    for path in sorted(sample_dir.iterdir())
    if path.suffix.lower() in {".pdf", ".docx", ".md"}
]

try:
    print("Belgeler indeksleniyor...")
    index_results = rag.index_files(files)
finally:
    for file in files:
        file.close()

for result in index_results:
    status = "atlandı" if result["skipped"] else "eklendi"
    print(f"- {result['file_name']}: {status}, chunk={result['chunk_count']}")

question = input("\nSorunu yaz: ").strip()
results = rag.search(question, top_k=5)

print("\nRetrieval sonuçları:\n")
for index, result in enumerate(results, start=1):
    preview = result["text"].replace("\n", " ")[:300]

    print(f"{index}. {result['file_name']}")
    print(f"   distance: {result['distance']}")
    if result.get("page_number"):
        print(f"   page: {result['page_number']}")
    if result.get("paragraph_number"):
        print(f"   paragraph: {result['paragraph_number']}")
    print(f"   text: {preview}")
    print()
