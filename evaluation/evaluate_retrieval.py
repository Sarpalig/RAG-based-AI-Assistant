import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RagPipeline


QUESTION_FILE = PROJECT_ROOT / "evaluation" / "test_questions.json"
SAMPLE_DOCUMENT_DIR = PROJECT_ROOT / "data" / "sample_documents"
EVALUATION_CHROMA_DIR = PROJECT_ROOT / ".tmp" / "retrieval_eval_chroma"


def load_questions(question_file=QUESTION_FILE):
    with open(question_file, "r", encoding="utf-8") as file:
        return json.load(file)


def open_sample_documents(sample_document_dir=SAMPLE_DOCUMENT_DIR):
    document_paths = sorted(
        path
        for path in sample_document_dir.iterdir()
        if path.suffix.lower() in {".pdf", ".docx", ".md"}
    )
    return [open(path, "rb") for path in document_paths]


def calculate_recall_at_k(questions, search_fn, top_k=5):
    total = 0
    correct = 0
    details = []

    for item in questions:
        if not item.get("answerable"):
            continue

        total += 1
        results = search_fn(item["question"], top_k=top_k)
        expected_file = item["expected_file"]
        found = any(result.get("file_name") == expected_file for result in results)

        if found:
            correct += 1

        details.append(
            {
                "question": item["question"],
                "expected_file": expected_file,
                "found": found,
                "returned_files": [result.get("file_name") for result in results],
            }
        )

    recall = correct / total if total else 0
    return {
        "top_k": top_k,
        "total": total,
        "correct": correct,
        "recall": recall,
        "details": details,
    }


def run_evaluation(top_k=5, chroma_dir=EVALUATION_CHROMA_DIR):
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    rag = RagPipeline(chroma_dir=str(chroma_dir), collection_name="retrieval_eval")
    sample_documents = open_sample_documents()

    try:
        rag.index_files(sample_documents)
    finally:
        for document in sample_documents:
            document.close()

    questions = load_questions()
    return calculate_recall_at_k(questions, rag.search, top_k=top_k)


def print_report(report):
    for item in report["details"]:
        status = "yes" if item["found"] else "no"
        print(f"Question: {item['question']}")
        print(f"Expected file: {item['expected_file']}")
        print(f"Returned files: {item['returned_files']}")
        print(f"Found in top {report['top_k']}: {status}")
        print()

    print(
        f"Recall@{report['top_k']}: "
        f"{report['correct']}/{report['total']} = {report['recall']:.2f}"
    )


if __name__ == "__main__":
    print_report(run_evaluation(top_k=5))
