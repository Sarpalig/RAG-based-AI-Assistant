import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
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
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def load_questions(question_file=QUESTION_FILE):
    with open(question_file, "r", encoding="utf-8") as file:
        return json.load(file)


def open_sample_documents(sample_document_dir=SAMPLE_DOCUMENT_DIR, document_names=None):
    allowed_names = set(document_names) if document_names else None
    document_paths = sorted(
        path
        for path in sample_document_dir.iterdir()
        if path.suffix.lower() in {".pdf", ".docx", ".md"}
        and (allowed_names is None or path.name in allowed_names)
    )
    return [open(path, "rb") for path in document_paths]


def calculate_recall_at_k(questions, search_fn, top_k=5):
    total = 0
    correct = 0
    details = []
    by_category = {}

    for item in questions:
        if not item.get("answerable"):
            details.append(
                {
                    "question": item["question"],
                    "category": item.get("category", "uncategorized"),
                    "answerable": False,
                    "expected_file": None,
                    "expected_answer": item.get("expected_answer"),
                    "found": None,
                    "rank": None,
                    "returned_files": [],
                }
            )
            continue

        total += 1
        results = search_fn(item["question"], top_k=top_k)
        expected_files = _expected_files(item)
        expected_file = expected_files[0] if expected_files else None
        returned_files = [result.get("file_name") for result in results]
        ranks = {
            file_name: _first_rank(returned_files, file_name)
            for file_name in expected_files
        }
        found = bool(ranks) and all(rank is not None for rank in ranks.values())
        rank = max(ranks.values()) if found else None

        if found:
            correct += 1

        category = item.get("category", "uncategorized")
        category_report = by_category.setdefault(
            category,
            {"total": 0, "correct": 0, "recall": 0},
        )
        category_report["total"] += 1
        if found:
            category_report["correct"] += 1

        details.append(
            {
                "question": item["question"],
                "category": category,
                "answerable": True,
                "expected_answer": item.get("expected_answer"),
                "expected_file": expected_file,
                "expected_files": expected_files,
                "expected_page": item.get("expected_page"),
                "expected_paragraph": item.get("expected_paragraph"),
                "found": found,
                "rank": rank,
                "returned_files": returned_files,
                "top_results": [
                    {
                        "file_name": result.get("file_name"),
                        "page_number": result.get("page_number"),
                        "paragraph_number": result.get("paragraph_number"),
                        "distance": result.get("distance"),
                    }
                    for result in results
                ],
            }
        )

    for category_report in by_category.values():
        category_report["recall"] = (
            category_report["correct"] / category_report["total"]
            if category_report["total"]
            else 0
        )

    recall = correct / total if total else 0
    return {
        "top_k": top_k,
        "total": total,
        "correct": correct,
        "recall": recall,
        "by_category": by_category,
        "details": details,
    }


def run_evaluation(
    top_k=5,
    chroma_dir=EVALUATION_CHROMA_DIR,
    save_results=True,
    question_file=QUESTION_FILE,
    document_names=None,
):
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    started_at = time.perf_counter()
    rag = RagPipeline(chroma_dir=str(chroma_dir), collection_name="retrieval_eval")
    questions = load_questions(question_file)
    document_names = document_names or _document_names_from_questions(questions)
    sample_documents = open_sample_documents(document_names=document_names)

    try:
        rag.index_files(sample_documents)
    finally:
        for document in sample_documents:
            document.close()

    report = calculate_recall_at_k(questions, rag.search, top_k=top_k)
    report["metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - started_at, 3),
        "question_count": len(questions),
        "answerable_count": report["total"],
        "unanswerable_count": len(questions) - report["total"],
        "sample_document_dir": str(SAMPLE_DOCUMENT_DIR),
        "question_file": str(question_file),
        "indexed_document_names": sorted(document_names) if document_names else None,
        "chroma_dir": str(chroma_dir),
        "collection_name": "retrieval_eval",
        "embedding_model_name": rag.embedding_model_name,
        "llm_provider": rag.llm_provider,
        "ollama_model": rag.ollama_model,
        "openrouter_model": rag.openrouter_model,
        "environment": {
            "CHROMA_DIR": os.getenv("CHROMA_DIR"),
            "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
            "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL"),
        },
    }

    if save_results:
        report["result_file"] = str(save_report(report))

    return report


def save_report(report, results_dir=RESULTS_DIR):
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"retrieval_report_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return result_file


def print_report(report):
    for item in report["details"]:
        if not item["answerable"]:
            print(f"Question: {item['question']}")
            print("Answerable: no")
            print()
            continue

        status = "yes" if item["found"] else "no"
        print(f"Question: {item['question']}")
        print(f"Category: {item['category']}")
        print(f"Expected file: {item['expected_file']}")
        print(f"Returned files: {item['returned_files']}")
        print(f"Found in top {report['top_k']}: {status}")
        print(f"Rank: {item['rank']}")
        print()

    print(
        f"Recall@{report['top_k']}: "
        f"{report['correct']}/{report['total']} = {report['recall']:.2f}"
    )
    print("By category:")
    for category, values in report["by_category"].items():
        print(
            f"- {category}: {values['correct']}/{values['total']} = "
            f"{values['recall']:.2f}"
        )
    metadata = report.get("metadata", {})
    if metadata:
        print(f"Duration: {metadata['duration_seconds']}s")
        print(f"Embedding model: {metadata['embedding_model_name']}")
        print(f"LLM provider: {metadata['llm_provider']}")
    if report.get("result_file"):
        print(f"Saved report: {report['result_file']}")


def _first_rank(returned_files, expected_file):
    for index, file_name in enumerate(returned_files, start=1):
        if file_name == expected_file:
            return index
    return None


def _expected_files(item):
    if item.get("expected_files"):
        return item["expected_files"]
    if item.get("expected_file"):
        return [item["expected_file"]]
    return []


def _document_names_from_questions(questions):
    document_names = set()
    for item in questions:
        document_names.update(_expected_files(item))
    return document_names


if __name__ == "__main__":
    question_file = Path(os.getenv("QUESTION_FILE", QUESTION_FILE))
    document_names = os.getenv("EVALUATION_DOCUMENTS")
    if document_names:
        document_names = [
            document_name.strip()
            for document_name in document_names.split(",")
            if document_name.strip()
        ]
    print_report(
        run_evaluation(
            top_k=int(os.getenv("TOP_K", "5")),
            question_file=question_file,
            document_names=document_names,
        )
    )
