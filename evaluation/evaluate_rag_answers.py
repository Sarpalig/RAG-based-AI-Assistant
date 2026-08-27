import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if os.getenv("ALLOW_MODEL_DOWNLOADS") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_retrieval import load_questions
from src.rag_pipeline import RagPipeline


SAMPLE_DOCUMENT_DIR = PROJECT_ROOT / "data" / "sample_documents"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
ANSWER_EVAL_CHROMA_DIR = PROJECT_ROOT / ".tmp" / "rag_answer_eval_chroma"
REJECTION_PATTERNS = [
    "bulunamad",
    "yer alm",
    "verilen belgelerde",
    "sağlanan belgelerde",
    "provided documents",
    "not found",
    "does not contain",
    "cannot be found",
]


def run_answer_evaluation(top_k=5, question_file=None, document_names=None):
    if ANSWER_EVAL_CHROMA_DIR.exists():
        shutil.rmtree(ANSWER_EVAL_CHROMA_DIR)

    question_file = question_file or PROJECT_ROOT / "evaluation" / "test_questions.json"
    questions = load_questions(question_file)
    document_names = document_names or _document_names_from_questions(questions)

    started_at = time.perf_counter()
    rag = RagPipeline(
        chroma_dir=str(ANSWER_EVAL_CHROMA_DIR),
        collection_name="rag_answer_eval",
    )
    index_results = _index_sample_documents(rag, document_names=document_names)

    details = []
    for item in questions:
        item_started_at = time.perf_counter()
        try:
            answer, citations = rag.answer_query(item["question"], top_k=top_k)
            details.append(
                _evaluate_answer_item(
                    item=item,
                    answer=answer,
                    citations=citations,
                    latency_seconds=round(time.perf_counter() - item_started_at, 3),
                )
            )
        except Exception as exc:
            details.append(
                {
                    "question": item["question"],
                    "category": item.get("category", "uncategorized"),
                    "answerable": item.get("answerable"),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_seconds": round(time.perf_counter() - item_started_at, 3),
                }
            )

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.perf_counter() - started_at, 3),
            "top_k": top_k,
            "question_count": len(details),
            "sample_document_dir": str(SAMPLE_DOCUMENT_DIR),
            "question_file": str(question_file),
            "indexed_document_names": sorted(document_names) if document_names else None,
            "chroma_dir": str(ANSWER_EVAL_CHROMA_DIR),
            "collection_name": "rag_answer_eval",
            "embedding_model_name": rag.embedding_model_name,
            "llm_provider": rag.llm_provider,
            "ollama_model": rag.ollama_model,
            "openrouter_model": rag.openrouter_model,
            "index_results": index_results,
        },
        "summary": _build_summary(details),
        "details": details,
    }
    report["result_file"] = str(save_report(report))
    return report


def _index_sample_documents(rag, document_names=None):
    allowed_names = set(document_names) if document_names else None
    files = [
        open(path, "rb")
        for path in sorted(SAMPLE_DOCUMENT_DIR.iterdir())
        if path.suffix.lower() in {".pdf", ".docx", ".md"}
        and (allowed_names is None or path.name in allowed_names)
    ]
    try:
        return rag.index_files(files)
    finally:
        for file in files:
            file.close()


def _evaluate_answer_item(item, answer, citations, latency_seconds):
    expected_files = _expected_files(item)
    expected_file = expected_files[0] if expected_files else None
    answerable = item.get("answerable")
    rejection_detected = _contains_any(answer, REJECTION_PATTERNS)
    expected_overlap = _expected_answer_overlap(
        expected_answer=item.get("expected_answer", ""),
        answer=answer,
    )
    expected_file_cited = bool(expected_files) and all(
        any(expected_file in citation for citation in citations)
        for expected_file in expected_files
    )

    if answerable:
        status = "pass" if expected_overlap >= 0.5 and expected_file_cited else "review"
    else:
        status = "pass" if rejection_detected else "review"

    return {
        "question": item["question"],
        "category": item.get("category", "uncategorized"),
        "answerable": answerable,
        "status": status,
        "expected_answer": item.get("expected_answer"),
        "expected_file": expected_file,
        "expected_files": expected_files,
        "answer": answer,
        "citations": citations,
        "expected_answer_overlap": expected_overlap,
        "expected_file_cited": expected_file_cited,
        "rejection_detected": rejection_detected,
        "latency_seconds": latency_seconds,
    }


def _build_summary(details):
    completed = [item for item in details if item["status"] != "error"]
    answerable = [item for item in completed if item["answerable"]]
    unanswerable = [item for item in completed if not item["answerable"]]
    errors = [item for item in details if item["status"] == "error"]
    needs_review = [item for item in completed if item["status"] == "review"]

    return {
        "completed": len(completed),
        "errors": len(errors),
        "needs_review": len(needs_review),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answerable_heuristic_pass": sum(
            1 for item in answerable if item["status"] == "pass"
        ),
        "rejection_pass": sum(1 for item in unanswerable if item["status"] == "pass"),
        "citation_pass": sum(1 for item in answerable if item["expected_file_cited"]),
        "average_latency_seconds": round(
            sum(item["latency_seconds"] for item in completed) / len(completed),
            3,
        )
        if completed
        else 0,
        "review_questions": [
            {
                "question": item["question"],
                "category": item["category"],
                "answerable": item["answerable"],
                "expected_answer_overlap": item.get("expected_answer_overlap"),
                "expected_file_cited": item.get("expected_file_cited"),
                "rejection_detected": item.get("rejection_detected"),
            }
            for item in needs_review
        ],
        "errors_by_question": [
            {
                "question": item["question"],
                "error_type": item["error_type"],
                "error": item["error"],
            }
            for item in errors
        ],
    }


def _expected_answer_overlap(expected_answer, answer):
    expected_tokens = _meaningful_tokens(expected_answer)
    if not expected_tokens:
        return 0
    answer_tokens = set(_meaningful_tokens(answer))
    return round(
        sum(1 for token in expected_tokens if token in answer_tokens)
        / len(expected_tokens),
        4,
    )


def _meaningful_tokens(text):
    stopwords = {
        "ve",
        "veya",
        "ile",
        "için",
        "bir",
        "bu",
        "şu",
        "the",
        "and",
        "or",
        "is",
        "are",
    }
    return [
        token
        for token in re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", text.casefold())
        if len(token) > 2 and token not in stopwords
    ]


def _contains_any(text, patterns):
    normalized = text.casefold()
    return any(pattern in normalized for pattern in patterns)


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


def save_report(report):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"rag_answer_report_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return result_file


def print_report(report):
    summary = report["summary"]
    print(f"Completed: {summary['completed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Needs review: {summary['needs_review']}")
    print(
        "Answerable heuristic pass: "
        f"{summary['answerable_heuristic_pass']}/{summary['answerable_count']}"
    )
    print(
        "Citation pass: "
        f"{summary['citation_pass']}/{summary['answerable_count']}"
    )
    print(
        "Rejection pass: "
        f"{summary['rejection_pass']}/{summary['unanswerable_count']}"
    )
    print(f"Average latency: {summary['average_latency_seconds']}s")

    if summary["review_questions"]:
        print("\nReview questions:")
        for item in summary["review_questions"]:
            print(
                f"- {item['category']} | {item['question']} | "
                f"overlap={item['expected_answer_overlap']} "
                f"cited={item['expected_file_cited']} "
                f"rejected={item['rejection_detected']}"
            )

    if summary["errors_by_question"]:
        print("\nErrors:")
        for item in summary["errors_by_question"]:
            print(f"- {item['question']}: {item['error_type']} {item['error']}")

    print(f"\nSaved report: {report['result_file']}")


if __name__ == "__main__":
    question_file = Path(os.getenv("QUESTION_FILE", PROJECT_ROOT / "evaluation" / "test_questions.json"))
    document_names = os.getenv("EVALUATION_DOCUMENTS")
    if document_names:
        document_names = [
            document_name.strip()
            for document_name in document_names.split(",")
            if document_name.strip()
        ]
    print_report(
        run_answer_evaluation(
            top_k=int(os.getenv("TOP_K", "5")),
            question_file=question_file,
            document_names=document_names,
        )
    )
