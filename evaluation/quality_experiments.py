import json
import os
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

from evaluation.evaluate_retrieval import calculate_recall_at_k, load_questions
from src.rag_pipeline import RagPipeline


SAMPLE_DOCUMENT_DIR = PROJECT_ROOT / "data" / "sample_documents"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
EXPERIMENT_CHROMA_ROOT = PROJECT_ROOT / ".tmp" / "quality_experiments_chroma"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_VALUES = [1, 3, 5, 10]
CHUNK_SETTINGS = [
    {"chunk_size": 150, "chunk_overlap": 0},
    {"chunk_size": 150, "chunk_overlap": 30},
    {"chunk_size": 300, "chunk_overlap": 30},
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 700, "chunk_overlap": 100},
]


def run_quality_experiments():
    questions = load_questions()
    embedding_models = _embedding_models_from_env()
    experiments = []

    if EXPERIMENT_CHROMA_ROOT.exists():
        shutil.rmtree(EXPERIMENT_CHROMA_ROOT)
    EXPERIMENT_CHROMA_ROOT.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    for embedding_model_name in embedding_models:
        for chunk_settings in CHUNK_SETTINGS:
            experiments.append(
                _run_single_experiment(
                    questions=questions,
                    embedding_model_name=embedding_model_name,
                    chunk_size=chunk_settings["chunk_size"],
                    chunk_overlap=chunk_settings["chunk_overlap"],
                )
            )

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.perf_counter() - started_at, 3),
            "question_count": len(questions),
            "answerable_count": sum(1 for item in questions if item.get("answerable")),
            "unanswerable_count": sum(
                1 for item in questions if not item.get("answerable")
            ),
            "top_k_values": TOP_K_VALUES,
            "sample_document_dir": str(SAMPLE_DOCUMENT_DIR),
            "chroma_root": str(EXPERIMENT_CHROMA_ROOT),
            "offline_model_loading": os.getenv("ALLOW_MODEL_DOWNLOADS") != "1",
        },
        "experiments": experiments,
    }
    report["recommendations"] = build_recommendations(experiments)
    report["result_file"] = str(save_report(report))
    return report


def _run_single_experiment(questions, embedding_model_name, chunk_size, chunk_overlap):
    label = f"{_safe_label(embedding_model_name)}_cs{chunk_size}_ov{chunk_overlap}"
    chroma_dir = EXPERIMENT_CHROMA_ROOT / label
    experiment_started_at = time.perf_counter()

    try:
        rag = RagPipeline(
            chroma_dir=str(chroma_dir),
            collection_name="quality_experiment",
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        index_results = _index_sample_documents(rag)
        reports_by_top_k = {
            str(top_k): calculate_recall_at_k(
                questions,
                rag.search,
                top_k=top_k,
            )
            for top_k in TOP_K_VALUES
        }

        return {
            "status": "ok",
            "embedding_model_name": embedding_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": sum(result["chunk_count"] for result in index_results),
            "duration_seconds": round(time.perf_counter() - experiment_started_at, 3),
            "reports_by_top_k": _compact_reports(reports_by_top_k),
            "unanswerable_probe": _probe_unanswerable_questions(questions, rag),
        }
    except Exception as exc:
        return {
            "status": "error",
            "embedding_model_name": embedding_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "duration_seconds": round(time.perf_counter() - experiment_started_at, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _index_sample_documents(rag):
    files = [
        open(path, "rb")
        for path in sorted(SAMPLE_DOCUMENT_DIR.iterdir())
        if path.suffix.lower() in {".pdf", ".docx", ".md"}
    ]
    try:
        return rag.index_files(files)
    finally:
        for file in files:
            file.close()


def _compact_reports(reports_by_top_k):
    compact = {}
    for top_k, report in reports_by_top_k.items():
        compact[top_k] = {
            "correct": report["correct"],
            "total": report["total"],
            "recall": report["recall"],
            "mean_reciprocal_rank": _mean_reciprocal_rank(report),
            "by_category": report["by_category"],
            "failures": [
                {
                    "question": item["question"],
                    "category": item["category"],
                    "expected_file": item["expected_file"],
                    "returned_files": item["returned_files"],
                    "rank": item["rank"],
                }
                for item in report["details"]
                if item["answerable"] and not item["found"]
            ],
            "non_rank1": [
                {
                    "question": item["question"],
                    "category": item["category"],
                    "expected_file": item["expected_file"],
                    "returned_files": item["returned_files"],
                    "rank": item["rank"],
                }
                for item in report["details"]
                if item["answerable"] and item["rank"] != 1
            ],
        }
    return compact


def _mean_reciprocal_rank(report):
    reciprocal_ranks = []
    for item in report["details"]:
        if not item["answerable"]:
            continue
        reciprocal_ranks.append(1 / item["rank"] if item["rank"] else 0)
    return (
        round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
        if reciprocal_ranks
        else 0
    )


def _probe_unanswerable_questions(questions, rag):
    probe = []
    for item in questions:
        if item.get("answerable"):
            continue
        results = rag.search(item["question"], top_k=3)
        probe.append(
            {
                "question": item["question"],
                "top_results": [
                    {
                        "file_name": result.get("file_name"),
                        "distance": result.get("distance"),
                        "preview": result.get("text", "").replace("\n", " ")[:180],
                    }
                    for result in results
                ],
            }
        )
    return probe


def build_recommendations(experiments):
    successful = [item for item in experiments if item["status"] == "ok"]
    if not successful:
        return ["No successful experiment completed."]

    top1_best = max(
        successful,
        key=lambda item: (
            item["reports_by_top_k"]["1"]["recall"],
            item["reports_by_top_k"]["1"]["mean_reciprocal_rank"],
            -item["chunk_count"],
        ),
    )
    top5_best = max(
        successful,
        key=lambda item: (
            item["reports_by_top_k"]["5"]["recall"],
            item["reports_by_top_k"]["5"]["mean_reciprocal_rank"],
            -item["chunk_count"],
        ),
    )

    recommendations = [
        _format_best_setting("Best Top-1 retrieval", top1_best, top_k="1"),
        _format_best_setting("Best Recall@5 retrieval", top5_best, top_k="5"),
    ]

    if top5_best["reports_by_top_k"]["5"]["recall"] == 1.0:
        recommendations.append(
            "Retrieval is already strong on the current test set; prioritize harder evaluation questions before adding hybrid search or reranking."
        )

    if any(item["status"] == "error" for item in experiments):
        recommendations.append(
            "At least one embedding experiment failed; inspect error entries before comparing embedding models."
        )

    return recommendations


def _format_best_setting(title, experiment, top_k):
    report = experiment["reports_by_top_k"][top_k]
    return (
        f"{title}: chunk_size={experiment['chunk_size']}, "
        f"chunk_overlap={experiment['chunk_overlap']}, "
        f"embedding={experiment['embedding_model_name']}, "
        f"recall={report['recall']:.2f}, "
        f"mrr={report['mean_reciprocal_rank']:.4f}, "
        f"chunks={experiment['chunk_count']}"
    )


def _embedding_models_from_env():
    configured = os.getenv("EXPERIMENT_EMBEDDING_MODELS")
    if not configured:
        return [os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)]
    return [model.strip() for model in configured.split(",") if model.strip()]


def _safe_label(value):
    return "".join(character if character.isalnum() else "_" for character in value)


def save_report(report):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"quality_experiments_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return result_file


def print_summary(report):
    for experiment in report["experiments"]:
        if experiment["status"] != "ok":
            print(
                "ERROR "
                f"model={experiment['embedding_model_name']} "
                f"chunk={experiment['chunk_size']}/{experiment['chunk_overlap']} "
                f"{experiment['error_type']}: {experiment['error']}"
            )
            continue

        metrics = []
        for top_k in TOP_K_VALUES:
            top_report = experiment["reports_by_top_k"][str(top_k)]
            metrics.append(
                f"R@{top_k}={top_report['recall']:.2f},"
                f"MRR={top_report['mean_reciprocal_rank']:.4f}"
            )
        print(
            f"OK model={experiment['embedding_model_name']} "
            f"chunk={experiment['chunk_size']}/{experiment['chunk_overlap']} "
            f"chunks={experiment['chunk_count']} "
            + " ".join(metrics)
        )

    print("\nRecommendations:")
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")
    print(f"\nSaved report: {report['result_file']}")


if __name__ == "__main__":
    print_summary(run_quality_experiments())
