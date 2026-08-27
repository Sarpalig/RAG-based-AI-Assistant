import json

from evaluation.evaluate_retrieval import calculate_recall_at_k, save_report


def test_calculate_recall_at_k_counts_expected_files():
    questions = [
        {
            "question": "Remote work approval?",
            "expected_file": "remote_work_policy.md",
            "answerable": True,
            "category": "policy",
        },
        {
            "question": "Password length?",
            "expected_file": "security_guidelines.md",
            "answerable": True,
            "category": "security",
        },
        {
            "question": "Unknown policy?",
            "expected_file": None,
            "answerable": False,
            "category": "unanswerable",
        },
    ]

    search_results = {
        "Remote work approval?": [
            {"file_name": "remote_work_policy.md"},
            {"file_name": "security_guidelines.md"},
        ],
        "Password length?": [
            {"file_name": "remote_work_policy.md"},
        ],
    }

    def fake_search(question, top_k=5):
        assert top_k == 5
        return search_results[question]

    report = calculate_recall_at_k(questions, fake_search, top_k=5)

    assert report["top_k"] == 5
    assert report["total"] == 2
    assert report["correct"] == 1
    assert report["recall"] == 0.5
    assert report["by_category"]["policy"]["recall"] == 1.0
    assert report["by_category"]["security"]["recall"] == 0.0
    assert report["details"][0]["found"] is True
    assert report["details"][0]["rank"] == 1
    assert report["details"][1]["found"] is False
    assert report["details"][1]["rank"] is None
    assert report["details"][2]["answerable"] is False


def test_calculate_recall_at_k_supports_multiple_expected_files():
    questions = [
        {
            "question": "Compare GROW and AMS.",
            "expected_files": ["grow.pdf", "ams.pdf"],
            "answerable": True,
            "category": "multi_document",
        },
        {
            "question": "Partial match.",
            "expected_files": ["grow.pdf", "missing.pdf"],
            "answerable": True,
            "category": "multi_document",
        },
    ]

    search_results = {
        "Compare GROW and AMS.": [
            {"file_name": "ams.pdf"},
            {"file_name": "grow.pdf"},
        ],
        "Partial match.": [
            {"file_name": "grow.pdf"},
        ],
    }

    def fake_search(question, top_k=5):
        return search_results[question]

    report = calculate_recall_at_k(questions, fake_search, top_k=5)

    assert report["total"] == 2
    assert report["correct"] == 1
    assert report["details"][0]["expected_files"] == ["grow.pdf", "ams.pdf"]
    assert report["details"][0]["found"] is True
    assert report["details"][0]["rank"] == 2
    assert report["details"][1]["found"] is False


def test_save_report_writes_json(tmp_path):
    report = {
        "top_k": 5,
        "total": 1,
        "correct": 1,
        "recall": 1.0,
        "details": [],
    }

    result_file = save_report(report, results_dir=tmp_path)

    assert result_file.exists()
    saved_report = json.loads(result_file.read_text(encoding="utf-8"))
    assert saved_report["recall"] == 1.0
