from evaluation.evaluate_retrieval import calculate_recall_at_k


def test_calculate_recall_at_k_counts_expected_files():
    questions = [
        {
            "question": "Remote work approval?",
            "expected_file": "remote_work_policy.md",
            "answerable": True,
        },
        {
            "question": "Password length?",
            "expected_file": "security_guidelines.md",
            "answerable": True,
        },
        {
            "question": "Unknown policy?",
            "expected_file": None,
            "answerable": False,
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
    assert report["details"][0]["found"] is True
    assert report["details"][1]["found"] is False
