from src import rag_pipeline
from src.rag_pipeline import RagPipeline


def test_format_search_results_flattens_chroma_response():
    chroma_results = {
        "ids": [["chunk-1", "chunk-2"]],
        "documents": [["First chunk text", "Second chunk text"]],
        "metadatas": [
            [
                {
                    "file_name": "remote_work_policy.md",
                    "page_number": 1,
                    "document_type": "md",
                },
                {
                    "file_name": "onboarding_process.docx",
                    "paragraph_number": 3,
                    "document_type": "docx",
                },
            ]
        ],
        "distances": [[0.12, 0.45]],
    }

    results = RagPipeline.format_search_results(chroma_results)

    assert results == [
        {
            "chunk_id": "chunk-1",
            "text": "First chunk text",
            "file_name": "remote_work_policy.md",
            "page_number": 1,
            "paragraph_number": None,
            "document_type": "md",
            "distance": 0.12,
        },
        {
            "chunk_id": "chunk-2",
            "text": "Second chunk text",
            "file_name": "onboarding_process.docx",
            "page_number": None,
            "paragraph_number": 3,
            "document_type": "docx",
            "distance": 0.45,
        },
    ]


def test_search_embeds_question_and_formats_results(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.collection = object()

    class FakeEmbeddingModel:
        def embed_query(self, question):
            assert question == "What is the remote work approval rule?"
            return [0.1, 0.2, 0.3]

    pipeline.embedding_model = FakeEmbeddingModel()

    def fake_similarity_search(collection, query_embedding, top_k=5):
        assert collection is pipeline.collection
        assert query_embedding == [0.1, 0.2, 0.3]
        assert top_k == 5
        return {
            "ids": [["chunk-1"]],
            "documents": [["Remote work requires manager approval."]],
            "metadatas": [[{"file_name": "remote_work_policy.md"}]],
            "distances": [[0.2]],
        }

    monkeypatch.setattr(rag_pipeline, "similarity_search", fake_similarity_search)

    results = pipeline.search("What is the remote work approval rule?", top_k=5)

    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["file_name"] == "remote_work_policy.md"
    assert results[0]["distance"] == 0.2


def test_index_files_uses_base_file_name(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    seen_file_names = []

    class FakeFile:
        name = r"C:\sample\documents\remote_work_policy.md"

    def fake_index_file(file, file_name=None):
        seen_file_names.append(file_name)
        return {"file_name": file_name}

    monkeypatch.setattr(pipeline, "index_file", fake_index_file)

    results = pipeline.index_files([FakeFile()])

    assert seen_file_names == ["remote_work_policy.md"]
    assert results == [{"file_name": "remote_work_policy.md"}]


def test_index_file_uses_configured_chunk_settings(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.collection = object()
    pipeline.chunk_size = 120
    pipeline.chunk_overlap = 20
    pipeline.embedding_model = object()

    monkeypatch.setattr(
        rag_pipeline,
        "document_exists",
        lambda collection, document_hash: False,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "extract_text",
        lambda file, file_name: [{"text": "Example document text.", "file_name": file_name}],
    )

    seen_settings = {}

    def fake_split_text(documents, chunk_size=300, chunk_overlap=30):
        seen_settings["chunk_size"] = chunk_size
        seen_settings["chunk_overlap"] = chunk_overlap
        return [{"chunk_id": "chunk-1", "text": "Example", "file_name": "doc.md"}]

    class FakeEmbeddingModel:
        def encode(self, texts):
            return [[0.1, 0.2]]

    def fake_add_documents(collection, chunks, embeddings):
        assert collection is pipeline.collection
        assert chunks[0]["chunk_id"] == "chunk-1"
        assert embeddings == [[0.1, 0.2]]

    pipeline.embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(rag_pipeline, "split_text", fake_split_text)
    monkeypatch.setattr(rag_pipeline, "add_documents", fake_add_documents)

    result = pipeline.index_file(b"content", file_name="doc.md")

    assert seen_settings == {"chunk_size": 120, "chunk_overlap": 20}
    assert result["chunk_count"] == 1


def test_chunk_settings_for_pdf_uses_larger_context_window():
    pipeline = object.__new__(RagPipeline)
    pipeline.chunk_size = 300
    pipeline.chunk_overlap = 30

    settings = pipeline.chunk_settings_for_documents(
        [{"text": "PDF text", "document_type": "pdf"}]
    )

    assert settings == {"chunk_size": 800, "chunk_overlap": 100}


def test_chunk_settings_for_markdown_uses_pipeline_defaults():
    pipeline = object.__new__(RagPipeline)
    pipeline.chunk_size = 300
    pipeline.chunk_overlap = 30

    settings = pipeline.chunk_settings_for_documents(
        [{"text": "Markdown text", "document_type": "md"}]
    )

    assert settings == {"chunk_size": 300, "chunk_overlap": 30}


def test_build_rag_prompt_includes_rules_question_and_sources():
    pipeline = object.__new__(RagPipeline)

    prompt = pipeline.build_rag_prompt(
        "Uzaktan çalışma için kimden onay alınmalıdır?",
        [
            {
                "file_name": "remote_work_policy.md",
                "text": "Uzaktan çalışma için onay birim yöneticisinden alınır.",
            }
        ],
    )

    assert "Use only the provided context" in prompt
    assert "Uzaktan çalışma için kimden onay alınmalıdır?" in prompt
    assert "[Kaynak 1]" in prompt
    assert "remote_work_policy.md" in prompt
    assert "Uzaktan çalışma için onay birim yöneticisinden alınır." in prompt


def test_build_rag_prompt_includes_seniority_guidance_when_profile_is_active():
    pipeline = object.__new__(RagPipeline)

    prompt = pipeline.build_rag_prompt(
        "What should I do first?",
        [{"file_name": "onboarding.md", "text": "Complete the first checklist item."}],
        {"seniority": "Intern"},
    )

    assert "Audience adaptation:" in prompt
    assert "The user is an intern" in prompt
    assert "Use only the provided context" in prompt
    assert "Complete the first checklist item." in prompt


def test_build_rag_prompt_includes_project_manager_role_guidance():
    pipeline = object.__new__(RagPipeline)

    prompt = pipeline.build_rag_prompt(
        "What should we prioritize?",
        [{"file_name": "roadmap.md", "text": "Prioritize defects blocking launch."}],
        {"role": "Senior Project Manager", "seniority": "Senior"},
    )

    assert "The user's role is Project Manager" in prompt
    assert "Focus on product quality" in prompt
    assert "Avoid low-level implementation details" in prompt
    assert "The user is senior-level" in prompt


def test_build_rag_prompt_includes_chat_history_as_context_only():
    pipeline = object.__new__(RagPipeline)

    prompt = pipeline.build_rag_prompt(
        "What about the approval?",
        [{"file_name": "policy.md", "text": "Manager approval is required."}],
        chat_history=[
            {"role": "user", "content": "We discussed remote work."},
            {"role": "assistant", "content": "Remote work has an approval flow."},
        ],
    )

    assert "Conversation context may be used only to understand references" in prompt
    assert "Konuşma bağlamı:" in prompt
    assert "Kullanıcı: We discussed remote work." in prompt
    assert "Asistan: Remote work has an approval flow." in prompt
    assert "Kaynaklar:" in prompt
    assert "Manager approval is required." in prompt


def test_build_chat_history_section_ignores_empty_or_unknown_messages():
    section = RagPipeline.build_chat_history_section(
        [
            {"role": "system", "content": "Hidden instruction"},
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "Previous answer"},
        ]
    )

    assert "Hidden instruction" not in section
    assert "Kullanıcı:" not in section
    assert "Asistan: Previous answer" in section


def test_build_role_guidance_matches_turkish_project_manager_role():
    guidance = RagPipeline.build_role_guidance("Proje Yoneticisi")

    assert "The user's role is Project Manager" in guidance


def test_build_role_guidance_matches_turkish_project_manager_role_with_diacritics():
    guidance = RagPipeline.build_role_guidance("Proje Yöneticisi")

    assert "The user's role is Project Manager" in guidance


def test_build_role_guidance_ignores_non_project_manager_role():
    guidance = RagPipeline.build_role_guidance("Backend Developer")

    assert guidance == ""


def test_build_audience_guidance_combines_project_manager_role_and_seniority():
    guidance = RagPipeline.build_audience_guidance(
        {"role": "Project Manager", "seniority": "Mid-level"}
    )

    assert "Focus on product quality" in guidance
    assert "Avoid low-level implementation details" in guidance
    assert "The user is mid-level" in guidance


def test_build_audience_guidance_ignores_missing_or_unknown_profile():
    assert RagPipeline.build_audience_guidance(None) == ""
    assert RagPipeline.build_audience_guidance({"seniority": "Unknown"}) == ""


def test_answer_query_searches_builds_prompt_and_calls_openrouter(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm_provider = "ollama"
    pipeline.ollama_base_url = "http://localhost:11434"
    pipeline.ollama_model = "qwen3.5:9b"
    pipeline.openrouter_api_key = "test-key"
    pipeline.openrouter_model = "test-model"
    pipeline.openrouter_fallback_model = "openrouter/free"

    search_results = [
        {
            "file_name": "remote_work_policy.md",
            "text": "Remote work requires manager approval.",
        }
    ]

    def fake_search(question, top_k=5):
        assert question == "Who approves remote work?"
        assert top_k == 3
        return search_results

    def fake_query_llm(
        provider,
        prompt,
        max_tokens=512,
        openrouter_api_key=None,
        openrouter_model=None,
        openrouter_fallback_model=None,
        ollama_base_url=None,
        ollama_model=None,
    ):
        assert provider == "ollama"
        assert openrouter_api_key == "test-key"
        assert openrouter_model == "test-model"
        assert openrouter_fallback_model == "openrouter/free"
        assert ollama_base_url == "http://localhost:11434"
        assert ollama_model == "qwen3.5:9b"
        assert "Who approves remote work?" in prompt
        assert "Remote work requires manager approval." in prompt
        assert max_tokens == 512
        return "Manager approval is required. Kaynak 1 [Kaynak 99]"

    monkeypatch.setattr(pipeline, "search", fake_search)
    monkeypatch.setattr(rag_pipeline, "query_llm", fake_query_llm)

    answer, citations = pipeline.answer_query("Who approves remote work?", top_k=3)

    assert answer == "Manager approval is required. [Kaynak 1] "
    assert citations == ["Kaynak 1: remote_work_policy.md"]


def test_answer_query_passes_user_profile_to_prompt(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm_provider = "ollama"
    pipeline.ollama_base_url = "http://localhost:11434"
    pipeline.ollama_model = "qwen3.5:9b"
    pipeline.openrouter_api_key = None
    pipeline.openrouter_model = None
    pipeline.openrouter_fallback_model = "openrouter/free"
    user_profile = {"seniority": "Senior"}
    seen = {}
    search_results = [{"file_name": "policy.md", "text": "Policy text."}]

    monkeypatch.setattr(pipeline, "search", lambda question, top_k=5: search_results)

    def fake_build_rag_prompt(question, results, profile=None, history=None):
        seen["profile"] = profile
        return "prompt"

    monkeypatch.setattr(pipeline, "build_rag_prompt", fake_build_rag_prompt)
    monkeypatch.setattr(
        rag_pipeline,
        "query_llm",
        lambda **kwargs: "Answer [Kaynak 1]",
    )

    answer, citations = pipeline.answer_query("Question?", user_profile=user_profile)

    assert seen["profile"] == user_profile
    assert answer == "Answer [Kaynak 1]"
    assert citations == ["Kaynak 1: policy.md"]


def test_answer_query_passes_chat_history_to_prompt(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm_provider = "ollama"
    pipeline.ollama_base_url = "http://localhost:11434"
    pipeline.ollama_model = "qwen3.5:9b"
    pipeline.openrouter_api_key = None
    pipeline.openrouter_model = None
    pipeline.openrouter_fallback_model = "openrouter/free"
    chat_history = [{"role": "user", "content": "Previous question"}]
    seen = {}
    search_results = [{"file_name": "policy.md", "text": "Policy text."}]

    monkeypatch.setattr(pipeline, "search", lambda question, top_k=5: search_results)

    def fake_build_rag_prompt(question, results, profile=None, history=None):
        seen["history"] = history
        return "prompt"

    monkeypatch.setattr(pipeline, "build_rag_prompt", fake_build_rag_prompt)
    monkeypatch.setattr(
        rag_pipeline,
        "query_llm",
        lambda **kwargs: "Answer [Kaynak 1]",
    )

    answer, citations = pipeline.answer_query("Question?", chat_history=chat_history)

    assert seen["history"] == chat_history
    assert answer == "Answer [Kaynak 1]"
    assert citations == ["Kaynak 1: policy.md"]


def test_answer_query_adds_project_manager_guidance_to_prompt(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm_provider = "ollama"
    pipeline.ollama_base_url = "http://localhost:11434"
    pipeline.ollama_model = "qwen3.5:9b"
    pipeline.openrouter_api_key = None
    pipeline.openrouter_model = None
    pipeline.openrouter_fallback_model = "openrouter/free"
    user_profile = {"role": "Project Manager", "seniority": "Senior"}
    search_results = [{"file_name": "quality.md", "text": "Launch defects block release."}]
    seen = {}

    monkeypatch.setattr(pipeline, "search", lambda question, top_k=5: search_results)

    def fake_query_llm(**kwargs):
        seen["prompt"] = kwargs["prompt"]
        return "Prioritize launch-blocking defects. [Kaynak 1]"

    monkeypatch.setattr(rag_pipeline, "query_llm", fake_query_llm)

    answer, citations = pipeline.answer_query(
        "What should we prioritize?",
        user_profile=user_profile,
    )

    assert "The user's role is Project Manager" in seen["prompt"]
    assert "Focus on product quality" in seen["prompt"]
    assert "Avoid low-level implementation details" in seen["prompt"]
    assert "The user is senior-level" in seen["prompt"]
    assert answer == "Prioritize launch-blocking defects. [Kaynak 1]"
    assert citations == ["Kaynak 1: quality.md"]


def test_answer_query_lists_only_sources_used_in_answer(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm_provider = "ollama"
    pipeline.ollama_base_url = "http://localhost:11434"
    pipeline.ollama_model = "qwen3.5:9b"
    pipeline.openrouter_api_key = None
    pipeline.openrouter_model = None
    pipeline.openrouter_fallback_model = "openrouter/free"

    search_results = [
        {"file_name": "first.md", "text": "First source."},
        {"file_name": "second.md", "text": "Second source."},
        {"file_name": "third.md", "text": "Third source."},
    ]

    monkeypatch.setattr(pipeline, "search", lambda question, top_k=5: search_results)
    monkeypatch.setattr(
        rag_pipeline,
        "query_llm",
        lambda **kwargs: "Answer is supported by Kaynak 2.",
    )

    answer, citations = pipeline.answer_query("Question?")

    assert answer == "Answer is supported by [Kaynak 2]."
    assert citations == ["Kaynak 2: second.md"]


def test_query_llm_receives_openrouter_fallback_model(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm_provider = "openrouter"
    pipeline.ollama_base_url = "http://localhost:11434"
    pipeline.ollama_model = "qwen3.5:9b"
    pipeline.openrouter_api_key = "test-key"
    pipeline.openrouter_model = "primary-model"
    pipeline.openrouter_fallback_model = "openrouter/free"

    def fake_query_llm(
        provider,
        prompt,
        max_tokens=512,
        openrouter_api_key=None,
        openrouter_model=None,
        openrouter_fallback_model=None,
        ollama_base_url=None,
        ollama_model=None,
    ):
        assert provider == "openrouter"
        assert openrouter_model == "primary-model"
        assert openrouter_fallback_model == "openrouter/free"
        return "OpenRouter answer"

    monkeypatch.setattr(rag_pipeline, "query_llm", fake_query_llm)

    answer = pipeline._query_llm("prompt")

    assert answer == "OpenRouter answer"


def test_answer_query_returns_not_found_when_no_search_results(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    monkeypatch.setattr(pipeline, "search", lambda question, top_k=5: [])

    answer, citations = pipeline.answer_query("Unknown question?")

    assert answer == "Bu bilgi verilen belgelerde bulunamadı."
    assert citations == []


def test_build_citations_includes_file_page_and_paragraph():
    citations = RagPipeline.build_citations(
        [
            {
                "file_name": "travel_expense_policy.pdf",
                "page_number": 2,
                "paragraph_number": None,
            },
            {
                "file_name": "onboarding_process.docx",
                "page_number": None,
                "paragraph_number": 4,
            },
        ]
    )

    assert citations == [
        "Kaynak 1: travel_expense_policy.pdf, sayfa 2",
        "Kaynak 2: onboarding_process.docx, paragraf 4",
    ]


def test_filter_invalid_citations_removes_unknown_source_numbers():
    answer = RagPipeline.filter_invalid_citations(
        "Cevap [Kaynak 1] ama bu uydurma [Kaynak 9].",
        source_count=2,
    )

    assert answer == "Cevap [Kaynak 1] ama bu uydurma ."


def test_extract_citation_numbers_returns_valid_unique_numbers_in_order():
    source_numbers = RagPipeline.extract_citation_numbers(
        "Cevap [Kaynak 2], tekrar [Kaynak 2], sonra [Kaynak 1] ve [Kaynak 9].",
        source_count=3,
    )

    assert source_numbers == [2, 1]


def test_rerank_search_results_uses_question_terms_as_tiebreaker():
    results = [
        {
            "file_name": "onboarding_process.docx",
            "text": "Çalışanlara şirket politikaları ve güvenlik kuralları anlatılır.",
        },
        {
            "file_name": "security_guidelines.md",
            "text": "Tüm çalışanlar şirket hesabı için iki faktörlü doğrulama kullanmalıdır.",
        },
    ]

    reranked = RagPipeline.rerank_search_results(
        "Hangi güvenlik yöntemi şirket hesabı için zorunludur?",
        results,
    )

    assert reranked[0]["file_name"] == "security_guidelines.md"


def test_diversify_search_results_prefers_distinct_files_before_duplicates():
    results = [
        {"chunk_id": "a1", "file_name": "a.pdf"},
        {"chunk_id": "a2", "file_name": "a.pdf"},
        {"chunk_id": "b1", "file_name": "b.pdf"},
        {"chunk_id": "c1", "file_name": "c.pdf"},
        {"chunk_id": "a3", "file_name": "a.pdf"},
    ]

    diversified = RagPipeline.diversify_search_results(results, top_k=4)

    assert [item["chunk_id"] for item in diversified] == ["a1", "b1", "c1", "a2"]
