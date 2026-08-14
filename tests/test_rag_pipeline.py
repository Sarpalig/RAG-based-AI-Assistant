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
