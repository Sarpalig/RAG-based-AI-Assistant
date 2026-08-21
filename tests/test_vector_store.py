import os
from src.vector_store import (
    add_documents,
    create_chroma_client,
    list_indexed_documents,
    similarity_search,
)


def test_chroma_add_and_query(tmp_path):
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir()

    client = create_chroma_client(str(db_dir))
    collection = client.get_or_create_collection(name="pytest_rag")

    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "This is a test chunk.",
            "file_name": "doc.md",
            "page_number": 1,
            "document_type": "md",
        }
    ]

    # simple dummy embedding vector
    embeddings = [[0.0] * 8]

    add_documents(collection, chunks, embeddings)

    res = similarity_search(collection, embeddings[0], top_k=1)

    # chroma returns lists inside lists for batched queries
    assert res["ids"][0][0] == "chunk-1"
    assert res["metadatas"][0][0]["file_name"] == "doc.md"


def test_list_indexed_documents_groups_chunks_by_document_hash(tmp_path):
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir()

    client = create_chroma_client(str(db_dir))
    collection = client.get_or_create_collection(name="pytest_documents")

    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "First chunk.",
            "file_name": "doc.md",
            "document_type": "md",
            "document_hash": "hash-1",
        },
        {
            "chunk_id": "chunk-2",
            "text": "Second chunk.",
            "file_name": "doc.md",
            "document_type": "md",
            "document_hash": "hash-1",
        },
    ]
    embeddings = [[0.0] * 8, [0.1] * 8]

    add_documents(collection, chunks, embeddings)

    documents = list_indexed_documents(collection)

    assert documents == [
        {
            "document_hash": "hash-1",
            "file_name": "doc.md",
            "document_type": "md",
            "chunk_count": 2,
            "skipped": False,
        }
    ]
