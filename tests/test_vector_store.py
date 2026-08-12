import os
from src.vector_store import create_chroma_client, get_or_create_collection, add_documents, similarity_search


def test_chroma_add_and_query(tmp_path):
    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir()

    client = create_chroma_client(str(db_dir))
    collection = get_or_create_collection(client, collection_name="pytest_rag")

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
