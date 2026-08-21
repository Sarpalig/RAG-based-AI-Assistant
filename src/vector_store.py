import chromadb
from chromadb.config import Settings


def create_chroma_client(chroma_dir):
    # Use the newer Settings keys: set persistence and directory
    return chromadb.Client(Settings(is_persistent=True, persist_directory=chroma_dir))


def add_documents(collection, chunks, embeddings):
    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        _metadata_from_chunk(chunk)
        for chunk in chunks
    ]
    # embeddings must align with chunks order
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def _metadata_from_chunk(chunk):
    metadata = {
        "file_name": chunk["file_name"],
        "page_number": chunk.get("page_number"),
        "paragraph_number": chunk.get("paragraph_number"),
        "document_type": chunk.get("document_type"),
        "document_hash": chunk.get("document_hash"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def document_exists(collection, document_hash):
    result = collection.get(where={"document_hash": document_hash}, limit=1)
    return bool(result["ids"])


def delete_document(collection, document_hash):
    collection.delete(where={"document_hash": document_hash})


def list_indexed_documents(collection):
    result = collection.get(include=["metadatas"])
    documents = {}

    for metadata in result.get("metadatas", []):
        if not metadata:
            continue

        document_hash = metadata.get("document_hash")
        if not document_hash:
            continue

        document = documents.setdefault(
            document_hash,
            {
                "document_hash": document_hash,
                "file_name": metadata.get("file_name", "Bilinmeyen dosya"),
                "document_type": metadata.get("document_type"),
                "chunk_count": 0,
                "skipped": False,
            },
        )
        document["chunk_count"] += 1

    return sorted(documents.values(), key=lambda item: item["file_name"])


def similarity_search(collection, query_embedding, top_k=5):
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
