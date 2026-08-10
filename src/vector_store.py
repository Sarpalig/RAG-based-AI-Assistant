import chromadb
from chromadb.config import Settings


def create_chroma_client(chroma_dir):
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))


def get_or_create_collection(client, collection_name="rag_documents"):
    if collection_name in [col.name for col in client.list_collections()]:
        return client.get_collection(collection_name)
    return client.create_collection(name=collection_name)


def add_documents(collection, chunks, embeddings):
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "file_name": chunk["file_name"],
            "page_number": chunk.get("page_number"),
            "document_type": chunk["document_type"],
        }
        for chunk in chunks
    ]
    documents = [chunk["text"] for chunk in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def similarity_search(collection, query_embedding, top_k=5):
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return result
