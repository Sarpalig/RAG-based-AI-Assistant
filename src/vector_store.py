import chromadb
from chromadb.config import Settings


def create_chroma_client(chroma_dir):
    # Use the newer Settings keys: set persistence and directory
    return chromadb.Client(Settings(is_persistent=True, persist_directory=chroma_dir))

def get_or_create_collection(client, collection_name="rag_documents"):
    if collection_name in [col.name for col in client.list_collections()]:
        return client.get_collection(collection_name)
    return client.create_collection(name=collection_name)

def add_documents(collection,chunks,embeddings):
    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "file_name": chunk["file_name"],
            "page_number": chunk.get("page_number"),
            "document_type": chunk.get("document_type"),
        }
        for chunk in chunks
    ]
    # embeddings must align with chunks order
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def similarity_search(collection, query_embedding, top_k=5):
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)