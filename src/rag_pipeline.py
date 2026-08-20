import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

from .chunking import split_text
from .embeddings import EmbeddingModel
from .generation import query_openrouter
from .loaders import extract_text
from .vector_store import (
    add_documents,
    create_chroma_client,
    delete_document,
    document_exists,
    similarity_search,
)


class RagPipeline:
    def __init__(
        self,
        chroma_dir=None,
        collection_name="rag_documents",
        embedding_model_name=None,
    ):
        load_dotenv()

        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "chroma_db")
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL")

        self.client = create_chroma_client(self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        self.embedding_model = EmbeddingModel(self.embedding_model_name)

    def index_file(self, file, file_name=None):
        file_content = self._read_file_content(file)
        file_name = file_name or getattr(file, "name", "uploaded_document")
        document_hash = self._hash_content(file_content)

        if document_exists(self.collection, document_hash):
            return {
                "file_name": file_name,
                "document_hash": document_hash,
                "chunk_count": 0,
                "skipped": True,
            }

        documents = extract_text(BytesIO(file_content), file_name)
        for document in documents:
            document["document_hash"] = document_hash

        chunks = split_text(documents)
        if not chunks:
            return {
                "file_name": file_name,
                "document_hash": document_hash,
                "chunk_count": 0,
                "skipped": False,
            }

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        add_documents(self.collection, chunks, embeddings)

        return {
            "file_name": file_name,
            "document_hash": document_hash,
            "chunk_count": len(chunks),
            "skipped": False,
        }

    def index_files(self, files):
        results = []
        for file in files:
            file_name = getattr(file, "name", None)
            if file_name:
                file_name = Path(file_name).name
            results.append(self.index_file(file, file_name))
        return results

    def delete_document(self, document_hash):
        delete_document(self.collection, document_hash)

    def search(self, question, top_k=5):
        query_embedding = self.embedding_model.embed_query(question)
        results = similarity_search(self.collection, query_embedding, top_k)
        return self.format_search_results(results)

    @staticmethod
    def format_search_results(results):
        formatted_results = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = distances[index] if index < len(distances) else None

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[index],
                    "file_name": metadata.get("file_name"),
                    "page_number": metadata.get("page_number"),
                    "paragraph_number": metadata.get("paragraph_number"),
                    "document_type": metadata.get("document_type"),
                    "distance": distance,
                }
            )

        return formatted_results

    def build_rag_prompt(self, question, results):
        system_prompt = """You are a retrieval-augmented generation (RAG) document assistant.

Your task is to answer the user's question using only the information provided in the retrieved context.

Rules:
1. Use only the provided context to answer the question.
2. Do not use outside knowledge, assumptions, or invented information.
3. If the context does not contain enough information to answer the question, clearly state that the answer could not be found in the provided documents.
4. If multiple context passages contain relevant information, combine them into one clear and coherent answer.
5. Do not mention information that is not supported by the context.
6. Preserve important technical terms, names, numbers, dates, and conditions exactly when relevant.
7. Answer in the same language as the user's question unless explicitly requested otherwise.
8. Keep the answer concise but complete.
9. When source information is available, cite the relevant source using the source identifiers provided in the context.
10. Never fabricate a source or citation.

The goal is to provide accurate, grounded, and traceable answers based strictly on the retrieved documents."""
        source_blocks = []

        for index, result in enumerate(results, start=1):
            source_blocks.append(
                f"""[Kaynak {index}]
Dosya: {result["file_name"]}
Metin:
{result["text"]}"""
            )

        sources_text = "\n\n".join(source_blocks)

        return f"""{system_prompt}

Soru:
{question}

Kaynaklar:
{sources_text}
"""

    def answer_query(self, question, top_k=5):
        search_results = self.search(question, top_k)
        if not search_results:
            return "Bu bilgi verilen belgelerde bulunamadı."

        prompt = self.build_rag_prompt(question, search_results)

        answer = query_openrouter(
            api_key=self.openrouter_api_key,
            model=self.openrouter_model,
            prompt=prompt,
            max_tokens=512,
        )
        return answer

    @staticmethod
    def _read_file_content(file):
        if isinstance(file, bytes):
            return file
        if hasattr(file, "seek"):
            file.seek(0)
        file_content = file.read()
        if isinstance(file_content, str):
            return file_content.encode("utf-8")
        return file_content

    @staticmethod
    def _hash_content(file_content):
        return sha256(file_content).hexdigest()
