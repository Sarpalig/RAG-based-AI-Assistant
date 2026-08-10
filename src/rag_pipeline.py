import os

from dotenv import load_dotenv

from .chunking import split_text
from .embeddings import embed_texts, get_embedding_model
from .generation import query_openrouter
from .loaders import extract_text
from .vector_store import (
    add_documents,
    create_chroma_client,
    get_or_create_collection,
    similarity_search,
)


class RagPipeline:
    def __init__(self):
        load_dotenv()
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
        self.chroma_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        self.client = create_chroma_client(self.chroma_dir)
        self.collection = get_or_create_collection(self.client)
        self.embedding_model = get_embedding_model(self.embedding_model_name)

    def index_files(self, files):
        documents = []
        for file in files:
            documents.extend(extract_text(file, file.name))
        chunks = split_text(documents)
        embeddings = embed_texts(self.embedding_model, [chunk["text"] for chunk in chunks])
        add_documents(self.collection, chunks, embeddings)

    def answer_query(self, query, top_k=5):
        query_embedding = self.embedding_model.embed_query(query)
        search_result = similarity_search(self.collection, query_embedding, top_k=top_k)
        retrieved_texts = search_result["documents"][0]
        retrieved_metadatas = search_result["metadatas"][0]
        citations = [
            f"{meta['file_name']} - sayfa {meta.get('page_number', 'N/A')}"
            for meta in retrieved_metadatas
        ]
        prompt = self.build_rag_prompt(query, retrieved_texts, citations)
        answer = query_openrouter(self.openrouter_api_key, self.openrouter_model, prompt)
        return answer, citations

    @staticmethod
    def build_rag_prompt(query, contexts, citations):
        context_blocks = []
        for index, (text, citation) in enumerate(zip(contexts, citations), start=1):
            context_blocks.append(f"[{index}] {citation}\n{text}")
        return (
            "Aşağıdaki kaynaklara dayanarak soruyu cevapla. "
            "Yanıtını sadece verilen parçalardan üret. Eğer bilgi yoksa 'bilgi bulunamadı' de.\n\n"
            f"Soru: {query}\n\n"
            "Kaynaklar:\n"
            + "\n\n".join(context_blocks)
        )
