from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Small wrapper around SentenceTransformer used by the RAG pipeline."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encode([text])[0]

