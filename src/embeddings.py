from sentence_transformers import SentenceTransformer
from typing import List, Any


class EmbeddingModelWrapper:
    """Wrap a SentenceTransformer to provide a consistent interface.

    Exposes `encode(texts)` and `embed_query(text)` methods used by the pipeline.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return arr.tolist()

    def embed_query(self, text: str):
        return self.encode([text])[0]


def get_embedding_model(model_name: str) -> Any:
    """Return an embedding model wrapper for the given model name."""
    return EmbeddingModelWrapper(model_name)


def embed_texts(embedding_model: Any, texts: List[str]):
    """Produce embeddings for a list of texts using the provided model wrapper."""
    if hasattr(embedding_model, "encode"):
        return embedding_model.encode(texts)
    # fallback: try raw SentenceTransformer API
    if hasattr(embedding_model, "model") and hasattr(embedding_model.model, "encode"):
        arr = embedding_model.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return arr.tolist()
    raise RuntimeError("Provided embedding_model does not support encoding")



