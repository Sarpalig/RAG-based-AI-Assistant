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
