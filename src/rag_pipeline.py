import os
import re
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

from .chunking import split_text
from .embeddings import EmbeddingModel
from .generation import query_llm
from .loaders import extract_text
from .vector_store import (
    add_documents,
    create_chroma_client,
    delete_document,
    document_exists,
    list_indexed_documents,
    similarity_search,
)
from .audience_guidance import (
    build_audience_guidance,
    build_role_guidance,
)


DEFAULT_CHUNK_SETTINGS = {
    "chunk_size": 300,
    "chunk_overlap": 30,
}
CHUNK_SETTINGS_BY_DOCUMENT_TYPE = {
    "pdf": {
        "chunk_size": 800,
        "chunk_overlap": 100,
    },
}


class RagPipeline:
    def __init__(
        self,
        chroma_dir=None,
        collection_name="rag_documents",
        embedding_model_name=None,
        ollama_model=None,
        chunk_size=DEFAULT_CHUNK_SETTINGS["chunk_size"],
        chunk_overlap=DEFAULT_CHUNK_SETTINGS["chunk_overlap"],
    ):
        load_dotenv()

        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "chroma_db")
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.llm_provider = os.getenv("LLM_PROVIDER", "ollama")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL")
        self.openrouter_fallback_model = os.getenv(
            "OPENROUTER_FALLBACK_MODEL",
            "openrouter/free",
        )
        self.answer_max_tokens = self._env_int("RAG_ANSWER_MAX_TOKENS", 1024)
        self.source_max_chars = self._env_int("RAG_SOURCE_MAX_CHARS", 0)
        self.query_embedding_cache_size = self._env_int("RAG_QUERY_CACHE_SIZE", 128)
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = create_chroma_client(self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        self.embedding_model = EmbeddingModel(self.embedding_model_name)
        self.query_embedding_cache = {}

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

        chunk_settings = self.chunk_settings_for_documents(documents)
        chunks = split_text(documents, **chunk_settings)
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

    def chunk_settings_for_documents(self, documents):
        document_type = self._document_type_for_chunking(documents)
        type_settings = CHUNK_SETTINGS_BY_DOCUMENT_TYPE.get(document_type)
        if type_settings:
            return type_settings.copy()

        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    @staticmethod
    def _document_type_for_chunking(documents):
        for document in documents:
            document_type = document.get("document_type")
            if document_type:
                return str(document_type).casefold()
        return None

    def delete_document(self, document_hash):
        delete_document(self.collection, document_hash)

    def list_indexed_documents(self):
        return list_indexed_documents(self.collection)

    def search(self, question, top_k=5):
        query_embedding = self.embed_query_cached(question)
        candidate_count = self._search_candidate_count(top_k)
        results = similarity_search(self.collection, query_embedding, candidate_count)
        formatted_results = self.format_search_results(results)
        reranked_results = self.rerank_search_results(question, formatted_results)
        return self.diversify_search_results(reranked_results, top_k)

    def embed_query_cached(self, question):
        cache_key = str(question).strip().casefold()
        if not cache_key:
            return self.embedding_model.embed_query(question)

        cache = getattr(self, "query_embedding_cache", None)
        if cache is None:
            cache = {}
            self.query_embedding_cache = cache

        if cache_key in cache:
            return cache[cache_key]

        embedding = self.embedding_model.embed_query(question)
        cache_size = getattr(self, "query_embedding_cache_size", 128)
        if cache_size > 0:
            if len(cache) >= cache_size:
                oldest_key = next(iter(cache))
                cache.pop(oldest_key, None)
            cache[cache_key] = embedding
        return embedding

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

    def build_rag_prompt(self, question, results, user_profile=None, chat_history=None):
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
11. Use the exact citation format [Kaynak 1], [Kaynak 2], etc. Do not write citations without square brackets.
12. Conversation context may be used only to understand references in the current question. Do not treat conversation context as source evidence.
13. Fit the answer inside the response token budget. If the available context is broad, summarize instead of starting a long answer that may be cut off.
14. Finish with a complete sentence. Do not leave a numbered list, bullet, or paragraph unfinished.

The goal is to provide accurate, grounded, and traceable answers based strictly on the retrieved documents."""
        audience_guidance = self.build_audience_guidance(user_profile)
        audience_section = ""
        if audience_guidance:
            audience_section = f"""

Audience adaptation:
{audience_guidance}"""
        history_section = self.build_chat_history_section(chat_history)
        source_blocks = []

        for index, result in enumerate(results, start=1):
            source_text = self.truncate_source_text(result["text"])
            source_blocks.append(
                f"""[Kaynak {index}]
Dosya: {result["file_name"]}
Metin:
{source_text}"""
            )

        sources_text = "\n\n".join(source_blocks)

        return f"""{system_prompt}{audience_section}{history_section}

Soru:
{question}

Kaynaklar:
{sources_text}
"""

    def answer_query(self, question, top_k=5, user_profile=None, chat_history=None):
        search_results = self.search(question, top_k)
        if not search_results:
            return "Bu bilgi verilen belgelerde bulunamadı.", []

        prompt = self.build_rag_prompt(
            question,
            search_results,
            user_profile,
            chat_history,
        )

        answer = self._query_llm(prompt)
        all_citations = self.build_citations(search_results)
        answer = self.normalize_citation_format(answer)
        answer = self.filter_invalid_citations(answer, len(all_citations))
        used_source_numbers = self.extract_citation_numbers(answer, len(all_citations))
        citations = self.select_used_citations(all_citations, used_source_numbers)
        return answer, citations

    @staticmethod
    def build_audience_guidance(user_profile):
        return build_audience_guidance(user_profile)

    @staticmethod
    def build_role_guidance(role):
        return build_role_guidance(role)

    @staticmethod
    def build_chat_history_section(chat_history):
        if not chat_history:
            return ""

        lines = []
        for message in chat_history:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role == "user":
                label = "Kullanıcı"
            elif role == "assistant":
                label = "Asistan"
            else:
                continue
            if content:
                lines.append(f"{label}: {content}")

        if not lines:
            return ""

        return f"""

Konuşma bağlamı:
{chr(10).join(lines)}"""

    def _query_llm(self, prompt):
        return query_llm(
            provider=self.llm_provider,
            prompt=prompt,
            max_tokens=getattr(self, "answer_max_tokens", 512),
            openrouter_api_key=self.openrouter_api_key,
            openrouter_model=self.openrouter_model,
            openrouter_fallback_model=self.openrouter_fallback_model,
            ollama_base_url=self.ollama_base_url,
            ollama_model=self.ollama_model,
            ollama_keep_alive=getattr(self, "ollama_keep_alive", "10m"),
        )

    @staticmethod
    def _env_int(name, default):
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default

        return value if value > 0 else default

    def truncate_source_text(self, text):
        max_chars = getattr(self, "source_max_chars", 0)
        if not max_chars or len(text) <= max_chars:
            return text

        shortened = text[:max_chars].rsplit(maxsplit=1)[0].rstrip()
        if not shortened:
            shortened = text[:max_chars].rstrip()
        return f"{shortened}..."

    @staticmethod
    def build_citations(results):
        citations = []

        for index, result in enumerate(results, start=1):
            file_name = result.get("file_name") or "Bilinmeyen dosya"
            location_parts = []

            if result.get("page_number") is not None:
                location_parts.append(f"sayfa {result['page_number']}")
            if result.get("paragraph_number") is not None:
                location_parts.append(f"paragraf {result['paragraph_number']}")

            location = ", ".join(location_parts)
            if location:
                citations.append(f"Kaynak {index}: {file_name}, {location}")
            else:
                citations.append(f"Kaynak {index}: {file_name}")

        return citations

    @staticmethod
    def normalize_citation_format(answer):
        return re.sub(
            r"(?<!\[)\bKaynak\s+(\d+)\b(?!\])",
            r"[Kaynak \1]",
            answer,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def filter_invalid_citations(answer, source_count):
        def replace_invalid(match):
            source_number = int(match.group(1))
            if 1 <= source_number <= source_count:
                return f"[Kaynak {source_number}]"
            return ""

        return re.sub(
            r"\[Kaynak\s+(\d+)\]",
            replace_invalid,
            answer,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def extract_citation_numbers(answer, source_count):
        source_numbers = []
        seen = set()

        for match in re.finditer(r"\[Kaynak\s+(\d+)\]", answer, flags=re.IGNORECASE):
            source_number = int(match.group(1))
            if not 1 <= source_number <= source_count:
                continue
            if source_number in seen:
                continue
            seen.add(source_number)
            source_numbers.append(source_number)

        return source_numbers

    @staticmethod
    def select_used_citations(citations, source_numbers):
        return [
            citations[source_number - 1]
            for source_number in source_numbers
            if 1 <= source_number <= len(citations)
        ]

    @staticmethod
    def diversify_search_results(results, top_k):
        selected_results = []
        selected_chunk_ids = set()
        seen_files = set()

        for result in results:
            file_name = result.get("file_name")
            if file_name in seen_files:
                continue
            selected_results.append(result)
            selected_chunk_ids.add(result.get("chunk_id"))
            seen_files.add(file_name)
            if len(selected_results) == top_k:
                return selected_results

        for result in results:
            chunk_id = result.get("chunk_id")
            if chunk_id in selected_chunk_ids:
                continue
            selected_results.append(result)
            selected_chunk_ids.add(chunk_id)
            if len(selected_results) == top_k:
                break

        return selected_results

    def _search_candidate_count(self, top_k):
        try:
            collection_count = self.collection.count()
        except (AttributeError, TypeError):
            return top_k

        if collection_count <= 0:
            return top_k

        return min(collection_count, max(top_k, top_k * 4, 12))

    @classmethod
    def rerank_search_results(cls, question, results):
        question_tokens = cls._meaningful_tokens(question)
        if not question_tokens:
            return results

        scored_results = []
        for semantic_rank, result in enumerate(results):
            lexical_score = cls._lexical_score(question_tokens, result.get("text", ""))
            combined_score = semantic_rank - (lexical_score * 5)
            scored_results.append((combined_score, semantic_rank, result))

        return [
            result
            for _, _, result in sorted(scored_results, key=lambda item: (item[0], item[1]))
        ]

    @classmethod
    def _lexical_score(cls, question_tokens, text):
        text_tokens = cls._meaningful_tokens(text)
        if not text_tokens:
            return 0

        unique_question_tokens = set(question_tokens)
        unique_text_tokens = set(text_tokens)
        token_overlap = len(unique_question_tokens & unique_text_tokens) / len(
            unique_question_tokens
        )

        normalized_text = cls._normalize_for_matching(text)
        question_bigrams = list(zip(question_tokens, question_tokens[1:]))
        if not question_bigrams:
            return token_overlap

        phrase_hits = 0
        for first_token, second_token in question_bigrams:
            if f"{first_token} {second_token}" in normalized_text:
                phrase_hits += 1

        phrase_overlap = phrase_hits / len(question_bigrams)
        return token_overlap + phrase_overlap

    @staticmethod
    def _meaningful_tokens(text):
        stopwords = {
            "ama",
            "bir",
            "bu",
            "daha",
            "da",
            "de",
            "gibi",
            "hangi",
            "ile",
            "için",
            "mı",
            "mi",
            "mu",
            "mü",
            "nasıl",
            "ne",
            "nedir",
            "ve",
            "veya",
            "and",
            "are",
            "the",
            "or",
            "what",
            "which",
            "who",
        }
        return [
            token
            for token in re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", text.casefold())
            if len(token) > 2 and token not in stopwords
        ]

    @classmethod
    def _normalize_for_matching(cls, text):
        return " ".join(cls._meaningful_tokens(text))

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
