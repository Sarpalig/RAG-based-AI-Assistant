from uuid import uuid4

from langchain_text_splitters.character import RecursiveCharacterTextSplitter


def split_text(documents, chunk_size=300, chunk_overlap=30):
    """Split extracted documents into overlapping chunks for retrieval.

    Args:
        documents (list[dict]): Extracted document pieces, each containing at least
            a "text" field and source metadata like file_name/page_number.
        chunk_size (int): Maximum tokens/characters per chunk.
        chunk_overlap (int): Overlap size between neighboring chunks.

    Returns:
        list[dict]: Chunk objects with text, metadata, and chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for document in documents:
        text = document.get("text", "")
        if not text or not text.strip():
            continue

        split_texts = splitter.split_text(text)
        for index, chunk_text in enumerate(split_texts, start=1):
            # Start with all document metadata
            chunk = {
                "chunk_id": str(uuid4()),
                "text": chunk_text,
                "source_text": text,
                "chunk_index": index,
            }
            # Preserve all document metadata fields
            for key, value in document.items():
                if key != "text":
                    chunk[key] = value
            chunks.append(chunk)

    return chunks
