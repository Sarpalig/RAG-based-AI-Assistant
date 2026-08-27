import re
from uuid import uuid4

from langchain_text_splitters.character import RecursiveCharacterTextSplitter


def split_text(documents, chunk_size=300, chunk_overlap=30):
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

        split_texts = []
        for segment in _document_segments(document):
            split_texts.extend(splitter.split_text(segment))

        for index, chunk_text in enumerate(split_texts, start=1):
            chunk = {
                "chunk_id": str(uuid4()),
                "text": chunk_text,
                "source_text": text,
                "chunk_index": index,
            }

            for key, value in document.items():
                if key != "text":
                    chunk[key] = value
            chunks.append(chunk)

    return chunks


def _document_segments(document):
    text = document.get("text", "")
    if document.get("document_type") != "md":
        return [text]

    return _markdown_sections(text)


def _markdown_sections(text):
    stripped_text = text.strip()
    if not stripped_text:
        return []

    parts = re.split(r"(?m)(?=^##\s+)", stripped_text)
    if len(parts) == 1:
        return [stripped_text]

    title = parts[0].strip()
    sections = []
    for part in parts[1:]:
        section = part.strip()
        if not section:
            continue
        if title:
            sections.append(f"{title}\n\n{section}")
        else:
            sections.append(section)

    return sections or [stripped_text]
