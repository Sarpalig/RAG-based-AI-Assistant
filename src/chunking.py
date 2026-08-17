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

        split_texts = splitter.split_text(text)
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
