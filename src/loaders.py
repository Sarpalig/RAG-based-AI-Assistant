from pathlib import Path

import docx
import fitz


def extract_pdf_text(file_bytes, file_name):
    document = fitz.open(stream=file_bytes.read(), filetype="pdf")
    pages = []
    for page_number in range(len(document)):
        page = document.load_page(page_number)
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "file_name": file_name,
                "page_number": page_number + 1,
                "document_type": "pdf",
            })
    return pages


def extract_docx_text(file_bytes, file_name):
    document = docx.Document(file_bytes)
    paragraphs = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(
                {
                    "text": paragraph.text,
                    "file_name": file_name,
                    "document_type": "docx",
                }
            )
    return paragraphs


def extract_md_text(file_bytes, file_name):
    text = file_bytes.read().decode("utf-8")
    return [
        {
            "text": text,
            "file_name": file_name,
            "document_type": "md",
        }
    ]


def extract_text(file, file_name):
    extension = Path(file_name).suffix.lower()
    if extension == ".pdf":
        return extract_pdf_text(file, file_name)
    if extension == ".docx":
        return extract_docx_text(file, file_name)
    if extension == ".md":
        return extract_md_text(file, file_name)
    raise ValueError(f"Desteklenmeyen dosya uzantısı: {extension}")
