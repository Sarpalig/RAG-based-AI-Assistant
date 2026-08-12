from io import BytesIO
from pathlib import Path

import docx
import fitz


class DocumentLoaderError(Exception):
    """A custom exception for document loading failures."""


def _read_file_bytes(file_bytes):
    try:
        return file_bytes.read()
    except Exception as exc:
        raise DocumentLoaderError("Dosya okunurken bir hata oluştu.") from exc


def extract_pdf_text(file_bytes, file_name):
    content = _read_file_bytes(file_bytes)
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise DocumentLoaderError(f"PDF dosyası açılamıyor: {file_name}") from exc

    pages = []
    for page_number in range(len(document)):
        try:
            page = document.load_page(page_number)
            text = page.get_text()
        except Exception as exc:
            raise DocumentLoaderError(
                f"PDF sayfa metni çıkarılamadı: {file_name}, sayfa {page_number + 1}"
            ) from exc
        if text and text.strip():
            pages.append(
                {
                    "text": text,
                    "file_name": file_name,
                    "page_number": page_number + 1,
                    "document_type": "pdf",
                }
            )
    return pages


def extract_docx_text(file_bytes, file_name):
    content = _read_file_bytes(file_bytes)
    try:
        document = docx.Document(BytesIO(content))
    except Exception as exc:
        raise DocumentLoaderError(f"DOCX dosyası açılamıyor: {file_name}") from exc

    paragraphs = []
    for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
        paragraph_text = paragraph.text.strip()
        if paragraph_text:
            paragraphs.append(
                {
                    "text": paragraph_text,
                    "file_name": file_name,
                    "document_type": "docx",
                    "paragraph_number": paragraph_number,
                }
            )
    return paragraphs


def extract_md_text(file_bytes, file_name):
    content = _read_file_bytes(file_bytes)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoaderError(
            f"Markdown dosyası UTF-8 olarak okunamadı: {file_name}"
        ) from exc

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    return [
        {
            "text": cleaned_text,
            "file_name": file_name,
            "document_type": "md",
        }
    ]


def extract_text(file, file_name):
    if not file_name:
        raise DocumentLoaderError("Dosya adı boş olamaz.")

    extension = Path(file_name).suffix.lower()
    if extension == ".pdf":
        return extract_pdf_text(file, file_name)
    if extension == ".docx":
        return extract_docx_text(file, file_name)
    if extension == ".md":
        return extract_md_text(file, file_name)

    raise DocumentLoaderError(f"Desteklenmeyen dosya uzantısı: {extension}")
