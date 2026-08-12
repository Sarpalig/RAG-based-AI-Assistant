import io
import json
from pathlib import Path

import pytest

from src.loaders import DocumentLoaderError, extract_text


def test_extract_md_text_success():
    file_bytes = io.BytesIO("# Başlık\nİçerik testi." .encode("utf-8"))
    results = extract_text(file_bytes, "test.md")
    assert len(results) == 1
    assert results[0]["document_type"] == "md"
    assert "Başlık" in results[0]["text"]


def test_extract_docx_text_success():
    from docx import Document

    document = Document()
    document.add_paragraph("Bir paragraf.")
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)

    results = extract_text(buffer, "test.docx")
    assert len(results) == 1
    assert results[0]["document_type"] == "docx"
    assert results[0]["paragraph_number"] == 1
    assert results[0]["text"] == "Bir paragraf."


def test_extract_pdf_text_success():
    import fitz

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF test içeriği.")
    buffer = io.BytesIO()
    pdf.save(buffer)
    buffer.seek(0)

    results = extract_text(buffer, "test.pdf")
    assert len(results) == 1
    assert results[0]["document_type"] == "pdf"
    assert results[0]["page_number"] == 1
    assert "PDF test" in results[0]["text"]
    assert "i" in results[0]["text"]


def test_extract_text_unsupported_extension_raises():
    file_bytes = io.BytesIO(b"dummy")
    with pytest.raises(DocumentLoaderError) as exc_info:
        extract_text(file_bytes, "test.txt")
    assert "Desteklenmeyen dosya uzantısı" in str(exc_info.value)


def test_extract_md_text_invalid_utf8_raises():
    file_bytes = io.BytesIO(b"\xff\xfe\xfd")
    with pytest.raises(DocumentLoaderError) as exc_info:
        extract_text(file_bytes, "test.md")
    assert "UTF-8" in str(exc_info.value)


def test_extract_docx_text_corrupt_raises():
    file_bytes = io.BytesIO(b"not-a-docx")
    with pytest.raises(DocumentLoaderError) as exc_info:
        extract_text(file_bytes, "test.docx")
    assert "DOCX dosyası açılamıyor" in str(exc_info.value)


def test_extract_pdf_text_corrupt_raises():
    file_bytes = io.BytesIO(b"not-a-pdf")
    with pytest.raises(DocumentLoaderError) as exc_info:
        extract_text(file_bytes, "test.pdf")
    assert "PDF dosyası açılamıyor" in str(exc_info.value)
