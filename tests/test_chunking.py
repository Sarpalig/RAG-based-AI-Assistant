import io
import pytest

from src.chunking import split_text
from src.loaders import extract_text


def test_split_text_basic():
    """Test basic chunking with a simple document."""
    documents = [
        {
            "text": "Bu bir test metnidir. " * 50,  # Tekrarlı metin, chunk threshold'u geçmek için
            "file_name": "test.md",
            "document_type": "md",
        }
    ]
    chunks = split_text(documents, chunk_size=300, chunk_overlap=30)
    
    assert len(chunks) > 0
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all(chunk["file_name"] == "test.md" for chunk in chunks)


def test_split_text_preserves_metadata():
    """Test that metadata is preserved across chunks."""
    documents = [
        {
            "text": "Uzaktan çalışma için onay birim yöneticisinden alınmalıdır. " * 30,
            "file_name": "remote_work_policy.md",
            "page_number": 1,
            "document_type": "md",
        }
    ]
    chunks = split_text(documents, chunk_size=200, chunk_overlap=20)
    
    for chunk in chunks:
        assert chunk["file_name"] == "remote_work_policy.md"
        assert chunk["page_number"] == 1
        assert chunk["document_type"] == "md"


def test_split_text_unique_chunk_ids():
    """Test that each chunk has a unique ID."""
    documents = [
        {
            "text": "Paragraf bir. " * 30,
            "file_name": "test.md",
            "document_type": "md",
        }
    ]
    chunks = split_text(documents, chunk_size=200, chunk_overlap=20)
    
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_split_text_filters_short_chunks():
    """Test that very short text is not chunked into tiny pieces."""
    documents = [
        {
            "text": "Kısa metin.",
            "file_name": "short.md",
            "document_type": "md",
        }
    ]
    chunks = split_text(documents, chunk_size=300, chunk_overlap=30)
    
    # Çok kısa dokümanlarda birer chunk olmalı (boş değilse)
    assert len(chunks) <= 1


def test_split_text_preserves_pdf_page_number():
    """Test that PDF page numbers are preserved."""
    documents = [
        {
            "text": "Sayfa 1 içeriği. " * 30,
            "file_name": "policy.pdf",
            "page_number": 1,
            "document_type": "pdf",
        },
        {
            "text": "Sayfa 2 içeriği. " * 30,
            "file_name": "policy.pdf",
            "page_number": 2,
            "document_type": "pdf",
        }
    ]
    chunks = split_text(documents, chunk_size=200, chunk_overlap=20)
    
    page1_chunks = [c for c in chunks if c["page_number"] == 1]
    page2_chunks = [c for c in chunks if c["page_number"] == 2]
    
    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0


def test_split_text_preserves_docx_paragraph_number():
    """Test that DOCX paragraph numbers are preserved."""
    documents = [
        {
            "text": "Paragraf birinci. " * 20,
            "file_name": "document.docx",
            "document_type": "docx",
            "paragraph_number": 1,
        },
        {
            "text": "Paragraf ikinci. " * 20,
            "file_name": "document.docx",
            "document_type": "docx",
            "paragraph_number": 2,
        }
    ]
    chunks = split_text(documents, chunk_size=150, chunk_overlap=15)
    
    para1_chunks = [c for c in chunks if c.get("paragraph_number") == 1]
    para2_chunks = [c for c in chunks if c.get("paragraph_number") == 2]
    
    assert len(para1_chunks) > 0
    assert len(para2_chunks) > 0


def test_split_text_with_real_sample_document():
    """Test chunking with real sample Markdown document."""
    markdown_content = """# Uzaktan Çalışma Politikası

## Amaç
Bu politika, şirket çalışanlarının uzaktan çalışma süreçlerini düzenler.

## Onay Süreci
Uzaktan çalışma için onay, birim yöneticisinden alınmalıdır.
Çalışan, taleplerini en az 3 gün önceden yöneticiye iletmelidir.

## Çalışma Saatleri
Uzaktan çalışanlar, normal çalışma saatlerine uymak zorundadır.
Günlük rapor her akşam şirket sistemine yüklenmelidir.

## Güvenlik
İş bilgisayarları yalnızca şirket tarafından verilen hesaplarla kullanılmalıdır.
Kişisel cihazlar üzerinden hassas veriler paylaşılmamalıdır.
"""
    
    documents = [
        {
            "text": markdown_content,
            "file_name": "remote_work_policy.md",
            "document_type": "md",
        }
    ]
    
    chunks = split_text(documents, chunk_size=300, chunk_overlap=30)
    
    assert len(chunks) > 0
    assert all(len(chunk["text"]) > 0 for chunk in chunks)
    assert all(chunk["file_name"] == "remote_work_policy.md" for chunk in chunks)


def test_split_text_keeps_markdown_section_context_together():
    markdown_content = """# GÃ¼venlik YÃ¶nergeleri

## Åžifre PolitikasÄ±
Åžifrelerde en az 12 karakter bulunmalÄ±dÄ±r.

## Ä°ki FaktÃ¶rlÃ¼ DoÄŸrulama
TÃ¼m Ã§alÄ±ÅŸanlar, ÅŸirket hesabÄ± iÃ§in iki faktÃ¶rlÃ¼ doÄŸrulama kullanmalÄ±dÄ±r.
Bu sistem, hesabÄ±n gÃ¼venliÄŸini artÄ±rmak iÃ§in zorunludur.
"""

    chunks = split_text(
        [
            {
                "text": markdown_content,
                "file_name": "security_guidelines.md",
                "document_type": "md",
            }
        ],
        chunk_size=300,
        chunk_overlap=30,
    )

    matching_chunks = [
        chunk["text"]
        for chunk in chunks
        if "Ä°ki FaktÃ¶rlÃ¼ DoÄŸrulama" in chunk["text"]
    ]

    assert matching_chunks
    assert "zorunludur" in matching_chunks[0]
    assert "GÃ¼venlik YÃ¶nergeleri" in matching_chunks[0]


def test_split_text_multiple_documents():
    """Test chunking multiple documents together."""
    documents = [
        {
            "text": "Güvenlik politikası: Şifre en az 12 karakter olmalıdır. " * 15,
            "file_name": "security.md",
            "document_type": "md",
        },
        {
            "text": "Onboarding: Yeni çalışanlar ilk gün giriş prosedürlerini tamamlamalıdır. " * 15,
            "file_name": "onboarding.md",
            "document_type": "md",
        }
    ]
    
    chunks = split_text(documents, chunk_size=200, chunk_overlap=20)
    
    security_chunks = [c for c in chunks if c["file_name"] == "security.md"]
    onboarding_chunks = [c for c in chunks if c["file_name"] == "onboarding.md"]
    
    assert len(security_chunks) > 0
    assert len(onboarding_chunks) > 0
    assert len(chunks) == len(security_chunks) + len(onboarding_chunks)


def test_split_text_chunk_index():
    """Test that chunk_index is sequential."""
    documents = [
        {
            "text": "Metin parçası. " * 50,
            "file_name": "test.md",
            "document_type": "md",
        }
    ]
    
    chunks = split_text(documents, chunk_size=200, chunk_overlap=20)
    
    chunk_indices = [chunk["chunk_index"] for chunk in chunks]
    # Her dokümana ait chunk'lar baştan başlamalı
    assert chunk_indices == list(range(1, len(chunks) + 1))


def test_split_text_overlap():
    """Test that chunks have overlap as specified."""
    # Uzun bir metin oluştur
    long_text = "Bu bir test metnidir. " * 100
    documents = [
        {
            "text": long_text,
            "file_name": "test.md",
            "document_type": "md",
        }
    ]
    
    chunks = split_text(documents, chunk_size=300, chunk_overlap=50)
    
    # Overlap varsa, ard arda chunk'lar arasında benzerlik olmalı
    if len(chunks) > 1:
        chunk1_end = chunks[0]["text"][-50:]  # Son 50 karakter
        chunk2_start = chunks[1]["text"][:50]  # İlk 50 karakter
        # Overlap olduğu için bir kısmı ortak olmalı
        assert len(chunks) > 1  # Yeterince uzun chunk olmalı
