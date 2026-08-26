import os

import streamlit as st
from dotenv import load_dotenv

from src.generation import LLMError
from src.loaders import DocumentLoaderError
from src.rag_pipeline import RagPipeline
from src.vector_store import create_chroma_client, list_indexed_documents


DEFAULT_COLLECTION_NAME = "rag_documents"


@st.cache_resource
def get_pipeline():
    return RagPipeline()


def load_pipeline():
    progress = st.progress(0)
    status = st.empty()

    status.info("RAG pipeline başlıyor...")
    progress.progress(20)

    status.info("Embedding modeli ve vector store hazırlanıyor...")
    pipeline = get_pipeline()
    if not hasattr(pipeline, "list_indexed_documents"):
        get_pipeline.clear()
        pipeline = get_pipeline()

    progress.progress(100)
    status.success("RAG pipeline tamamlandı.")

    return pipeline


def initialize_session_state():
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = {}
    if "indexed_documents_loaded" not in st.session_state:
        st.session_state.indexed_documents_loaded = False


def load_indexed_documents_on_startup():
    if st.session_state.indexed_documents_loaded:
        return

    try:
        load_dotenv()
        chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
        client = create_chroma_client(chroma_dir)
        collection = client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
        documents = list_indexed_documents(collection)
        st.session_state.indexed_documents = {
            document["document_hash"]: document
            for document in documents
        }
        st.session_state.indexed_documents_loaded = True
    except Exception as exc:
        st.session_state.indexed_documents_loaded = True
        st.warning(f"Yüklü belge listesi okunamadı: {exc}")


def remember_index_result(result):
    document_hash = result.get("document_hash")
    if not document_hash:
        return

    st.session_state.indexed_documents[document_hash] = {
        "file_name": result.get("file_name", "Bilinmeyen dosya"),
        "chunk_count": result.get("chunk_count", 0),
        "skipped": result.get("skipped", False),
    }


def refresh_indexed_documents(rag):
    documents = rag.list_indexed_documents()
    st.session_state.indexed_documents = {
        document["document_hash"]: document
        for document in documents
    }


def show_indexed_documents():
    st.write("#### Yüklenen belgeler")

    if not st.session_state.indexed_documents:
        st.caption("Henüz yüklenen belge yok.")
        return

    rows = []
    for document in st.session_state.indexed_documents.values():
        if document["skipped"]:
            status = "Zaten yüklü"
        elif document["chunk_count"] > 0:
            status = "Hazır"
        else:
            status = "Metin bulunamadı"

        rows.append(
            {
                "Dosya": document["file_name"],
                "Parça sayısı": document["chunk_count"],
                "Durum": status,
            }
        )

    st.dataframe(rows, hide_index=True, width="stretch")


def index_uploaded_files(uploaded_files):
    if not uploaded_files:
        st.warning("Önce en az bir belge yükleyin.")
        return

    try:
        rag = load_pipeline()
    except Exception as exc:
        st.error(f"Pipeline başlatılırken hata oluştu: {exc}")
        return

    with st.spinner("Belgeler hazırlanıyor..."):
        for uploaded_file in uploaded_files:
            try:
                result = rag.index_file(uploaded_file, uploaded_file.name)
                remember_index_result(result)

                if result["skipped"]:
                    st.info(f"{result['file_name']} zaten yüklü.")
                elif result["chunk_count"] > 0:
                    st.success(
                        f"{result['file_name']} hazırlandı "
                        f"({result['chunk_count']} parça)."
                    )
                else:
                    st.warning(f"{result['file_name']} icinde okunabilir metin bulunamadi.")
            except DocumentLoaderError as exc:
                st.error(f"{uploaded_file.name}: {exc}")
            except Exception as exc:
                st.error(f"{uploaded_file.name}: Belge hazırlanırken hata oluştu: {exc}")

    refresh_indexed_documents(rag)


def refresh_document_list():
    try:
        rag = load_pipeline()
        refresh_indexed_documents(rag)
        st.success("Yüklenen belge listesi yenilendi.")
    except Exception as exc:
        st.error(f"Belge listesi yenilenirken hata oluştu: {exc}")


def answer_question(query):
    if not query.strip():
        st.warning("Once bir soru yazin.")
        return

    try:
        rag = load_pipeline()
        with st.spinner("Cevap hazırlanıyor..."):
            answer, citations = rag.answer_query(query)
    except LLMError as exc:
        st.error(f"LLM hatası: {exc}")
        return
    except Exception as exc:
        st.error(f"Soru cevaplanırken hata oluştu: {exc}")
        return

    st.write("### Cevap")
    st.write(answer)

    if citations:
        with st.expander("Kaynaklar", expanded=True):
            for citation in citations:
                st.write(f"- {citation}")
    else:
        st.info("Bu cevap için gösterilecek kaynak bulunamadı.")


def main():
    st.set_page_config(page_title="Kurumsal Doküman Asistanı", layout="wide")
    initialize_session_state()
    load_indexed_documents_on_startup()

    st.title("Kurumsal Doküman Asistanı")
    st.info("PDF, DOCX ve Markdown belgelerinden kaynaklı yanıtlar üreten RAG uygulaması.")

    with st.expander("Belge yükleme", expanded=True):
        uploaded_files = st.file_uploader(
            "Belgeleri yükleyin",
            type=["pdf", "docx", "md"],
            accept_multiple_files=True,
        )
        if st.button("Belgeleri hazırla", type="primary"):
            index_uploaded_files(uploaded_files)

        if st.button("Listeyi yenile"):
            refresh_document_list()

        show_indexed_documents()

    query = st.text_input("Soru sor", value="")
    if st.button("Ara", type="primary"):
        answer_question(query)


if __name__ == "__main__":
    main()
