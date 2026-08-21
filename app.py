import streamlit as st

from src.generation import OpenRouterError
from src.loaders import DocumentLoaderError
from src.rag_pipeline import RagPipeline


@st.cache_resource
def get_pipeline():
    return RagPipeline()


def load_pipeline():
    progress = st.progress(0)
    status = st.empty()

    status.info("RAG pipeline baslatiliyor...")
    progress.progress(20)

    status.info("Embedding modeli ve vector store hazirlaniyor...")
    pipeline = get_pipeline()
    if not hasattr(pipeline, "list_indexed_documents"):
        get_pipeline.clear()
        pipeline = get_pipeline()

    progress.progress(100)
    status.success("RAG pipeline hazir.")

    return pipeline


def initialize_session_state():
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = {}


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
    st.write("#### Indekslenen belgeler")

    if not st.session_state.indexed_documents:
        st.caption("Henuz indekslenen belge yok.")
        return

    rows = []
    for document in st.session_state.indexed_documents.values():
        if document["skipped"]:
            status = "Zaten indeksli"
        elif document["chunk_count"] > 0:
            status = "Indekslendi"
        else:
            status = "Metin bulunamadi"

        rows.append(
            {
                "Dosya": document["file_name"],
                "Chunk sayisi": document["chunk_count"],
                "Durum": status,
            }
        )

    st.dataframe(rows, hide_index=True, width="stretch")


def index_uploaded_files(uploaded_files):
    if not uploaded_files:
        st.warning("Once en az bir belge yukleyin.")
        return

    try:
        rag = load_pipeline()
    except Exception as exc:
        st.error(f"Pipeline baslatilirken hata olustu: {exc}")
        return

    with st.spinner("Belgeler indeksleniyor..."):
        for uploaded_file in uploaded_files:
            try:
                result = rag.index_file(uploaded_file, uploaded_file.name)
                remember_index_result(result)

                if result["skipped"]:
                    st.info(f"{result['file_name']} zaten indeksli.")
                elif result["chunk_count"] > 0:
                    st.success(
                        f"{result['file_name']} indekslendi "
                        f"({result['chunk_count']} chunk)."
                    )
                else:
                    st.warning(f"{result['file_name']} icinde okunabilir metin bulunamadi.")
            except DocumentLoaderError as exc:
                st.error(f"{uploaded_file.name}: {exc}")
            except Exception as exc:
                st.error(f"{uploaded_file.name}: Indeksleme sirasinda hata olustu: {exc}")

    refresh_indexed_documents(rag)


def refresh_document_list():
    try:
        rag = load_pipeline()
        refresh_indexed_documents(rag)
        st.success("Indekslenen belge listesi yenilendi.")
    except Exception as exc:
        st.error(f"Belge listesi yenilenirken hata olustu: {exc}")


def answer_question(query):
    if not query.strip():
        st.warning("Once bir soru yazin.")
        return

    try:
        rag = load_pipeline()
        with st.spinner("Cevap hazirlaniyor..."):
            answer, citations = rag.answer_query(query)
    except OpenRouterError as exc:
        st.error(f"LLM/API hatasi: {exc}")
        return
    except Exception as exc:
        st.error(f"Soru cevaplanirken hata olustu: {exc}")
        return

    st.write("### Cevap")
    st.write(answer)

    if citations:
        with st.expander("Kaynaklar", expanded=True):
            for citation in citations:
                st.write(f"- {citation}")
    else:
        st.info("Bu cevap icin gosterilecek kaynak bulunamadi.")


def main():
    st.set_page_config(page_title="Kurumsal Dokuman Asistani", layout="wide")
    initialize_session_state()

    st.title("Kurumsal Dokuman Asistani")
    st.info("PDF, DOCX ve Markdown belgelerinden kaynakli yanitlar ureten RAG uygulamasi.")

    with st.expander("Belge yukleme ve indeksleme", expanded=True):
        uploaded_files = st.file_uploader(
            "Belgeleri yukleyin",
            type=["pdf", "docx", "md"],
            accept_multiple_files=True,
        )
        if st.button("Dokumani isle", type="primary"):
            index_uploaded_files(uploaded_files)

        if st.button("Listeyi yenile"):
            refresh_document_list()

        show_indexed_documents()

    query = st.text_input("Soru sor", value="")
    if st.button("Ara", type="primary"):
        answer_question(query)


if __name__ == "__main__":
    main()
