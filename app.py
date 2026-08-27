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


def load_pipeline(show_status=False):
    if not show_status:
        pipeline = get_pipeline()
        if not hasattr(pipeline, "list_indexed_documents"):
            get_pipeline.clear()
            pipeline = get_pipeline()
        return pipeline

    with st.status("RAG pipeline hazirlaniyor...", expanded=False) as status:
        status.write("Embedding modeli ve vector store yukleniyor.")
        pipeline = get_pipeline()
        if not hasattr(pipeline, "list_indexed_documents"):
            get_pipeline.clear()
            pipeline = get_pipeline()
        status.update(label="RAG pipeline hazir.", state="complete")

    return pipeline


def initialize_session_state():
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = {}
    if "indexed_documents_loaded" not in st.session_state:
        st.session_state.indexed_documents_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


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
        st.warning(f"Yuklu belge listesi okunamadi: {exc}")


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


def document_status_label(document):
    if document["skipped"]:
        return "Zaten yuklu"
    if document["chunk_count"] > 0:
        return "Hazir"
    return "Metin bulunamadi"


def show_indexed_documents():
    if not st.session_state.indexed_documents:
        st.caption("Henuz yuklenen belge yok.")
        return

    for document in st.session_state.indexed_documents.values():
        status = document_status_label(document)
        st.markdown(f"**{document['file_name']}**")
        st.caption(f"{status} - {document['chunk_count']} parca")


def index_uploaded_files(uploaded_files):
    if not uploaded_files:
        st.warning("Once en az bir belge yukleyin.")
        return

    try:
        rag = load_pipeline(show_status=True)
    except Exception as exc:
        st.error(f"Pipeline baslatilirken hata olustu: {exc}")
        return

    with st.spinner("Belgeler hazirlaniyor..."):
        for uploaded_file in uploaded_files:
            try:
                result = rag.index_file(uploaded_file, uploaded_file.name)
                remember_index_result(result)

                if result["skipped"]:
                    st.info(f"{result['file_name']} zaten yuklu.")
                elif result["chunk_count"] > 0:
                    st.success(
                        f"{result['file_name']} hazirlandi "
                        f"({result['chunk_count']} parca)."
                    )
                else:
                    st.warning(
                        f"{result['file_name']} icinde okunabilir metin bulunamadi."
                    )
            except DocumentLoaderError as exc:
                st.error(f"{uploaded_file.name}: {exc}")
            except Exception as exc:
                st.error(f"{uploaded_file.name}: Belge hazirlanirken hata olustu: {exc}")

    refresh_indexed_documents(rag)


def refresh_document_list():
    try:
        rag = load_pipeline(show_status=True)
        refresh_indexed_documents(rag)
        st.success("Yuklenen belge listesi yenilendi.")
    except Exception as exc:
        st.error(f"Belge listesi yenilenirken hata olustu: {exc}")


def answer_question(query):
    if not query.strip():
        raise ValueError("Once bir soru yazin.")

    try:
        rag = load_pipeline()
        with st.spinner("Yaziyor..."):
            return rag.answer_query(query)
    except LLMError as exc:
        raise LLMError(f"LLM hatasi: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Soru cevaplanirken hata olustu: {exc}") from exc


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations") or []
        if citations:
            with st.expander("Kaynaklar", expanded=False):
                for citation in citations:
                    st.write(f"- {citation}")


def render_chat():
    if not st.session_state.messages:
        st.info(
            "Belgeler hazirsa asagidan soru sorabilirsin. "
            "Cevaplar sadece indekslenen dokuman parcalarina dayanir."
        )

    for message in st.session_state.messages:
        render_message(message)


def render_sidebar():
    with st.sidebar:
        st.header("Documents")
        st.caption("PDF, DOCX veya Markdown dosyalarini indeksle.")

        uploaded_files = st.file_uploader(
            "Dosya yukle",
            type=["pdf", "docx", "md"],
            accept_multiple_files=True,
        )

        if st.button("Belgeleri hazirla", type="primary", use_container_width=True):
            index_uploaded_files(uploaded_files)

        if st.button("Listeyi yenile", use_container_width=True):
            refresh_document_list()

        st.divider()
        st.subheader("Indekslenen belgeler")
        show_indexed_documents()

        st.divider()
        if st.button("Sohbeti temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_assistant_response(query):
    with st.chat_message("assistant"):
        try:
            answer, citations = answer_question(query)
            st.markdown(answer)
            if citations:
                with st.expander("Kaynaklar", expanded=True):
                    for citation in citations:
                        st.write(f"- {citation}")
            else:
                st.info("Bu cevap icin gosterilecek kaynak bulunamadi.")
        except Exception as exc:
            answer = str(exc)
            citations = []
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
        }
    )


def main():
    st.set_page_config(page_title="Kurumsal Dokuman Asistani", layout="wide")
    initialize_session_state()
    load_indexed_documents_on_startup()

    render_sidebar()

    st.title("Kurumsal Dokuman Asistani")
    st.caption("Kaynakli cevap ureten RAG tabanli dokuman sohbeti.")

    render_chat()

    query = st.chat_input("Dokumanlar hakkinda soru sor")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        render_assistant_response(query)


if __name__ == "__main__":
    main()
