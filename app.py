import os

import streamlit as st
from dotenv import load_dotenv

from src.generation import LLMError
from src.loaders import DocumentLoaderError
from src.rag_pipeline import RagPipeline
from src.vector_store import create_chroma_client, list_indexed_documents


DEFAULT_COLLECTION_NAME = "rag_documents"
OLLAMA_MODEL_OPTIONS = ["qwen3.5:9b", "gemma4:e4b"]


@st.cache_resource
def get_pipeline(ollama_model):
    return RagPipeline(ollama_model=ollama_model)


def configured_ollama_model():
    load_dotenv()
    env_model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_OPTIONS[0])
    if env_model in OLLAMA_MODEL_OPTIONS:
        return env_model
    return OLLAMA_MODEL_OPTIONS[0]


def selected_ollama_model():
    return st.session_state.get("selected_ollama_model", configured_ollama_model())


def selected_page():
    return st.session_state.get("selected_page", "Sohbet")


def load_pipeline(show_status=False, ollama_model=None):
    model = ollama_model or selected_ollama_model()

    if not show_status:
        pipeline = get_pipeline(model)
        if not hasattr(pipeline, "list_indexed_documents"):
            get_pipeline.clear()
            pipeline = get_pipeline(model)
        return pipeline

    with st.status("RAG pipeline hazirlaniyor...", expanded=False) as status:
        status.write("Embedding modeli ve vector store yukleniyor.")
        pipeline = get_pipeline(model)
        if not hasattr(pipeline, "list_indexed_documents"):
            get_pipeline.clear()
            pipeline = get_pipeline(model)
        status.update(label="RAG pipeline hazir.", state="complete")

    return pipeline


def initialize_session_state():
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = {}
    if "indexed_documents_loaded" not in st.session_state:
        st.session_state.indexed_documents_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_ollama_model" not in st.session_state:
        st.session_state.selected_ollama_model = configured_ollama_model()
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Sohbet"
    if "document_notice" not in st.session_state:
        st.session_state.document_notice = None


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


def delete_indexed_document(document_hash):
    document = st.session_state.indexed_documents.get(document_hash)
    file_name = document.get("file_name", "Belge") if document else "Belge"

    try:
        rag = load_pipeline(show_status=True)
        rag.delete_document(document_hash)
        refresh_indexed_documents(rag)
        st.session_state.document_notice = {
            "type": "success",
            "message": f"{file_name} indeksten kaldirildi.",
        }
    except Exception as exc:
        st.error(f"{file_name} silinirken hata olustu: {exc}")


def answer_question(query, ollama_model=None):
    if not query.strip():
        raise ValueError("Once bir soru yazin.")

    try:
        rag = load_pipeline(ollama_model=ollama_model)
        with st.spinner("Yaziyor..."):
            return rag.answer_query(query)
    except LLMError as exc:
        raise LLMError(f"LLM hatasi: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Soru cevaplanirken hata olustu: {exc}") from exc


def render_citations(citations, expanded=False):
    if citations:
        with st.expander("Kaynaklar", expanded=expanded):
            for citation in citations:
                st.write(f"- {citation}")
    else:
        st.info("Bu cevap icin gosterilecek kaynak bulunamadi.")


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations") or []
        if citations:
            render_citations(citations, expanded=False)


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
        st.radio(
            "Ekran",
            options=["Sohbet", "Belgeler"],
            key="selected_page",
        )

        st.divider()
        st.header("Model")
        current_model = selected_ollama_model()
        selected_index = (
            OLLAMA_MODEL_OPTIONS.index(current_model)
            if current_model in OLLAMA_MODEL_OPTIONS
            else 0
        )
        st.selectbox(
            "Ollama modeli",
            options=OLLAMA_MODEL_OPTIONS,
            index=selected_index,
            key="selected_ollama_model",
        )
        st.caption("Model degisikligi sonraki cevaplarda kullanilir.")

        st.divider()
        if st.button("Sohbeti temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_document_upload():
    st.subheader("Belge ekle")
    uploaded_files = st.file_uploader(
        "PDF, DOCX veya Markdown dosyasi sec",
        type=["pdf", "docx", "md"],
        accept_multiple_files=True,
    )

    actions = st.columns([1, 1, 4])
    with actions[0]:
        if st.button("Belgeleri hazirla", type="primary", use_container_width=True):
            index_uploaded_files(uploaded_files)
    with actions[1]:
        if st.button("Listeyi yenile", use_container_width=True):
            refresh_document_list()


def render_document_list():
    st.subheader("Indekslenen belgeler")

    if not st.session_state.indexed_documents:
        st.info("Henuz indekslenen belge yok.")
        return

    header = st.columns([4, 1, 1, 1])
    header[0].markdown("**Dosya**")
    header[1].markdown("**Tur**")
    header[2].markdown("**Parca**")
    header[3].markdown("**Islem**")

    for document_hash, document in st.session_state.indexed_documents.items():
        row = st.columns([4, 1, 1, 1])
        row[0].write(document.get("file_name", "Bilinmeyen dosya"))
        row[1].write(document.get("document_type", "-"))
        row[2].write(document.get("chunk_count", 0))
        if row[3].button(
            "Sil",
            key=f"delete_{document_hash}",
            use_container_width=True,
        ):
            delete_indexed_document(document_hash)
            st.rerun()


def render_documents_page():
    st.title("Belgeler")
    st.caption("RAG indeksine belge ekle, mevcut belgeleri gor ve gerekmeyenleri kaldir.")
    notice = st.session_state.get("document_notice")
    if notice:
        if notice["type"] == "success":
            st.success(notice["message"])
        else:
            st.info(notice["message"])
        st.session_state.document_notice = None
    render_document_upload()
    st.divider()
    render_document_list()


def render_chat_page():
    st.title("Kurumsal Dokuman Asistani")
    st.caption("Kaynakli cevap ureten RAG tabanli dokuman sohbeti.")

    render_chat()

    query = st.chat_input("Dokumanlar hakkinda soru sor")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        render_assistant_response(query)


def render_assistant_response(query):
    with st.chat_message("assistant"):
        try:
            answer, citations = answer_question(query)
            st.markdown(answer)
            render_citations(citations, expanded=True)
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

    if selected_page() == "Belgeler":
        render_documents_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
