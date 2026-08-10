import streamlit as st

from src.rag_pipeline import RagPipeline


def main():
    st.set_page_config(page_title="Kurumsal Doküman Asistanı", layout="wide")
    st.title("Kurumsal Doküman Asistanı")

    st.info("PDF, DOCX ve Markdown belgelerinden yanıtlar üreten RAG uygulaması.")

    rag = RagPipeline()

    with st.expander("Belge yükleme ve indeksleme"):
        uploaded_files = st.file_uploader(
            "Belgeleri yükleyin", type=["pdf", "docx", "md"], accept_multiple_files=True
        )
        if st.button("İndeksle") and uploaded_files:
            rag.index_files(uploaded_files)
            st.success("Belgeler indekslendi.")

    query = st.text_input("Soru sor", value="")
    if query and st.button("Ara"):
        answer, citations = rag.answer_query(query)
        st.write("### Cevap")
        st.write(answer)
        st.write("### Kaynaklar")
        for citation in citations:
            st.write(f"- {citation}")


if __name__ == "__main__":
    main()
