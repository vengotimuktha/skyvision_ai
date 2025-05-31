import os
import uuid
import streamlit as st
from utils import (
    extract_text_from_pdf,
    extract_text_from_csv,
    create_faiss_index,
    answer_query,
)

# ────────────── App Config ──────────────
st.set_page_config(page_title="SkyVision AI - Document Q&A", layout="wide")
st.title("📄 SkyVision AI — Ask Questions About Any PDF or CSV")

UPLOAD_DIR = "data/input"
INDEX_DIR = "data/skyvision_faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ────────────── Session State ──────────────
if "indexed_docs" not in st.session_state:
    st.session_state["indexed_docs"] = {}
if "prev_doc" not in st.session_state:
    st.session_state["prev_doc"] = ""
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "clear_query" not in st.session_state:
    st.session_state["clear_query"] = False

# ────────────── File Upload ──────────────
st.markdown("### 📤 Upload PDF or CSV")
uploaded_files = st.file_uploader(
    "Upload files below",
    type=["pdf", "csv"],
    accept_multiple_files=True
)
st.caption("You can upload multiple files. After uploading, click 'Extract & Index'.")

# ────────────── Index Uploaded Files ──────────────
if uploaded_files:
    for uploaded_file in uploaded_files:
        doc_id = str(uuid.uuid4())
        file_name = uploaded_file.name
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file_name}")

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"✅ Uploaded: {file_name}")

        if st.button(f"📌 Extract & Index: {file_name}", key=file_name):
            with st.spinner("🔍 Indexing..."):
                if file_name.lower().endswith(".csv"):
                    text = extract_text_from_csv(file_path)
                else:
                    text = extract_text_from_pdf(file_path)

                index_path = os.path.join(
                    INDEX_DIR,
                    f"{doc_id}_{file_name.replace('.pdf', '').replace('.csv', '')}"
                )
                create_faiss_index(text, index_path)
                st.session_state["indexed_docs"][file_name] = index_path

            st.success(f"📚 Indexed: {file_name}")

# ────────────── Question Answering ──────────────
if st.session_state["indexed_docs"]:
    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    selected_file = st.selectbox("📂 Choose document", list(st.session_state["indexed_docs"].keys()))

    if selected_file != st.session_state["prev_doc"]:
        st.session_state["prev_doc"] = selected_file
        st.session_state["query"] = ""

    if st.session_state["clear_query"]:
        st.session_state["query"] = ""
        st.session_state["clear_query"] = False
        st.rerun()

    query = st.text_input("🔎 Your question:", key="query")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("❌ Clear"):
            st.session_state["clear_query"] = True
    with col2:
        if st.button("🚀 Get Answer") and query:
            index_path = st.session_state["indexed_docs"][selected_file]
            with st.spinner("💬 Generating answer..."):
                answer, sources = answer_query(index_path, query)

                st.markdown("#### ✅ Answer:")
                st.markdown(f"""
                    <div style="background-color:#f9f9f9; padding:14px; border-radius:8px; border: 1px solid #ddd;">
                        {answer}
                    </div>
                """, unsafe_allow_html=True)

                if sources:
                    st.markdown("#### 📌 Source Chunks:")
                    for i, chunk in enumerate(sources):
                        with st.expander(f"📄 Chunk {i+1}"):
                            st.markdown(chunk)
