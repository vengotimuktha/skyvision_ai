from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import streamlit as st
import os
from utils import extract_text_from_pdf, create_faiss_index, answer_query
import uuid

# App Configuration
st.set_page_config(page_title="SkyVision AI - PDF Q&A", layout="wide")
st.title(" SkyVision AI — Ask Questions About Any PDF")

# Directory Setup
UPLOAD_DIR = "data/input"
INDEX_DIR = "data/skyvision_faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Session State Initialization
for key in ["indexed_docs", "prev_pdf", "query", "clear_query"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key == "query" else {}

# UI: Upload Section
st.markdown("###  Upload PDF(s) to Begin")
uploaded_files = st.file_uploader("Drag and drop PDF files here", type=["pdf"], accept_multiple_files=True)
st.caption("You can upload multiple PDF documents. After uploading, click 'Extract & Index' to enable Q&A.")

# File Handling & Indexing
if uploaded_files:
    for uploaded_file in uploaded_files:
        doc_id = str(uuid.uuid4())
        pdf_name = uploaded_file.name
        save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{pdf_name}")

        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f" Uploaded: {pdf_name}")

        if st.button(f" Extract & Index: {pdf_name}"):
            with st.spinner(" Extracting text and creating FAISS index..."):
                text = extract_text_from_pdf(save_path)
                index_path = os.path.join(INDEX_DIR, f"{doc_id}_{pdf_name.replace('.pdf','')}")
                create_faiss_index(text, index_path)
                st.session_state["indexed_docs"][pdf_name] = index_path
            st.success(f" {pdf_name} indexed successfully!")

# Q&A Section
if st.session_state["indexed_docs"]:
    st.markdown("---")
    st.markdown("###  Ask Questions About Your Indexed PDFs")

    selected_pdf = st.selectbox(" Choose a PDF to query", list(st.session_state["indexed_docs"].keys()))

    # Clear previous query if PDF changes
    if selected_pdf != st.session_state["prev_pdf"]:
        st.session_state["prev_pdf"] = selected_pdf
        st.session_state["query"] = ""

    # Clear query logic
    if st.session_state["clear_query"]:
        st.session_state["query"] = ""
        st.session_state["clear_query"] = False

    # Question form
    with st.form(key="question_form"):
        query = st.text_input(" Ask a question about the PDF:", key="query")
        submitted = st.form_submit_button(" Get Answer")

    # Clear Question Button
    if st.button(" Clear Question"):
        st.session_state["clear_query"] = True
        st.rerun()

    # Display Answer
    if submitted and query:
        index_path = st.session_state["indexed_docs"][selected_pdf]
        result = answer_query(index_path, query)
        st.markdown("####  Answer:")
        st.markdown(f"""
            <div style="background-color:#f9f9f9; padding:14px; border-radius:8px; border: 1px solid #ddd;">
                {result}
            </div>
        """, unsafe_allow_html=True)
