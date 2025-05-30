import os
import fitz  # PyMuPDF
import pandas as pd
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from config import OPENAI_API_KEY  # loads from .env or constant


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_text_from_csv(csv_path: str) -> str:
    """Extract text from a CSV by converting rows to text."""
    df = pd.read_csv(csv_path)
    rows_as_text = df.astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()
    return "\n".join(rows_as_text)


def create_faiss_index(text: str, index_path: str) -> None:
    """Split text, create embeddings, and save FAISS index."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Set it for langchain_openai
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    vectorstore.save_local(index_path)
    print(f" FAISS index saved to: {index_path}")


def answer_query(index_path: str, query: str) -> tuple[str, List[str]]:
    """Load FAISS index and answer user query with context."""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    answer = result['result']
    sources = [doc.page_content for doc in result.get("source_documents", [])]

    return answer, sources
