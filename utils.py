import os
from PyPDF2 import PdfReader
import pandas as pd
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from config import OPENAI_API_KEY  # securely loaded from .env


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_csv(csv_path: str) -> str:
    """Extract text from a CSV by joining rows."""
    df = pd.read_csv(csv_path)
    rows_as_text = df.astype(str).apply(lambda row: " | ".join(row.values), axis=1).tolist()
    return "\n".join(rows_as_text)


def create_faiss_index(text: str, index_path: str) -> None:
    """Split text into chunks, embed them, and save FAISS index locally."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please check your .env file or config.py.")

    # Ensure it's visible to langchain_openai internally
    #os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    #embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding_model=embeddings)
    vectorstore.save_local(index_path)
    print(f" FAISS index saved to: {index_path}")


def answer_query(index_path: str, query: str) -> tuple[str, List[str]]:
    """Use the FAISS index to answer user query using OpenAI + RetrievalQA."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")

    #os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    #embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = [doc.page_content for doc in result.get("source_documents", [])]

    return answer, sources
