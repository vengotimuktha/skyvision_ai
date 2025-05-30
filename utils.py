# utils.py

import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def create_faiss_index(text: str, index_path: str) -> None:
    """Split text, create embeddings, and save FAISS index."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    vectorstore.save_local(index_path)
    print(f" FAISS index saved to: {index_path}")


def answer_query(index_path: str, query: str) -> str:
    """Load FAISS index and answer user query with context."""
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    answer = result['result']
    sources = result.get("source_documents", [])

    # Print for logs/debugging
    print(f"\n🔹 Answer:\n{answer}\n")
    if sources:
        print(" Retrieved Chunks:")
        for doc in sources:
            print(f"• {doc.page_content[:300]}...\n")

    return answer
