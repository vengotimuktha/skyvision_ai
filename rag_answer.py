# rag_answer.py

from dotenv import load_dotenv
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores.faiss import FAISS

from langchain.chains import RetrievalQA


# Load environment variables (API key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local(
    "data/skyvision_faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True  
)


# Set up retriever
retriever = vectorstore.as_retriever(search_type="similarity", k=5)

# Load OpenAI LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = input(" Enter your question about the PDF: ")
result = qa_chain(query)

# Show result
print("\n Answer:\n", result['result'])
print("\n  Retrieved Chunks:\n")
for doc in result["source_documents"]:
    print("•", doc.page_content[:200], "\n")
