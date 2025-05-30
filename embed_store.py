# embed_store.py
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores.faiss import FAISS

import os

# Set your OpenAI API key (optionally from environment variables)
import os
openai_api_key = os.getenv("OPENAI_API_KEY")


# Load the text
with open("data/extracted/output_text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
texts = text_splitter.split_text(raw_text)

# Create embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_texts(texts, embedding_model)

# Save locally
vectorstore.save_local("data/skyvision_faiss_index")

print("FAISS index created and saved.")
