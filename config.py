# config.py

import os
from dotenv import load_dotenv
load_dotenv()

# App-wide directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "input")
INDEX_DIR = os.path.join(BASE_DIR, "data", "skyvision_faiss_index")

# Load your OpenAI API key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
