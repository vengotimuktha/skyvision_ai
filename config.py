import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# App-wide directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "input")
INDEX_DIR = os.path.join(BASE_DIR, "data", "skyvision_faiss_index")

# Secure API key load
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: Export for libraries expecting it as env var
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
