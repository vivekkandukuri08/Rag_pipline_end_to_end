import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/budget_speech.pdf"

MODEL_NAME = "all-MiniLM-L6-v2"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

DIMENSION = 384

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")