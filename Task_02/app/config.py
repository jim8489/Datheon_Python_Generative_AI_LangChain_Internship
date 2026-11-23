
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHROMA_PERSIST_DIRECTORY = (
    BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "document_knowledge",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
# Minimum accepted Chroma relevance score.
# Adjust this after testing with your own documents.
MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.0")
)
DOCUMENTS_DIRECTORY = BASE_DIR / "data" / "sample_documents"


def validate_configuration():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_actual_gemini_api_key":
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Please configure it in the .env file."
        )

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative "
            "and smaller than CHUNK_SIZE."
        )

    if RETRIEVAL_K <= 0:
        raise ValueError("RETRIEVAL_K must be greater than zero.")

    if not 0.0 <= MIN_RELEVANCE_SCORE <= 1.0:
       raise ValueError(
        "MIN_RELEVANCE_SCORE must be between 0 and 1."
    )
