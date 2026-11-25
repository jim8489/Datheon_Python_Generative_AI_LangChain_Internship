"""Environment-backed configuration for DocuMind Task 2."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash",
).strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()

CHROMA_PERSIST_DIRECTORY = (
    BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
)
CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "document_knowledge",
).strip()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))
DOCUMENTS_DIRECTORY = BASE_DIR / "data" / "sample_documents"

_API_KEY_PLACEHOLDERS = {
    "your_gemini_api_key_here",
    "your_actual_gemini_api_key",
    "replace_me",
}


def validate_configuration() -> None:
    """Fail fast when required API or retrieval settings are invalid."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.lower() in _API_KEY_PLACEHOLDERS:
        raise ValueError(
            "GEMINI_API_KEY is missing or still contains the example placeholder. "
            "Configure a valid key in Task_02/.env."
        )
    if not GEMINI_MODEL:
        raise ValueError("GEMINI_MODEL cannot be empty.")
    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL cannot be empty.")
    if not CHROMA_COLLECTION_NAME:
        raise ValueError("CHROMA_COLLECTION_NAME cannot be empty.")
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
        )
    if RETRIEVAL_K <= 0:
        raise ValueError("RETRIEVAL_K must be greater than zero.")
    if not 0.0 <= MIN_RELEVANCE_SCORE <= 1.0:
        raise ValueError("MIN_RELEVANCE_SCORE must be between 0 and 1.")
