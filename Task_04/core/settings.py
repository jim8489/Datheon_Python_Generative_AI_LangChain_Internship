"""Task 4 configuration, with Task 2 kept as the source of RAG settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

TASK4_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK4_ROOT.parent

# Task 4 has only chat-specific settings. Gemini and Chroma credentials/settings
# remain in Task 2's .env and are loaded by Task 2's own configuration module.
load_dotenv(TASK4_ROOT / ".env", override=False)

_task2_path = os.getenv("TASK2_PATH", "../Task_02")
_task2_candidate = Path(_task2_path).expanduser()
if not _task2_candidate.is_absolute():
    _task2_candidate = (TASK4_ROOT / _task2_candidate).resolve()
TASK2_ROOT = _task2_candidate

_database_path = Path(os.getenv("CHAT_DATABASE_PATH", "data/conversations.sqlite3")).expanduser()
if not _database_path.is_absolute():
    _database_path = (TASK4_ROOT / _database_path).resolve()
CHAT_DATABASE_PATH = _database_path

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_CONVERSATIONS = int(os.getenv("MAX_CONVERSATIONS", "20"))
DEFAULT_RETRIEVAL_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "6"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
BROWSER_COOKIE_NAME = "documind_browser"
BROWSER_COOKIE_TTL_DAYS = int(os.getenv("BROWSER_COOKIE_TTL_DAYS", "365"))
SESSION_SIGNING_KEY = os.getenv("SESSION_SIGNING_KEY", "").strip() or None


def validate_settings() -> None:
    """Fail early for invalid chat-specific configuration."""
    if MAX_HISTORY_MESSAGES < 2:
        raise ValueError("MAX_HISTORY_MESSAGES must be at least 2.")
    if MAX_CONVERSATIONS < 1:
        raise ValueError("MAX_CONVERSATIONS must be at least 1.")
    if DEFAULT_RETRIEVAL_K < 1:
        raise ValueError("DEFAULT_RETRIEVAL_K must be at least 1.")
    if MAX_QUESTION_CHARS < 100:
        raise ValueError("MAX_QUESTION_CHARS must be at least 100.")
    if BROWSER_COOKIE_TTL_DAYS < 1:
        raise ValueError("BROWSER_COOKIE_TTL_DAYS must be at least 1.")
    if not TASK2_ROOT.is_dir():
        raise ValueError(
            f"Task 2 directory was not found at {TASK2_ROOT}. "
            "Set TASK2_PATH in Task_04/.env if your folder layout differs."
        )
