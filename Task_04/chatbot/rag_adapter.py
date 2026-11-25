"""A thin integration boundary around Task 2's existing RAG services.

Task 4 intentionally does not reimplement document loading, chunking,
embeddings, Chroma access, retrieval, or source formatting.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from core.exceptions import Task2IntegrationError
from core.settings import TASK2_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task2Services:
    """References to the public service functions already implemented in Task 2."""

    config: ModuleType
    rag: ModuleType
    vector_store: ModuleType

    def search(self, query: str, k: int, document_id: str | None = None) -> list[Any]:
        return self.vector_store.search_similar_documents(
            query=query,
            k=k,
            source=document_id,
        )

    def format_context(self, documents: list[Any]) -> tuple[str, list[dict[str, Any]]]:
        return self.rag.format_context(documents)

    def list_documents(self) -> list[dict[str, Any]]:
        return self.vector_store.list_indexed_documents()

    def count_chunks(self, document_id: str | None = None) -> int:
        return self.vector_store.get_indexed_chunk_count(document_id)

    def invoke_with_fallback(self, messages: list[dict[str, str]]) -> Any:
        return self.rag.invoke_with_fallback(messages)

    def extract_response_text(self, content: Any) -> str:
        return self.rag.extract_response_text(content)

    def get_language_model(self, model_name: str | None = None) -> Any:
        return self.rag.get_language_model(model_name)

    @property
    def system_prompt(self) -> str:
        return self.rag.SYSTEM_PROMPT

    @property
    def insufficient_information(self) -> str:
        return self.rag.INSUFFICIENT_INFORMATION

    @property
    def model_names(self) -> tuple[str, str | None]:
        return (
            self.config.GEMINI_MODEL,
            self.config.GEMINI_FALLBACK_MODEL,
        )

    @property
    def retrieval_k(self) -> int:
        return self.config.RETRIEVAL_K


def validate_task2_root(task2_root: str | Path = TASK2_ROOT) -> Path:
    """Validate the expected Task 2 package layout."""
    root = Path(task2_root).expanduser().resolve()
    required = (
        root / "app" / "__init__.py",
        root / "app" / "config.py",
        root / "app" / "services" / "rag_service.py",
        root / "app" / "services" / "vector_store.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Task2IntegrationError(
            "Task 2 could not be located or its expected service files are missing. "
            f"Configured Task2 path: {root}. Missing: {', '.join(missing)}. "
            "Set TASK2_PATH in Task_04/.env if your folder layout differs."
        )
    return root


def load_task2_services(task2_root: str | Path = TASK2_ROOT) -> Task2Services:
    """Load Task 2's `app` package without copying or duplicating its code."""
    root = validate_task2_root(task2_root)
    package_file = (root / "app" / "__init__.py").resolve()

    # The integration uses Task 2's existing absolute `app.*` imports.
    # Refuse an ambiguous already-loaded `app` package instead of silently
    # connecting to the wrong project.
    loaded_app = sys.modules.get("app")
    loaded_file = getattr(loaded_app, "__file__", None) if loaded_app else None
    if loaded_file and Path(loaded_file).resolve() != package_file:
        raise Task2IntegrationError(
            "A different Python package named 'app' is already loaded. "
            "Run Task 4 with its provided streamlit_app.py entry point and "
            "avoid importing another top-level 'app' package first."
        )

    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    importlib.invalidate_caches()

    try:
        config = importlib.import_module("app.config")
        rag = importlib.import_module("app.services.rag_service")
        vector_store = importlib.import_module("app.services.vector_store")
    except Exception as exc:
        logger.exception("Unable to import Task 2's RAG service modules.")
        raise Task2IntegrationError(
            "Task 2's RAG services could not be imported. Install the dependencies "
            "from Task_02/requirements.txt and verify Task_02/.env."
        ) from exc

    required_rag = (
        "format_context",
        "invoke_with_fallback",
        "get_language_model",
        "SYSTEM_PROMPT",
        "INSUFFICIENT_INFORMATION",
    )
    required_vector = (
        "search_similar_documents",
        "list_indexed_documents",
        "get_indexed_chunk_count",
    )
    missing_api = [
        f"rag_service.{name}" for name in required_rag if not hasattr(rag, name)
    ] + [
        f"vector_store.{name}" for name in required_vector if not hasattr(vector_store, name)
    ]
    if missing_api:
        raise Task2IntegrationError(
            "Task 2 does not expose the service interface expected by Task 4: "
            + ", ".join(missing_api)
        )

    logger.info("Connected Task 4 to Task 2 services at %s", root)
    return Task2Services(config=config, rag=rag, vector_store=vector_store)
