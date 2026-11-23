
import hashlib
import logging
import math
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    MIN_RELEVANCE_SCORE,
)
from app.services.embedding_service import get_embedding_model


logger = logging.getLogger(__name__)


def squared_l2_relevance(distance: float) -> float:
    """Convert squared L2 distance into a bounded score between 0 and 1."""
    distance = max(float(distance), 0.0)
    score = 1.0 - (math.sqrt(distance) / 2.0)
    return max(0.0, min(1.0, score))


def get_vector_store() -> Chroma:
    CHROMA_PERSIST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
    )


def _legacy_document_id(source: str) -> str:
    """
    Provide a stable identity for older or manually created documents
    that do not yet contain a document_id.
    """
    return f"legacy:{source}"


def _ensure_document_id(document: Document) -> str:
    """
    Ensure every incoming chunk has a document_id.

    Documents loaded through document_loader already have a content-based
    ID. The legacy fallback preserves compatibility with older callers.
    """
    source = str(document.metadata.get("source", "unknown"))

    document_id = document.metadata.get("document_id")

    if not document_id:
        document_id = _legacy_document_id(source)
        document.metadata["document_id"] = document_id

    return str(document_id)


def _document_filter(document_id: str) -> dict:
    """
    Build the appropriate Chroma filter.

    Legacy documents are identified by filename because their original
    Chroma records may not contain document_id metadata.
    """
    if document_id.startswith("legacy:"):
        return {
            "source": document_id[len("legacy:"):]
        }

    return {
        "document_id": document_id
    }


def generate_chunk_id(
    document: Document,
    index: int,
    batch_id: str | None = None,
) -> str:
    document_id = _ensure_document_id(document)

    page = document.metadata.get("page", "unknown")
    start_index = document.metadata.get("start_index", index)

    unique_value = (
        f"{batch_id or 'default'}:"
        f"{document_id}:{page}:{start_index}:{index}"
    )

    return hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()


def add_documents_to_vector_store(
    documents: list[Document],
) -> int:
    """
    Index document chunks.

    Replacement is performed by document_id, not filename.
    New chunks are inserted before old chunks are deleted, preserving
    the previous version if insertion fails.
    """
    if not documents:
        raise ValueError("No document chunks provided.")

    vector_store = get_vector_store()

    document_ids = {
        _ensure_document_id(document)
        for document in documents
    }

    old_ids_by_document = {}

    # Find existing chunks belonging to these exact documents.
    for document_id in document_ids:
        existing = vector_store.get(
            where=_document_filter(document_id),
            include=[],
        )

        old_ids_by_document[document_id] = existing.get("ids", [])

    batch_id = uuid.uuid4().hex

    new_ids = [
        generate_chunk_id(document, index, batch_id)
        for index, document in enumerate(documents)
    ]

    # Insert new chunks before deleting the old version.
    try:
        vector_store.add_documents(
            documents=documents,
            ids=new_ids,
        )

    except Exception:
        try:
            vector_store.delete(ids=new_ids)
        except Exception:
            logger.exception(
                "Could not clean up partially inserted chunks."
            )

        logger.exception(
            "Failed to index replacement documents. "
            "Existing indexed data has been preserved."
        )
        raise

    # Delete old chunks only after successful insertion.
    for document_id, old_ids in old_ids_by_document.items():
        if not old_ids:
            continue

        try:
            vector_store.delete(ids=old_ids)

        except Exception:
            logger.exception(
                "New chunks were indexed, but old chunks for "
                "document '%s' could not be removed.",
                document_id,
            )

    logger.info(
        "Indexed %d chunks from %d document(s).",
        len(documents),
        len(document_ids),
    )

    return len(documents)


def list_indexed_documents() -> list[dict]:
    """
    Return unique indexed documents.

    Each item contains:
        document_id
        source
        legacy
    """
    vector_store = get_vector_store()

    result = vector_store.get(
        include=["metadatas"],
    )

    indexed_documents = {}

    for metadata in result.get("metadatas", []) or []:
        if not metadata:
            continue

        source = metadata.get("source")

        if not source:
            continue

        source = str(source)

        document_id = metadata.get("document_id")

        legacy = not bool(document_id)

        if legacy:
            document_id = _legacy_document_id(source)

        document_id = str(document_id)

        indexed_documents[document_id] = {
            "document_id": document_id,
            "source": source,
            "legacy": legacy,
        }

    return sorted(
        indexed_documents.values(),
        key=lambda item: (
            item["source"].lower(),
            item["document_id"],
        ),
    )


def list_indexed_sources() -> list[str]:
    """
    Backward-compatible helper returning unique filenames.
    """
    return sorted({
        document["source"]
        for document in list_indexed_documents()
    })


def get_indexed_chunk_count(
    source: str | None = None,
) -> int:
    """
    Count indexed chunks.

    The source argument now accepts a document_id.
    None means all indexed chunks.
    """
    vector_store = get_vector_store()

    search_filter = (
        _document_filter(source)
        if source
        else None
    )

    result = vector_store.get(
        where=search_filter,
        include=[],
    )

    return len(result.get("ids", []))


def search_similar_documents(
    query: str,
    k: int = 6,
    source: str | None = None,
) -> list[Document]:
    """
    Retrieve relevant chunks.

    The source argument accepts a document_id, not a filename.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    vector_store = get_vector_store()

    search_filter = (
        _document_filter(source)
        if source
        else None
    )

    raw_results = vector_store.similarity_search_with_score(
        query=query.strip(),
        k=k,
        filter=search_filter,
    )

    relevant_documents = []

    for document, distance in raw_results:
        if source:
            expected_source = (
                source[len("legacy:"):]
                if source.startswith("legacy:")
                else None
            )

            actual_document_id = document.metadata.get("document_id")
            actual_source = document.metadata.get("source")

            if expected_source is not None:
                # Supports old records without document_id.
                if actual_source != expected_source:
                    continue
            elif actual_document_id != source:
                logger.warning(
                    "Ignoring unexpected document ID: %s",
                    actual_document_id,
                )
                continue

        score = squared_l2_relevance(distance)

        if score < MIN_RELEVANCE_SCORE:
            continue

        document.metadata["relevance_score"] = round(score, 4)
        document.metadata["vector_distance"] = round(
            float(distance),
            6,
        )

        relevant_documents.append(document)

    logger.info(
        "Retrieval completed | document_id=%s | candidates=%d | accepted=%d",
        source or "all documents",
        len(raw_results),
        len(relevant_documents),
    )

    return relevant_documents


def clear_vector_store() -> None:
    vector_store = get_vector_store()

    existing = vector_store.get(include=[])
    ids = existing.get("ids", [])

    if ids:
        vector_store.delete(ids=ids)
