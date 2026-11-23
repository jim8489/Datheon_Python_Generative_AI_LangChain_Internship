import hashlib
import logging
import math

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
    """
    Convert Chroma's squared L2 distance into a bounded
    retrieval score between 0 and 1.

    This conversion assumes normalized embeddings.
    Our embedding service uses normalize_embeddings=True.
    """

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


def generate_chunk_id(document: Document, index: int) -> str:
    source = document.metadata.get("source", "unknown")
    page = document.metadata.get("page", "unknown")
    start_index = document.metadata.get("start_index", index)

    unique_value = f"{source}:{page}:{start_index}:{index}"

    return hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()


def add_documents_to_vector_store(
    documents: list[Document],
) -> int:

    if not documents:
        raise ValueError("No document chunks provided.")

    vector_store = get_vector_store()

    sources = {
        str(document.metadata.get("source", "unknown"))
        for document in documents
    }

    # Replace old chunks belonging to the same source.
    for source in sources:

        existing = vector_store.get(
            where={"source": source},
            include=[],
        )

        existing_ids = existing.get("ids", [])

        if existing_ids:
            vector_store.delete(ids=existing_ids)

    ids = [
        generate_chunk_id(document, index)
        for index, document in enumerate(documents)
    ]

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    logger.info(
        "Indexed %d chunks from %d document(s).",
        len(documents),
        len(sources),
    )

    return len(documents)


def list_indexed_sources() -> list[str]:
    vector_store = get_vector_store()

    result = vector_store.get(
        include=["metadatas"],
    )

    sources = set()

    for metadata in result.get("metadatas", []) or []:

        if metadata and metadata.get("source"):
            sources.add(str(metadata["source"]))

    return sorted(sources)


def get_indexed_chunk_count(
    source: str | None = None,
) -> int:
    """
    Count indexed chunks, optionally restricted to one source.
    """

    vector_store = get_vector_store()

    search_filter = (
        {"source": source}
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
    Retrieve the top matching document chunks.

    Uses raw Chroma distances and converts them ourselves,
    avoiding LangChain's default relevance conversion.

    The minimum score is configurable. Set it to 0.0 while
    validating retrieval so potentially useful chunks are
    not discarded prematurely.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    vector_store = get_vector_store()

    search_filter = (
        {"source": source}
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

        document_source = document.metadata.get("source")

        # Defensive check: never return a different source
        # when the user selected a specific document.
        if source and document_source != source:
            logger.warning(
                "Ignoring unexpected source: %s",
                document_source,
            )
            continue

        score = squared_l2_relevance(distance)

        if score < MIN_RELEVANCE_SCORE:
            continue

        document.metadata["relevance_score"] = round(
            score,
            4,
        )

        document.metadata["vector_distance"] = round(
            float(distance),
            6,
        )

        relevant_documents.append(document)

    logger.info(
        "Retrieval completed | source=%s | candidates=%d | accepted=%d",
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