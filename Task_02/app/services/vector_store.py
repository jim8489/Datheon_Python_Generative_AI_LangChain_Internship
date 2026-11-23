
import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    MIN_RELEVANCE_SCORE,
)
from app.services.embedding_service import get_embedding_model


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
    start_index = document.metadata.get("start_index", index)

    unique_value = f"{source}:{start_index}:{index}"

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

    # Replace previously indexed chunks for the same source.
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

    return len(documents)


def list_indexed_sources() -> list[str]:
    """
    Return unique source filenames stored in ChromaDB.
    """

    vector_store = get_vector_store()

    result = vector_store.get(
        include=["metadatas"],
    )

    sources = set()

    for metadata in result.get("metadatas", []) or []:

        if metadata and metadata.get("source"):
            sources.add(str(metadata["source"]))

    return sorted(sources)


def search_similar_documents(
    query: str,
    k: int = 4,
    source: str | None = None,
) -> list[Document]:
    """
    Retrieve relevant chunks and attach their relevance scores.

    If source is supplied, search only within that document.
    Chunks below MIN_RELEVANCE_SCORE are rejected.
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

    results = vector_store.similarity_search_with_relevance_scores(
        query=query.strip(),
        k=k,
        filter=search_filter,
    )

    relevant_documents = []

    for document, score in results:

        score = float(score)

        if score < MIN_RELEVANCE_SCORE:
            continue

        # Store the score for citation transparency.
        document.metadata["relevance_score"] = score

        relevant_documents.append(document)

    return relevant_documents


def clear_vector_store() -> None:
    vector_store = get_vector_store()

    existing = vector_store.get(include=[])

    ids = existing.get("ids", [])

    if ids:
        vector_store.delete(ids=ids)
