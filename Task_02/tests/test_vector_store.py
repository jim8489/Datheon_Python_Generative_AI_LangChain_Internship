import pytest

from langchain_core.documents import Document

from app.services import vector_store
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import (
    add_documents_to_vector_store,
    get_indexed_chunk_count,
    search_similar_documents,
    squared_l2_relevance,
)


@pytest.fixture
def isolated_vector_store(monkeypatch, tmp_path):

    monkeypatch.setattr(
        vector_store,
        "CHROMA_PERSIST_DIRECTORY",
        tmp_path / "test_chroma",
    )

    monkeypatch.setattr(
        vector_store,
        "CHROMA_COLLECTION_NAME",
        "test_document_knowledge",
    )

    monkeypatch.setattr(
        vector_store,
        "MIN_RELEVANCE_SCORE",
        0.0,
    )

    yield

    vector_store.clear_vector_store()


def test_embedding_generation():

    embeddings = generate_embeddings([
        "Artificial intelligence in healthcare."
    ])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


def test_squared_l2_relevance():

    assert squared_l2_relevance(0.0) == 1.0
    assert squared_l2_relevance(1.0) == 0.5
    assert squared_l2_relevance(4.0) == 0.0

    assert 0.0 <= squared_l2_relevance(2.5) <= 1.0


def test_chroma_add_and_search(isolated_vector_store):

    documents = [
        Document(
            page_content=(
                "Deep learning is useful for medical image analysis."
            ),
            metadata={"source": "test_medical.txt"},
        ),
        Document(
            page_content=(
                "Python is a popular programming language."
            ),
            metadata={"source": "test_python.txt"},
        ),
    ]

    count = add_documents_to_vector_store(documents)

    assert count == 2

    results = search_similar_documents(
        query="How is AI used for medical images?",
        k=2,
    )

    assert len(results) == 2

    assert any(
        "medical image" in document.page_content.lower()
        for document in results
    )

    assert all(
        0.0 <= document.metadata["relevance_score"] <= 1.0
        for document in results
    )

    assert get_indexed_chunk_count() == 2


def test_selected_document_filter(isolated_vector_store):

    documents = [
        Document(
            page_content=(
                "The assessment asks candidates to build "
                "a full-stack application."
            ),
            metadata={"source": "blinto-assessment.pdf"},
        ),
        Document(
            page_content=(
                "The research study investigates thyroid "
                "classification using machine learning."
            ),
            metadata={"source": "thyroid-study.pdf"},
        ),
    ]

    add_documents_to_vector_store(documents)

    results = search_similar_documents(
        query="What should the candidate build?",
        k=6,
        source="blinto-assessment.pdf",
    )

    assert results

    assert all(
        document.metadata["source"] == "blinto-assessment.pdf"
        for document in results
    )

    assert get_indexed_chunk_count(
        "blinto-assessment.pdf"
    ) == 1


def test_unknown_document_returns_no_results(isolated_vector_store):

    document = Document(
        page_content="Artificial intelligence supports healthcare.",
        metadata={"source": "healthcare.txt"},
    )

    add_documents_to_vector_store([document])

    results = search_similar_documents(
        query="What is artificial intelligence?",
        source="missing-document.pdf",
    )

    assert results == []