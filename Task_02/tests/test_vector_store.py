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
            metadata={
                "source": "report.pdf",
                "document_id": "assessment-document-id",
            },
        ),
        Document(
            page_content=(
                "The research study investigates thyroid "
                "classification using machine learning."
            ),
            metadata={
                "source": "report.pdf",
                "document_id": "thyroid-document-id",
            },
        ),
    ]

    add_documents_to_vector_store(documents)

    results = search_similar_documents(
        query="What should the candidate build?",
        k=6,
        source="assessment-document-id",
    )

    assert results

    assert all(
        document.metadata["document_id"] == "assessment-document-id"
        for document in results
    )

    assert get_indexed_chunk_count(
        "assessment-document-id"
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


class FakeVectorStore:
    """Minimal Chroma substitute for testing document replacement."""

    def __init__(self):
        self.records = {}
        self.fail_on_add = False

    def get(self, where=None, include=None):
        records = self.records

        if where:
            key, expected_value = next(iter(where.items()))

            records = {
                chunk_id: document
                for chunk_id, document in records.items()
                if document.metadata.get(key) == expected_value
            }

        return {"ids": list(records.keys())}

    def add_documents(self, documents, ids):
        for index, (document, chunk_id) in enumerate(zip(documents, ids)):
            self.records[chunk_id] = document

            # Simulate a failure after a partial insertion.
            if self.fail_on_add and index == 0:
                raise RuntimeError("Simulated Chroma insertion failure.")

    def delete(self, ids):
        for chunk_id in ids:
            self.records.pop(chunk_id, None)


def test_successful_document_replacement(monkeypatch):
    fake_store = FakeVectorStore()

    old_document = Document(
        page_content="Old version of the assessment.",
        metadata={"source": "assessment.pdf"},
    )

    fake_store.records["old-chunk-id"] = old_document

    monkeypatch.setattr(
        vector_store,
        "get_vector_store",
        lambda: fake_store,
    )

    new_document = Document(
        page_content="Updated version of the assessment.",
        metadata={"source": "assessment.pdf"},
    )

    count = add_documents_to_vector_store([new_document])

    assert count == 1
    assert "old-chunk-id" not in fake_store.records
    assert len(fake_store.records) == 1

    indexed_document = next(iter(fake_store.records.values()))

    assert indexed_document.page_content == (
        "Updated version of the assessment."
    )


def test_failed_replacement_preserves_old_documents(monkeypatch):
    fake_store = FakeVectorStore()

    old_document = Document(
        page_content="Previously indexed assessment content.",
        metadata={"source": "assessment.pdf"},
    )

    fake_store.records["old-chunk-id"] = old_document
    fake_store.fail_on_add = True

    monkeypatch.setattr(
        vector_store,
        "get_vector_store",
        lambda: fake_store,
    )

    new_document = Document(
        page_content="Replacement assessment content.",
        metadata={"source": "assessment.pdf"},
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated Chroma insertion failure",
    ):
        add_documents_to_vector_store([new_document])

    # The previous document must still exist.
    assert "old-chunk-id" in fake_store.records

    assert (
        fake_store.records["old-chunk-id"].page_content
        == "Previously indexed assessment content."
    )

    # The partially inserted replacement must be cleaned up.
    assert len(fake_store.records) == 1
def test_duplicate_filenames_are_independent(isolated_vector_store):

    documents = [
        Document(
            page_content=(
                "The first report discusses artificial intelligence "
                "in medical image analysis."
            ),
            metadata={
                "source": "report.pdf",
                "document_id": "medical-report-id",
            },
        ),
        Document(
            page_content=(
                "The second report discusses Python programming "
                "and software engineering."
            ),
            metadata={
                "source": "report.pdf",
                "document_id": "programming-report-id",
            },
        ),
    ]

    count = add_documents_to_vector_store(documents)

    assert count == 2

    assert get_indexed_chunk_count("medical-report-id") == 1
    assert get_indexed_chunk_count("programming-report-id") == 1

    medical_results = search_similar_documents(
        query="What does the medical report discuss?",
        source="medical-report-id",
    )

    programming_results = search_similar_documents(
        query="What does the programming report discuss?",
        source="programming-report-id",
    )

    assert medical_results
    assert programming_results

    assert all(
        document.metadata["document_id"] == "medical-report-id"
        for document in medical_results
    )

    assert all(
        document.metadata["document_id"] == "programming-report-id"
        for document in programming_results
    )

    assert get_indexed_chunk_count() == 2