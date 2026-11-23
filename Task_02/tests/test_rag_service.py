import pytest

from langchain_core.documents import Document

from app.services import rag_service


class FakeGeminiResponse:
    content = "RAG retrieves relevant information from documents [S1]."


class FakeGemini:
    def invoke(self, messages):
        return FakeGeminiResponse()


def test_format_context():
    documents = [
        Document(
            page_content="RAG combines retrieval with generation.",
            metadata={"source": "research.pdf", "page": 2},
        )
    ]

    context, sources = rag_service.format_context(documents)

    assert "[S1]" in context
    assert "research.pdf" in context
    assert "Page 2" in context
    assert len(sources) == 1
    assert sources[0]["page"] == 2


def test_answer_question(monkeypatch):
    documents = [
        Document(
            page_content="RAG retrieves relevant information.",
            metadata={"source": "research.pdf", "page": 3},
        )
    ]

    monkeypatch.setattr(
        rag_service,
        "search_similar_documents",
        lambda query, k: documents,
    )

    monkeypatch.setattr(
        rag_service,
        "get_language_model",
        lambda: FakeGemini(),
    )

    result = rag_service.answer_question(
        "What does RAG do?"
    )

    assert result["question"] == "What does RAG do?"
    assert "retrieves relevant information" in result["answer"]
    assert result["sources"][0]["source"] == "research.pdf"
    assert result["sources"][0]["page"] == 3


def test_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        rag_service.answer_question("   ")


def test_invalid_retrieval_count():
    with pytest.raises(ValueError, match="greater than zero"):
        rag_service.answer_question("What is RAG?", k=0)


def test_no_retrieved_documents(monkeypatch):
    monkeypatch.setattr(
        rag_service,
        "search_similar_documents",
        lambda query, k: [],
    )

    result = rag_service.answer_question("What is RAG?")

    assert result["sources"] == []
    assert "could not find sufficient information" in result["answer"]