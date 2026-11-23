from langchain_core.documents import Document

from app.services.embedding_service import generate_embeddings
from app.services.vector_store import (
    add_documents_to_vector_store,
    search_similar_documents,
)


def test_embedding_generation():
    embeddings = generate_embeddings([
        "Artificial intelligence in healthcare."
    ])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384


def test_chroma_add_and_search():
    documents = [
        Document(
            page_content="Deep learning is useful for medical image analysis.",
            metadata={"source": "test_medical.txt"},
        ),
        Document(
            page_content="Python is a popular programming language.",
            metadata={"source": "test_python.txt"},
        ),
    ]

    count = add_documents_to_vector_store(documents)

    assert count == 2

    results = search_similar_documents(
        "How is AI used for medical images?",
        k=2,
    )

    assert len(results) == 2
    assert any("medical image" in doc.page_content.lower() for doc in results)