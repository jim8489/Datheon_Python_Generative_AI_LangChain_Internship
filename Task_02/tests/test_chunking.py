from langchain_core.documents import Document

from app.services.chunking_service import split_documents


def test_split_documents():
    document = Document(
        page_content="Artificial intelligence. " * 200,
        metadata={"source": "sample.txt"},
    )

    chunks = split_documents([document])

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 1000
        assert chunk.metadata["source"] == "sample.txt"


def test_chunk_metadata_preserved():
    document = Document(
        page_content="Healthcare research information.",
        metadata={"source": "research.pdf", "page": 3},
    )

    chunks = split_documents([document])

    assert chunks[0].metadata["source"] == "research.pdf"
    assert chunks[0].metadata["page"] == 3


def test_empty_document_list():
    import pytest

    with pytest.raises(ValueError):
        split_documents([])