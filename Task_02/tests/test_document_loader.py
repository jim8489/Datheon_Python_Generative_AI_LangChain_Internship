import pytest

from app.loaders.document_loader import load_document, load_documents


def test_load_txt_document(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Artificial intelligence supports healthcare.",
        encoding="utf-8",
    )

    documents = load_document(file_path)

    assert len(documents) == 1
    assert "Artificial intelligence" in documents[0].page_content
    assert documents[0].metadata["source"] == "sample.txt"


def test_load_empty_txt_document(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_document(file_path)


def test_unsupported_file_type(tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("Sample content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_document("missing_document.txt")


def test_load_documents_from_directory(tmp_path):
    (tmp_path / "first.txt").write_text("First document.", encoding="utf-8")
    (tmp_path / "second.txt").write_text("Second document.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2