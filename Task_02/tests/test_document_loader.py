"""Tests for the production Task 2 document loader."""

from pathlib import Path

import pytest

from app.loaders.document_loader import (
    generate_document_id,
    load_document,
    load_documents,
    normalize_pdf_text,
)


def test_different_files_with_same_filename_get_different_ids(tmp_path: Path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_file = first_dir / "report.txt"
    second_file = second_dir / "report.txt"
    first_file.write_text("Research about artificial intelligence.", encoding="utf-8")
    second_file.write_text("Research about thyroid classification.", encoding="utf-8")

    first_document = load_document(first_file)[0]
    second_document = load_document(second_file)[0]

    assert first_document.metadata["source"] == "report.txt"
    assert second_document.metadata["source"] == "report.txt"
    assert first_document.metadata["document_id"] != second_document.metadata["document_id"]


def test_same_file_generates_same_document_id(tmp_path: Path):
    file_path = tmp_path / "report.txt"
    file_path.write_text("Research about artificial intelligence.", encoding="utf-8")

    first_document = load_document(file_path)[0]
    second_document = load_document(file_path)[0]

    assert first_document.metadata["document_id"] == second_document.metadata["document_id"]


def test_upload_source_name_is_preserved_in_production_loader(tmp_path: Path):
    file_path = tmp_path / "temporary-upload.txt"
    content = "Research about artificial intelligence."
    file_path.write_text(content, encoding="utf-8")

    document = load_document(file_path, source_name="report.txt")[0]

    assert document.metadata["source"] == "report.txt"
    assert document.metadata["document_id"] == generate_document_id(
        "report.txt", content.encode("utf-8")
    )


def test_load_documents_assigns_document_ids_to_cli_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("First document.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Second document.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [doc.metadata["source"] for doc in documents] == ["a.txt", "b.txt"]
    assert all(len(doc.metadata["document_id"]) == 64 for doc in documents)
    assert len({doc.metadata["document_id"] for doc in documents}) == 2


def test_normalize_pdf_text_preserves_paragraphs_and_joins_layout_lines():
    normalized = normalize_pdf_text("A line\ncontinued here.\n\nNew paragraph.")
    assert normalized == "A line continued here.\n\nNew paragraph."


def test_empty_and_unsupported_files_are_rejected(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")
    unsupported_file = tmp_path / "notes.docx"
    unsupported_file.write_bytes(b"not a supported document")

    with pytest.raises(ValueError, match="empty"):
        load_document(empty_file)
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(unsupported_file)


def test_missing_directory_is_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
