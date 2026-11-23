
import hashlib
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from app.loaders.document_loader import generate_document_id
from langchain_community.document_loaders import PyPDFLoader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def generate_document_id(
    filename: str,
    file_content: bytes,
) -> str:
    """
    Generate a stable ID from the original filename and file content.

    Different files with the same filename receive different IDs.
    Re-uploading the same file produces the same ID.
    """

    filename_bytes = filename.encode("utf-8")
    identity = (
        len(filename_bytes).to_bytes(8, "big")
        + filename_bytes
        + file_content
    )

    return hashlib.sha256(identity).hexdigest()


def normalize_pdf_text(text: str) -> str:
    """Clean PDF text while preserving paragraph boundaries."""

    text = unicodedata.normalize("NFKC", text)

    text = (
        text.replace("\x00", "")
        .replace("\u00ad", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = re.sub(
        r"[ \t]*\n[ \t]*\n+",
        "\n\n",
        text,
    )

    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n\n *", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_document(
    file_path: str | Path,
    source_name: str | None = None,
) -> list[Document]:
    """
    Load a PDF or TXT document and attach a stable document ID.

    source_name allows temporary upload files to retain their
    original filename in metadata and document identity.
    """

    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Only PDF and TXT files are supported."
        )

    original_name = source_name or path.name
    file_bytes = path.read_bytes()

    if not file_bytes:
        raise ValueError(f"The document is empty: {original_name}")

    document_id = generate_document_id(
        original_name,
        file_bytes,
    )

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()

        for document in documents:
            document.page_content = normalize_pdf_text(
                document.page_content
            )

            document.metadata.update({
                "source": original_name,
                "document_id": document_id,
                "file_type": "pdf",
            })

            # Convert zero-based PDF page numbers to one-based.
            if "page" in document.metadata:
                document.metadata["page"] += 1

        if not documents or not any(
            document.page_content.strip()
            for document in documents
        ):
            raise ValueError(
                f"The PDF contains no readable text: {original_name}"
            )

        return documents

    text = file_bytes.decode("utf-8")

    if not text.strip():
        raise ValueError(f"The document is empty: {original_name}")

    return [
        Document(
            page_content=text,
            metadata={
                "source": original_name,
                "document_id": document_id,
                "file_path": str(path),
                "file_type": "txt",
            },
        )
    ]


def load_documents(directory: str | Path) -> list[Document]:
    """Load every supported PDF and TXT file from a directory."""

    directory_path = Path(directory).expanduser().resolve()

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise ValueError(
            f"Expected a directory: {directory_path}"
        )

    documents = []

    for file_path in sorted(directory_path.iterdir()):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            documents.extend(load_document(file_path))

    if not documents:
        raise ValueError(
            f"No supported PDF or TXT documents found in {directory_path}"
        )

    return documents
def test_different_files_with_same_filename_get_different_ids(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_dir.mkdir()
    second_dir.mkdir()

    first_file = first_dir / "report.txt"
    second_file = second_dir / "report.txt"

    first_file.write_text(
        "Research about artificial intelligence.",
        encoding="utf-8",
    )

    second_file.write_text(
        "Research about thyroid classification.",
        encoding="utf-8",
    )

    first_document = load_document(first_file)
    second_document = load_document(second_file)

    assert first_document[0].metadata["source"] == "report.txt"
    assert second_document[0].metadata["source"] == "report.txt"

    assert (
        first_document[0].metadata["document_id"]
        != second_document[0].metadata["document_id"]
    )


def test_same_file_generates_same_document_id(tmp_path):
    file_path = tmp_path / "report.txt"

    file_path.write_text(
        "Research about artificial intelligence.",
        encoding="utf-8",
    )

    first_document = load_document(file_path)
    second_document = load_document(file_path)

    assert (
        first_document[0].metadata["document_id"]
        == second_document[0].metadata["document_id"]
    )


def test_upload_source_name_is_used_for_document_identity(tmp_path):
    file_path = tmp_path / "temporary-upload.txt"

    content = "Research about artificial intelligence."
    file_path.write_text(content, encoding="utf-8")

    document = load_document(
        file_path,
        source_name="report.txt",
    )[0]

    expected_id = generate_document_id(
        "report.txt",
        content.encode("utf-8"),
    )

    assert document.metadata["source"] == "report.txt"
    assert document.metadata["document_id"] == expected_id