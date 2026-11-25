"""Load PDF/TXT files and attach stable document-level identities."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def generate_document_id(filename: str, file_content: bytes) -> str:
    """Return a stable ID for a filename/content pair.

    Including the original filename prevents two differently named files with
    identical bytes from unintentionally sharing an identity. Including the
    bytes ensures that two different files with the same name remain distinct.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A non-empty original filename is required.")
    if not isinstance(file_content, bytes):
        raise TypeError("file_content must be bytes.")

    filename_bytes = filename.encode("utf-8")
    identity = (
        len(filename_bytes).to_bytes(8, "big")
        + filename_bytes
        + file_content
    )
    return hashlib.sha256(identity).hexdigest()


def normalize_pdf_text(text: str) -> str:
    """Clean common PDF layout artifacts while preserving paragraphs."""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\x00", "")
        .replace("\u00ad", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    text = re.sub(r"[ \t]*\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n\n *", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(
    file_path: str | Path,
    source_name: str | None = None,
) -> list[Document]:
    """Load one PDF/TXT file, preserving its original name and document ID.

    ``source_name`` is used by Streamlit uploads because those files are first
    written to temporary paths. It preserves the user's filename in source
    metadata and in the stable identity calculation.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. Only PDF and TXT files are supported."
        )

    original_name = (source_name or path.name).strip()
    if not original_name:
        raise ValueError("The original document filename cannot be empty.")

    file_bytes = path.read_bytes()
    if not file_bytes:
        raise ValueError(f"The document is empty: {original_name}")

    document_id = generate_document_id(original_name, file_bytes)

    if extension == ".pdf":
        loaded = PyPDFLoader(str(path)).load()
        documents: list[Document] = []

        for document in loaded:
            document.page_content = normalize_pdf_text(document.page_content)
            document.metadata.update(
                {
                    "source": original_name,
                    "document_id": document_id,
                    "file_path": str(path),
                    "file_type": "pdf",
                }
            )
            # PyPDFLoader uses zero-based page indices; display one-based pages.
            if "page" in document.metadata:
                document.metadata["page"] = int(document.metadata["page"]) + 1
            documents.append(document)

        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError(
                f"The PDF contains no readable text: {original_name}. "
                "Scanned PDFs may require OCR before indexing."
            )
        return documents

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"The TXT file is not valid UTF-8: {original_name}") from exc

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
    """Load every supported PDF/TXT file in a directory in sorted order."""
    directory_path = Path(directory).expanduser().resolve()
    if not directory_path.exists():
        raise FileNotFoundError(f"Document directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise ValueError(f"Expected a directory: {directory_path}")

    documents: list[Document] = []
    for file_path in sorted(directory_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_document(file_path))

    if not documents:
        raise ValueError(f"No supported PDF or TXT documents found in {directory_path}")
    return documents
