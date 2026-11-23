from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file_path: str | Path) -> list[Document]:
    """
    Load a PDF or TXT document and return LangChain Document objects.
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

    # Load PDF documents
    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()

        for document in documents:
            document.metadata["source"] = path.name

            # Convert page numbering from zero-based to one-based.
            if "page" in document.metadata:
                document.metadata["page"] += 1

        if not documents or not any(
            document.page_content.strip() for document in documents
        ):
            raise ValueError(f"The PDF contains no readable text: {path.name}")

        return documents

    # Load TXT documents
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"The document is empty: {path.name}")

    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "file_type": "txt",
            },
        )
    ]


def load_documents(directory: str | Path) -> list[Document]:
    """
    Load all supported PDF and TXT documents from a directory.
    """

    directory_path = Path(directory).expanduser().resolve()

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise ValueError(f"Expected a directory: {directory_path}")

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