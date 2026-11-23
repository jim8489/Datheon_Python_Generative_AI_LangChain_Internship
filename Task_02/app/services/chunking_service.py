from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split loaded documents into smaller overlapping chunks.
    Preserve the original document metadata.
    """

    if not documents:
        raise ValueError("No documents provided for chunking.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Remove chunks containing only whitespace.
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("Chunking produced no usable text.")

    return chunks