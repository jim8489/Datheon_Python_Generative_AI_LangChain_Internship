import argparse
import sys

from app.config import DOCUMENTS_DIRECTORY
from app.loaders.document_loader import load_documents
from app.services.chunking_service import split_documents
from app.services.rag_service import answer_question
from app.services.vector_store import add_documents_to_vector_store


def index_documents(directory):
    """Load, chunk, and index documents."""

    print("\nLoading documents...")

    try:
        documents = load_documents(directory)

        print(f"Loaded {len(documents)} document pages/files.")

        print("\nSplitting documents into chunks...")

        chunks = split_documents(documents)

        print(f"Created {len(chunks)} chunks.")

        print("\nGenerating embeddings and indexing in ChromaDB...")

        indexed_count = add_documents_to_vector_store(chunks)

        print(f"\nSuccessfully indexed {indexed_count} chunks.")
        print("Your documents are ready for questions.")

    except Exception as error:
        print(f"\nIndexing failed: {error}")
        sys.exit(1)


def ask_question(question):
    """Ask a question using the RAG pipeline."""

    try:
        print("\nSearching documents and generating an answer...\n")

        result = answer_question(question)

        print("=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result["answer"])

        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)

        if result["sources"]:
            for source in result["sources"]:
                location = (
                    f"Page {source['page']}"
                    if source["page"] is not None
                    else "Page unavailable"
                )

                print(
                    f"\n[{source['citation_id']}] "
                    f"{source['source']} — {location}"
                )
                print(f"Excerpt: {source['excerpt']}")
        else:
            print("No supporting sources were retrieved.")

        print("=" * 70)

    except Exception as error:
        print(f"\nQuestion answering failed: {error}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Document Question Answering Assistant using RAG"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Load and index documents.",
    )

    index_parser.add_argument(
        "--directory",
        default=str(DOCUMENTS_DIRECTORY),
        help="Directory containing PDF or TXT documents.",
    )

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question about indexed documents.",
    )

    ask_parser.add_argument(
        "--question",
        required=True,
        help="Your question.",
    )

    args = parser.parse_args()

    if args.command == "index":
        index_documents(args.directory)

    elif args.command == "ask":
        ask_question(args.question)


if __name__ == "__main__":
    main()