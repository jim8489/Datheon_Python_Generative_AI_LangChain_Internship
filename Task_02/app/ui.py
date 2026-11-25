"""Streamlit interface for uploading documents and asking grounded questions."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from app.loaders.document_loader import load_document
from app.services.chunking_service import split_documents
from app.services.rag_service import answer_question
from app.services.vector_store import (
    add_documents_to_vector_store,
    get_indexed_chunk_count,
    list_indexed_documents,
)

logger = logging.getLogger(__name__)


def load_document_from_upload(uploaded_file):
    """Load an uploaded file while preserving its original filename/identity."""
    suffix = Path(uploaded_file.name).suffix.lower()
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temporary_path = Path(temporary_file.name)
        return load_document(temporary_path, source_name=uploaded_file.name)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def display_sources(sources: list[dict]) -> None:
    for source in sources:
        page_number = source.get("page")
        location = f"Page {page_number}" if page_number is not None else "Page unavailable"
        score = source.get("relevance_score")
        score_text = f" | Relevance: {score:.3f}" if isinstance(score, (int, float)) else ""
        st.markdown(
            f"**[{source.get('citation_id', 'Source')}] "
            f"{source.get('source', 'Unknown document')} — {location}{score_text}**"
        )
        excerpt = source.get("excerpt")
        if excerpt:
            st.caption(excerpt)


st.set_page_config(
    page_title="DocuMind | AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title { font-size: 2.3rem; font-weight: 750; margin-bottom: 0.2rem; }
        .subtitle { color: #888888; font-size: 1rem; margin-bottom: 1.5rem; }
        .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("AI-powered document question answering")
    st.divider()

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to build your searchable knowledge base.",
    )

    if st.button(
        "Index uploaded documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        total_chunks = 0
        failed_files: list[str] = []
        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                status.write(f"Processing {uploaded_file.name}...")
                documents = load_document_from_upload(uploaded_file)
                chunks = split_documents(documents)
                total_chunks += add_documents_to_vector_store(chunks)
            except Exception:
                logger.exception("Could not index uploaded document %s", uploaded_file.name)
                failed_files.append(uploaded_file.name)

            progress.progress((index + 1) / len(uploaded_files))

        status.empty()
        if total_chunks:
            st.success(f"Successfully indexed {total_chunks} chunk(s).")
        for filename in failed_files:
            st.error(f"Could not index {filename}. Check the application logs for details.")
        st.rerun()

    st.divider()
    st.subheader("Knowledge base")
    try:
        indexed_documents = list_indexed_documents()
    except Exception:
        logger.exception("Could not load the indexed document list")
        indexed_documents = []
        st.error("Could not load indexed documents. Check the application logs.")

    filename_counts: dict[str, int] = {}
    for document in indexed_documents:
        filename = str(document["source"])
        filename_counts[filename] = filename_counts.get(filename, 0) + 1

    document_labels: dict[str, str] = {}
    for document in indexed_documents:
        document_id = str(document["document_id"])
        filename = str(document["source"])
        label = f"{filename} ({document_id[:10]})" if filename_counts[filename] > 1 else filename
        if document.get("legacy"):
            label += " [legacy]"
        document_labels[document_id] = label
        st.write(f"📄 {label}")

    if not indexed_documents:
        st.caption("No documents indexed yet.")

    st.divider()
    st.subheader("Search settings")
    document_options = [None] + [str(doc["document_id"]) for doc in indexed_documents]
    selected_document_id = st.selectbox(
        "Search within",
        options=document_options,
        format_func=lambda value: "All documents" if value is None else document_labels[value],
        help="Select one document to restrict retrieval, or choose All documents.",
    )
    selected_document = (
        "All documents" if selected_document_id is None else document_labels[selected_document_id]
    )
    st.caption("Selected: " + selected_document)

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="main-title">Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions and receive answers grounded in your documents.</div>',
    unsafe_allow_html=True,
)
st.caption(f"Current search scope: **{selected_document}**")

if not st.session_state.messages:
    st.info("Upload PDF or TXT documents from the sidebar, index them, and start asking questions.")
    st.subheader("Try asking:")
    example_questions = [
        "What is the main objective of this document?",
        "What methodology does the document describe?",
        "What are its main findings?",
        "What limitations are mentioned?",
    ]
    columns = st.columns(2)
    for index, question_text in enumerate(example_questions):
        with columns[index % 2]:
            if st.button(question_text, use_container_width=True):
                st.session_state.pending_question = question_text
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View supporting sources"):
                display_sources(message["sources"])
            if message.get("selected_document"):
                st.caption("Search scope: " + message["selected_document"])

pending_question = st.session_state.pop("pending_question", None)
user_question = st.chat_input("Ask a question about your documents...")
question = user_question or pending_question

if question:
    question = question.strip()
    if not question:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = answer_question(question=question, source=selected_document_id)

            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("View supporting sources", expanded=True):
                    display_sources(result["sources"])
            else:
                indexed_count = get_indexed_chunk_count(selected_document_id)
                if indexed_count == 0:
                    st.warning(
                        "No indexed chunks were found for the selected document. "
                        "Upload and index it before asking questions."
                    )
                else:
                    st.info(
                        "No sufficiently relevant passages were retrieved for this question. "
                        "Try rephrasing it or reviewing the selected document."
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "selected_document": selected_document,
                }
            )
        except Exception:
            logger.exception("Question answering failed")
            error_message = (
                "Something went wrong while generating the answer. "
                "Check the Gemini configuration and application logs, then try again."
            )
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "selected_document": selected_document,
                }
            )
