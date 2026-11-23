
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.loaders.document_loader import load_document
from app.services.chunking_service import split_documents
from app.services.rag_service import answer_question
from app.services.vector_store import (
    add_documents_to_vector_store,
    list_indexed_documents,
    get_indexed_chunk_count,
)

# --------------------------------------------------
# Upload helper
# --------------------------------------------------

def load_document_from_upload(uploaded_file):

    suffix = Path(uploaded_file.name).suffix.lower()

    with NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        delete=False,
    ) as temporary_file:

        temporary_file.write(uploaded_file.getvalue())
        temporary_path = Path(temporary_file.name)

    try:
        documents = load_document(
        temporary_path,
        source_name=uploaded_file.name,
    )

        return documents

    finally:
        Path(temporary_path).unlink(missing_ok=True)


# --------------------------------------------------
# Source display helper
# --------------------------------------------------

def display_sources(sources):

    for source in sources:

        page_number = source.get("page")

        location = (
            f"Page {page_number}"
            if page_number is not None
            else "Page unavailable"
        )

        score = source.get("relevance_score")

        score_text = (
            f" | Relevance: {score:.3f}"
            if score is not None
            else ""
        )

        st.markdown(
            f"**[{source['citation_id']}] "
            f"{source['source']} — {location}{score_text}**"
        )

        st.caption(source["excerpt"])


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind | AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #888888;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .stChatMessage {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("📚 DocuMind")

    st.caption("AI-powered document question answering")

    st.divider()

    # Upload section
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
        upload_errors = []

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                status.write(
                    f"Processing {uploaded_file.name}..."
                )

                documents = load_document_from_upload(
                    uploaded_file
                )

                chunks = split_documents(documents)

                count = add_documents_to_vector_store(
                    chunks
                )

                total_chunks += count

            except Exception as error:

                upload_errors.append(
                    f"{uploaded_file.name}: {error}"
                )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        status.empty()

        if total_chunks:
            st.success(
                f"Successfully indexed {total_chunks} chunks."
            )

        for error in upload_errors:
            st.error(error)

        if not total_chunks and not upload_errors:
            st.warning("No document chunks were indexed.")

        st.rerun()

    st.divider()

    # Knowledge base section

    # Knowledge base section
    st.subheader("Knowledge base")

    try:
        indexed_documents = list_indexed_documents()

    except Exception as error:
        indexed_documents = []
        st.error(f"Could not load indexed documents: {error}")

    # Count how many indexed documents share each filename.
    filename_counts = {}

    for document in indexed_documents:
        filename = document["source"]
        filename_counts[filename] = (
            filename_counts.get(filename, 0) + 1
        )

    # Create readable labels while retaining unique document IDs.
    document_labels = {}

    for document in indexed_documents:
        document_id = document["document_id"]
        filename = document["source"]

        if filename_counts[filename] > 1:
            label = f"{filename} ({document_id[:10]})"
        else:
            label = filename

        if document.get("legacy"):
            label += " [legacy]"

        document_labels[document_id] = label

    if indexed_documents:
        for document in indexed_documents:
            document_id = document["document_id"]
            st.write(f"📄 {document_labels[document_id]}")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # Document selection
    st.subheader("Search settings")

    document_options = (
        [None]
        + [
            document["document_id"]
            for document in indexed_documents
        ]
    )

    selected_document_id = st.selectbox(
        "Search within",
        options=document_options,
        format_func=lambda value: (
            "All documents"
            if value is None
            else document_labels[value]
        ),
        help=(
            "Select one document to restrict retrieval, "
            "or choose All documents."
        ),
    )

    selected_document = (
        "All documents"
        if selected_document_id is None
        else document_labels[selected_document_id]
    )

    # This ID is passed to the vector store and RAG service.
    source_filter = selected_document_id

    st.caption("Selected: " + selected_document)


    st.divider()

    if st.button(
        "Clear chat history",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Main content
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Document Q&A Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions and receive answers grounded in your documents.'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"Current search scope: **{selected_document}**"
)


# --------------------------------------------------
# Welcome screen
# --------------------------------------------------

if not st.session_state.messages:

    st.info(
        "Upload PDF or TXT documents from the sidebar, "
        "index them, and start asking questions."
    )

    st.subheader("Try asking:")

    example_questions = [
        "What is the main objective of this document?",
        "What methodology does the document describe?",
        "What are its main findings?",
        "What limitations are mentioned?",
    ]

    columns = st.columns(2)

    for index, question in enumerate(example_questions):

        with columns[index % 2]:

            if st.button(
                question,
                use_container_width=True,
            ):

                st.session_state.pending_question = question
                st.rerun()


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View supporting sources"):

                display_sources(
                    message["sources"]
                )

            if message.get("selected_document"):
                st.caption(
                    "Search scope: "
                    + message["selected_document"]
                )


# --------------------------------------------------
# Question processing
# --------------------------------------------------

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

user_question = st.chat_input(
    "Ask a question about your documents..."
)

question = user_question or pending_question

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating an answer..."
        ):

            try:

                result = answer_question(
                    question=question,
                    source=source_filter,
                )

                st.markdown(result["answer"])

                if result["sources"]:

                    with st.expander(
                        "View supporting sources",
                        expanded=True,
                    ):

                        display_sources(
                            result["sources"]
                        )

                else:

                   indexed_count = get_indexed_chunk_count(source_filter)

                   if indexed_count == 0:

                      st.warning(
                        "No indexed chunks were found for the selected document. "
                          "Please upload and index it again."
                       )
                   else:

                      st.warning(
                      f"The selected document contains {indexed_count} indexed "
                      "chunks, but retrieval returned no candidates. "
                      "Check the retrieval configuration and application logs."
                    )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "selected_document": selected_document,
                    }
                )

            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
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
