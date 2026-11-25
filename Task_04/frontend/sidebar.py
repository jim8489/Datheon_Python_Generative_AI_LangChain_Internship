"""Sidebar composition for conversation navigation and RAG scope selection."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from conversation.manager import (
    clear_active_conversation,
    create_conversation,
    delete_active_conversation,
    list_conversations,
    set_active_conversation,
)


def _document_label(document: dict[str, Any], duplicate_names: set[str]) -> str:
    filename = str(document.get("source", "Unnamed document"))
    document_id = str(document.get("document_id", ""))
    if filename in duplicate_names:
        return f"{filename} · {document_id[:8]}"
    return filename


def render_sidebar(
    rag: Any,
    state: Any,
    persist: Callable[[], None] | None = None,
) -> tuple[str | None, str, int]:
    """Render navigation and return (document_id, document_label, retrieval_k)."""
    def save_changes() -> None:
        if persist is not None:
            persist()

    with st.sidebar:
        st.markdown("## ✦ DocuMind")
        st.caption("Conversational document intelligence")
        st.divider()

        if st.button("＋  New conversation", use_container_width=True, type="primary"):
            create_conversation(state)
            save_changes()
            st.rerun()

        st.markdown("**Your conversations**")
        conversations = list_conversations(state)
        active_id = state["active_conversation_id"]
        for conversation in conversations:
            conversation_id = conversation["id"]
            title = conversation.get("title", "New conversation")
            prefix = "● " if conversation_id == active_id else "  "
            label = prefix + (title if len(title) <= 34 else title[:31] + "…")
            if st.button(
                label,
                key=f"conversation-{conversation_id}",
                use_container_width=True,
                type="secondary" if conversation_id != active_id else "tertiary",
            ):
                set_active_conversation(state, conversation_id)
                save_changes()
                st.rerun()

        st.divider()
        st.markdown("**Knowledge base**")
        try:
            documents = rag.list_documents()
            total_chunks = rag.count_chunks()
        except Exception as exc:
            documents = []
            total_chunks = 0
            st.error("Task 2 knowledge base could not be read.")
            st.caption(str(exc))

        st.caption(f"{len(documents)} indexed document(s) · {total_chunks} chunk(s)")
        if not documents:
            st.info(
                "No indexed documents were found. Open Task 2, upload and index your "
                "documents, then return here."
            )
            document_options: list[str | None] = [None]
            labels = {None: "All indexed documents"}
        else:
            counts: dict[str, int] = {}
            for document in documents:
                filename = str(document.get("source", "Unnamed document"))
                counts[filename] = counts.get(filename, 0) + 1
            duplicate_names = {name for name, count in counts.items() if count > 1}

            document_options = [None] + [str(doc["document_id"]) for doc in documents]
            labels = {None: "All indexed documents"}
            for document in documents:
                labels[str(document["document_id"])] = _document_label(document, duplicate_names)

        selected_id = st.selectbox(
            "Retrieval scope",
            options=document_options,
            format_func=lambda value: labels.get(value, "All indexed documents"),
            help="Choose the whole Task 2 knowledge base or restrict retrieval to one document.",
        )
        selected_label = labels.get(selected_id, "All indexed documents")

        default_k = max(1, min(12, int(getattr(rag, "retrieval_k", 6))))
        retrieval_k = st.slider(
            "Retrieved passages",
            min_value=1,
            max_value=12,
            value=default_k,
            help="Number of relevant Task 2 chunks supplied to Gemini for each answer.",
        )

        st.divider()
        with st.expander("Conversation controls"):
            if st.button("Clear current conversation", use_container_width=True):
                clear_active_conversation(state)
                save_changes()
                st.rerun()
            if st.button("Delete current conversation", use_container_width=True):
                delete_active_conversation(state)
                save_changes()
                st.rerun()

        st.divider()
        st.markdown(
            '<p class="small-muted">Conversation history is saved for this browser. '
            'Other browser profiles receive separate chat histories. Document indexing '
            'remains managed by Task 2.</p>',
            unsafe_allow_html=True,
        )

    return selected_id, selected_label, retrieval_k
