"""DocuMind Chat — Task 4 Streamlit entry point."""

from __future__ import annotations

import logging

import streamlit as st
from streamlit_cookies_controller import CookieController

from chatbot.chat_service import ChatService
from chatbot.rag_adapter import load_task2_services
from conversation.manager import (
    append_message,
    get_active_conversation,
    get_messages,
    get_recent_history,
    initialize_state,
)
from core.browser_identity import get_or_create_browser_id, load_signing_key
from core.exceptions import ConversationStoreError, Task2IntegrationError
from core.logging_config import configure_logging
from core.settings import CHAT_DATABASE_PATH, MAX_QUESTION_CHARS, validate_settings
from frontend.components import render_message, render_sources
from frontend.sidebar import render_sidebar
from frontend.styles import APP_CSS
from storage.sqlite_store import SQLiteConversationStore

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DocuMind Chat | Task 4",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


def _request_uses_https() -> bool:
    """Set the Secure cookie flag when Streamlit is served over HTTPS."""
    try:
        request_url = str(st.context.url)
        forwarded_proto = str(st.context.headers.get("X-Forwarded-Proto", ""))
    except Exception:
        request_url, forwarded_proto = "", ""
    return request_url.lower().startswith("https://") or forwarded_proto.split(",")[0].strip().lower() == "https"


def _restore_and_persist_state() -> tuple[SQLiteConversationStore, str]:
    """Resolve browser identity, hydrate its chats, and ensure one chat exists."""
    validate_settings()
    cookie_controller = CookieController(key="documind-cookie-controller")
    signing_key = load_signing_key()
    browser_id = get_or_create_browser_id(
        cookie_controller,
        signing_key,
        secure=_request_uses_https(),
        fallback_token=st.context.cookies.get("documind_browser"),
    )
    store = SQLiteConversationStore(CHAT_DATABASE_PATH)

    # A browser refresh creates a new Streamlit session. Rehydrate that session
    # from SQLite using only the browser ID verified from its signed cookie.
    if st.session_state.get("_browser_id") != browser_id:
        persisted = store.load_browser_state(browser_id)
        st.session_state["conversations"] = persisted["conversations"]
        st.session_state["active_conversation_id"] = persisted["active_conversation_id"]
        st.session_state["_browser_id"] = browser_id

    initialize_state(st.session_state)
    store.save_browser_state(browser_id, st.session_state)
    return store, browser_id


try:
    store, browser_id = _restore_and_persist_state()
    rag = load_task2_services()
except (Task2IntegrationError, ConversationStoreError, ValueError, OSError) as exc:
    st.error("DocuMind Chat could not start.")
    st.exception(exc)
    st.markdown(
        "Check that `Task_04` is beside `Task_02`, install Task 2 and Task 4 dependencies, "
        "and configure `Task_02/.env`. Ensure the `Task_04/data` directory is writable. "
        "See `Task_04/README.md` for setup instructions."
    )
    st.stop()


def persist_session() -> None:
    """Write the current browser's complete chat state to its SQLite records."""
    store.save_browser_state(browser_id, st.session_state)


chat_service = ChatService(rag)
selected_document_id, selected_document_label, retrieval_k = render_sidebar(
    rag,
    st.session_state,
    persist=persist_session,
)

conversation = get_active_conversation(st.session_state)
message_count = len(conversation["messages"])

st.markdown(
    """
    <section class="chat-hero">
      <div class="eyebrow">Task 04 · Streaming RAG Chat</div>
      <h1>Ask. Explore. Follow the thread.</h1>
      <p>A conversational interface over your existing DocuMind knowledge base, with context-aware follow-ups and live token streaming.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

meta_left, meta_right = st.columns([3, 1])
with meta_left:
    st.caption(f"Knowledge scope: **{selected_document_label}**")
with meta_right:
    st.caption(f"{message_count} message(s) in this conversation")

suggested_question = st.session_state.pop("_suggested_question", None)
messages = get_messages(st.session_state)
if not messages and not suggested_question:
    st.markdown("### Start a conversation")
    st.write(
        "Ask a question about the documents already indexed in Task 2. You can ask "
        "follow-up questions naturally; the retrieval query will be contextualized using this conversation."
    )
    example_columns = st.columns(2)
    suggestions = [
        "What is the main topic of the indexed documents?",
        "Summarize the key findings.",
        "What methodology is described?",
        "What limitations are discussed?",
    ]
    for index, suggestion in enumerate(suggestions):
        with example_columns[index % 2]:
            if st.button(suggestion, key=f"suggestion-{index}", use_container_width=True):
                st.session_state["_suggested_question"] = suggestion
                st.rerun()
else:
    for message in messages:
        render_message(message)

user_question = st.chat_input(
    "Ask a question about your indexed documents…",
    max_chars=MAX_QUESTION_CHARS,
)
question = user_question or suggested_question

if question:
    question = question.strip()
    if not question:
        st.stop()

    # Capture prior messages before appending the current user question.
    history = get_recent_history(st.session_state)
    append_message(st.session_state, "user", question)
    persist_session()

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        try:
            with st.status("Preparing your answer…", expanded=False) as status:
                turn = chat_service.prepare_turn(
                    question=question,
                    history=history,
                    document_id=selected_document_id,
                    document_label=selected_document_label,
                    retrieval_k=retrieval_k,
                )
                if turn.retrieval_query != question:
                    status.write("Resolved the follow-up question using conversation context.")
                if turn.is_conversational:
                    status_label = "Responding conversationally…"
                elif turn.has_context:
                    status_label = "Generating a grounded answer…"
                else:
                    status_label = "No relevant passages found"
                status.update(label=status_label, state="complete", expanded=False)

            answer = st.write_stream(turn.tokens)
            if not isinstance(answer, str):
                answer = str(answer or "")
            if not answer.strip():
                answer = rag.insufficient_information
                st.markdown(answer)

            append_message(
                st.session_state,
                "assistant",
                answer,
                sources=turn.sources,
            )
            persist_session()
            render_sources(turn.sources)

        except Exception as exc:
            logger.exception("Chat turn failed.")
            safe_message = (
                "I couldn't complete that response. Check the Gemini API configuration "
                "and Task 2 integration, then try again."
            )
            st.error(safe_message)
            st.caption(str(exc))
            append_message(st.session_state, "assistant", safe_message)
            persist_session()
