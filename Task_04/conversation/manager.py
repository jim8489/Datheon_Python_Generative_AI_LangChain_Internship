"""In-memory conversation lifecycle helpers used by the Streamlit UI and tests.

Persistence is handled separately by ``storage.sqlite_store``. Keeping this
module independent from Streamlit and SQLite makes its state transitions easy
 to test and allows the same manager to be used with another storage backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, MutableMapping
from uuid import uuid4

from core.settings import MAX_CONVERSATIONS, MAX_HISTORY_MESSAGES

State = MutableMapping[str, Any]
_ALLOWED_ROLES = {"user", "assistant"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_conversation() -> dict[str, Any]:
    now = _utc_now()
    return {
        "id": uuid4().hex,
        "title": "New conversation",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def initialize_state(state: State) -> None:
    """Initialize a session and repair malformed persisted/in-memory records."""
    conversations = state.get("conversations")
    if not isinstance(conversations, dict):
        conversations = {}
        state["conversations"] = conversations

    for conversation_id, conversation in list(conversations.items()):
        if not isinstance(conversation, dict):
            del conversations[conversation_id]
            continue
        conversation.setdefault("id", conversation_id)
        conversation.setdefault("title", "New conversation")
        conversation.setdefault("messages", [])
        conversation.setdefault("created_at", _utc_now())
        conversation.setdefault("updated_at", conversation["created_at"])
        if not isinstance(conversation["messages"], list):
            conversation["messages"] = []

    active_id = state.get("active_conversation_id")
    if not conversations or active_id not in conversations:
        if conversations:
            active_id = max(
                conversations,
                key=lambda key: conversations[key].get("updated_at", ""),
            )
            state["active_conversation_id"] = active_id
        else:
            conversation = _new_conversation()
            conversations[conversation["id"]] = conversation
            state["active_conversation_id"] = conversation["id"]


def create_conversation(state: State) -> str:
    """Create and activate a conversation; enforce the per-browser chat limit."""
    if not isinstance(state.get("conversations"), dict) or not state.get("conversations"):
        state["conversations"] = {}
        conversation = _new_conversation()
        state["conversations"][conversation["id"]] = conversation
        state["active_conversation_id"] = conversation["id"]
        return conversation["id"]

    initialize_state(state)
    conversations = state["conversations"]
    conversation = _new_conversation()
    conversations[conversation["id"]] = conversation
    state["active_conversation_id"] = conversation["id"]

    while len(conversations) > MAX_CONVERSATIONS:
        oldest_id = min(
            (key for key in conversations if key != conversation["id"]),
            key=lambda key: conversations[key].get("updated_at", ""),
        )
        del conversations[oldest_id]

    return conversation["id"]


def set_active_conversation(state: State, conversation_id: str) -> None:
    initialize_state(state)
    if conversation_id not in state["conversations"]:
        raise KeyError("The requested conversation does not exist in this browser.")
    state["active_conversation_id"] = conversation_id


def get_active_conversation(state: State) -> dict[str, Any]:
    initialize_state(state)
    return state["conversations"][state["active_conversation_id"]]


def get_messages(state: State) -> list[dict[str, Any]]:
    """Return the active conversation's message list."""
    return get_active_conversation(state)["messages"]


def get_recent_history(
    state: State,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict[str, str]]:
    """Return a bounded copy of recent user/assistant messages."""
    if max_messages < 1:
        raise ValueError("max_messages must be greater than zero.")

    recent = get_messages(state)[-max_messages:]
    return [
        {"role": message["role"], "content": message["content"]}
        for message in recent
        if message.get("role") in _ALLOWED_ROLES
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]


def append_message(
    state: State,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append a chat message and update the conversation title/timestamp."""
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"Unsupported message role: {role}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty.")

    conversation = get_active_conversation(state)
    message = {
        "id": uuid4().hex,
        "role": role,
        "content": content.strip(),
        "sources": sources or [],
        "created_at": _utc_now(),
    }
    conversation["messages"].append(message)
    conversation["updated_at"] = message["created_at"]

    if role == "user" and conversation["title"] == "New conversation":
        title = " ".join(content.split())
        conversation["title"] = title[:52] + ("…" if len(title) > 52 else "")

    return message


def clear_active_conversation(state: State) -> None:
    """Clear messages in the active conversation while preserving its identity."""
    conversation = get_active_conversation(state)
    conversation["messages"] = []
    conversation["title"] = "New conversation"
    conversation["updated_at"] = _utc_now()


def delete_active_conversation(state: State) -> str:
    """Delete the active conversation and activate another/new one."""
    initialize_state(state)
    current_id = state["active_conversation_id"]
    del state["conversations"][current_id]

    if not state["conversations"]:
        conversation = _new_conversation()
        state["conversations"][conversation["id"]] = conversation
        state["active_conversation_id"] = conversation["id"]
        return conversation["id"]

    next_id = max(
        state["conversations"],
        key=lambda key: state["conversations"][key].get("updated_at", ""),
    )
    state["active_conversation_id"] = next_id
    return next_id


def list_conversations(state: State) -> list[dict[str, Any]]:
    """Return conversation metadata, newest activity first."""
    initialize_state(state)
    return sorted(
        state["conversations"].values(),
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )
