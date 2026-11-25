from uuid import uuid4

import pytest

from core.exceptions import ConversationOwnershipError
from storage.sqlite_store import SQLiteConversationStore


def _browser_id():
    return uuid4().hex


def _state(conversation_id=None, *, content="A saved question"):
    conversation_id = conversation_id or uuid4().hex
    return {
        "active_conversation_id": conversation_id,
        "conversations": {
            conversation_id: {
                "id": conversation_id,
                "title": "Saved conversation",
                "created_at": "2026-11-25T10:00:00+00:00",
                "updated_at": "2026-11-25T10:01:00+00:00",
                "messages": [
                    {
                        "id": uuid4().hex,
                        "role": "user",
                        "content": content,
                        "sources": [],
                        "created_at": "2026-11-25T10:00:00+00:00",
                    },
                    {
                        "id": uuid4().hex,
                        "role": "assistant",
                        "content": "A cited response [S1].",
                        "sources": [
                            {
                                "citation_id": "S1",
                                "source": "paper.pdf",
                                "page": 3,
                                "excerpt": "Supporting text.",
                                "relevance_score": 0.91,
                            }
                        ],
                        "created_at": "2026-11-25T10:01:00+00:00",
                    },
                ],
            }
        },
    }


def test_state_round_trip_preserves_messages_sources_and_active_conversation(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chats.sqlite3")
    owner = _browser_id()
    state = _state()

    store.save_browser_state(owner, state)
    loaded = store.load_browser_state(owner)

    assert loaded["active_conversation_id"] == state["active_conversation_id"]
    conversation = loaded["conversations"][state["active_conversation_id"]]
    assert conversation["title"] == "Saved conversation"
    assert conversation["messages"][0]["content"] == "A saved question"
    assert conversation["messages"][1]["sources"][0]["citation_id"] == "S1"


def test_browser_histories_are_isolated(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chats.sqlite3")
    first_owner, second_owner = _browser_id(), _browser_id()
    first_state = _state(content="Private message for browser one")
    second_state = _state(content="Private message for browser two")

    store.save_browser_state(first_owner, first_state)
    store.save_browser_state(second_owner, second_state)

    first_loaded = store.load_browser_state(first_owner)
    second_loaded = store.load_browser_state(second_owner)
    first_message = next(iter(first_loaded["conversations"].values()))["messages"][0]["content"]
    second_message = next(iter(second_loaded["conversations"].values()))["messages"][0]["content"]

    assert first_message == "Private message for browser one"
    assert second_message == "Private message for browser two"
    assert first_state["active_conversation_id"] != second_state["active_conversation_id"]


def test_store_rejects_attempt_to_claim_another_browsers_conversation(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chats.sqlite3")
    owner_one, owner_two = _browser_id(), _browser_id()
    state = _state()
    store.save_browser_state(owner_one, state)

    with pytest.raises(ConversationOwnershipError):
        store.save_browser_state(owner_two, state)

    assert store.load_browser_state(owner_two)["conversations"] == {}
    assert len(store.load_browser_state(owner_one)["conversations"]) == 1


def test_removed_conversations_are_deleted_from_storage(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chats.sqlite3")
    owner = _browser_id()
    first = _state()
    store.save_browser_state(owner, first)

    second = _state()
    store.save_browser_state(owner, second)
    loaded = store.load_browser_state(owner)

    assert list(loaded["conversations"]) == [second["active_conversation_id"]]


def test_invalid_browser_id_is_rejected(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chats.sqlite3")
    with pytest.raises(ValueError, match="valid 32-character browser identity"):
        store.load_browser_state("not-a-browser-id")


def test_database_file_is_created_automatically(tmp_path):
    db_path = tmp_path / "nested" / "data" / "chats.sqlite3"
    SQLiteConversationStore(db_path)
    assert db_path.is_file()
