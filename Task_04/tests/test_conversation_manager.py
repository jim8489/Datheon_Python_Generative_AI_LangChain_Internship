from conversation.manager import (
    append_message,
    clear_active_conversation,
    create_conversation,
    delete_active_conversation,
    get_active_conversation,
    get_messages,
    get_recent_history,
    initialize_state,
    set_active_conversation,
)


def test_initialization_creates_one_session_conversation():
    state = {}
    initialize_state(state)

    assert len(state["conversations"]) == 1
    assert state["active_conversation_id"] in state["conversations"]
    assert get_messages(state) == []


def test_create_from_empty_state_creates_only_one_conversation():
    state = {}
    conversation_id = create_conversation(state)

    assert list(state["conversations"]) == [conversation_id]
    assert state["active_conversation_id"] == conversation_id


def test_conversation_switching_isolated_by_id():
    state = {}
    initialize_state(state)
    first_id = state["active_conversation_id"]
    append_message(state, "user", "Question in first chat")

    second_id = create_conversation(state)
    append_message(state, "user", "Question in second chat")
    set_active_conversation(state, first_id)

    assert get_messages(state)[0]["content"] == "Question in first chat"
    set_active_conversation(state, second_id)
    assert get_messages(state)[0]["content"] == "Question in second chat"


def test_first_user_message_sets_chat_title():
    state = {}
    initialize_state(state)
    append_message(state, "user", "Explain retrieval augmented generation")

    assert get_active_conversation(state)["title"] == "Explain retrieval augmented generation"


def test_history_is_bounded_and_returns_safe_copy():
    state = {}
    initialize_state(state)
    for index in range(5):
        append_message(state, "user", f"Question {index}")
        append_message(state, "assistant", f"Answer {index}")

    history = get_recent_history(state, max_messages=4)
    assert len(history) == 4
    assert history[0]["content"] == "Question 3"
    history[0]["content"] = "mutated"
    assert get_messages(state)[-4]["content"] == "Question 3"


def test_clear_preserves_conversation_identity():
    state = {}
    initialize_state(state)
    conversation_id = state["active_conversation_id"]
    append_message(state, "user", "A question")

    clear_active_conversation(state)

    assert state["active_conversation_id"] == conversation_id
    assert get_messages(state) == []
    assert get_active_conversation(state)["title"] == "New conversation"


def test_delete_last_chat_creates_exactly_one_replacement():
    state = {}
    initialize_state(state)
    original_id = state["active_conversation_id"]

    replacement_id = delete_active_conversation(state)

    assert replacement_id != original_id
    assert list(state["conversations"]) == [replacement_id]
    assert state["active_conversation_id"] == replacement_id


def test_delete_active_chat_activates_most_recent_remaining_chat():
    state = {}
    initialize_state(state)
    first_id = state["active_conversation_id"]
    second_id = create_conversation(state)

    assert delete_active_conversation(state) == first_id
    assert second_id not in state["conversations"]


def test_invalid_role_and_empty_content_are_rejected():
    state = {}
    initialize_state(state)

    try:
        append_message(state, "system", "Not allowed")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid role should raise ValueError")

    try:
        append_message(state, "user", "   ")
    except ValueError:
        pass
    else:
        raise AssertionError("Empty message should raise ValueError")


def test_browser_sessions_use_independent_state_objects():
    first_session = {}
    second_session = {}
    initialize_state(first_session)
    initialize_state(second_session)

    first_id = first_session["active_conversation_id"]
    second_id = second_session["active_conversation_id"]
    append_message(first_session, "user", "Private first-session message")

    assert first_id != second_id
    assert get_messages(second_session) == []


def test_conversation_limit_is_enforced_per_session():
    state = {}
    initialize_state(state)
    for _ in range(25):
        create_conversation(state)

    assert len(state["conversations"]) <= 20
    assert state["active_conversation_id"] in state["conversations"]


def test_malformed_state_is_recovered_safely():
    state = {
        "conversations": {"broken": None},
        "active_conversation_id": "broken",
    }
    initialize_state(state)

    active = get_active_conversation(state)
    assert active["id"] == state["active_conversation_id"]
    assert active["messages"] == []
    assert len(state["conversations"]) == 1
