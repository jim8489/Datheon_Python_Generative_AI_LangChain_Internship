from types import SimpleNamespace

import pytest

from chatbot.chat_service import ChatService
from core.exceptions import ChatGenerationError


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.last_messages = None

    def stream(self, messages):
        self.last_messages = messages
        for item in self.chunks:
            if isinstance(item, Exception):
                raise item
            yield FakeChunk(item)
        if self.error:
            raise self.error


class FakeRAG:
    system_prompt = "Answer strictly from supplied document context."
    insufficient_information = "I could not find sufficient information in the provided documents."
    model_names = ("primary-model", "fallback-model")
    retrieval_k = 6

    def __init__(self):
        self.search_calls = []
        self.invoke_calls = []
        self.messages = None
        self.models = {}
        self.documents = [object()]
        self.context = "[S1]\nSource: paper.pdf\nPage: 2\nContent: Test context."
        self.sources = [{
            "citation_id": "S1",
            "source": "paper.pdf",
            "page": 2,
            "excerpt": "Test context.",
            "relevance_score": 0.9,
        }]
        self.rewrite_response = SimpleNamespace(content="rewritten standalone query")

    def invoke_with_fallback(self, messages):
        self.invoke_calls.append(messages)
        return self.rewrite_response

    @staticmethod
    def extract_response_text(content):
        return content if isinstance(content, str) else str(content)

    def search(self, query, k, document_id=None):
        self.search_calls.append((query, k, document_id))
        return self.documents

    def format_context(self, documents):
        return self.context, self.sources

    def get_language_model(self, model_name=None):
        selected = model_name or "primary-model"
        return self.models[selected]


def test_standalone_question_skips_query_rewrite_and_streams_tokens():
    rag = FakeRAG()
    rag.models["primary-model"] = FakeModel(["A grounded ", "answer [S1]."])
    service = ChatService(rag)

    turn = service.prepare_turn(
        question="What is the main finding?",
        history=[],
        document_id="doc-123",
        document_label="paper.pdf",
        retrieval_k=4,
    )

    assert rag.invoke_calls == []
    assert rag.search_calls == [("What is the main finding?", 4, "doc-123")]
    assert "What is the main finding?" in turn.retrieval_query
    assert "".join(turn.tokens) == "A grounded answer [S1]."
    assert turn.sources[0]["citation_id"] == "S1"
    assert turn.has_context is True


def test_follow_up_is_rewritten_using_history_and_history_is_in_prompt():
    rag = FakeRAG()
    rag.models["primary-model"] = FakeModel(["The paper reports a limitation [S1]."])
    service = ChatService(rag, query_rewrite_enabled=True)

    history = [
        {"role": "user", "content": "Summarize the paper."},
        {"role": "assistant", "content": "It introduces a new method."},
    ]
    turn = service.prepare_turn(
        question="What are its limitations?",
        history=history,
        document_id=None,
    )

    assert len(rag.invoke_calls) == 1
    rewrite_prompt = " ".join(item["content"] for item in rag.invoke_calls[0])
    assert "Summarize the paper" in rewrite_prompt
    assert rag.search_calls[0][0] == "rewritten standalone query"

    answer = "".join(turn.tokens)
    assert "limitation" in answer
    model_messages = rag.models["primary-model"].last_messages
    combined_prompt = "\n".join(message["content"] for message in model_messages)
    assert "Summarize the paper" in combined_prompt
    assert "It introduces a new method." in combined_prompt
    assert "Retrieved document context" in combined_prompt


def test_previous_history_is_included_in_final_generation_messages():
    rag = FakeRAG()
    rag.models["primary-model"] = FakeModel(["Answer [S1]."])

    captured = {}
    service = ChatService(rag, query_rewrite_enabled=False)
    turn = service.prepare_turn(
        question="What about limitations?",
        history=[
            {"role": "user", "content": "Explain the method."},
            {"role": "assistant", "content": "It uses a two-stage approach."},
        ],
        document_label="All indexed documents",
    )
    list(turn.tokens)

    messages = rag.models["primary-model"].last_messages
    messages_text = "\n".join(message["content"] for message in messages)
    assert "Explain the method" in messages_text
    assert "It uses a two-stage approach" in messages_text
    assert "Retrieved document context" in messages_text


def test_no_retrieved_chunks_returns_insufficient_information_without_llm():
    rag = FakeRAG()
    rag.documents = []
    service = ChatService(rag)

    turn = service.prepare_turn("Question?", history=[])

    assert turn.has_context is False
    assert turn.sources == []
    assert list(turn.tokens) == [rag.insufficient_information]
    assert rag.models == {}


def test_invalid_question_is_rejected():
    service = ChatService(FakeRAG())
    with pytest.raises(ValueError, match="cannot be empty"):
        service.prepare_turn("  ", history=[])


def test_primary_failure_before_tokens_uses_fallback():
    rag = FakeRAG()
    rag.models["primary-model"] = FakeModel(error=RuntimeError("primary unavailable"))
    rag.models["fallback-model"] = FakeModel(["Fallback response."])
    service = ChatService(rag, query_rewrite_enabled=False)

    turn = service.prepare_turn("Question", history=[])

    assert "".join(turn.tokens) == "Fallback response."


def test_primary_failure_after_partial_output_does_not_duplicate_with_fallback():
    rag = FakeRAG()
    rag.models["primary-model"] = FakeModel(
        ["Partial response", RuntimeError("connection dropped")]
    )
    rag.models["fallback-model"] = FakeModel(["This must not be used."])
    service = ChatService(rag, query_rewrite_enabled=False)

    turn = service.prepare_turn("Question", history=[])
    output = "".join(turn.tokens)

    assert output.startswith("Partial response")
    assert "interrupted" in output.lower()
    assert "This must not be used" not in output


def test_all_models_failing_before_output_raises_clear_error():
    rag = FakeRAG()
    rag.models["primary-model"] = FakeModel(error=RuntimeError("primary unavailable"))
    rag.models["fallback-model"] = FakeModel(error=RuntimeError("fallback unavailable"))
    service = ChatService(rag, query_rewrite_enabled=False)

    turn = service.prepare_turn("Question", history=[])
    with pytest.raises(ChatGenerationError, match="could not generate"):
        list(turn.tokens)


def test_rewrite_failure_falls_back_to_original_question():
    rag = FakeRAG()
    rag.invoke_with_fallback = lambda messages: (_ for _ in ()).throw(RuntimeError("rewrite unavailable"))
    rag.models["primary-model"] = FakeModel(["Answer [S1]."])
    service = ChatService(rag, query_rewrite_enabled=True)

    turn = service.prepare_turn(
        "What about the limitations?",
        history=[{"role": "user", "content": "Tell me about the paper."}],
    )

    assert rag.search_calls[0][0] == "What about the limitations?"
    assert "".join(turn.tokens) == "Answer [S1]."
def test_greeting_does_not_trigger_retrieval_or_llm():
    rag = FakeRAG()
    service = ChatService(rag)

    turn = service.prepare_turn("Hi!", history=[])

    assert turn.is_conversational is True
    assert turn.has_context is False
    assert turn.sources == []
    assert rag.search_calls == []
    assert rag.invoke_calls == []
    assert list(turn.tokens)[0].startswith("Hello!")


def test_broad_question_prompt_separates_unrelated_subjects():
    rag = FakeRAG()
    rag.models["primary-model"] = FakeModel(["Grounded answer [S1]."])

    service = ChatService(rag, query_rewrite_enabled=False)

    turn = service.prepare_turn(
        question="What is data?",
        history=[],
        document_label="All indexed documents",
    )

    list(turn.tokens)

    system_prompt = rag.models["primary-model"].last_messages[0]["content"]

    assert "unrelated subjects" in system_prompt
    assert "clearly separated sections" in system_prompt
    assert "source citations" in system_prompt