# DocuMind Chat — Conversational RAG with Streaming

**Internship:** Datheon Python + Generative AI & LangChain Internship  
**Task:** Task 04 — Interactive Full-Stack AI Chatbot with Streamlit

DocuMind Chat turns the existing Task 02 DocuMind RAG pipeline into a multi-turn conversational application. It adds conversation history, session-state management, follow-up question contextualization, and token-by-token response streaming without copying or reimplementing Task 02's document ingestion, chunking, embedding, ChromaDB, retrieval, or source-formatting services.

## What Task 4 adds

- **Interactive chat UI:** Uses Streamlit's `st.chat_message` and `st.chat_input` components.
- **Streaming output:** Streams LangChain Gemini chunks into the conversation with `st.write_stream`.
- **Conversation history:** Maintains separate conversations and replays the active conversation when the app reruns.
- **Session-state management:** Stores chats in `st.session_state`; each browser session has its own isolated conversation dictionary.
- **Follow-up contextualization:** Rewrites context-dependent follow-up questions into standalone retrieval queries using recent conversation history.
- **History-aware RAG prompting:** Supplies recent user/assistant messages alongside the current retrieved evidence to Gemini. Factual answers remain grounded in the retrieved document context.
- **Document scope:** Allows retrieval across the Task 02 knowledge base or within one Task 02 document ID.
- **Source inspection:** Shows the source filename, page, relevance score, and excerpt returned by Task 02's source formatter.
- **Conversation controls:** Create, switch, clear, and delete conversations within the current browser session.
- **Graceful model fallback:** Attempts Task 02's configured fallback Gemini model only if the primary stream fails before producing text. It never restarts after partial output, avoiding duplicated answers.

## What Task 4 deliberately reuses

Task 4 integrates with these existing Task 02 services:

| Task 02 service | How Task 4 uses it |
|---|---|
| `app.services.vector_store.search_similar_documents()` | Retrieve relevant chunks using the rewritten follow-up query and optional document ID |
| `app.services.rag_service.format_context()` | Format retrieved chunks and generate matching source records |
| `app.services.rag_service.SYSTEM_PROMPT` | Preserve DocuMind's document-grounding and citation rules |
| `app.services.rag_service.get_language_model()` | Create the configured Gemini chat model for streaming |
| `app.services.rag_service.invoke_with_fallback()` | Contextualize follow-up questions using the configured primary/fallback models |
| `app.services.rag_service.extract_response_text()` | Normalize the query-rewrite response |
| `app.services.vector_store.list_indexed_documents()` | Populate the document-scope selector |

Task 4 does **not** upload, parse, chunk, embed, or index documents. Those responsibilities remain in Task 02. Index documents from the Task 02 application first; Task 4 is the conversational interface over that same knowledge base.

## Architecture

```text
User
  │
  ▼
Streamlit chat interface
(st.chat_message / st.chat_input)
  │
  ▼
Session State conversation manager
  ├── active conversation ID
  ├── message history
  └── independent conversations per browser session
  │
  ▼
Task 4 chat orchestration
  ├── bounded history selection
  ├── follow-up question contextualization
  └── history-aware prompt construction
  │
  ▼
Task 02 RAG services (reused, not copied)
  ├── search_similar_documents()
  └── format_context()
  │
  ▼
Task 02 Gemini model configuration
  ├── primary model streaming
  └── fallback model before first token only
  │
  ▼
Streamlit st.write_stream()
  │
  ▼
Assistant response + supporting source references
```

### Follow-up question example

1. User: “Summarize the main contribution of this paper.”
2. Assistant answers using retrieved Task 02 document passages.
3. User: “What are its limitations?”
4. Task 4 sends the recent conversation to the contextualization model and rewrites the follow-up as a standalone retrieval query, such as “What limitations of the paper's proposed contribution are described?”
5. Task 4 retrieves new evidence from Task 02's existing Chroma collection.
6. Gemini receives the recent conversation plus the newly retrieved evidence. The final answer is streamed into the chat and its sources remain expandable below the message.

The rewritten query is used for retrieval; it does not replace the user's displayed question.

## Project structure

```text
Task_04/
├── streamlit_app.py
├── chatbot/
│   ├── rag_adapter.py
│   └── chat_service.py
├── conversation/
│   └── manager.py
├── frontend/
│   ├── components.py
│   ├── sidebar.py
│   └── styles.py
├── core/
│   ├── settings.py
│   ├── exceptions.py
│   └── logging_config.py
├── tests/
│   ├── test_chat_service.py
│   └── test_conversation_manager.py
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Repository layout requirement

Keep `Task_02` and `Task_04` as sibling directories:

```text
Datheon_Python_Generative_AI_LangChain_Internship/
├── Task_01/
├── Task_02/
│   ├── app/
│   ├── data/
│   ├── chroma_db/       # created locally by Task 02; not committed
│   ├── .env             # local Gemini configuration; not committed
│   └── requirements.txt
└── Task_04/
    ├── streamlit_app.py
    └── ...
```

Task 4's adapter adds the sibling `Task_02` directory to Python's import path and imports its existing `app.services` modules. The entry point is named `streamlit_app.py` intentionally: Task 2 already has a top-level Python package named `app`, so using another `app.py` entry point could create an ambiguous import.

## Installation

Use Python 3.11, consistent with the preceding internship tasks.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install Task 4's requirements from inside its folder. The requirements file includes Task 2's dependency list, so the shared LangChain, Gemini, Chroma, embedding, Streamlit, and test dependencies stay aligned:

```bash
cd Task_04
pip install -r requirements.txt
```

## Configuration

Task 4 reuses the Gemini API key, model names, embedding configuration, Chroma directory, and collection configured for Task 2. **Do not create a second API key or a second vector database for Task 4.**

1. Make sure `Task_02/.env` exists and contains the valid configuration used by Task 2. At minimum, it should include the Gemini API key and model configuration. See `Task_02/.env.example`.
2. In `Task_04`, create the local chat settings file:

```bash
cp .env.example .env
```

3. The default `TASK2_PATH=../Task_02` is correct when the folders are siblings. Change it only if your folder layout differs.

Task 2's own `app/config.py` loads `Task_02/.env` and resolves its persistent Chroma directory relative to `Task_02`. Task 4 reads the same settings by importing Task 2's services.

## Prepare the knowledge base

If Task 2 already has indexed documents in its local `Task_02/chroma_db` directory, no re-indexing is required. Otherwise:

```bash
cd ../Task_02
PYTHONPATH=. python -m streamlit run app/ui.py
```

Upload and index your PDF/TXT files in Task 2, then stop that Streamlit process. Return to Task 4. Both applications use the same Task 02 Chroma collection and embedding configuration. Avoid running both applications as simultaneous writers to the same local Chroma directory; Task 4 is read-only with respect to document indexing.

## Run Task 4

From the `Task_04` directory:

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

### Typical workflow

1. Create a new conversation from the sidebar.
2. Select **All indexed documents** or a specific Task 02 document.
3. Ask a question about the indexed knowledge base.
4. Watch the answer stream into the chat.
5. Expand **Supporting sources** to inspect the retrieved passages.
6. Ask a follow-up naturally, such as “Can you explain that in more detail?” or “What limitations does it mention?”
7. Switch between conversations, clear the active conversation, or delete it from the sidebar.

## Session state and memory behavior

- Conversations live in `st.session_state` and are scoped to the active browser session.
- Each browser session has a separate conversation dictionary and active conversation ID; one session does not load another session's messages.
- Multiple conversations can be created and switched within one browser session.
- Conversation messages are held in application memory. They are not a permanent chat archive and can be lost when the session ends or the Streamlit process restarts.
- Only the most recent `MAX_HISTORY_MESSAGES` messages are included in each generation prompt. This bounds context growth while retaining recent conversational context.
- Document vectors are persisted by Task 02 in ChromaDB and are separate from chat history.

This is session-level conversation memory as requested by the assignment. Persistent cross-device chat history would require an authenticated identity and a durable chat database, which are outside this task's stated scope.

## Streaming and error handling

The final answer uses Task 2's configured Gemini model through LangChain's `.stream()` method and Streamlit's `st.write_stream()` renderer. If the primary model fails before producing text, Task 4 tries the configured fallback model. If a stream fails after partial output, Task 4 does not restart with a fallback because that would duplicate or contradict the already displayed answer; it displays an interruption notice instead.

If follow-up contextualization fails, Task 4 logs the error and falls back to the user's original question for retrieval. If the final model generation fails, the UI displays a safe error message without exposing credentials.

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `TASK2_PATH` | `../Task_02` | Location of the existing Task 02 package |
| `MAX_HISTORY_MESSAGES` | `12` | Number of recent messages included in generation context |
| `MAX_CONVERSATIONS` | `20` | Maximum conversation count held in one browser session |
| `DEFAULT_RETRIEVAL_K` | `6` | Fallback retrieval count if Task 2 does not expose one |
| `MAX_QUESTION_CHARS` | `4000` | Maximum submitted question length |
| `QUERY_REWRITE_ENABLED` | `true` | Contextualize follow-up questions before retrieval |
| `LOG_LEVEL` | `INFO` | Application logging verbosity |

Gemini, embedding, and Chroma variables remain in `Task_02/.env` and are documented in `Task_02/.env.example`.

## Tests

Run tests from `Task_04`:

```bash
python -m pytest tests/ -v
```

The Task 4 unit tests use fake RAG services and model streams, so the suite does not require a Gemini API request, downloaded embedding model, or populated Chroma database. Tests cover:

- Session initialization and session-scoped conversation IDs.
- Conversation creation, switching, title generation, history limits, clearing, and deletion.
- Follow-up question contextualization.
- Task 2 document-scope forwarding and retrieved-context prompt construction.
- No-context behavior and insufficient-information response.
- Token extraction from LangChain stream chunks.
- Primary-to-fallback streaming before the first token.
- No fallback restart after partial output.
- Clear failure behavior when all model attempts fail.

Run Task 2's test suite separately from its own directory:

```bash
cd ../Task_02
PYTHONPATH=. python -m pytest -v
```

## Optional Docker deployment

Build from the internship repository root so the image can include the Task 02 service code:

```bash
docker build -f Task_04/Dockerfile -t documind-task04 .
```

Run using Task 02's environment settings and persistent Chroma directory:

```bash
docker run --rm -p 8501:8501 \
  --env-file Task_02/.env \
  -e TASK2_PATH=/app/Task_02 \
  -v "$(pwd)/Task_02/chroma_db:/app/Task_02/chroma_db" \
  documind-task04
```

The first Docker build installs the dependencies declared by Task 02, including its local embedding stack. The Chroma volume keeps the indexed knowledge base available across container restarts.

## Security and operational notes

- Keep `Task_02/.env` and `Task_04/.env` out of Git; both `.gitignore` files exclude them.
- Task 4 does not persist original chat transcripts to disk or expose chat state across browser sessions.
- The Task 02 knowledge base is shared by applications that point to the same local Chroma collection. Do not put private documents in a shared deployment without adding authentication and document-level authorization.
- Retrieved document text is treated as untrusted evidence, not as system instructions.
- Source citations are evidence pointers to retrieved chunks. Review cited passages for important decisions.
- The first use of Task 2 may download the Hugging Face embedding model if it is not already cached.

---

**Task:** Task 04 — Interactive Full-Stack AI Chatbot with Streamlit  
**Pipeline:** Task 02 DocuMind RAG, reused through its existing service interface  
**Primary additions:** Streaming token output, conversation history, follow-up-aware retrieval, and explicit session-state management
