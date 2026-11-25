# DocuMind Chat — Persistent Conversational RAG

**Program:** Datheon Python + Generative AI & LangChain Internship  
**Task:** Task 04 — Conversational AI Chatbot with Streamlit

DocuMind Chat extends Task 02's existing document RAG pipeline with a persistent conversational interface. Task 04 reuses Task 02's document ingestion, chunking, embeddings, ChromaDB, retrieval, Gemini configuration, and source formatting. It adds follow-up-aware retrieval, streamed responses, conversation management, and browser-isolated SQLite history.

## Features

- Streamlit chat UI using `st.chat_message` and `st.chat_input`.
- Token streaming using `st.write_stream`.
- Conversation-aware follow-up question contextualization.
- Retrieval over all indexed documents or one selected document.
- Supporting source details, including filename, page, relevance score, and excerpt.
- Create, switch, clear, and delete conversations.
- SQLite persistence for conversations, ordered messages, and source references.
- Signed anonymous browser identity using an HMAC-protected cookie.
- Browser-scoped conversation ownership checks.
- Citation filtering that retains only citation IDs returned by Task 2.
- Primary/fallback model handling before output is emitted; avoids starting a second model after partial output has already been shown.
- Baseline-aware incremental persistence to reduce lost updates between tabs sharing one browser profile.

## Architecture

```text
Browser profile
  |
  | opaque HMAC-signed browser cookie
  v
Streamlit UI
  |
  +-- Conversation manager (session state)
  |
  +-- Per-tab persistence baseline
  |
  v
SQLite conversation store
  +-- browser_profiles
  +-- conversations
  +-- messages and source metadata
  |
  v
Task 4 ChatService
  +-- bounded history selection
  +-- follow-up question contextualization
  +-- Task 2 retrieval adapter
  +-- streaming citation filtering
  +-- Gemini primary/fallback streaming
  |
  v
Task 2 services (reused; not copied)
  +-- document retrieval
  +-- context/source formatting
  +-- Gemini model configuration
  +-- indexed document listing
```

### Task 2 integration boundary

`chatbot/rag_adapter.py` validates the sibling Task 02 package and exposes a small service interface to Task 4. Task 4 does not implement a second document loader, embedding pipeline, Chroma client, or retrieval implementation. This keeps Task 02 as the source of truth for the knowledge base.

## Project structure

```text
Task_04/
├── streamlit_app.py
├── chatbot/
│   ├── rag_adapter.py
│   └── chat_service.py
├── conversation/
│   └── manager.py
├── storage/
│   └── sqlite_store.py
├── frontend/
│   ├── components.py
│   ├── sidebar.py
│   └── styles.py
├── core/
│   ├── browser_identity.py
│   ├── settings.py
│   ├── exceptions.py
│   └── logging_config.py
├── tests/
│   ├── test_browser_identity.py
│   ├── test_chat_service.py
│   ├── test_conversation_manager.py
│   ├── test_rag_adapter.py
│   └── test_storage.py
├── .streamlit/config.toml
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Browser identity and privacy boundary

This application implements anonymous browser-profile identity, not user authentication. A random browser ID is signed with HMAC and placed in a browser cookie. The server verifies the signature and then scopes database reads and writes to that browser ID.

Important boundaries:

- People using the same browser profile share that profile's history.
- A different browser or device receives a different anonymous identity.
- A copied valid cookie can be replayed. This is not a replacement for authentication.
- The cookie controller is JavaScript-based, so the cookie is not `HttpOnly`.
- HTTPS should be used for remote deployments. The cookie is marked `Secure` when the request is detected as HTTPS and uses `SameSite=Strict`.
- Clearing cookies removes the browser's link to its previous anonymous history. The database records remain until database maintenance removes them.
- Do not use this anonymous identity model for sensitive, regulated, or account-specific information. A production application should authenticate users and map conversations to a verified user ID.

## Persistence and concurrent tabs

SQLite stores browser profiles, conversation metadata, messages, message order, and source references. It is the durable source of truth; Streamlit session state is used for the active tab's current view.

Each Streamlit tab retains its own last-saved snapshot. Saves compare the current tab state with that baseline and apply only the changes:

- New conversations and messages are inserted without replacing unrelated records created by another tab.
- A conversation is deleted only when it was present in the tab's baseline and is now absent from that tab's state.
- A stale tab does not recreate a conversation or message that was already deleted from SQLite.
- Clearing a conversation removes messages known to that tab's baseline. Messages added concurrently by another tab that this tab has not loaded are preserved; refresh the browser to load the merged history.
- Message order for independent writes from different tabs follows database commit order. It cannot reconstruct a perfect real-time ordering between simultaneous user actions.

The store retains a full-snapshot method for backwards compatibility with direct callers and existing tests. The Streamlit application uses baseline-aware incremental saves.

SQLite is suitable for a lightweight, single-instance internship deployment. For multiple application replicas, network filesystems, or high write concurrency, move the persistence interface to a managed database such as PostgreSQL.

## Installation and setup

Keep `Task_02` and `Task_04` as sibling directories:

```text
Datheon_Python_Generative_AI_LangChain_Internship/
├── Task_02/
└── Task_04/
```

From the repository root, create an environment and install Task 4's requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\\Scripts\\activate      # Windows PowerShell
python -m pip install --upgrade pip
cd Task_04
python -m pip install -r requirements.txt
```

Task 4's requirements file includes `../Task_02/requirements.txt` and the cookie controller dependency. Install from `Task_04` so the relative requirements include resolves correctly.

Create the Task 4 environment file:

```bash
cp .env.example .env
```

Make sure `Task_02/.env` exists and contains the working Gemini settings. Task 4 deliberately reads Gemini, embedding, and Chroma configuration from Task 2 rather than maintaining a second set of credentials.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TASK2_PATH` | `../Task_02` | Path to the sibling Task 2 directory |
| `CHAT_DATABASE_PATH` | `data/conversations.sqlite3` | SQLite database path; relative to Task 4 |
| `MAX_HISTORY_MESSAGES` | `12` | Recent messages included in prompts |
| `MAX_CONVERSATIONS` | `20` | Maximum conversations retained per browser |
| `DEFAULT_RETRIEVAL_K` | `6` | Fallback retrieval count |
| `MAX_QUESTION_CHARS` | `4000` | Maximum question length |
| `QUERY_REWRITE_ENABLED` | `true` | Enable contextualized follow-up queries |
| `BROWSER_COOKIE_TTL_DAYS` | `365` | Anonymous browser cookie lifetime |
| `SESSION_SIGNING_KEY` | Auto-generated local key | Optional stable HMAC key, at least 32 bytes |
| `LOG_LEVEL` | `INFO` | Server logging level |

Task 4 creates `Task_04/data/` and a local `.browser_signing_key` if `SESSION_SIGNING_KEY` is not configured. Both the database and key are excluded from Git. For deployment, configure a randomly generated key and preserve it across restarts:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Changing or losing the key invalidates existing cookies. Users then receive new anonymous browser IDs and cannot retrieve history associated with their previous cookies.

## Prepare the knowledge base

If Task 02 already contains indexed documents in `Task_02/chroma_db`, Task 4 can use them directly. Otherwise, from the repository root:

```bash
cd Task_02
source .venv/bin/activate       # activate the environment containing Task 2 dependencies
PYTHONPATH=. python -m streamlit run app/ui.py
```

Upload and index PDF/TXT documents, then stop Task 2 and return to Task 4. Both applications use the same Chroma collection and embedding configuration. Task 4 reads the knowledge base; avoid running multiple processes that write to the same local Chroma directory at the same time.

## Run Task 4

From `Task_04`:

```bash
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

Typical workflow:

1. Create a conversation from the sidebar.
2. Select all indexed documents or one document.
3. Ask a question and watch the response stream.
4. Expand supporting sources to inspect retrieved passages.
5. Ask a follow-up, such as “What are its limitations?”
6. Switch, clear, or delete conversations.
7. Refresh the page to restore history for the same browser profile.

## Tests

Run from `Task_04`:

```bash
python -m pytest tests/ -v
```

Tests use fake RAG/model objects and temporary SQLite databases. They do not make live Gemini requests or require a populated Chroma collection. Coverage includes:

- Conversation creation, switching, bounded history, clearing, deleting, and malformed-state recovery.
- Browser token signing, verification, expiry, tamper rejection, cookie reuse, and signing-key persistence.
- SQLite round trips, source preservation, active-conversation restoration, browser isolation, and ownership checks.
- Incremental concurrent-tab message merging, stale conversation deletion, and protection against later stale writes.
- Follow-up question contextualization, document-scope forwarding, streaming behavior, primary/fallback handling, and citation filtering.
- Task 2 package-layout validation.

Run Task 2's test suite separately from its directory:

```bash
cd ../Task_02
PYTHONPATH=. python -m pytest -v
```

The Task 2 embedding test may download the local Hugging Face model on its first run. Neither test suite uses a live Gemini API request unless you explicitly run the application manually.

## Docker

Build from the repository root:

```bash
docker build -f Task_04/Dockerfile -t documind-task04 .
```

Create local persistent data directories and run:

```bash
mkdir -p Task_04/data

docker run --rm -p 8501:8501 \
  --env-file Task_02/.env \
  -e TASK2_PATH=/app/Task_02 \
  -v "$(pwd)/Task_02/chroma_db:/app/Task_02/chroma_db" \
  -v "$(pwd)/Task_04/data:/app/Task_04/data" \
  documind-task04
```

The Chroma volume makes Task 2's indexed knowledge base available to the container. The Task 4 data volume preserves chat history and the generated browser signing key across container restarts. Do not run multiple replicas against a local SQLite file shared over a network filesystem.

## Operational notes and limitations

- The application is intended as a single-instance internship project, not a production multi-tenant service.
- Anonymous browser identity is not authentication and should not protect sensitive records.
- A copied browser cookie can be replayed until it expires or the signing key changes.
- Citation filtering checks reference IDs, not semantic entailment between a source passage and every sentence in the answer.
- If streaming fails after partial text has been emitted, the UI marks the response as interrupted rather than starting a fallback model and duplicating content.
- The first Task 2 use may download the configured embedding model if it is not cached.
- Users should verify important generated answers against the supporting document passages.

## AI development disclosure

Google Gemini is used at runtime for document-grounded answer generation and follow-up contextualization. ChatGPT was used as a development assistant for implementation planning, debugging discussion, test ideas, and documentation review. The developer remains responsible for reviewing, running, and validating the submitted code.

---

**Task:** Task 04 — Conversational AI Chatbot with Streamlit  
**RAG pipeline:** Task 02 DocuMind, reused through its service interface  
**Primary additions:** streaming chat, history-aware retrieval, source-aware responses, and browser-isolated SQLite persistence
