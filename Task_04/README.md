# DocuMind Chat — Persistent Conversational RAG

**Internship:** Datheon Python + Generative AI & LangChain Internship
**Task:** Task 04 — Interactive Full-Stack AI Chatbot with Streamlit

DocuMind Chat extends the existing Task 02 RAG pipeline with a conversational Streamlit interface. Task 4 reuses Task 2's document ingestion, chunking, embeddings, ChromaDB, retrieval, Gemini configuration, and source formatting. It adds follow-up-aware retrieval, token streaming, separate conversations, and durable SQLite chat history.

## What Task 4 provides

- Interactive chat using Streamlit's `st.chat_message` and `st.chat_input`.
- Token-by-token model output through `st.write_stream`.
- Multi-turn conversation history and contextualized follow-up questions.
- Retrieval across all indexed documents or within one selected document.
- Expandable source references with filename, page, relevance score, and excerpt.
- Create, switch, clear, and delete conversation controls.
- SQLite persistence: conversations and message sources survive refreshes and app restarts.
- Anonymous browser-level isolation using an opaque, HMAC-signed browser identifier stored in a browser cookie.
- Citation validation during streaming: only citation IDs returned by Task 2 are retained.
- Primary-to-fallback model handling only before any output is emitted, preventing duplicate text after partial output.

## Architecture

```text
Browser
  │
  ├── Signed anonymous browser cookie
  │      └── HMAC verification (server-side signing key)
  │
  ▼
Streamlit UI + Conversation Manager
  │
  ├── Session State (current UI state)
  │
  ▼
SQLite Conversation Store
  ├── browser_profiles
  ├── conversations
  └── messages + serialized source references
  │
  ▼
Task 4 Chat Service
  ├── bounded history selection
  ├── follow-up question contextualization
  ├── Task 2 retrieval adapter
  ├── streaming citation validation
  └── Gemini primary/fallback streaming
  │
  ▼
Task 02 RAG Services (reused, not copied)
  ├── search_similar_documents()
  ├── format_context()
  ├── get_language_model()
  └── list_indexed_documents()
```

### Important identity and privacy boundary

This task uses **anonymous browser identity**, not user accounts. Each browser profile receives a random identifier in a signed cookie. The server verifies the signature before using that identifier to load chat history. Every database read and write is scoped to that browser identifier; a conversation ID alone is never treated as authorization.

This is suitable for demonstrating multi-visitor isolation in an internship MVP without adding a full authentication system. It is not equivalent to authenticated user identity:

- Users in the same browser profile share that profile's chat history.
- A user cannot automatically access the same chats from another device or browser profile.
- Anyone who obtains and replays the browser cookie may act as that anonymous browser. Use proper authentication for accounts, sensitive information, or cross-device history.
- The cookie controller is JavaScript-based, so the cookie is not `HttpOnly`. Serve deployments over HTTPS, keep dependencies updated, and do not put sensitive personal data in an anonymous chat.
- Clearing browser cookies loses the browser's ability to retrieve its previous history. The old database records remain until removed through database maintenance.

For an authenticated product, replace anonymous browser identity with a verified user ID from an authentication provider and keep the same owner-scoped database access pattern.

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

## Repository layout requirement

Keep `Task_02` and `Task_04` as sibling directories:

```text
Datheon_Python_Generative_AI_LangChain_Internship/
├── Task_01/
├── Task_02/
│   ├── app/
│   ├── data/
│   ├── chroma_db/       # created locally by Task 2; not committed
│   ├── .env             # local Gemini configuration; not committed
│   └── requirements.txt
└── Task_04/
    ├── streamlit_app.py
    ├── data/            # generated locally; ignored by Git
    └── ...
```

Task 4 imports Task 2's existing `app.services` modules. Its entry point is named `streamlit_app.py` because Task 2 already has a top-level Python package named `app`.

## Installation

Use Python 3.11, consistent with the preceding internship tasks. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cd Task_04
pip install -r requirements.txt
```

Task 4's requirements include Task 2's dependency file and the cookie controller used for browser identity. Do not create a second vector database or a second Gemini key for Task 4.

## Configuration

1. Make sure `Task_02/.env` exists and has the working Gemini configuration used by Task 2. Refer to `Task_02/.env.example`.
2. From `Task_04`, create the local settings file:

   ```bash
   cp .env.example .env
   ```

3. The default `TASK2_PATH=../Task_02` is correct when both tasks are sibling directories.
4. The default chat database is `Task_04/data/conversations.sqlite3`. Task 4 creates the directory and database automatically.
5. Task 4 creates `Task_04/data/.browser_signing_key` automatically when `SESSION_SIGNING_KEY` is not configured. Both the database and this signing key are excluded from Git.

For a stable deployment, set `SESSION_SIGNING_KEY` to a randomly generated secret of at least 32 bytes and preserve it across restarts. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Keep the key private. Rotating or losing it invalidates existing browser cookies, so users receive new browser identities and cannot retrieve their earlier histories with the old cookie.

## Prepare the knowledge base

If Task 2 already has indexed documents in `Task_02/chroma_db`, no re-indexing is required. Otherwise, start Task 2 from its directory:

```bash
cd ../Task_02
PYTHONPATH=. python -m streamlit run app/ui.py
```

Upload and index PDF/TXT files in Task 2, then stop that Streamlit process and return to Task 4. Both apps use the same Task 2 Chroma collection and embedding configuration. Avoid running both apps as simultaneous writers to the same local Chroma directory; Task 4 only reads from it.

## Run Task 4

From the `Task_04` directory:

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

Typical workflow:

1. Create a conversation from the sidebar.
2. Select **All indexed documents** or one document.
3. Ask a question and watch the response stream.
4. Expand **Supporting sources** to inspect the retrieved passages.
5. Ask a follow-up such as “What are its limitations?”
6. Switch, clear, or delete conversations. The changes are saved to SQLite.
7. Refresh the browser. The same browser profile should see its prior conversation list and active conversation restored.

## Persistence and isolation behavior

- `st.session_state` is used for the active Streamlit interaction, not as the long-term database.
- SQLite persists conversation metadata, ordered messages, source references, and the active conversation ID.
- The signed cookie identifies one anonymous browser profile. The identifier is validated before any history is loaded.
- Queries use the browser ID as an ownership filter. Guessing another conversation ID is not sufficient to read or change it.
- The default maximum is 20 conversations per browser profile. When the limit is exceeded, the least recently active conversations are removed.
- The default browser cookie lifetime is 365 days. `BROWSER_COOKIE_TTL_DAYS` can change this value.
- Only the most recent `MAX_HISTORY_MESSAGES` are added to each generation prompt. The full transcript remains stored for display.
- SQLite is appropriate for a lightweight, single-instance internship deployment. For multiple application replicas or high write concurrency, move this storage interface to a managed database such as PostgreSQL.

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `TASK2_PATH` | `../Task_02` | Location of the existing Task 02 package |
| `CHAT_DATABASE_PATH` | `data/conversations.sqlite3` | SQLite database path; relative paths resolve from Task 04 |
| `MAX_HISTORY_MESSAGES` | `12` | Recent messages included in the generation context |
| `MAX_CONVERSATIONS` | `20` | Maximum conversations retained per browser |
| `DEFAULT_RETRIEVAL_K` | `6` | Fallback retrieval count if Task 2 does not expose one |
| `MAX_QUESTION_CHARS` | `4000` | Maximum submitted question length |
| `QUERY_REWRITE_ENABLED` | `true` | Contextualize follow-up questions before retrieval |
| `BROWSER_COOKIE_TTL_DAYS` | `365` | Browser identity cookie lifetime |
| `SESSION_SIGNING_KEY` | auto-generated local key | Optional deployment key used to sign browser identities |
| `LOG_LEVEL` | `INFO` | Application logging verbosity |

Gemini, embedding, and Chroma settings remain in `Task_02/.env` and are documented in `Task_02/.env.example`.

## Tests

Run from `Task_04`:

```bash
python -m pytest tests/ -v
```

The unit tests use fake RAG services and model streams. They do not make Gemini API calls or require a populated Chroma database. Coverage includes:

- Conversation creation, switching, history limits, clearing, deleting, and malformed-state recovery.
- Signed browser token creation, signature verification, expiry, cookie reuse, and key persistence.
- SQLite round trips, source preservation, active-conversation restoration, deletion synchronization, and browser-to-browser isolation.
- Rejection of attempts to claim another browser's conversation record.
- Follow-up question contextualization, document-scope forwarding, context construction, and greeting behavior.
- Token extraction, valid/invalid citation filtering across chunk boundaries, and model fallback behavior.

Run Task 2's tests separately from its own directory:

```bash
cd ../Task_02
PYTHONPATH=. python -m pytest -v
```

## Docker deployment

Build from the internship repository root. The ZIP also includes a root-level `.dockerignore`; place it at the repository root beside `Task_02` and `Task_04`.

```bash
docker build -f Task_04/Dockerfile -t documind-task04 .
```

Create the persistent data directory and run the container:

```bash
mkdir -p Task_04/data
docker run --rm -p 8501:8501 \
  --env-file Task_02/.env \
  -e TASK2_PATH=/app/Task_02 \
  -v "$(pwd)/Task_02/chroma_db:/app/Task_02/chroma_db" \
  -v "$(pwd)/Task_04/data:/app/Task_04/data" \
  documind-task04
```

The Chroma volume keeps Task 2's knowledge base available, while the Task 4 data volume preserves conversations and the automatically generated browser signing key across container restarts. Do not run multiple replicas against a local SQLite file shared over a network filesystem.

## Security and operational notes

- Keep both `Task_02/.env` and `Task_04/.env` out of Git.
- Keep the SQLite database and browser signing key out of Git and Docker build contexts.
- Serve remote deployments over HTTPS. The cookie is marked `Secure` when the request URL or forwarded protocol indicates HTTPS, and uses `SameSite=Strict`.
- Browser identity is anonymous and not suitable for regulated or highly sensitive information. Add authentication and authorization for a production service.
- Retrieved document text is treated as untrusted evidence, not as system instructions.
- Citation filtering removes references that were not present in the retrieved source list. It does not prove that a cited passage semantically supports every claim.
- The first use of Task 2 may download its configured embedding model if it is not already cached.

---

**Task:** Task 04 — Conversational AI Chatbot with Streamlit
**RAG pipeline:** Task 02 DocuMind, reused through its service interface
**Primary additions:** streaming chat, history-aware retrieval, browser-isolated SQLite persistence, and source-aware responses
