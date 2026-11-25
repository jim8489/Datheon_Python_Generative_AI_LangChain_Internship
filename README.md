# Datheon Python + Generative AI & LangChain Internship

This repository contains practical task implementations for the Datheon Python + Generative AI & LangChain Internship. The projects cover Python application development, structured LLM extraction, retrieval-augmented generation, conversational AI, and integration of existing services into user-facing applications.

## Repository structure

```text
Datheon_Python_Generative_AI_LangChain_Internship/
├── Task_01/    # Structured LLM Prompt Pipeline & JSON Extractor
├── Task_02/    # DocuMind document-based question answering with RAG
├── Task_04/    # Persistent conversational RAG chatbot with Streamlit
├── .dockerignore
└── README.md
```

Each task has its own README, dependency file, configuration example, and tests where applicable.

## Completed tasks

### Task 01 — Structured LLM Prompt Pipeline & JSON Extractor

A Python application that extracts structured information from resumes and invoices using Google's Gemini API.

**Key features:** Gemini API integration, Jinja2 prompt templates, few-shot prompting, Pydantic schema validation, retry/fallback handling, PDF/TXT support, and automated tests.

See [Task 01 README](Task_01/README.md).

### Task 02 — DocuMind: Document-Based Question Answering with RAG

A Streamlit application that ingests PDF and TXT documents, chunks their contents, creates Hugging Face embeddings, stores vectors in ChromaDB, and generates context-grounded answers with Gemini. It supports document-specific retrieval and supporting source references.

**Technologies:** Python, LangChain, Google Gemini, Sentence Transformers, ChromaDB, PyPDF, Streamlit, Pytest.

See [Task 02 README](Task_02/README.md).

### Task 04 — DocuMind Chat: Persistent Conversational RAG

A conversational layer built on top of Task 02's existing RAG services. It adds streaming responses, follow-up question contextualization, conversation management, source inspection, and SQLite persistence. Anonymous browser identities are signed and verified to isolate each browser profile's chat history without introducing an account system.

**Technologies:** Python, Streamlit, LangChain, Google Gemini, SQLite, ChromaDB, Pytest.

See [Task 04 README](Task_04/README.md).

## Running a task

Each task is self-contained within its own folder. Create and activate a Python virtual environment, install that task's requirements, configure its `.env` using the provided `.env.example`, and follow its README.

Task 04 deliberately reuses Task 02. Keep `Task_02` and `Task_04` as sibling directories and configure the Gemini key in `Task_02/.env`.

## Tests

Run tests from the relevant task directory, using the commands documented in its README. Task 04 tests use fake RAG/model objects and a temporary SQLite database, so they do not require live Gemini requests or a populated Chroma collection.

## Security

Never commit real API credentials, environment files, local SQLite databases, browser signing keys, or Chroma data. The `.gitignore` and root `.dockerignore` files exclude local secrets and generated data from source control and Docker build contexts.

## Internship

**Program:** Datheon Python + Generative AI & LangChain Internship  
**Repository purpose:** Practical task implementations and project submissions.
