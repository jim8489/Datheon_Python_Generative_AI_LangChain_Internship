# Datheon Python + Generative AI & LangChain Internship

This repository contains the completed practical project work for the Datheon Python + Generative AI & LangChain Internship. The implementations demonstrate structured LLM extraction, document-based Retrieval-Augmented Generation (RAG), and a persistent conversational interface built on the existing RAG services.

## Repository structure

```text
Datheon_Python_Generative_AI_LangChain_Internship/
├── Task_01/    # Structured LLM prompt pipeline and JSON extractor
├── Task_02/    # DocuMind document-based question answering with RAG
├── Task_04/    # DocuMind Chat: persistent conversational RAG
├── .dockerignore
├── .gitignore
└── README.md
```

Each task contains its own README, dependency configuration, and tests where applicable.

## Projects

### Task 01 — Structured LLM Prompt Pipeline & JSON Extractor

Extracts structured resume and invoice information from TXT/PDF documents using Gemini, Jinja2 prompt templates, few-shot examples, and Pydantic schemas. The application includes retry/fallback handling and a command-line interface.

[Read Task 01 documentation](Task_01/README.md)

### Task 02 — DocuMind: Document-Based Question Answering with RAG

A Streamlit application for uploading PDF/TXT documents, chunking and embedding their contents, indexing them in ChromaDB, and answering questions using retrieved context and Gemini. It supports document-level retrieval and source references.

[Read Task 02 documentation](Task_02/README.md)

### Task 04 — DocuMind Chat: Persistent Conversational RAG

A conversational layer on top of Task 02. It adds follow-up question contextualization, streamed responses, source inspection, conversation controls, and browser-profile-scoped SQLite persistence. It reuses Task 02's retrieval and model services rather than copying the RAG pipeline.

[Read Task 04 documentation](Task_04/README.md)

## Quick setup

Use Python 3.11 or newer. Clone the repository:

```bash
git clone https://github.com/jim8489/Datheon_Python_Generative_AI_LangChain_Internship.git
cd Datheon_Python_Generative_AI_LangChain_Internship
```

Each task has its own installation and configuration instructions. Task 4 must remain beside Task 2 because it imports the existing Task 2 services.

## Running tests

Run each suite from its task directory:

```bash
# Task 01
cd Task_01
python -m pytest tests/ -v

# Task 02
cd ../Task_02
PYTHONPATH=. python -m pytest tests/ -v

# Task 04
cd ../Task_04
python -m pytest tests/ -v
```

Task 1 and Task 2 tests mock Gemini calls where appropriate. Task 2's vector-store test uses the local embedding model and may download it on first use. Task 4 uses fake RAG/model objects and temporary SQLite databases; it does not require a live Gemini API request.

## Configuration and secrets

- Copy each `.env.example` to a local `.env` and add your own credentials.
- Task 4 uses Task 2's Gemini and Chroma configuration; keep `Task_02` and `Task_04` as sibling directories.
- Never commit API keys, `.env` files, signing keys, SQLite databases, Chroma data, or virtual environments.
- The `.gitignore` and `.dockerignore` files exclude local secrets and generated data.

## AI development disclosure

Google Gemini is used at runtime in the LLM features described in the task READMEs. ChatGPT was used as a development assistant for implementation planning, debugging discussion, test ideas, and documentation review. The developer remains responsible for reviewing, running, and validating the submitted code.

---

**Program:** Datheon Python + Generative AI & LangChain Internship  
**Repository:** [jim8489/Datheon_Python_Generative_AI_LangChain_Internship](https://github.com/jim8489/Datheon_Python_Generative_AI_LangChain_Internship)
