# DocuMind — Document-Based Question Answering with RAG

DocuMind is a document-grounded question-answering application. Users upload PDF or TXT files, index their contents, and ask questions through a Streamlit interface. The application retrieves relevant passages from ChromaDB and supplies those passages to Google Gemini as evidence for an answer.

The project separates document loading, chunking, embedding generation, vector storage, retrieval, and answer generation into independent modules.

## Features

- Upload PDF and TXT documents from the Streamlit interface.
- Extract text and preserve source filename and PDF page metadata.
- Normalize common PDF line-break artifacts while retaining paragraph boundaries.
- Split documents into overlapping chunks.
- Generate local semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store chunks and metadata in persistent ChromaDB storage.
- Search the complete knowledge base or restrict retrieval to one selected document.
- Display source references, page numbers, relevance scores, and readable excerpts.
- Re-index a document without intentionally removing unrelated documents.
- Support duplicate filenames by identifying a document using both its original filename and file bytes.
- Run the ingestion and question-answering pipeline from either the Streamlit interface or CLI.

## Technology stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Application runtime |
| Streamlit | Upload and question-answering interface |
| LangChain | Document, splitter, and RAG components |
| Google Gemini | Context-grounded answer generation |
| Hugging Face Sentence Transformers | Local semantic embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| ChromaDB | Persistent vector database |
| PyPDF / PyPDFLoader | PDF text extraction |
| Pytest | Automated tests |

## Project structure

```text
Task_02/
├── app/
│   ├── loaders/
│   │   └── document_loader.py
│   ├── services/
│   │   ├── chunking_service.py
│   │   ├── embedding_service.py
│   │   ├── rag_service.py
│   │   └── vector_store.py
│   ├── utils/
│   ├── config.py
│   ├── main.py
│   └── ui.py
├── tests/
│   ├── test_chunking.py
│   ├── test_document_loader.py
│   ├── test_rag_service.py
│   └── test_vector_store.py
├── data/
│   └── sample_documents/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Architecture and workflow

1. **Ingestion:** The application reads a PDF or UTF-8 TXT file. Uploads retain the filename supplied by the user even though Streamlit temporarily stores the bytes in a temporary file.
2. **Document identity:** The loader calculates a SHA-256 ID from the original filename and file bytes. The ID is attached to every page/chunk from that document.
3. **Text normalization:** PDF text is normalized to join layout line breaks while preserving paragraph boundaries. PDF page numbers are converted to one-based numbering for display.
4. **Chunking:** `RecursiveCharacterTextSplitter` creates overlapping chunks and carries source metadata forward.
5. **Embedding:** The cached Hugging Face embedding model converts each chunk into a vector locally.
6. **Indexing:** ChromaDB stores each chunk, its vector, and metadata. Re-indexing replaces chunks belonging to the same document ID after the new chunk batch is inserted.
7. **Retrieval:** A question is embedded and matched against the complete collection or a selected document ID.
8. **Generation:** Retrieved passages are formatted with citation IDs and sent to Gemini. The answer is displayed with source records so users can inspect the evidence.

## Document identity and duplicate filenames

A filename alone is not a reliable document identifier. Two different files may both be named `report.pdf`.

DocuMind computes the ID using the original filename and the raw file bytes:

```text
SHA256(length(filename) + filename + file_bytes)
```

Consequences:

- The same filename and same bytes produce the same ID.
- Different contents with the same filename produce different IDs.
- Streamlit uploads retain the original upload filename for identity and display.
- CLI directory indexing uses the same production loader and identity logic.
- Retrieval filters by document ID rather than filename, so documents with duplicate names remain independently selectable.
- Older Chroma records without a `document_id` use a legacy filename-based fallback. Legacy records sharing a filename cannot be separated reliably until they are re-indexed through the current loader.

## Installation

Clone the repository and enter Task 02:

```bash
git clone https://github.com/jim8489/Datheon_Python_Generative_AI_LangChain_Internship.git
cd Datheon_Python_Generative_AI_LangChain_Internship/Task_02
```

Create an environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\\Scripts\\activate      # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env` from the example:

```bash
cp .env.example .env
```

Configure your Gemini API key and desired model:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODEL=gemini-3.8-flash

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHROMA_PERSIST_DIRECTORY=chroma_db
CHROMA_COLLECTION_NAME=document_knowledge

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=6
MIN_RELEVANCE_SCORE=0.0
```

Use model identifiers available to your Gemini API account. Never commit `.env` or expose the API key.

## Run the application

From the `Task_02` directory, with the virtual environment activated:

```bash
PYTHONPATH=. python -m streamlit run app/ui.py
```

Open the local URL printed by Streamlit.

### Typical workflow

1. Upload one or more PDF/TXT files in the sidebar.
2. Select **Index uploaded documents**.
3. Wait for the per-file indexing process to finish.
4. Select **All documents** or one indexed document under Search settings.
5. Ask a question about the selected knowledge scope.
6. Inspect the supporting source references beneath the answer.

### CLI indexing and question answering

Index every supported PDF/TXT file in a directory:

```bash
PYTHONPATH=. python -m app.main index --directory data/sample_documents
```

Ask a question:

```bash
PYTHONPATH=. python -m app.main ask --question "What is the main topic of the indexed documents?"
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | Required | Gemini API key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Primary generation model |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.8-flash` | Fallback generation model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_PERSIST_DIRECTORY` | `chroma_db` | Persistent ChromaDB directory |
| `CHROMA_COLLECTION_NAME` | `document_knowledge` | ChromaDB collection |
| `CHUNK_SIZE` | `1000` | Maximum chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `6` | Maximum passages requested from retrieval |
| `MIN_RELEVANCE_SCORE` | `0.0` | Minimum normalized relevance score accepted |

The relevance score is a normalized display/filtering heuristic derived from the vector distance; it is not a calibrated probability that a passage supports an answer.

## Tests

Run from `Task_02`:

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

The loader tests import the production loader and cover document identity, duplicate filenames, Streamlit's `source_name` upload path, directory/CLI indexing, normalization, and invalid files. Other tests cover chunking, RAG answer formatting, vector retrieval, document-level filters, and replacement behavior.

The unit tests do not require live Gemini calls. The vector-store embedding test uses the configured Hugging Face model locally; the first run may download model files and requires sufficient memory and disk space.

## Limitations and security notes

- Scanned/image-only PDFs need OCR; OCR is not included.
- Retrieval and answer quality depend on extraction quality, chunking, embedding similarity, and the content of the indexed documents.
- Citation ID validation confirms that a citation identifier came from the retrieved source list; it does not independently prove that every cited passage entails the associated claim.
- Gemini requests require a working API key, network access, quota, and supported model identifiers.
- Uploaded document text is treated as untrusted evidence, not as instructions to follow.
- The default application is a local/single-instance demonstration, not a hardened multi-tenant document service. Add authentication, upload limits, access controls, and operational monitoring before handling private or regulated documents.

## AI development disclosure

Google Gemini is used at runtime to generate answers from retrieved document context. ChatGPT was used as a development assistant for implementation planning, debugging discussion, test ideas, and documentation review. The developer remains responsible for reviewing, running, and validating the submitted code.

---

**Program:** Datheon Python + Generative AI & LangChain Internship  
**Task:** Task 02 — DocuMind: Document-Based Question Answering with RAG
