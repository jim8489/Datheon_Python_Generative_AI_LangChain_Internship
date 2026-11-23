# DocuMind — Document-Based Question Answering with RAG

DocuMind is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF and TXT documents, build a searchable knowledge base, and ask questions about the uploaded content.

Instead of relying only on the language model's general knowledge, DocuMind retrieves relevant passages from indexed documents and uses that context to generate answers. Retrieved information can be inspected through supporting source references.

The application is built with Python, LangChain, Hugging Face sentence embeddings, ChromaDB, Google Gemini, and Streamlit.

## Features

- **Document upload:** Upload PDF and TXT documents through a Streamlit interface.
- **PDF and TXT processing:** Extract text from supported documents before indexing.
- **Text chunking:** Split document content into smaller, overlapping chunks for retrieval.
- **Semantic embeddings:** Generate vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
- **Persistent vector storage:** Store document embeddings and metadata in ChromaDB.
- **Retrieval-Augmented Generation:** Retrieve relevant document passages and provide them as context to Gemini.
- **Document-specific search:** Ask questions across the entire knowledge base or restrict retrieval to one selected document.
- **Duplicate filename support:** Index different documents even when they share the same filename.
- **Stable document identity:** Generate document IDs from the original filename and file content.
- **Safe re-indexing:** Re-indexing a document replaces its existing chunks without removing unrelated documents.
- **Source references:** Display supporting document information with generated answers.
- **Configurable retrieval:** Adjust chunk size, overlap, retrieval count, and relevance settings through environment variables.
- **Automated testing:** Test document loading, chunking, vector storage, and RAG behavior with Pytest.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core programming language |
| LangChain | RAG pipeline and component integration |
| Google Gemini | Context-based answer generation |
| `langchain-google-genai` | Gemini integration |
| Hugging Face Sentence Transformers | Semantic embedding generation |
| `all-MiniLM-L6-v2` | Text embedding model |
| ChromaDB | Persistent vector database |
| `langchain-chroma` | LangChain integration with ChromaDB |
| PyPDF | PDF text extraction |
| Streamlit | Interactive web interface |
| python-dotenv | Environment variable management |
| Pytest | Automated testing |

## Project Structure

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
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Architecture and Workflow

DocuMind follows a modular Retrieval-Augmented Generation workflow.

### 1. Document ingestion

Users upload PDF or TXT files through the Streamlit interface. The document loader extracts their text and preserves relevant metadata, including the original filename.

### 2. Document identification

Each newly indexed document receives a stable document ID generated using SHA-256 from its original filename and file content.

This allows two different documents with the same filename to remain independently identifiable.

### 3. Text chunking

The extracted text is divided into smaller chunks with configurable overlap. Chunking helps the retrieval system identify relevant sections without processing an entire document as one large text block.

### 4. Embedding generation

The application uses the Hugging Face embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

The model converts text chunks into numerical vector representations suitable for semantic similarity search.

### 5. Vector storage

The generated embeddings, document chunks, and associated metadata are stored in a persistent ChromaDB collection.

### 6. Question retrieval

When a user submits a question, the application searches the vector database for relevant chunks.

Users can search across all indexed documents or select a particular document to restrict retrieval.

### 7. Answer generation

The retrieved passages are supplied as context to the configured Google Gemini model. Gemini generates an answer based on the retrieved information.

The application also provides supporting source information so users can inspect the document context behind an answer.

## Handling Duplicate Filenames

A key improvement in DocuMind is its document identity mechanism.

A filename alone is not sufficient to identify a document. For example, two users may upload different files named `report.pdf`.

To avoid treating them as the same document, DocuMind generates a SHA-256 document ID using the original filename and file content.

The resulting behavior is:

- Different contents with the same filename receive different document IDs.
- The same filename and content produce the same document ID.
- Re-indexing an existing document replaces its previous chunks.
- Re-indexing one document does not remove chunks belonging to another document.
- Document selection and retrieval use document IDs rather than relying only on filenames.

Older indexed records without document IDs are handled through a legacy compatibility mechanism. They remain identifiable using their available filename metadata.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Datheon_Python_GenAI_LangChain_Internship/Task_02
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual repository URL.

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the `Task_02` directory using `.env.example` as a template.

```bash
cp .env.example .env
```

Configure the following values:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHROMA_PERSIST_DIRECTORY=chroma_db
CHROMA_COLLECTION_NAME=document_knowledge

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=6
MIN_RELEVANCE_SCORE=0.0
```

Replace the placeholder with a valid Gemini API key.

**Security:** Never commit the `.env` file or expose API credentials publicly. The `.env.example` file contains configuration placeholders only.

## Running the Application

From the `Task_02` directory, activate the virtual environment and run:

```bash
PYTHONPATH=. python -m streamlit run app/ui.py
```

The Streamlit interface will be available at the local URL displayed in the terminal.

### Typical usage

1. Open the DocuMind application.
2. Upload a PDF or TXT document.
3. Allow the application to process and index the document.
4. Select a specific indexed document or choose the entire knowledge base.
5. Enter a question about the document content.
6. Review the generated answer and its supporting source information.

When selecting a specific document, retrieval is restricted to that document rather than the entire knowledge base.

## Configuration

The following environment variables control the main application settings.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | Required | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Answer-generation model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHROMA_PERSIST_DIRECTORY` | `chroma_db` | Persistent vector database directory |
| `CHROMA_COLLECTION_NAME` | `document_knowledge` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Maximum configured chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `6` | Number of relevant chunks to retrieve |
| `MIN_RELEVANCE_SCORE` | `0.0` | Configured minimum relevance threshold |

## Running Tests

The project includes automated tests for its main application components.

From the `Task_02` directory, run:

```bash
PYTHONPATH=. python -m pytest -v
```

The test suite covers:

- Document loading and validation.
- Document metadata and stable identity generation.
- Text chunking behavior.
- Vector database indexing and retrieval.
- Document-specific retrieval.
- Handling of documents with duplicate filenames.
- Safe document replacement and preservation of unrelated indexed documents.
- RAG service behavior.

Tests should be run from the `Task_02` directory with the project root on `PYTHONPATH` so that application modules can be imported consistently.

## Design Considerations

- **Modularity:** Document loading, chunking, embedding, vector storage, and answer generation are separated into dedicated modules.
- **Persistent storage:** ChromaDB retains indexed data between application sessions.
- **Document isolation:** Stable document IDs prevent filename collisions from mixing distinct documents.
- **Safe replacement:** New chunks are inserted before old chunks for the same document are removed, reducing the risk of losing existing indexed content during a failed replacement.
- **Configurable behavior:** Important retrieval and processing settings are managed through environment variables.
- **Context-based answers:** Retrieved document passages provide the context for Gemini responses.
- **Legacy compatibility:** Existing records without document IDs are supported through filename-based fallback handling.

## Limitations

- Answer quality depends on the quality and completeness of the uploaded document text and the retrieved context.
- Scanned PDFs that contain only images may require OCR before their contents can be searched.
- Gemini API access requires a valid API key and may be subject to API quotas or service availability.
- The application is designed for document-based question answering; generated answers should be checked against the supporting source material.

## AI Tools and Development Disclosure

Google Gemini is used as the language model for generating answers from retrieved document context.

AI assistance was also used during development for implementation guidance, debugging, test development, and documentation. The application architecture, integration, and testing were reviewed and refined as part of the development process.

## Internship Project

**Program:** Datheon Python + Generative AI & LangChain Internship  
**Task:** Task 02 — DocuMind: Document-Based Question Answering with RAG