# Structured LLM Prompt Pipeline & JSON Extractor

A Python-based document information extraction system that uses Google's Gemini API to convert unstructured resumes and invoices into structured, validated JSON.

The project demonstrates parameterized prompt engineering, few-shot prompting, LLM integration, structured output generation, Pydantic validation, retry handling, and automated testing.

## Features

- **Resume extraction:** Extracts names, contact information, professional summaries, skills, education, experience, and certifications.
- **Invoice extraction:** Extracts invoice metadata, seller and buyer details, line items, quantities, prices, taxes, and totals.
- **Parameterized prompts:** Uses Jinja2 templates to dynamically construct prompts.
- **Few-shot prompting:** Provides examples and extraction guidance to improve consistency.
- **Structured JSON output:** Uses Gemini's JSON response mode and Pydantic schemas.
- **Data validation:** Validates generated responses against predefined data models.
- **Automatic retry and fallback:** Retries temporary API failures and switches to a fallback Gemini model.
- **Document support:** Reads TXT and text-based PDF files.
- **Automated testing:** Includes unit tests with mocked API responses.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core programming language |
| Google Gemini API | LLM-powered information extraction |
| Google Gen AI SDK | Gemini API integration |
| Pydantic | Structured data validation |
| Jinja2 | Dynamic prompt templates |
| PyPDF | PDF text extraction |
| python-dotenv | Environment configuration |
| Pytest | Automated testing |

## Project Structure

```text
Task_01/
├── app/
│   ├── schemas/
│   │   ├── resume.py
│   │   └── invoice.py
│   ├── services/
│   │   └── llm_service.py
│   ├── prompts/
│   │   ├── resume.jinja2
│   │   ├── invoice.jinja2
│   │   └── few_shot_examples.py
│   ├── utils/
│   │   └── pdf_reader.py
│   ├── config.py
│   └── main.py
├── examples/
│   ├── sample_resume.txt
│   └── sample_invoice.txt
├── tests/
│   ├── test_schemas.py
│   └── test_llm_service.py
├── outputs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Datheon_Python_GenAI_LangChain_Internship/Task_01
```

### 2. Create and activate a virtual environment

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
pip install -r requirements.txt
```

### 4. Configure the Gemini API

Create a `.env` file in the `Task_01` directory:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite
MAX_RETRIES=3
```

Replace the placeholder with your own Gemini API key.

Never commit your `.env` file or expose your API key publicly.

## Usage

### Extract information from a resume

```bash
python -m app.main \
    --type resume \
    --input examples/sample_resume.txt \
    --output outputs/resume.json
```

### Extract information from an invoice

```bash
python -m app.main \
    --type invoice \
    --input examples/sample_invoice.txt \
    --output outputs/invoice.json
```

### PDF document extraction

The application also accepts text-based PDF documents:

```bash
python -m app.main \
    --type resume \
    --input examples/resume.pdf \
    --output outputs/resume.json
```

Scanned PDFs that contain only images require OCR, which is not currently included.

## Testing

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

The tests cover:

- Resume and invoice schema validation.
- Missing and invalid field handling.
- Prompt template rendering.
- Empty document rejection.
- Retryable API error detection.
- Automatic fallback model handling.
- Structured response validation.
- Invalid JSON rejection.

Tests use mocked API responses where appropriate, so the automated test suite does not require live Gemini API requests.

## Architecture

The application follows a modular extraction pipeline:

1. **Document input:** Reads a TXT or PDF document.
2. **Prompt construction:** Loads the relevant Jinja2 template and incorporates few-shot examples.
3. **LLM processing:** Sends the constructed prompt to the configured Gemini model.
4. **Error handling:** Retries temporary failures and attempts the fallback model.
5. **Structured response:** Requests JSON output according to the selected Pydantic schema.
6. **Validation:** Validates the generated JSON before accepting it.
7. **Export:** Saves the validated result as a JSON file.

## Design Considerations

- Missing information is represented using `null` or empty lists where appropriate.
- The model is instructed not to invent information absent from the source document.
- Document content is treated as untrusted input rather than as instructions.
- API configuration is managed through environment variables.
- Extraction and validation are separated into reusable modules.
- API failures are handled without silently accepting invalid output.

## Current Test Results

**10 automated tests passed.**

The application has also been tested with live Gemini API requests for both resume and invoice extraction, including successful fallback-model execution after temporary HTTP 503 errors.

## Future Improvements

- OCR support for scanned documents.
- Additional document types and customizable extraction schemas.
- More comprehensive evaluation using a larger labeled test dataset.
- Batch document processing.
- A web interface for uploading documents and viewing extracted information.

---

**Internship Project:** Datheon Python + Generative AI & LangChain Internship  
**Task:** Structured LLM Prompt Pipeline & JSON Extractor
