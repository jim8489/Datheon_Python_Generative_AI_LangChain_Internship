# Datheon Python + Generative AI & LangChain Internship

This repository contains my practical projects and task submissions for the Datheon Python + Generative AI & LangChain Internship.

The projects focus on Python development, Generative AI integration, prompt engineering, structured LLM applications, and building practical AI-powered solutions.

## Repository Structure

```text
Datheon_Python_+_Generative_AI_&_LangChain_Internship/
│
├── Task_01/
│   ├── app/
│   ├── examples/
│   ├── tests/
│   ├── outputs/
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md
```

## Completed Tasks

### Task 01 — Structured LLM Prompt Pipeline & JSON Extractor

A Python application that extracts structured information from resumes and invoices using Google's Gemini API.

**Key features:**
- Gemini API integration using the Google Gen AI SDK.
- Parameterized prompt templates using Jinja2.
- Few-shot prompting with extraction examples.
- Structured JSON generation.
- Pydantic schema validation.
- Automatic retry and fallback model handling.
- TXT and text-based PDF document support.
- Automated tests using Pytest and mocked API responses.

**Technologies:** Python, Google Gemini API, Pydantic, Jinja2, PyPDF, python-dotenv, Pytest.

**Test status:** 10 automated tests passed.

See [Task 01 README](Task_01/README.md) for installation, configuration, architecture, and usage instructions.

## Technologies and Tools

- Python
- Generative AI and LLM APIs
- Prompt Engineering
- LangChain (for applicable tasks)
- Pydantic
- REST APIs
- Pytest
- Git and GitHub

## Running a Task

Each task has its own folder and README containing its dependencies, setup instructions, and execution steps.

For example, to run Task 01:

```bash
cd Task_01
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the required environment variables using the task's `.env.example` file, then follow the task-specific README.

## Security

API keys and environment-specific configuration files are excluded from version control. Never commit real API credentials.

## Internship

**Program:** Datheon Python + Generative AI & LangChain Internship  
**Repository purpose:** Practical task implementations and project submissions.
