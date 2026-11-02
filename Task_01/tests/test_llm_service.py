
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.resume import ResumeData
from app.services.llm_service import (
    GeminiExtractionService,
    LLMExtractionError,
)


def create_test_service():
    """Create a service without initializing the Gemini API client."""

    service = object.__new__(GeminiExtractionService)

    prompts_directory = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "prompts"
    )

    service.environment = Environment(
        loader=FileSystemLoader(prompts_directory),
        undefined=StrictUndefined,
        autoescape=False,
    )

    return service


def test_resume_prompt_rendering():
    service = create_test_service()

    prompt = service.render_prompt(
        template_name="resume.jinja2",
        document_text="John Doe. Python developer.",
        examples=[
            {
                "input": "Jane Doe. Django developer.",
                "guidance": "Extract the name and skills.",
            }
        ],
    )

    assert "John Doe" in prompt
    assert "Python developer" in prompt
    assert "Jane Doe" in prompt
    assert "Extract the name and skills" in prompt


def test_empty_document_is_rejected():
    service = create_test_service()

    with pytest.raises(ValueError):
        service.render_prompt(
            template_name="resume.jinja2",
            document_text="   ",
        )


def test_retryable_status_codes():
    assert GeminiExtractionService.is_retryable(
        SimpleNamespace(code=503)
    )

    assert GeminiExtractionService.is_retryable(
        SimpleNamespace(code=429)
    )

    assert GeminiExtractionService.is_retryable(
        SimpleNamespace(code=500)
    )

    assert not GeminiExtractionService.is_retryable(
        SimpleNamespace(code=400)
    )


def test_fallback_model_after_primary_model_failure():
    service = create_test_service()

    service.client = MagicMock()
    service.models = ["primary-model", "fallback-model"]

    class TemporaryAPIError(Exception):
        code = 503

    fallback_response = SimpleNamespace(
        text='{"name": "John Doe"}'
    )

    service.client.models.generate_content.side_effect = [
        TemporaryAPIError("Service unavailable"),
        fallback_response,
    ]

    with patch("app.services.llm_service.MAX_RETRIES", 1):
        with patch("app.services.llm_service.time.sleep"):
            result = service.generate_with_retry(
                prompt="Extract the resume.",
                response_schema=ResumeData,
            )

    assert result == '{"name": "John Doe"}'
    assert service.client.models.generate_content.call_count == 2


def test_extract_validates_structured_response():
    service = create_test_service()

    valid_response = """
    {
        "name": "John Doe",
        "email": null,
        "phone": null,
        "location": null,
        "professional_summary": "Python developer",
        "skills": ["Python", "Django"],
        "education": [],
        "experience": [],
        "certifications": []
    }
    """

    with patch.object(
        service,
        "generate_with_retry",
        return_value=valid_response,
    ):
        result = service.extract(
            document_text="John Doe is a Python developer.",
            template_name="resume.jinja2",
            response_schema=ResumeData,
        )

    assert isinstance(result, ResumeData)
    assert result.name == "John Doe"
    assert "Python" in result.skills


def test_invalid_json_is_rejected():
    service = create_test_service()

    with patch.object(
        service,
        "generate_with_retry",
        return_value='{"name": "John Doe", "skills": "not-a-list"}',
    ):
        with pytest.raises(LLMExtractionError):
            service.extract(
                document_text="John Doe",
                template_name="resume.jinja2",
                response_schema=ResumeData,
            )
