
import logging
import time
from pathlib import Path
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODEL,
    MAX_RETRIES,
    validate_config,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMExtractionError(Exception):
    """Raised when structured extraction fails."""


class GeminiExtractionService:

    def __init__(self):

        validate_config()

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompts_directory = (
            Path(__file__).resolve().parents[1] / "prompts"
        )

        self.environment = Environment(
            loader=FileSystemLoader(prompts_directory),
            undefined=StrictUndefined,
            autoescape=False,
        )

        # Primary and fallback models
        self.models = list(dict.fromkeys([
            GEMINI_MODEL,
            GEMINI_FALLBACK_MODEL,
        ]))

        logger.info(
            "Gemini service initialized. Models: %s",
            self.models,
        )

    def render_prompt(
        self,
        template_name: str,
        document_text: str,
        examples: list | None = None,
    ) -> str:

        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty.")

        try:
            template = self.environment.get_template(
                template_name
            )

            return template.render(
                document_text=document_text,
                examples=examples or [],
            )

        except Exception as error:
            raise LLMExtractionError(
                f"Prompt rendering failed: {error}"
            ) from error

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """
        Determine whether an API error is temporary.
        """

        status_code = getattr(
            error,
            "code",
            getattr(error, "status_code", None),
        )

        return status_code in (
            408, 429, 500, 502, 503, 504
        )

    def generate_with_retry(
        self,
        prompt: str,
        response_schema: Type[T],
    ):

        last_error = None

        # Try primary model, then fallback model
        for model in self.models:

            for attempt in range(1, MAX_RETRIES + 1):

                try:

                    logger.info(
                        "Using model %s (attempt %s/%s)",
                        model,
                        attempt,
                        MAX_RETRIES,
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            temperature=0.1,
                        ),
                    )

                    if not response.text:
                        raise LLMExtractionError(
                            "Gemini returned an empty response."
                        )

                    logger.info(
                        "Response received from %s",
                        model,
                    )

                    return response.text

                except Exception as error:

                    last_error = error

                    if not self.is_retryable(error):
                        logger.error(
                            "Non-retryable error from %s: %s",
                            model,
                            error,
                        )
                        raise LLMExtractionError(
                            f"Gemini request failed: {error}"
                        ) from error

                    if attempt < MAX_RETRIES:

                        wait_time = 2 ** attempt

                        logger.warning(
                            "Temporary API failure from %s. "
                            "Retrying in %s seconds.",
                            model,
                            wait_time,
                        )

                        time.sleep(wait_time)

                    else:
                        logger.warning(
                            "Model %s failed after %s attempts.",
                            model,
                            MAX_RETRIES,
                        )

        raise LLMExtractionError(
            "All configured Gemini models failed. "
            f"Last error: {last_error}"
        )

    def extract(
        self,
        document_text: str,
        template_name: str,
        response_schema: Type[T],
        examples: list | None = None,
    ) -> T:

        prompt = self.render_prompt(
            template_name=template_name,
            document_text=document_text,
            examples=examples,
        )

        try:

            logger.info(
                "Starting extraction using %s",
                template_name,
            )

            response_text = self.generate_with_retry(
                prompt=prompt,
                response_schema=response_schema,
            )

            # Independent Pydantic validation
            result = response_schema.model_validate_json(
                response_text
            )

            logger.info(
                "Extraction and validation completed."
            )

            return result

        except ValidationError as error:

            logger.exception(
                "Pydantic validation failed."
            )

            raise LLMExtractionError(
                f"Invalid structured response: {error}"
            ) from error

        except LLMExtractionError:
            raise

        except Exception as error:

            logger.exception(
                "Unexpected extraction failure."
            )

            raise LLMExtractionError(
                f"Extraction failed: {error}"
            ) from error
