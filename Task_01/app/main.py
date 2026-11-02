import argparse
import logging
from pathlib import Path

from app.schemas.resume import ResumeData
from app.schemas.invoice import InvoiceData
from app.services.llm_service import (
    GeminiExtractionService,
    LLMExtractionError,
)
from app.utils.pdf_reader import DocumentReader
from app.prompts.few_shot_examples import (
    RESUME_EXAMPLES,
    INVOICE_EXAMPLES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_json(data, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        data.model_dump_json(indent=4),
        encoding="utf-8",
    )

    print(f"\nJSON successfully saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Structured LLM JSON Extractor"
    )

    parser.add_argument(
        "--type",
        choices=["resume", "invoice"],
        required=True,
        help="Type of document to analyze",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input TXT or PDF document",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the extracted JSON will be saved",
    )

    args = parser.parse_args()

    try:
        print("\nReading document...")

        document_text = DocumentReader.read_document(args.input)

        if not document_text.strip():
            raise ValueError("Document contains no text.")

        if args.type == "resume":
            schema = ResumeData
            template = "resume.jinja2"
            examples = RESUME_EXAMPLES
        else:
            schema = InvoiceData
            template = "invoice.jinja2"
            examples = INVOICE_EXAMPLES

        print("Initializing Gemini...")

        service = GeminiExtractionService()

        print("Extracting structured information...")

        extracted_data = service.extract(
            document_text=document_text,
            template_name=template,
            response_schema=schema,
            examples=examples,
        )

        print("\n========== EXTRACTED JSON ==========\n")
        print(extracted_data.model_dump_json(indent=4))

        save_json(extracted_data, args.output)

        print("\nExtraction completed successfully!")

    except (ValueError, FileNotFoundError, LLMExtractionError) as error:
        logger.error("Extraction failed: %s", error)
        raise SystemExit(1)

    except Exception:
        logger.exception("Unexpected application error.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()