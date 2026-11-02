from pathlib import Path
from pypdf import PdfReader


class DocumentReader:
    """
    Extracts readable text from TXT and PDF documents.
    """

    @staticmethod
    def read_txt(file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        return path.read_text(encoding="utf-8")

    @staticmethod
    def read_pdf(file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        reader = PdfReader(str(path))

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if page_text:
                extracted_pages.append(page_text)

        text = "\n".join(extracted_pages)

        if not text.strip():
            raise ValueError(
                "No readable text found in PDF. "
                "It may be a scanned document requiring OCR."
            )

        return text

    @classmethod
    def read_document(cls, file_path: str) -> str:
        """
        Automatically detects the document format.
        """

        extension = Path(file_path).suffix.lower()

        if extension == ".txt":
            return cls.read_txt(file_path)

        if extension == ".pdf":
            return cls.read_pdf(file_path)

        raise ValueError(
            "Unsupported file format. "
            "Please provide a TXT or PDF file."
        )