from pydantic import ValidationError

from app.schemas.resume import ResumeData, Education
from app.schemas.invoice import InvoiceData, InvoiceItem


def test_valid_resume():

    resume = ResumeData(
        name="John Smith",
        email="john@example.com",
        skills=["Python", "Django"],
        education=[
            Education(
                degree="BSc Computer Science",
                institution="ABC University",
            )
        ],
    )

    assert resume.name == "John Smith"
    assert "Python" in resume.skills


def test_missing_resume_fields():

    resume = ResumeData()

    assert resume.name is None
    assert resume.skills == []
    assert resume.experience == []


def test_invalid_email():

    try:
        ResumeData(email="invalid-email")
        assert False, "Expected validation error"
    except ValidationError:
        assert True


def test_valid_invoice():

    invoice = InvoiceData(
        invoice_number="INV-001",
        seller_name="ABC Technology",
        items=[
            InvoiceItem(
                description="Website Development",
                quantity=1,
                unit_price=50000,
                total=50000,
            )
        ],
        total_amount=50000,
    )

    assert invoice.invoice_number == "INV-001"
    assert invoice.total_amount == 50000