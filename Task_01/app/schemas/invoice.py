
from typing import Optional
from pydantic import BaseModel, Field


class InvoiceItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None

    seller_name: Optional[str] = None
    buyer_name: Optional[str] = None

    currency: Optional[str] = None

    items: list[InvoiceItem] = Field(default_factory=list)

    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total_amount: Optional[float] = None
