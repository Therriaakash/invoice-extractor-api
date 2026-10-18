
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="Invoice Structured Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)


class InvoiceFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


def extract_vendor(text: str) -> Optional[str]:
    # Prefer an explicitly labelled vendor or supplier.
    label_patterns = [
        r"(?im)^\s*(?:vendor|supplier|seller|issued\s+by|billed\s+by)"
        r"\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*from\s*:\s*(.+?)\s*$",
    ]

    for pattern in label_patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip().strip('"\'')
            value = re.sub(r"\s+(?:invoice|invoice\s+number)\s*:.*$", "", value, flags=re.I)
            if value:
                return value.rstrip(" ,;")

    # Recognize common legal business-name suffixes, including
    # randomized names such as Acme-X7K2 Industries Ltd.
    company_pattern = (
        r"\b([A-Z][A-Za-z0-9&'().,\-]*(?:\s+[A-Z0-9][A-Za-z0-9&'().,\-]*){0,8}"
        r"\s+(?:Industries|Incorporated|Corporation|Company|Limited|"
        r"Inc\.?|Ltd\.?|LLC|Corp\.?|GmbH))\b"
    )
    match = re.search(company_pattern, text)
    if match:
        return match.group(1).strip().rstrip(".,;")

    # Last resort: extract a value after a common business label.
    match = re.search(
        r"(?im)^\s*(?:company|business\s+name)\s*:\s*(.+?)\s*$",
        text,
    )
    if match:
        return match.group(1).strip()

    return None


def extract_invoice(text: str) -> InvoiceFields:
    vendor = extract_vendor(text)

    # Prefer totals explicitly described as due, then invoice totals.
    amount_patterns = [
        r"(?im)^\s*(?:total\s+amount\s+due|total\s+due|amount\s+due|"
        r"balance\s+due|payment\s+due)\s*[:\-]?\s*"
        r"(?:(USD|EUR|GBP)\s*)?[$€£]?\s*"
        r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(USD|EUR|GBP)?\b",
        r"(?im)^\s*(?:grand\s+total|invoice\s+total|total)\s*"
        r"[:\-]?\s*(?:(USD|EUR|GBP)\s*)?[$€£]?\s*"
        r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(USD|EUR|GBP)?\b",
    ]

    amount = None
    amount_currency = None

    for pattern in amount_patterns:
        match = re.search(pattern, text)
        if match:
            amount = float(match.group(2).replace(",", ""))
            amount_currency = match.group(1) or match.group(3)
            break

    if amount is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract invoice amount",
        )

    currency = amount_currency
    if currency is None:
        currency_match = re.search(r"\b(USD|EUR|GBP)\b", text, re.I)
        if currency_match:
            currency = currency_match.group(1).upper()
        else:
            symbol_match = re.search(r"[$€£]", text)
            if symbol_match:
                currency = {"$": "USD", "€": "EUR", "£": "GBP"}[
                    symbol_match.group(0)
                ]

    if not currency:
        raise HTTPException(
            status_code=422,
            detail="Could not extract invoice currency",
        )

    # Prefer dates explicitly attached to payment-due labels.
    date_patterns = [
        r"(?im)(?:payment\s+due\s+date|due\s+date|payment\s+due|"
        r"payable\s+by|due\s+by)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
        r"(?im)^\s*date\s+due\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
    ]

    date_value = None
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            date_value = date_match.group(1)
            break

    if not date_value:
        raise HTTPException(
            status_code=422,
            detail="Could not extract payment due date",
        )

    if not vendor:
        raise HTTPException(
            status_code=422,
            detail="Could not extract vendor name",
        )

    # Validate the date as a real calendar date, not just a matching string.
    from datetime import date

    try:
        date.fromisoformat(date_value)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid due date")

    return InvoiceFields(
        vendor=vendor,
        amount=amount,
        currency=currency.upper(),
        date=date_value,
    )


@app.post("/extract", response_model=InvoiceFields)
async def extract(request: ExtractRequest) -> InvoiceFields:
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail="Invoice text is empty")

    return extract_invoice(text)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
