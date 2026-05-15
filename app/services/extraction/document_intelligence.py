"""Lightweight document classification and structured field extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class DocumentIntelligenceResult:
    document_type: str
    domain: str
    structure: str
    structured_fields: dict[str, Any] = field(default_factory=dict)


class DocumentIntelligence:
    """Heuristic document classifier and field extractor."""

    DOCUMENT_PATTERNS: dict[str, list[str]] = {
        "identity_document": [
            "driver license",
            "driving license",
            "license number",
            "passport",
            "date of birth",
            "dob",
            "expiry date",
        ],
        "invoice": [
            "invoice",
            "amount due",
            "total due",
            "invoice number",
            "bill to",
            "vendor",
        ],
        "receipt": ["receipt", "subtotal", "tax", "cashier", "payment method"],
        "bank_statement": [
            "bank statement",
            "account number",
            "transaction",
            "available balance",
            "statement period",
        ],
        "research_paper": [
            "abstract",
            "references",
            "methodology",
            "authors",
            "keywords",
        ],
        "architecture_document": [
            "architecture",
            "system design",
            "layered",
            "service layer",
            "repository layer",
        ],
        "code_document": ["def ", "class ", "import ", "function", "package"],
        "meeting_notes": ["agenda", "meeting notes", "action items", "attendees"],
        "screenshot": ["screenshot", "screen capture", "ui", "button", "dialog"],
        "presentation": ["slide", "presentation", "deck", "speaker notes"],
        "medical_document": ["patient", "diagnosis", "prescription", "clinical"],
        "legal_document": ["agreement", "contract", "clause", "party", "hereby"],
    }

    DOCUMENT_DOMAINS: dict[str, str] = {
        "identity_document": "identity",
        "invoice": "finance",
        "receipt": "finance",
        "bank_statement": "finance",
        "research_paper": "research",
        "architecture_document": "engineering",
        "code_document": "engineering",
        "meeting_notes": "operations",
        "screenshot": "ui",
        "presentation": "presentation",
        "medical_document": "healthcare",
        "legal_document": "legal",
    }

    STRUCTURE_BY_TYPE: dict[str, str] = {
        "identity_document": "semi_structured",
        "invoice": "semi_structured",
        "receipt": "semi_structured",
        "bank_statement": "semi_structured",
        "research_paper": "structured",
        "architecture_document": "semi_structured",
        "code_document": "structured",
        "meeting_notes": "semi_structured",
        "presentation": "semi_structured",
        "medical_document": "semi_structured",
        "legal_document": "semi_structured",
        "screenshot": "unstructured",
    }

    def classify(
        self,
        text: str,
        filename: str | None = None,
        title: str | None = None,
        source_type: str | None = None,
        mime_type: str | None = None,
    ) -> DocumentIntelligenceResult:
        haystack_parts = [
            text,
            filename or "",
            title or "",
            source_type or "",
            mime_type or "",
        ]
        haystack = " ".join(part for part in haystack_parts if part).lower()

        best_type = "generic_document"
        best_score = 0
        for document_type, patterns in self.DOCUMENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in haystack)
            if score > best_score:
                best_score = score
                best_type = document_type

        domain = self.DOCUMENT_DOMAINS.get(best_type, "general")
        structure = self.STRUCTURE_BY_TYPE.get(best_type, "unstructured")
        structured_fields = self.extract_structured_fields(
            best_type, text, filename, title
        )

        return DocumentIntelligenceResult(
            document_type=best_type,
            domain=domain,
            structure=structure,
            structured_fields=structured_fields,
        )

    def extract_structured_fields(
        self,
        document_type: str,
        text: str,
        filename: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        text = text or ""
        if document_type == "invoice":
            return self._extract_invoice_fields(text)
        if document_type == "identity_document":
            return self._extract_identity_fields(text)
        if document_type == "bank_statement":
            return self._extract_bank_statement_fields(text)
        if document_type == "research_paper":
            return self._extract_research_fields(text, filename, title)
        if document_type == "meeting_notes":
            return self._extract_meeting_notes_fields(text, title)
        if document_type == "presentation":
            return self._extract_presentation_fields(text, title)
        return {}

    @staticmethod
    def _first_match(patterns: list[str], text: str) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                if match.lastindex:
                    return match.group(1).strip()
                return match.group(0).strip()
        return None

    @staticmethod
    def _normalize_date(value: str | None) -> str | None:
        if not value:
            return None
        cleaned = value.strip()
        parsed_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]
        for date_format in parsed_formats:
            try:
                return datetime.strptime(cleaned, date_format).date().isoformat()
            except ValueError:
                continue
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", cleaned)
        if iso_match:
            return iso_match.group(1)
        return None

    @staticmethod
    def _normalize_money(value: str | None) -> str | None:
        if not value:
            return None
        cleaned = value.replace(",", "").strip()
        match = re.search(r"-?\d+(?:\.\d{1,2})?", cleaned)
        return match.group(0) if match else cleaned

    def _extract_invoice_fields(self, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        invoice_number = self._first_match(
            [
                r"invoice\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9\-/]+)",
                r"inv\.?\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9\-/]+)",
            ],
            text,
        )
        total_amount = self._first_match(
            [
                r"(?:total\s+due|amount\s+due|grand\s+total|total)\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{1,2})?)",
            ],
            text,
        )
        due_date = self._first_match(
            [r"due\s+date\s*[:\-]?\s*([A-Za-z0-9,\-/ ]{6,})"], text
        )
        vendor_name = self._first_match(
            [
                r"vendor\s*[:\-]\s*(.+)",
                r"bill\s+from\s*[:\-]\s*(.+)",
            ],
            text,
        )

        if invoice_number:
            fields["invoice_number"] = invoice_number
        if total_amount:
            fields["total_amount"] = self._normalize_money(total_amount)
        if due_date:
            fields["due_date"] = self._normalize_date(due_date) or due_date.strip()
        if vendor_name:
            fields["vendor_name"] = vendor_name.splitlines()[0].strip()
        return fields

    def _extract_identity_fields(self, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        name = self._first_match([r"(?:full\s+name|name)\s*[:\-]\s*(.+)"], text)
        license_number = self._first_match(
            [
                r"(?:license|licence)\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9\-/]+)",
            ],
            text,
        )
        dob = self._first_match(
            [r"(?:date\s+of\s+birth|dob)\s*[:\-]?\s*([A-Za-z0-9,\-/ ]{6,})"],
            text,
        )
        expiry_date = self._first_match(
            [r"(?:expiry|expiration)\s+date\s*[:\-]?\s*([A-Za-z0-9,\-/ ]{6,})"],
            text,
        )

        if name:
            fields["full_name"] = name.splitlines()[0].strip()
        if license_number:
            fields["license_number"] = license_number
        if dob:
            fields["dob"] = self._normalize_date(dob) or dob.strip()
        if expiry_date:
            fields["expiry_date"] = (
                self._normalize_date(expiry_date) or expiry_date.strip()
            )
        return fields

    def _extract_bank_statement_fields(self, text: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        account_number = self._first_match(
            [r"account\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9\-/]+)"], text
        )
        balance = self._first_match(
            [
                r"(?:balance|available\s+balance)\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{1,2})?)"
            ],
            text,
        )
        transaction_date = self._first_match(
            [r"(?:transaction\s+date|date)\s*[:\-]?\s*([A-Za-z0-9,\-/ ]{6,})"],
            text,
        )
        transaction_amount = self._first_match(
            [
                r"(?:transaction\s+amount|amount)\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{1,2})?)"
            ],
            text,
        )

        if account_number:
            fields["account_number"] = account_number
        if balance:
            fields["balance"] = self._normalize_money(balance)
        if transaction_date:
            fields["transaction_date"] = (
                self._normalize_date(transaction_date) or transaction_date.strip()
            )
        if transaction_amount:
            fields["transaction_amount"] = self._normalize_money(transaction_amount)
        return fields

    def _extract_research_fields(
        self, text: str, filename: str | None = None, title: str | None = None
    ) -> dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        fields: dict[str, Any] = {}
        if title:
            fields["title"] = title
        elif lines:
            fields["title"] = lines[0]

        authors = self._first_match([r"authors?\s*[:\-]\s*(.+)"], text)
        if authors:
            fields["authors"] = [
                part.strip() for part in re.split(r",| and ", authors) if part.strip()
            ]

        abstract = self._first_match(
            [r"abstract\s*[:\-]?\s*(.+?)(?:\n\n|\n[A-Z][^\n]{0,80}:|references|$)"],
            text,
        )
        if abstract:
            fields["abstract"] = abstract.strip()

        return fields

    def _extract_meeting_notes_fields(
        self, text: str, title: str | None = None
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if title:
            fields["title"] = title
        attendees = self._first_match([r"attendees?\s*[:\-]\s*(.+)"], text)
        if attendees:
            fields["attendees"] = [
                part.strip() for part in re.split(r",| and ", attendees) if part.strip()
            ]
        action_items = self._first_match([r"action\s+items?\s*[:\-]?\s*(.+)"], text)
        if action_items:
            fields["action_items"] = action_items.strip()
        return fields

    def _extract_presentation_fields(
        self, text: str, title: str | None = None
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if title:
            fields["title"] = title
        elif lines:
            fields["title"] = lines[0]
        speaker = self._first_match([r"speaker\s*[:\-]\s*(.+)"], text)
        if speaker:
            fields["speaker"] = speaker.strip()
        return fields


__all__ = ["DocumentIntelligence", "DocumentIntelligenceResult"]
