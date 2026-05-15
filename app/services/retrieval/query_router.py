"""Query intent classification and retrieval strategy routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(slots=True)
class QueryRouteDecision:
    intent: str
    strategy: str
    normalized_query: str
    field_candidates: list[str] = field(default_factory=list)
    document_type_hints: list[str] = field(default_factory=list)


class QueryRouter:
    """Classify queries into deterministic or hybrid retrieval routes."""

    FIELD_ALIASES: dict[str, list[str]] = {
        "full_name": ["full name", "name on", "name of", "name"],
        "license_number": ["license number", "licence number", "driver license number"],
        "dob": ["date of birth", "dob", "birth date"],
        "expiry_date": ["expiry date", "expiration date", "valid until"],
        "invoice_number": ["invoice number", "invoice no", "invoice #"],
        "total_amount": [
            "invoice amount",
            "total amount",
            "amount due",
            "grand total",
            "total",
        ],
        "due_date": ["due date", "payment due"],
        "vendor_name": ["vendor name", "vendor", "merchant", "supplier"],
        "account_number": ["account number", "account no", "acct number"],
        "transaction_amount": ["transaction amount", "amount spent", "amount paid"],
        "transaction_date": [
            "transaction date",
            "transaction time",
            "date of transaction",
        ],
        "title": ["title"],
        "authors": ["authors", "author"],
        "abstract": ["abstract"],
        "sections": ["section", "sections"],
        "attendees": ["attendees", "participants"],
        "action_items": ["action items", "next steps", "follow ups"],
        "speaker": ["speaker"],
    }

    DOCUMENT_HINTS: dict[str, list[str]] = {
        "identity_document": ["license", "licence", "passport", "identity", "id card"],
        "invoice": ["invoice", "bill", "vendor", "amount due"],
        "receipt": ["receipt", "subtotal", "cashier"],
        "bank_statement": ["bank statement", "account", "transaction", "balance"],
        "research_paper": ["paper", "research", "abstract", "authors"],
        "architecture_document": ["architecture", "system design", "retrieval"],
        "code_document": ["code", "function", "class", "module"],
        "meeting_notes": ["meeting", "notes", "agenda", "attendees"],
        "presentation": ["presentation", "slides", "deck"],
        "medical_document": ["medical", "patient", "prescription", "diagnosis"],
        "legal_document": ["legal", "contract", "agreement", "clause"],
        "screenshot": ["screenshot", "screen", "ui"],
    }

    TEMPORAL_TERMS = [
        "today",
        "yesterday",
        "last week",
        "last month",
        "this week",
        "this month",
        "ago",
        "week",
        "month",
        "year",
    ]

    def classify(self, query: str) -> QueryRouteDecision:
        normalized = " ".join(query.split())
        lowered = normalized.lower()

        field_candidates = self._field_candidates(lowered)
        document_type_hints = self._document_type_hints(lowered)

        if field_candidates:
            return QueryRouteDecision(
                intent="field_lookup",
                strategy="structured_field",
                normalized_query=normalized,
                field_candidates=field_candidates,
                document_type_hints=document_type_hints,
            )

        if any(term in lowered for term in self.TEMPORAL_TERMS):
            return QueryRouteDecision(
                intent="temporal_query",
                strategy="hybrid_temporal",
                normalized_query=normalized,
                document_type_hints=document_type_hints,
            )

        if re.search(r"\b(summarize|summary|overview|briefly)\b", lowered):
            return QueryRouteDecision(
                intent="summarization",
                strategy="semantic",
                normalized_query=normalized,
                document_type_hints=document_type_hints,
            )

        if re.search(r"\b(compare|difference|versus|vs\.)\b", lowered):
            return QueryRouteDecision(
                intent="comparison",
                strategy="hybrid",
                normalized_query=normalized,
                document_type_hints=document_type_hints,
            )

        if re.search(
            r"\b(who|what|where|when|why|how|explain|reason|why did)\b", lowered
        ):
            return QueryRouteDecision(
                intent="semantic_question",
                strategy="hybrid",
                normalized_query=normalized,
                document_type_hints=document_type_hints,
            )

        return QueryRouteDecision(
            intent="entity_lookup",
            strategy="hybrid",
            normalized_query=normalized,
            document_type_hints=document_type_hints,
        )

    def _field_candidates(self, lowered_query: str) -> list[str]:
        candidates: list[str] = []
        for field_name, aliases in self.FIELD_ALIASES.items():
            if any(alias in lowered_query for alias in aliases):
                candidates.append(field_name)
        return candidates

    def _document_type_hints(self, lowered_query: str) -> list[str]:
        hints: list[str] = []
        for document_type, aliases in self.DOCUMENT_HINTS.items():
            if any(alias in lowered_query for alias in aliases):
                hints.append(document_type)
        return hints


__all__ = ["QueryRouter", "QueryRouteDecision"]
