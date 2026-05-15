"""Document repository."""

from __future__ import annotations

from typing import Any

from pymongo.collection import Collection

from app.models.document import DocumentRecord, DocumentStatus


class DocumentRepository:
    def __init__(self, collection: Collection):
        self.collection = collection

    def create(self, record: DocumentRecord) -> None:
        self.collection.insert_one(record.model_dump())

    def get(self, document_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"document_id": document_id}, {"_id": 0})

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        update_fields: dict[str, Any] = {"status": status.value}
        if error_message is not None:
            update_fields["error_message"] = error_message
        if metadata is not None:
            update_fields["metadata"] = metadata
        self.collection.update_one(
            {"document_id": document_id},
            {"$set": update_fields},
        )
