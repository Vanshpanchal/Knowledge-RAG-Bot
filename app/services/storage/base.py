"""Object storage abstractions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class StorageProvider(Protocol):
    def save(self, document_id: str, filename: str, payload: bytes) -> tuple[str, str]:
        """Return `(storage_url, storage_path)`."""

    def read(self, storage_path: str) -> bytes:
        """Read previously stored bytes."""


class LocalStorageProvider:
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, filename: str, payload: bytes) -> tuple[str, str]:
        safe_name = filename.replace("/", "_").replace("\\", "_")
        output_path = self.root / f"{document_id}__{safe_name}"
        output_path.write_bytes(payload)
        return str(output_path), str(output_path)

    def read(self, storage_path: str) -> bytes:
        return Path(storage_path).read_bytes()
