"""Cloud storage provider implementations."""

from __future__ import annotations

from typing import Tuple

import requests

from app.services.storage.base import StorageProvider


class AppwriteStorageProvider(StorageProvider):
    """Appwrite storage provider using the REST API.

    - `save` uploads bytes and returns (download_url, bucket_id:file_id)
    - `read` downloads bytes using the configured API key
    """

    def __init__(
        self,
        endpoint: str,
        project_id: str,
        api_key: str,
        bucket_id: str,
        timeout_seconds: float = 30.0,
        file_id: str | None = None,
        read_roles: str | None = None,
        write_roles: str | None = None,
    ):
        if not endpoint or not project_id or not api_key or not bucket_id:
            raise ValueError("Appwrite settings are missing or incomplete.")

        self.endpoint = endpoint.rstrip("/")
        self.project_id = project_id
        self.bucket_id = bucket_id
        self.timeout = timeout_seconds
        self.file_id = file_id
        self.read_roles = read_roles
        self.write_roles = write_roles
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-Appwrite-Project": project_id,
                "X-Appwrite-Key": api_key,
            }
        )

    def save(self, document_id: str, filename: str, payload: bytes) -> Tuple[str, str]:
        safe_name = filename.replace("/", "_").replace("\\", "_")
        url = f"{self.endpoint}/storage/buckets/{self.bucket_id}/files"
        file_id = self.file_id or document_id
        if file_id == "document_id":
            file_id = document_id
        data = {"fileId": file_id}
        if self.read_roles:
            data["read"] = self.read_roles
        if self.write_roles:
            data["write"] = self.write_roles
        files = {"file": (safe_name, payload)}

        try:
            response = self._session.post(
                url, data=data, files=files, timeout=self.timeout
            )
            response.raise_for_status()
            payload_json = response.json()
        except Exception as exc:  # pragma: no cover - runtime safety
            raise RuntimeError(f"Appwrite upload failed: {exc}") from exc

        file_id = payload_json.get("$id") or payload_json.get("id") or document_id
        storage_url = (
            f"{self.endpoint}/storage/buckets/{self.bucket_id}/files/{file_id}/download"
        )
        storage_path = f"{self.bucket_id}:{file_id}"
        return storage_url, storage_path

    def read(self, storage_path: str) -> bytes:
        bucket_id, file_id = self._split_storage_path(storage_path)
        url = f"{self.endpoint}/storage/buckets/{bucket_id}/files/{file_id}/download"

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # pragma: no cover - runtime safety
            raise RuntimeError(
                f"Failed to read Appwrite object {bucket_id}:{file_id}: {exc}"
            ) from exc

    def _split_storage_path(self, storage_path: str) -> tuple[str, str]:
        if ":" in storage_path:
            bucket_id, file_id = storage_path.split(":", 1)
        else:
            bucket_id, file_id = self.bucket_id, storage_path

        if not bucket_id or not file_id:
            raise ValueError("Invalid Appwrite storage_path format.")
        return bucket_id, file_id
