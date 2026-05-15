"""Upload validation logic."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.utils.validators import extension_of, sniff_signature


@dataclass
class ValidationResult:
    extension: str
    mime_type: str
    signature: str


class UploadValidator:
    def validate(
        self, filename: str, content_type: str | None, payload: bytes
    ) -> ValidationResult:
        extension = extension_of(filename)
        if extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {extension}")

        if len(payload) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"File size exceeds limit of {settings.MAX_UPLOAD_SIZE_BYTES} bytes"
            )

        incoming_mime = (content_type or "").split(";")[0].strip().lower()
        if incoming_mime and incoming_mime not in settings.ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported MIME type: {incoming_mime}")

        signature = sniff_signature(payload[:16])
        if extension in {".pdf"} and signature != "application/pdf":
            raise ValueError("PDF signature check failed.")
        if extension in {".png"} and signature != "image/png":
            raise ValueError("PNG signature check failed.")
        if extension in {".jpg", ".jpeg"} and signature != "image/jpeg":
            raise ValueError("JPEG signature check failed.")
        if extension in {".docx"} and signature not in {"application/zip-based"}:
            raise ValueError("DOCX signature check failed.")
        if extension in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            allowed_audio_signatures = {
                "unknown",
                "audio/wav",
                "audio/mpeg",
                "audio/flac",
                "audio/ogg",
                "audio/mp4",
            }
            if signature not in allowed_audio_signatures:
                raise ValueError(f"Audio signature check failed: {signature}")

        mime_type = incoming_mime or self._default_mime(extension)
        return ValidationResult(
            extension=extension, mime_type=mime_type, signature=signature
        )

    @staticmethod
    def _default_mime(extension: str) -> str:
        mime_map = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        return mime_map.get(extension, "application/octet-stream")
