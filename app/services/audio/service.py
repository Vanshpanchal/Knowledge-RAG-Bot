"""Audio transcription and synthesis helpers."""

from __future__ import annotations

import base64
from io import BytesIO
import logging

import requests  # type: ignore[import-untyped]

from app.core.config import settings

logger = logging.getLogger(__name__)


class AudioService:
    def __init__(
        self,
        api_key: str,
        transcription_model: str | None = None,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = api_key
        self.transcription_model = (
            transcription_model or settings.ELEVENLABS_TRANSCRIPTION_MODEL
        )
        self.tts_model = tts_model or settings.ELEVENLABS_TTS_MODEL
        self.tts_voice = tts_voice or settings.ELEVENLABS_VOICE_ID
        self.timeout = timeout or settings.HTTP_TIMEOUT_SECONDS

    def transcribe(self, payload: bytes, filename: str, mime_type: str | None) -> str:
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for audio transcription.")

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": self.api_key}
        files = {
            "file": (
                filename,
                BytesIO(payload),
                mime_type or "application/octet-stream",
            )
        }
        data = {
            "model_id": self.transcription_model,
        }
        logger.info(
            "ElevenLabs transcription endpoint called: model=%s filename=%s mime_type=%s bytes=%d",
            self.transcription_model,
            filename,
            mime_type,
            len(payload),
        )
        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        transcript = str(body.get("text") or body.get("transcript") or "").strip()
        if not transcript:
            raise ValueError("Audio transcription returned an empty transcript.")
        return transcript

    def synthesize_bytes(self, text: str) -> tuple[str, bytes]:
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for audio synthesis.")
        if not self.tts_voice:
            raise ValueError("ELEVENLABS_VOICE_ID is required for audio synthesis.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.tts_voice}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.tts_model,
        }
        logger.info(
            "ElevenLabs speech endpoint called: model=%s voice_id=%s text_chars=%d",
            self.tts_model,
            self.tts_voice,
            len(text),
        )
        response = requests.post(
            url,
            headers=headers,
            params={"output_format": "mp3_44100_128"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        audio_bytes = response.content
        if not audio_bytes:
            raise ValueError("Audio synthesis returned empty audio.")
        return "audio/mpeg", audio_bytes

    def synthesize(self, text: str) -> tuple[str, str]:
        audio_mime_type, audio_bytes = self.synthesize_bytes(text)
        return audio_mime_type, base64.b64encode(audio_bytes).decode("ascii")
