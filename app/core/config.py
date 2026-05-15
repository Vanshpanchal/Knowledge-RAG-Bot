"""Application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

AllowedStorageProvider = Literal["local", "appwrite"]
AllowedEmbeddingProvider = Literal["mock", "gemini"]
AllowedLlmProvider = Literal["mock", "gemini"]
AllowedOcrProvider = Literal["none", "azure"]


class Settings(BaseSettings):
    """Application settings loaded from `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_ENV: str = "development"
    APP_TITLE: str = "Knowledge RAG API"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Production-oriented modular RAG backend."
    LOG_LEVEL: str = "INFO"

    MONGODB_URI: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "knowledge_rag"
    DOCUMENT_COLLECTION: str = "documents"
    CHUNK_COLLECTION: str = "chunks"
    VECTOR_INDEX_NAME: str = "chunk_embedding_vector_idx"
    VECTOR_DIMENSIONS: int = 768

    STORAGE_PROVIDER: AllowedStorageProvider = "appwrite"
    LOCAL_STORAGE_PATH: str = str(PROJECT_ROOT / "temp_uploads")
    APPWRITE_ENDPOINT: str = ""
    APPWRITE_PROJECT_ID: str = ""
    APPWRITE_API_KEY: str = ""
    APPWRITE_BUCKET_ID: str = ""
    APPWRITE_FILE_ID: str = "unique()"
    APPWRITE_READ_ROLES: str = '["role:all"]'
    APPWRITE_WRITE_ROLES: str = '["role:all"]'

    OCR_PROVIDER: AllowedOcrProvider = "azure"
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = ""
    AZURE_DOC_INTELLIGENCE_KEY: str = ""
    AZURE_VISION_ENDPOINT: str = ""
    AZURE_VISION_KEY: str = ""

    EMBEDDING_PROVIDER: AllowedEmbeddingProvider = "gemini"
    EMBEDDING_MODEL: str = "models/gemini-embedding-2"
    GEMINI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""

    LLM_PROVIDER: AllowedLlmProvider = "gemini"
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_P: float = 0.8
    LLM_TOP_K: int = 32

    ELEVENLABS_TRANSCRIPTION_MODEL: str = "scribe_v1"
    ELEVENLABS_TTS_MODEL: str = "eleven_multilingual_v2"
    ELEVENLABS_VOICE_ID: str = "JBFqnCBsd6RMkjVDRZzb"

    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024
    DEFAULT_TOP_K: int = 6
    MAX_TOP_K: int = 20
    CHUNK_MAX_TOKENS: int = 700
    CHUNK_MIN_TOKENS: int = 450
    CHUNK_OVERLAP_TOKENS: int = 120

    ATLAS_SEARCH_ENABLED: bool = False
    ATLAS_SEARCH_INDEX_NAME: str = "chunk_search_idx"

    HTTP_TIMEOUT_SECONDS: float = 30.0
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_SECONDS: float = 0.75
    API_KEYS: str = ""  # Comma-separated list of valid API keys

    PROMETHEUS_ENABLED: bool = False
    API_PREFIX: str = "/api/v1"

    ALLOWED_EXTENSIONS: tuple[str, ...] = Field(
        default=(
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".docx",
            ".txt",
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
        )
    )
    ALLOWED_MIME_TYPES: tuple[str, ...] = Field(
        default=(
            "application/pdf",
            "image/png",
            "image/jpeg",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/mpeg",
            "audio/mp3",
            "audio/mp4",
            "audio/x-m4a",
            "audio/flac",
            "audio/ogg",
        )
    )


settings = Settings()
