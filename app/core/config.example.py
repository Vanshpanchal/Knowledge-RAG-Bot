"""Application configuration template - COPY TO .env and fill in your values."""

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

    # === Application Configuration ===
    APP_ENV: str = "development"
    APP_TITLE: str = "Knowledge RAG API"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Production-oriented modular RAG backend."
    LOG_LEVEL: str = "INFO"

    # === MongoDB Configuration ===
    MONGODB_URI: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "knowledge_rag"
    DOCUMENT_COLLECTION: str = "documents"
    CHUNK_COLLECTION: str = "chunks"
    VECTOR_INDEX_NAME: str = "chunk_embedding_vector_idx"
    VECTOR_DIMENSIONS: int = 768

    # === Storage Provider Configuration ===
    STORAGE_PROVIDER: AllowedStorageProvider = "local"  # "local" or "appwrite"
    LOCAL_STORAGE_PATH: str = str(PROJECT_ROOT / "temp_uploads")

    # Appwrite Configuration (leave empty if using local storage)
    APPWRITE_ENDPOINT: str = ""  # e.g., "https://nyc.cloud.appwrite.io/v1"
    APPWRITE_PROJECT_ID: str = ""
    APPWRITE_API_KEY: str = ""
    APPWRITE_BUCKET_ID: str = ""
    APPWRITE_FILE_ID: str = "unique()"
    APPWRITE_READ_ROLES: str = '["role:all"]'
    APPWRITE_WRITE_ROLES: str = '["role:all"]'

    # === OCR Provider Configuration ===
    OCR_PROVIDER: AllowedOcrProvider = "none"  # "none" or "azure"

    # Azure Document Intelligence (leave empty if not using)
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = ""
    AZURE_DOC_INTELLIGENCE_KEY: str = ""

    # Azure Vision/OCR (leave empty if not using)
    AZURE_VISION_ENDPOINT: str = ""
    AZURE_VISION_KEY: str = ""

    # === Embedding Provider Configuration ===
    EMBEDDING_PROVIDER: AllowedEmbeddingProvider = "gemini"  # "mock" or "gemini"
    EMBEDDING_MODEL: str = "models/gemini-embedding-2"
    GEMINI_API_KEY: str = ""  # Required if EMBEDDING_PROVIDER="gemini"

    # === LLM Provider Configuration ===
    LLM_PROVIDER: AllowedLlmProvider = "gemini"  # "mock" or "gemini"
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_P: float = 0.8
    LLM_TOP_K: int = 32

    # === Audio Service Configuration ===
    # ElevenLabs for transcription and TTS
    ELEVENLABS_API_KEY: str = ""  # Required for audio features
    ELEVENLABS_TRANSCRIPTION_MODEL: str = "scribe_v1"
    ELEVENLABS_TTS_MODEL: str = "eleven_multilingual_v2"
    ELEVENLABS_VOICE_ID: str = "JBFqnCBsd6RMkjVDRZzb"

    # OpenAI (optional, fallback for audio if ElevenLabs not configured)
    OPENAI_API_KEY: str = ""

    # === Chunking Configuration ===
    MAX_UPLOAD_SIZE_BYTES: int = 20 * 1024 * 1024
    DEFAULT_TOP_K: int = 6
    MAX_TOP_K: int = 20
    CHUNK_MAX_TOKENS: int = 700
    CHUNK_MIN_TOKENS: int = 450
    CHUNK_OVERLAP_TOKENS: int = 120

    # === Search Configuration ===
    ATLAS_SEARCH_ENABLED: bool = False
    ATLAS_SEARCH_INDEX_NAME: str = "chunk_search_idx"

    # === Network Configuration ===
    HTTP_TIMEOUT_SECONDS: float = 30.0
    RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_SECONDS: float = 0.75

    # === Monitoring Configuration ===
    PROMETHEUS_ENABLED: bool = False
    API_PREFIX: str = "/api/v1"

    # === File Type Configuration ===
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
