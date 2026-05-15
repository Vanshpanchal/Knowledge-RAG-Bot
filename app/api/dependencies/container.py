"""Dependency container for FastAPI."""

from typing import Union
from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.embeddings.providers import (
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
)
from app.services.audio.service import AudioService
from app.services.retrieval.retrieval_service import RetrievalService
from app.services.retrieval.query_rewriter import QueryRewriter
from app.services.retrieval.reranker import Reranker
from app.services.llm.gemini_generator import GeminiGenerator
from app.services.llm.providers import GeminiLlmProvider
from app.services.upload.uploader import UploadService
from app.services.upload.validator import UploadValidator
from app.services.storage.base import LocalStorageProvider
from app.services.storage.cloud import AppwriteStorageProvider
from app.services.storage.base import StorageProvider
from app.services.chunking.semantic_chunker import SemanticChunkerService
from app.services.extraction.document_extractor import DocumentExtractor
from app.workers.ingestion_worker import IngestionWorker
from app.services.ocr.base import (
    AzureDocumentIntelligenceProvider,
    AzureVisionOcrProvider,
    NullOcrProvider,
)

logger = get_logger(__name__)


def get_collection(collection_name: str):
    """Get a MongoDB collection."""
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi

    client: MongoClient = MongoClient(
        settings.MONGODB_URI,
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=7000,
    )
    return client[settings.DATABASE_NAME][collection_name]


def _get_ocr_provider():
    """Initialize OCR provider based on configuration."""
    if settings.OCR_PROVIDER == "azure":
        try:
            return AzureDocumentIntelligenceProvider(
                endpoint=settings.AZURE_DOC_INTELLIGENCE_ENDPOINT,
                api_key=settings.AZURE_DOC_INTELLIGENCE_KEY,
            )
        except Exception as e:
            logger.warning(f"Azure OCR initialization failed: {e}, using null provider")
            return NullOcrProvider()
    else:
        return NullOcrProvider()


def _get_vision_ocr_provider():
    """Initialize Vision OCR provider based on configuration."""
    if settings.OCR_PROVIDER == "azure":
        try:
            return AzureVisionOcrProvider(
                endpoint=settings.AZURE_VISION_ENDPOINT,
                api_key=settings.AZURE_VISION_KEY,
            )
        except Exception as e:
            logger.warning(
                f"Azure Vision OCR initialization failed: {e}, using null provider"
            )
            return NullOcrProvider()
    else:
        return NullOcrProvider()


def build_container() -> dict:
    """Build the dependency container with all services."""
    setup_logging(False)

    try:
        # Get MongoDB collections
        chunk_collection = get_collection(settings.CHUNK_COLLECTION)
        document_collection = get_collection(settings.DOCUMENT_COLLECTION)

        # Initialize repositories
        chunk_repository = ChunkRepository(chunk_collection)
        document_repository = DocumentRepository(document_collection)

        # Initialize embedding provider
        if settings.EMBEDDING_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
            embedding_provider: Union[
                GeminiEmbeddingProvider, MockEmbeddingProvider
            ] = GeminiEmbeddingProvider(
                api_key=settings.GEMINI_API_KEY,
                model=settings.EMBEDDING_MODEL,
                dimensions=settings.VECTOR_DIMENSIONS,
            )
            logger.info("Initialized Gemini embedding provider")
        else:
            embedding_provider = MockEmbeddingProvider(
                dimensions=settings.VECTOR_DIMENSIONS
            )
            if settings.EMBEDDING_PROVIDER == "gemini":
                logger.warning(
                    "GEMINI_API_KEY is empty; falling back to Mock embedding provider"
                )
            else:
                logger.info("Initialized Mock embedding provider")

        # Initialize query rewriter (if enabled)
        query_rewriter = None
        if settings.EMBEDDING_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
            try:
                llm_provider = GeminiLlmProvider(
                    api_key=settings.GEMINI_API_KEY, model="gemini-2.5-flash"
                )
                query_rewriter = QueryRewriter(llm=llm_provider)
                logger.info("Query rewriter initialized")
            except Exception as e:
                logger.warning(f"Query rewriter initialization failed: {e}")

        # Initialize reranker
        reranker = Reranker(embedding_provider)

        # Initialize retrieval service
        retrieval_service = RetrievalService(
            chunk_repository=chunk_repository,
            embedding_provider=embedding_provider,
            vector_index_name=settings.VECTOR_INDEX_NAME,
            query_rewriter=query_rewriter,
            reranker=reranker,
        )
        logger.info("Retrieval service initialized")

        # Initialize generation service
        generation_service = GeminiGenerator() if settings.GEMINI_API_KEY else None
        if generation_service is None:
            logger.warning("GEMINI_API_KEY is empty; generation service is disabled")
        else:
            logger.info("Generation service initialized")

        audio_service = AudioService(
            api_key=settings.ELEVENLABS_API_KEY,
            transcription_model=settings.ELEVENLABS_TRANSCRIPTION_MODEL,
            tts_model=settings.ELEVENLABS_TTS_MODEL,
            tts_voice=settings.ELEVENLABS_VOICE_ID,
            timeout=settings.HTTP_TIMEOUT_SECONDS,
        )
        logger.info("Audio service initialized")

        # Initialize upload services
        validator = UploadValidator()

        # Choose storage provider based on configuration
        storage_provider: StorageProvider | None = None
        if settings.STORAGE_PROVIDER == "local":
            storage_provider = LocalStorageProvider(
                root_path=settings.LOCAL_STORAGE_PATH
            )
            logger.info("Using LocalStorageProvider")
        elif settings.STORAGE_PROVIDER == "appwrite":
            try:
                storage_provider = AppwriteStorageProvider(
                    endpoint=settings.APPWRITE_ENDPOINT,
                    project_id=settings.APPWRITE_PROJECT_ID,
                    api_key=settings.APPWRITE_API_KEY,
                    bucket_id=settings.APPWRITE_BUCKET_ID,
                    timeout_seconds=settings.HTTP_TIMEOUT_SECONDS,
                    file_id=settings.APPWRITE_FILE_ID,
                    read_roles=settings.APPWRITE_READ_ROLES,
                    write_roles=settings.APPWRITE_WRITE_ROLES,
                )
                logger.info("Using AppwriteStorageProvider")
            except Exception as e:
                logger.error(f"Failed to initialize Appwrite provider: {e}")
                raise
        else:
            raise ValueError(
                f"Unsupported storage provider: {settings.STORAGE_PROVIDER}"
            )

        assert storage_provider is not None

        upload_service = UploadService(
            validator=validator,
            storage_provider=storage_provider,
            document_repository=document_repository,
        )
        logger.info("Upload service initialized")

        # Initialize ingestion worker
        # Get OCR providers
        doc_ocr = _get_ocr_provider()
        vision_ocr = _get_vision_ocr_provider()

        extractor = DocumentExtractor(
            doc_ocr=doc_ocr,
            vision_ocr=vision_ocr,
            audio_service=audio_service,
        )
        chunker = SemanticChunkerService(
            min_tokens=settings.CHUNK_MIN_TOKENS,
            max_tokens=settings.CHUNK_MAX_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        )
        ingestion_worker = IngestionWorker(
            storage=storage_provider,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            extractor=extractor,
            chunker=chunker,
            embedding_provider=embedding_provider,
        )
        logger.info("Ingestion worker initialized")

        return {
            "chunk_repository": chunk_repository,
            "document_repository": document_repository,
            "document_repo": document_repository,
            "embedding_provider": embedding_provider,
            "storage_provider": storage_provider,
            "retrieval_service": retrieval_service,
            "generation_service": generation_service,
            "audio_service": audio_service,
            "query_rewriter": query_rewriter,
            "reranker": reranker,
            "upload_service": upload_service,
            "ingestion_worker": ingestion_worker,
        }

    except Exception as e:
        logger.error(f"Container initialization failed: {str(e)}", exc_info=True)
        raise


# Container instance
container: dict = {}


def get_container() -> dict:
    """Get the container instance."""
    return container


def initialize_container():
    """Initialize the container at startup."""
    global container
    container = build_container()
    return container
