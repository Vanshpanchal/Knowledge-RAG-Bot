"""End-to-end knowledge pipeline."""
import asyncio
from typing import Optional, Dict, Any
from app.services.ocr.extractor import OCRExtractor
from app.services.text_processing import TextCleaner, SemanticChunker
from app.services.embeddings import MongoDBVectorStore
from app.services.retrieval import HybridRetriever
from app.services.prompt_builder import PromptBuilder
from app.services.gemini_client import GeminiClient
from app.core.logging import get_logger

logger = get_logger(__name__)


class KnowledgePipeline:
    """End-to-end pipeline for document ingestion and querying."""

    def __init__(self):
        self.extractor = OCRExtractor()
        self.cleaner = TextCleaner()
        self.chunker = SemanticChunker()
        self.vector_store = MongoDBVectorStore()
        self.retriever = HybridRetriever()
        
    async def initialize(self, show_logs: bool = False) -> None:
        """Initialize the pipeline components."""
        if show_logs:
            logger.info("Initializing KnowledgePipeline")
        await self.retriever.initialize(show_logs=show_logs)
        if show_logs:
            logger.info("KnowledgePipeline initialized successfully")

    async def ingest_document(
        self,
        file_url: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        show_logs: bool = False
    ) -> None:
        """Process and store a document."""
        if show_logs:
            logger.info(f"Starting ingestion of {file_type} document: {file_url}")
        
        try:
            # Step 1: Extract text
            extracted = await self.extractor.extract(file_url, file_type, show_logs=show_logs)
            
            # Step 2: Clean text
            cleaned_text = self.cleaner.clean(extracted.text, show_logs=show_logs)
            
            # Step 3: Chunk text
            chunks = self.chunker.chunk(cleaned_text, show_logs=show_logs)
            
            # Prepare metadata
            metadata_dict = metadata if metadata is not None else {}
            doc_metadata = {
                "document": file_url,
                "document_type": file_type,
                **metadata_dict
            }
            
            # Step 4: Store embeddings
            await self.vector_store.store_chunks(
                chunks=chunks,
                metadata=doc_metadata,
                show_logs=show_logs
            )
            
            if show_logs:
                logger.info("Document ingestion completed successfully")
                
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            raise

    async def query(
        self,
        question: str,
        show_logs: bool = False
    ) -> str:
        """Answer a question using the knowledge base."""
        if show_logs:
            logger.info(f"Processing query: {question}")
        
        try:
            # Step 1: Retrieve relevant context
            context = await self.retriever.retrieve(question, show_logs=show_logs)
            
            # Step 2: Build prompt
            prompt = PromptBuilder.build_prompt(context, question, show_logs=show_logs)
            
            # Step 3: Generate answer
            answer = await GeminiClient.generate_text(
                prompt=prompt,
                model="gemini-2.5-pro",
                temperature=0.2,
                show_logs=show_logs
            )
            
            if show_logs:
                logger.info("Query processing completed successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise