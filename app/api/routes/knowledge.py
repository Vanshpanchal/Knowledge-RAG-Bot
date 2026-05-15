"""RAG knowledge endpoints with show_logs support."""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from app.api.dependencies.container import get_container
from app.core.logging import setup_logging
import logging

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])


@router.post("/ingest")
async def ingest_document(
    file_url: str,
    file_type: str = Query(..., description="File type (pdf, png, jpg, jpeg)"),
    show_logs: bool = Query(False, description="Enable detailed logging"),
    metadata: Optional[dict] = None,
    container: dict = Depends(get_container),
):
    """
    Ingest a document into the knowledge base.

    Args:
        file_url: URL of the file to ingest
        file_type: Type of the file (pdf, png, jpg, jpeg)
        show_logs: Enable detailed logging (default: False)
        metadata: Optional metadata to associate with the document
    """
    if show_logs:
        setup_logging(True)
        logger = logging.getLogger(__name__)
        logger.info(f"Starting ingestion of {file_type} document: {file_url}")

    valid_types = ["pdf", "png", "jpg", "jpeg"]
    if file_type.lower() not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Must be one of: {', '.join(valid_types)}",
        )

    try:
        ingestion_service = container.get("ingestion_service")
        if ingestion_service:
            ingestion_service.ingest_document(
                file_url=file_url, file_type=file_type, metadata=metadata
            )

        if show_logs:
            logger = logging.getLogger(__name__)
            logger.info("Document ingestion completed successfully")

        return {
            "status": "success",
            "message": "Document ingestion initiated",
            "file_url": file_url,
            "file_type": file_type,
        }
    except Exception as e:
        if show_logs:
            setup_logging(True)
            logger = logging.getLogger(__name__)
            logger.error(f"Document ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query")
async def query_knowledge_base(
    question: str = Query(..., description="Question to ask the knowledge base"),
    show_logs: bool = Query(False, description="Enable detailed logging"),
    top_k: int = Query(6, description="Number of context chunks to retrieve"),
    container: dict = Depends(get_container),
):
    """
    Query the knowledge base with retrieval and generation.

    Args:
        question: Question to ask
        show_logs: Enable detailed logging (default: False)
        top_k: Number of context chunks to retrieve (default: 6)
    """
    if show_logs:
        setup_logging(True)
        logger = logging.getLogger(__name__)
        logger.info(f"Processing query: {question}")

    try:
        retrieval_service = container.get("retrieval_service")
        generation_service = container.get("generation_service")

        if not retrieval_service:
            raise HTTPException(
                status_code=500, detail="Retrieval service not available"
            )

        if show_logs:
            logger = logging.getLogger(__name__)
            logger.info("Performing hybrid retrieval")

        contexts = retrieval_service.retrieve(
            question=question, top_k=top_k, filters=None
        )

        if not contexts:
            return {
                "question": question,
                "answer": "No relevant context found for the question.",
                "status": "success",
                "citations": [],
            }

        if not generation_service:
            return {
                "question": question,
                "answer": "Gemini is not configured. Add GEMINI_API_KEY to .env to enable answer generation.",
                "status": "success",
                "citations": [],
            }

        if show_logs:
            logger = logging.getLogger(__name__)
            logger.info(f"Retrieved {len(contexts)} contexts, generating response")

        response = generation_service.answer(question, contexts)

        if show_logs:
            logger = logging.getLogger(__name__)
            logger.info("Query processing completed successfully")

        return response

    except Exception as e:
        if show_logs:
            setup_logging(True)
            logger = logging.getLogger(__name__)
            logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., description="Search query"),
    show_logs: bool = Query(False, description="Enable detailed logging"),
    top_k: int = Query(6, description="Number of results to return"),
    container: dict = Depends(get_container),
):
    """
    Search the knowledge base without generation.

    Args:
        query: Search query
        show_logs: Enable detailed logging (default: False)
        top_k: Number of results to return (default: 6)
    """
    if show_logs:
        setup_logging(True)
        logger = logging.getLogger(__name__)
        logger.info(f"Searching knowledge base for: {query}")

    try:
        retrieval_service = container.get("retrieval_service")

        if not retrieval_service:
            raise HTTPException(
                status_code=500, detail="Retrieval service not available"
            )

        results = retrieval_service.retrieve(question=query, top_k=top_k, filters=None)

        if show_logs:
            logger = logging.getLogger(__name__)
            logger.info(f"Search returned {len(results)} results")

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "status": "success",
        }

    except Exception as e:
        if show_logs:
            setup_logging(True)
            logger = logging.getLogger(__name__)
            logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
