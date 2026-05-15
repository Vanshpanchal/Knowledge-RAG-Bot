"""Document upload and metadata routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Form,
    UploadFile,
    status,
)

from app.api.dependencies.container import get_container
from app.core.auth import verify_api_key
from app.core.rate_limiter import check_rate_limit
from app.models.response import UploadResponse, TextEntryRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _audio_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").strip() or "audio note"


def _schedule_ingestion(
    background_tasks: BackgroundTasks, ingestion_worker, document_id: str
) -> None:
    background_tasks.add_task(ingestion_worker.run, document_id)
    logger.info("Scheduled ingestion background task for document_id=%s", document_id)


@router.post(
    "/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    container: dict = Depends(get_container),
    api_key: str = Depends(check_rate_limit),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    payload = await file.read()
    upload_service = container["upload_service"]
    ingestion_worker = container["ingestion_worker"]
    try:
        record = upload_service.upload(
            filename=file.filename,
            content_type=file.content_type,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # schedule ingestion in background and log scheduling so it's visible in server logs
    _schedule_ingestion(background_tasks, ingestion_worker, record.document_id)
    return UploadResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status.value,
        message="Uploaded successfully. Ingestion started asynchronously.",
    )


@router.post(
    "/audio", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED
)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    source_url: str | None = Form(default=None),
    sensitivity: str = Form(default="normal"),
    container: dict = Depends(get_container),
    api_key: str = Depends(check_rate_limit),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    payload = await file.read()
    upload_service = container["upload_service"]
    ingestion_worker = container["ingestion_worker"]

    normalized_tags = [
        tag.strip().lower() for tag in (tags or "").split(",") if tag.strip()
    ]

    try:
        record = upload_service.upload(
            filename=file.filename,
            content_type=file.content_type,
            payload=payload,
            source_type="audio",
            title=title or _audio_title(file.filename),
            tags=normalized_tags,
            sensitivity=sensitivity,
            source_url=source_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _schedule_ingestion(background_tasks, ingestion_worker, record.document_id)
    return UploadResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status.value,
        message=(
            "Audio uploaded successfully. Transcription and chunking started asynchronously."
        ),
    )


@router.post(
    "/text", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED
)
def create_text_entry(
    payload: TextEntryRequest,
    background_tasks: BackgroundTasks,
    container: dict = Depends(get_container),
    api_key: str = Depends(check_rate_limit),
):
    upload_service = container["upload_service"]
    ingestion_worker = container["ingestion_worker"]

    try:
        record = upload_service.ingest_text_entry(
            title=payload.title,
            content=payload.content,
            tags=payload.tags,
            source_type=payload.source_type,
            source_url=payload.source_url,
            sensitivity=payload.sensitivity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _schedule_ingestion(background_tasks, ingestion_worker, record.document_id)
    return UploadResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status.value,
        message="Text entry saved successfully. Ingestion started asynchronously.",
    )


@router.get("/{document_id}")
def get_document(
    document_id: str,
    container: dict = Depends(get_container),
    api_key: str = Depends(verify_api_key),
):
    document_repo = container["document_repo"]
    document = document_repo.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document
