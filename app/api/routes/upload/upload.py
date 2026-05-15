import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app.api.dependencies.container import get_container

router = APIRouter()


@router.post("/pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accept PDF, validate and store via `UploadService`, then enqueue ingestion worker.

    Returns a document id immediately (202 semantics could be used by the caller).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    # Use central configured max size if available
    from app.core.config import settings

    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400, detail="File size exceeds configured limit"
        )

    # Use dependency container to get upload service and ingestion worker
    container = get_container()
    upload_service = container.get("upload_service")
    ingestion_worker = container.get("ingestion_worker")

    if upload_service is None:
        raise HTTPException(status_code=500, detail="Upload service not available")

    # Persist upload (validator + storage + DB record)
    record = upload_service.upload(file.filename, file.content_type, content)

    # Enqueue ingestion in background (non-blocking). In production use a real queue.
    if ingestion_worker is not None:
        background_tasks.add_task(ingestion_worker.run, record.document_id)

    return {"document_id": record.document_id, "filename": record.filename}


# settings = Settings()
# routes = router
