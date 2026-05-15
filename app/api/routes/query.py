"""RAG query route."""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies.container import get_container
from app.core.auth import verify_api_key
from app.core.rate_limiter import check_rate_limit
from app.models.response import AudioQueryResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


def _build_filters(payload: QueryRequest) -> dict:
    filters = dict(payload.filters or {})
    if payload.tags:
        normalized_tags = [tag.strip().lower() for tag in payload.tags if tag.strip()]
        if normalized_tags:
            filters.setdefault("metadata.tags", {"$in": normalized_tags})

    sensitivity_levels = ["normal", "high", "classified"]
    sensitivity_filter = (
        payload.sensitivity_filter.lower() if payload.sensitivity_filter else "normal"
    )
    if sensitivity_filter not in sensitivity_levels:
        sensitivity_filter = "normal"
    filters.setdefault(
        "metadata.sensitivity",
        {"$in": sensitivity_levels[: sensitivity_levels.index(sensitivity_filter) + 1]},
    )
    return filters


def _resolve_question(
    question: str,
    top_k: int,
    filters: dict | None,
    container: dict,
) -> QueryResponse:
    retrieval_service = container["retrieval_service"]
    generation_service = container["generation_service"]

    logger.info("Retrieving contexts for question: %s", question)
    resolution = retrieval_service.resolve(
        question=question,
        top_k=top_k,
        filters=filters or None,
    )
    logger.info(
        "Resolved query intent=%s strategy=%s contexts=%d",
        resolution.intent,
        resolution.strategy,
        len(resolution.contexts),
    )

    if resolution.structured_answer:
        return QueryResponse(
            answer=resolution.structured_answer,
            citations=resolution.citations,
            strategy=resolution.strategy,
        )

    if not resolution.contexts:
        return QueryResponse(
            answer="Information not found in document.",
            citations=[],
            strategy=resolution.strategy,
        )

    if generation_service is None:
        return QueryResponse(
            answer=(
                "Gemini is not configured. Add GEMINI_API_KEY to .env to enable "
                "answer generation."
            ),
            citations=resolution.citations,
            strategy=resolution.strategy,
        )

    response = generation_service.answer(question, resolution.contexts)
    response.strategy = resolution.strategy
    return response


def _audio_output_filename(source_filename: str | None) -> str:
    stem = (source_filename or "query_audio").rsplit(".", 1)[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "audio"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"AI_Audio_{safe_stem}_{timestamp}.mp3"


@router.post("", response_model=QueryResponse)
def query_knowledge_base(
    payload: QueryRequest,
    container: dict = Depends(get_container),
    api_key: str = Depends(check_rate_limit),
):
    try:
        filters = _build_filters(payload)
        return _resolve_question(
            question=payload.question,
            top_k=payload.top_k,
            filters=filters,
            container=container,
        )
    except Exception as exc:
        logger.exception(f"Retrieval failed for question: {payload.question}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc


@router.post("/audio", response_model=AudioQueryResponse)
async def query_audio_knowledge_base(
    file: UploadFile = File(...),
    top_k: int = Form(default=6),
    filters: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    sensitivity_filter: str = Form(default="normal"),
    container: dict = Depends(get_container),
    api_key: str = Depends(check_rate_limit),
):
    upload_service = container["upload_service"]
    audio_service = container["audio_service"]
    storage_provider = container.get("storage_provider")
    payload = await file.read()

    try:
        upload_service.validator.validate(
            file.filename or "audio", file.content_type, payload
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        transcript = audio_service.transcribe(
            payload,
            file.filename or "query_audio",
            file.content_type,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Audio transcription failed: {exc}"
        ) from exc

    form_filters: dict = {}
    if filters:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid filters JSON: {exc}"
            ) from exc
        if isinstance(parsed_filters, dict):
            form_filters.update(parsed_filters)

    normalized_tags = [
        tag.strip().lower() for tag in (tags or "").split(",") if tag.strip()
    ]
    if normalized_tags:
        form_filters.setdefault("metadata.tags", {"$in": normalized_tags})

    query_payload = QueryRequest(
        question=transcript,
        top_k=top_k,
        filters=form_filters or None,
        tags=normalized_tags,
        sensitivity_filter=sensitivity_filter,
    )

    try:
        response = _resolve_question(
            question=query_payload.question,
            top_k=query_payload.top_k,
            filters=_build_filters(query_payload),
            container=container,
        )
    except Exception as exc:
        logger.exception("Audio query failed after transcription")
        raise HTTPException(
            status_code=500, detail=f"Audio query failed: {exc}"
        ) from exc

    audio_mime_type: str | None = None
    audio_base64: str | None = None
    audio_error: str | None = None
    audio_storage_url: str | None = None
    audio_storage_path: str | None = None
    try:
        audio_mime_type, audio_bytes = audio_service.synthesize_bytes(response.answer)
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
        if storage_provider is not None:
            output_filename = _audio_output_filename(file.filename)
            audio_storage_url, audio_storage_path = storage_provider.save(
                document_id=f"audio-{uuid4()}",
                filename=output_filename,
                payload=audio_bytes,
            )
    except Exception as exc:
        audio_error = str(exc)

    return AudioQueryResponse(
        question=transcript,
        transcript=transcript,
        answer=response.answer,
        citations=response.citations,
        status=response.status,
        strategy=response.strategy,
        audio_mime_type=audio_mime_type,
        audio_base64=audio_base64,
        audio_error=audio_error,
        audio_storage_url=audio_storage_url,
        audio_storage_path=audio_storage_path,
    )
