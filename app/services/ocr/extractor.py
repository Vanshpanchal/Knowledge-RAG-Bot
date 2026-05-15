"""OCR Extraction Router for Azure Document Intelligence and Computer Vision."""

import httpx
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel
from app.core.config import settings
from app.core.logging import get_logger
from app.services.ocr.poller import poll_operation_result_async

logger = get_logger(__name__)


class ExtractedText(BaseModel):
    text: str
    document_type: str  # "pdf", "image", "text"
    metadata: Dict[str, Any]


class AzureDocumentIntelligence:
    """Extracts text from PDFs using Azure Document Intelligence."""

    ENDPOINT = "https://doc-sintelligence-my.cognitiveservices.azure.com"
    MODEL = "prebuilt-read"
    API_VERSION = "2024-11-30"

    @classmethod
    async def extract_from_pdf(cls, url: str, show_logs: bool = False) -> ExtractedText:
        """Extract text from a PDF using Azure Document Intelligence."""
        if show_logs:
            logger.info(f"Starting PDF extraction from: {url}")

        headers = {
            "Ocp-Apim-Subscription-Key": settings.AZURE_DOCUMENT_INTELLIGENCE_KEY,
            "Content-Type": "application/json",
        }
        payload = {"urlSource": url}

        async with httpx.AsyncClient() as client:
            try:
                # Submit analysis request
                submit_url = f"{cls.ENDPOINT}/documentintelligence/documentModels/{cls.MODEL}:analyze?api-version={cls.API_VERSION}"
                response = await client.post(submit_url, headers=headers, json=payload)
                response.raise_for_status()
                # some Azure responses use 'Operation-Location' or 'operation-location'
                operation_url = response.headers.get(
                    "Operation-Location"
                ) or response.headers.get("operation-location")

                if show_logs:
                    logger.info(f"Analysis submitted. Operation URL: {operation_url}")

                # Poll for results using shared async poller
                poll_data = await poll_operation_result_async(
                    client, operation_url, headers
                )
                if poll_data.get("status") != "succeeded":
                    raise Exception("Azure Document Intelligence extraction failed")

                # fetch final result (some APIs return a URL in analyzeResult.content)
                result_url = poll_data.get("analyzeResult", {}).get("content")
                result_response = await client.get(result_url, headers=headers)
                result_response.raise_for_status()
                text_content = "\n".join(
                    [
                        page["content"]
                        for page in result_response.json().get("pages", [])
                    ]
                )

                if show_logs:
                    logger.info(
                        f"Successfully extracted {len(text_content)} characters from PDF"
                    )

                return ExtractedText(
                    text=text_content,
                    document_type="pdf",
                    metadata={
                        "source": url,
                        "page_count": len(result_response.json().get("pages", [])),
                        "model": cls.MODEL,
                    },
                )

            except Exception as e:
                logger.error(f"PDF extraction failed: {str(e)}")
                raise


class AzureComputerVision:
    """Extracts text from images using Azure Computer Vision."""

    ENDPOINT = "https://computer-vision-ocr-my.cognitiveservices.azure.com/vision/v3.2/read/analyze"
    # API_VERSION = "2024-02-01"

    @classmethod
    async def extract_from_image(
        cls, url: str, show_logs: bool = False
    ) -> ExtractedText:
        """Extract text from an image using Azure Computer Vision OCR."""
        if show_logs:
            logger.info(f"Starting image OCR extraction from: {url}")

        headers = {
            "Ocp-Apim-Subscription-Key": settings.AZURE_COMPUTER_VISION_KEY,
            "Content-Type": "application/json",
        }
        payload = {"url": url}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{cls.ENDPOINT}",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

                # Extract text from OCR response
                ocr_result = response.json()
                lines = []
                for region in ocr_result.get("regions", []):
                    for line in region.get("lines", []):
                        lines.append(
                            " ".join([word["text"] for word in line.get("words", [])])
                        )

                text_content = "\n".join(lines)
                if show_logs:
                    logger.info(
                        f"Successfully extracted {len(text_content)} characters from image"
                    )

                return ExtractedText(
                    text=text_content,
                    document_type="image",
                    metadata={
                        "source": url,
                        "language": ocr_result.get("language", "unknown"),
                    },
                )

            except Exception as e:
                logger.error(f"Image OCR failed: {str(e)}")
                raise


class OCRExtractor:
    """Routes files to the appropriate OCR service."""

    @classmethod
    async def extract(
        cls, file_url: str, file_type: str, show_logs: bool = False
    ) -> ExtractedText:
        """Route to the correct OCR service based on file type."""
        if show_logs:
            logger.info(
                f"Routing {file_type} file to appropriate OCR service: {file_url}"
            )

        if file_type == "pdf":
            return await AzureDocumentIntelligence.extract_from_pdf(file_url, show_logs)
        elif file_type in ["png", "jpg", "jpeg"]:
            return await AzureComputerVision.extract_from_image(file_url, show_logs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
