"""Gemini LLM client for generation."""
import httpx
from typing import Optional, Dict, Any
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Client for interacting with Gemini LLM."""

    @classmethod
    async def generate_text(
        cls,
        prompt: str,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 32,
        show_logs: bool = False
    ) -> str:
        """Generate text using Gemini LLM."""
        if show_logs:
            logger.info(f"Generating text with model {model}")
            logger.info(f"Prompt size: {len(prompt)} characters")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                generated_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                if show_logs:
                    logger.info(f"Generated text of length {len(generated_text)} characters")
                
                return generated_text
                
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise