"""
LLM Client Factory for Google Gemini and Groq Cloud.
"""

import logging
import os
from typing import Any, Optional, Tuple
from groq import Groq

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    LLM_BACKEND,
)
from .gemini_rotator import gemini_rotator
from .fallback_service import GroqWithGeminiFallbackLLMService

logger = logging.getLogger("hindi-ai-calling.llm.factory")


def get_gemini_client() -> Tuple[Optional[Any], str]:
    """Get or create Google GenAI client dynamically with key rotation."""
    key = gemini_rotator.get_current_key() or GEMINI_API_KEY
    if key and not key.startswith("your-"):
        try:
            from google import genai
            return genai.Client(api_key=key), key
        except Exception as e:
            logger.error(f"Error initializing Google GenAI Client: {e}")
            return None, key
    return None, ""


def get_groq_client() -> Optional[Groq]:
    """Get or create Groq client dynamically."""
    key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if key and not key.startswith("your-"):
        return Groq(api_key=key)
    return None


def create_pipecat_llm_service(system_prompt: str, temperature: float = 0.3, max_tokens: int = 160):
    """
    Create Pipecat LLM service.
    If Groq is enabled (or hybrid): Uses Groq (qwen/qwen3.8-27b, ~150-200ms TTFT) as primary,
    with automatic seamless failover to Google Gemini (gemini-3.5-flash-lite) if Groq hits
    rate limits, token exhaustion, or API errors.
    If Groq is disabled: Uses Google Gemini directly.
    """
    from pipecat.services.google.llm import GoogleLLMService
    from pipecat.services.groq.llm import GroqLLMService

    # 1. Prepare Google Gemini fallback service with multi-key rotation
    active_gemini_key = gemini_rotator.get_next_key() or GEMINI_API_KEY
    gemini_settings = GoogleLLMService.Settings(
        model=GEMINI_MODEL,
        system_instruction=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    gemini_service = GoogleLLMService(api_key=active_gemini_key, settings=gemini_settings)

    # 2. Check Groq configuration
    groq_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if LLM_BACKEND in ("groq", "hybrid") and groq_key and not groq_key.startswith("your-"):
        groq_settings = GroqLLMService.Settings(
            model=GROQ_LLM_MODEL,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        logger.info(
            f"Created Hybrid LLM service (Primary: Groq {GROQ_LLM_MODEL} | Fallback: Gemini {GEMINI_MODEL})"
        )
        return GroqWithGeminiFallbackLLMService(
            groq_api_key=groq_key,
            groq_settings=groq_settings,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # 3. Default: Pure Gemini service
    logger.info(
        f"Created Gemini LLM service (model={GEMINI_MODEL}, key: ...{active_gemini_key[-6:] if len(active_gemini_key) >= 6 else active_gemini_key})"
    )
    return gemini_service
