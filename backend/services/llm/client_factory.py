"""
LLM Client Factory for Google Gemini and Groq Cloud.
"""

import logging
import os
from typing import Any, Optional, Tuple
from groq import Groq

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_LLM_MODEL, LLM_BACKEND
from .gemini_rotator import gemini_rotator

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


def create_pipecat_llm_service(system_prompt: str, temperature: float = 0.2, max_tokens: int = 140):
    """Create Pipecat LLM service based on LLM_BACKEND."""
    if LLM_BACKEND == "gemini":
        from pipecat.services.google.llm import GoogleLLMService

        active_key = gemini_rotator.get_next_key() or GEMINI_API_KEY
        settings = GoogleLLMService.Settings(
            model=GEMINI_MODEL,
            system_instruction=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"Created Gemini LLM service (model={GEMINI_MODEL}, key: ...{active_key[-6:] if len(active_key) >= 6 else active_key})")
        return GoogleLLMService(api_key=active_key, settings=settings)
    else:
        from pipecat.services.groq.llm import GroqLLMService

        settings = GroqLLMService.Settings(
            model=GROQ_LLM_MODEL,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        logger.info(f"Created Groq LLM service (model={GROQ_LLM_MODEL})")
        return GroqLLMService(api_key=GROQ_API_KEY, settings=settings)
