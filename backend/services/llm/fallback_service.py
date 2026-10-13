"""
Groq LLM Service with automatic Google Gemini Fallback.
Primary: Ultra-fast Groq LPU inference (qwen/qwen3.8-27b, ~150-200ms).
Fallback: Direct Google Gemini 3.5 Flash Lite with multi-key rotation if Groq is exhausted or rate-limited.
"""

import asyncio
import logging
from typing import Optional

from google.genai import types
from pipecat.frames.frames import LLMFullResponseEndFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.groq.llm import GroqLLMService

from backend.config import GEMINI_MODEL
from .gemini_rotator import gemini_rotator

logger = logging.getLogger("hindi-ai-calling.llm.fallback")


class GroqWithGeminiFallbackLLMService(GroqLLMService):
    """
    Hybrid LLM Service:
    Executes primary queries using Groq LPU (sub-200ms TTFT).
    If Groq fails due to rate-limiting (429), token exhaustion, or API errors,
    it seamlessly falls back to Google Gemini with rotating keys without dropping the call.
    """

    def __init__(
        self,
        groq_api_key: str,
        groq_settings: GroqLLMService.Settings,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(api_key=groq_api_key, settings=groq_settings, **kwargs)
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._groq_failed = False
        self._emitted_text_count = 0

    def _sanitize_context(self, context: LLMContext):
        """
        Sanitize conversation history after barge-ins:
        1. Remove empty user/assistant turns resulting from interrupted speech.
        2. If previous user message was an interrupted/aborted query, replace it with the new query
           rather than concatenating unrelated questions.
        3. Only merge short speech fragments (caller pause/continuation of same sentence).
        4. Prevent models like Qwen from returning empty string on duplicated turns.
        """
        raw_messages = (
            context.get_messages()
            if hasattr(context, "get_messages")
            else getattr(context, "messages", [])
        )
        if not raw_messages:
            return

        cleaned = []
        for msg in raw_messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "system":
                cleaned.append(dict(msg))
                continue
            if not content:
                continue

            if cleaned and cleaned[-1].get("role") == "user" and role == "user":
                prev_text = cleaned[-1]["content"]
                # If the previous user turn was a short speech fragment (continuation of same sentence):
                if len(prev_text.split()) < 6 or not any(k in prev_text for k in ("?", "।", "क्या", "बताओ")):
                    cleaned[-1]["content"] = f"{prev_text} {content}"
                else:
                    # Previous user turn was an aborted question that the caller interrupted.
                    # Replace with the caller's new question so the assistant answers the active question!
                    cleaned[-1]["content"] = content
            elif cleaned and cleaned[-1].get("role") == "assistant" and role == "assistant":
                cleaned[-1]["content"] = f"{cleaned[-1]['content']} {content}"
            else:
                cleaned.append(dict(msg))

        if cleaned and hasattr(context, "set_messages"):
            context.set_messages(cleaned)

    async def _push_llm_text(self, text: str):
        if text and text.strip():
            self._emitted_text_count += 1
        await super()._push_llm_text(text)

    async def _process_context(self, context: LLMContext):
        self._sanitize_context(context)
        self._emitted_text_count = 0

        if not self._groq_failed:
            try:
                logger.info(f"LLM (Primary: Groq LPU) requesting completion (model={self._settings.model})...")
                await asyncio.wait_for(super()._process_context(context), timeout=5.0)
                if self._emitted_text_count > 0:
                    return
                logger.warning("Groq returned 0 tokens. Failing over seamlessly to Gemini fallback...")
            except Exception as groq_err:
                logger.warning(
                    f"Groq LLM error, timeout or rate-limit ({groq_err.__class__.__name__}: {groq_err}). "
                    f"Failing over seamlessly to Gemini fallback (gemini-3.5-flash-lite)..."
                )
                self._groq_failed = True

        # Fallback to Google Gemini using key rotation
        await self._run_gemini_fallback(context)

        # Safety fallback: if both LLMs emitted 0 tokens, provide a conversational prompt so bot never stays silent
        if self._emitted_text_count == 0:
            logger.warning("Both primary and fallback LLMs emitted 0 tokens. Emitting safety conversational prompt.")
            await self._push_llm_text("जी, कृपया अपना सवाल दोबारा बताएं।")

    async def _run_gemini_fallback(self, context: LLMContext):
        try:
            from .client_factory import get_gemini_client

            client, active_key = get_gemini_client()
            if not client:
                raise RuntimeError("No valid Gemini API key available for fallback")

            # Extract messages into Gemini format
            contents = []
            for msg in getattr(context, "messages", []):
                role = msg.get("role")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    contents.append({"role": "user", "parts": [{"text": str(content)}]})
                elif role in ("assistant", "model"):
                    contents.append({"role": "model", "parts": [{"text": str(content)}]})

            if not contents:
                logger.warning("Empty context messages for Gemini fallback")
                return

            logger.info(f"LLM (Fallback: Google Gemini) generating response with model {GEMINI_MODEL}...")
            loop = asyncio.get_running_loop()

            def _call_gemini_stream():
                return client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._system_prompt,
                        temperature=self._temperature,
                        max_output_tokens=self._max_tokens,
                    ),
                )

            stream = await loop.run_in_executor(None, _call_gemini_stream)
            for chunk in stream:
                if chunk.text:
                    await self._push_llm_text(chunk.text)

            logger.info("LLM (Fallback: Google Gemini) completed streaming response successfully.")

        except Exception as gem_err:
            logger.error(f"Gemini fallback also failed: {gem_err}")
            gemini_rotator.mark_error(gem_err)
