"""
FastAPI Application Factory & Lifecycle Manager for IndiaWalls AI Voice Calling.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import torch

from backend.api import auth_router, calls_router, system_router, websockets_router
from backend.config import (
    GEMINI_MODEL,
    KOKORO_MODE,
    KOKORO_TTS_VOICE,
    LLM_BACKEND,
    ROOT_DIR,
    SHERPA_MODEL_DIR,
    SHERPA_NUM_THREADS,
    STT_BACKEND,
)
import backend.database as db
from backend.services.llm import get_gemini_client
from backend.services.stt import get_shared_recognizer
from backend.services.tts import get_kokoro_pipeline

# Optimize PyTorch CPU inference threads for TTS
try:
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    torch.set_grad_enabled(False)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
# Suppress noisy third-party loggers
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("hindi-ai-calling")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm CPU models, database, and connections to achieve 0ms first-call latency."""
    logger.info("Initializing IndiaWalls AI Calling System...")

    # 1. Pre-warm Kokoro TTS
    if KOKORO_MODE == "inproc":
        logger.info(f"Pre-warming in-process Kokoro TTS model (voice='{KOKORO_TTS_VOICE}')...")
        loop = asyncio.get_running_loop()

        def _prewarm_tts():
            try:
                pipe = get_kokoro_pipeline()
                with torch.inference_mode():
                    for _ in pipe("नमस्ते", voice=KOKORO_TTS_VOICE):
                        pass
            except Exception as e:
                logger.warning(f"TTS pre-warm notice: {e}")

        await loop.run_in_executor(None, _prewarm_tts)
        logger.info("Kokoro TTS model pre-warmed & ready in memory.")

    # 2. Pre-warm IndicConformer STT
    if STT_BACKEND == "local":
        logger.info(f"Pre-warming shared IndicConformer STT model from: {SHERPA_MODEL_DIR}...")
        loop = asyncio.get_running_loop()

        def _prewarm_stt():
            try:
                rec = get_shared_recognizer(SHERPA_MODEL_DIR, SHERPA_NUM_THREADS)
                stream = rec.create_stream()
                stream.accept_waveform(16000, np.zeros(1600, dtype=np.float32))
                rec.decode_stream(stream)
            except Exception as e:
                logger.warning(f"STT pre-warm notice: {e}")

        await loop.run_in_executor(None, _prewarm_stt)
        logger.info("IndicConformer STT pre-warmed & ready in memory.")

    # 3. Initialize & Auto-Seed Supabase Cloud Database
    try:
        seed_report = db.verify_and_seed_database()
        logger.info(f"Supabase Database initialized. Status: {seed_report.get('supabase_status')}, Calls: {seed_report.get('supabase_calls_count', 0)}.")
    except Exception as e:
        logger.error(f"Supabase database initialization & seed error: {e}")

    # 4. Pre-warm Gemini LLM connection pool
    if LLM_BACKEND == "gemini":
        logger.info(f"Pre-warming Gemini LLM connection ({GEMINI_MODEL})...")
        loop = asyncio.get_running_loop()

        def _prewarm_gemini():
            try:
                client, _ = get_gemini_client()
                if client:
                    from google.genai import types
                    client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents="ping",
                        config=types.GenerateContentConfig(max_output_tokens=2),
                    )
            except Exception as e:
                logger.warning(f"Gemini pre-warm notice: {e}")

        await loop.run_in_executor(None, _prewarm_gemini)
        logger.info("Gemini LLM connection pool pre-warmed & ready.")

    yield


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="IndiaWalls AI Voice Calling System",
        description="Production Speech-to-Speech Hindi AI Calling pipeline with Supabase & Exotel",
        version="2.5.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_origin_regex=r".*",
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    # Include all modular routers
    app.include_router(auth_router)
    app.include_router(calls_router)
    app.include_router(system_router)
    app.include_router(websockets_router)

    # Serve Testing Dashboard
    @app.get("/")
    async def serve_index():
        index_file = ROOT_DIR / "index.html"
        return FileResponse(
            str(index_file),
            media_type="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return app


app = create_app()
