"""
Production CLI entry point to launch the IndiaWalls Voice Calling Server.
Usage:
    python backend/run.py
"""

import os
import signal
import sys
import uvicorn

# Ensure project root is on Python sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import app
from backend.config import (
    GEMINI_MODEL,
    GROQ_LLM_MODEL,
    GROQ_STT_MODEL,
    KOKORO_MODE,
    KOKORO_TTS_URL,
    KOKORO_TTS_VOICE,
    LLM_BACKEND,
    SHERPA_MODEL_DIR,
    STT_BACKEND,
    SUPABASE_URL,
    WS_HOST,
    WS_PORT,
)
from backend.services.llm import gemini_rotator
import backend.database as db
import logging

logger = logging.getLogger("hindi-ai-calling")


def sigint_handler(signum, frame):
    logger.info("\nShutdown signal received. Exiting...")
    os._exit(0)


def main():
    try:
        signal.signal(signal.SIGINT, sigint_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, sigint_handler)
    except Exception:
        pass

    # 1. Check & auto-download local STT model files if missing
    if STT_BACKEND == "local":
        try:
            from scripts.download_models import ensure_models_exist
            ensure_models_exist(SHERPA_MODEL_DIR)
        except Exception as e:
            logger.error(f"Error checking/downloading STT models: {e}")

    stt_info = (
        f"LOCAL IndicConformer ({SHERPA_MODEL_DIR})"
        if STT_BACKEND == "local"
        else f"Groq Whisper ({GROQ_STT_MODEL})"
    )
    tts_info = (
        f"IN-PROCESS (Kokoro-82M, voice={KOKORO_TTS_VOICE})"
        if KOKORO_MODE == "inproc"
        else f"REMOTE ({KOKORO_TTS_URL})"
    )
    if LLM_BACKEND == "gemini":
        k_count = gemini_rotator.total_keys
        rot_label = f"{k_count} key(s) rotating" if k_count > 1 else f"{k_count} active key"
        llm_info = f"Gemini ({GEMINI_MODEL}, {rot_label})"
    else:
        llm_info = f"Groq ({GROQ_LLM_MODEL})"
    # 1. Verify Supabase Cloud connection & auto-seed if empty
    seed_report = db.verify_and_seed_database()
    supa_st = seed_report.get("supabase_status", "Checking...")
    calls_count = seed_report.get("supabase_calls_count", 0)
    db_status = f"Supabase Cloud ({supa_st}, {calls_count} calls)"

    logger.info("==================================================")
    logger.info("  IndiaWalls AI Voice Calling System (Production) ")
    logger.info("==================================================")
    logger.info(f"  Database Engine:    {db_status}")
    logger.info(f"  STT Backend:        {stt_info}")
    logger.info(f"  LLM Service:        {llm_info}")
    logger.info(f"  TTS Engine:         {tts_info}")
    logger.info(f"  Dashboard Console:  http://{WS_HOST}:{WS_PORT}/")
    logger.info(f"  Browser WebSocket:  ws://{WS_HOST}:{WS_PORT}/ws")
    logger.info(f"  Exotel Telephony:   ws://{WS_HOST}:{WS_PORT}/exotel-ws")
    logger.info("==================================================")

    config = uvicorn.Config(
        app=app,
        host=WS_HOST,
        port=WS_PORT,
        loop="asyncio",
        timeout_graceful_shutdown=1,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
