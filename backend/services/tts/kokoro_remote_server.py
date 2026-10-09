"""
Kokoro-82M TTS API Server — standalone, ultra-low-latency streaming.

Optimized for real-time speech-to-speech pipelines:
    - Multi-threaded CPU inference via torch.inference_mode()
    - Chunked streaming endpoint (/v1/audio/speech/stream) for instant audio delivery
    - Standard OpenAI-compatible (/v1/audio/speech) endpoint

Run:
    python kokoro_tts_server.py
"""

import io
import os
import logging
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# Optimize PyTorch CPU inference threads
try:
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    torch.set_grad_enabled(False)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kokoro-tts")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KOKORO_LANG = os.environ.get("KOKORO_LANG", "h")           # h = Hindi
KOKORO_DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "hf_alpha")
KOKORO_SAMPLE_RATE = 24000
TTS_HOST = os.environ.get("TTS_HOST", "0.0.0.0")
TTS_PORT = int(os.environ.get("TTS_PORT", "8880"))

# Available Hindi voices
HINDI_VOICES = {
    "hf_alpha": {"name": "Alpha", "gender": "female", "lang": "hi"},
    "hf_beta":  {"name": "Beta",  "gender": "female", "lang": "hi"},
    "hm_omega": {"name": "Omega", "gender": "male",   "lang": "hi"},
    "hm_psi":   {"name": "Psi",   "gender": "male",   "lang": "hi"},
}

# ---------------------------------------------------------------------------
# Load Kokoro pipeline at startup
# ---------------------------------------------------------------------------
logger.info("Loading Kokoro-82M model with optimized inference...")
from kokoro import KPipeline  # noqa: E402

pipeline = KPipeline(lang_code=KOKORO_LANG)
logger.info(f"Kokoro-82M loaded. Lang: {KOKORO_LANG}, default voice: {KOKORO_DEFAULT_VOICE}")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kokoro TTS API",
    description="Ultra-low latency streaming TTS endpoint powered by Kokoro-82M",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    model: str = Field(default="kokoro", description="Model name")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="hf_alpha", description="Voice ID")
    response_format: str = Field(default="wav", description="Audio format: wav or pcm")
    speed: float = Field(default=1.0, description="Speaking speed")


class VoiceInfo(BaseModel):
    id: str
    name: str
    gender: str
    language: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "Kokoro TTS API",
        "model": "Kokoro-82M",
        "language": KOKORO_LANG,
        "default_voice": KOKORO_DEFAULT_VOICE,
        "endpoints": ["/v1/audio/speech", "/v1/audio/speech/stream", "/v1/voices", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "kokoro-82m", "language": KOKORO_LANG}


@app.get("/v1/voices", response_model=list[VoiceInfo])
async def list_voices():
    return [
        VoiceInfo(id=vid, name=v["name"], gender=v["gender"], language=v["lang"])
        for vid, v in HINDI_VOICES.items()
    ]


def to_numpy_audio(audio) -> np.ndarray:
    """Safely convert PyTorch Tensor or array to 1D float32 numpy array."""
    if isinstance(audio, torch.Tensor):
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


@app.post("/v1/audio/speech/stream")
async def stream_speech(request: SpeechRequest):
    """
    Ultra-low-latency streaming endpoint:
    Yields raw PCM chunks immediately as each sentence/clause finishes synthesis.
    """
    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    voice = request.voice or KOKORO_DEFAULT_VOICE

    def generate_chunks():
        t0 = time.time()
        chunk_count = 0
        with torch.inference_mode():
            for _gs, _ps, audio in pipeline(text, voice=voice):
                if audio is not None and len(audio) > 0:
                    chunk_count += 1
                    audio_np = to_numpy_audio(audio)
                    pcm_data = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    if chunk_count == 1:
                        logger.info(f"First TTS audio chunk generated in {round((time.time() - t0)*1000)}ms")
                    yield pcm_data

    return StreamingResponse(
        generate_chunks(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(KOKORO_SAMPLE_RATE),
            "X-Channels": "1",
        },
    )


@app.post("/v1/audio/speech")
async def synthesize_speech(request: SpeechRequest):
    """OpenAI-compatible speech synthesis endpoint."""
    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    voice = request.voice or KOKORO_DEFAULT_VOICE
    t0 = time.time()

    try:
        audio_chunks = []
        with torch.inference_mode():
            for _gs, _ps, audio in pipeline(text, voice=voice):
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(to_numpy_audio(audio))

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        full_audio = np.concatenate(audio_chunks)
        elapsed_ms = round((time.time() - t0) * 1000)
        logger.info(f"TTS complete: {elapsed_ms}ms, {len(full_audio)} samples ({len(full_audio)/KOKORO_SAMPLE_RATE:.2f}s audio)")

        if request.response_format == "pcm":
            pcm_data = (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            return Response(
                content=pcm_data,
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(KOKORO_SAMPLE_RATE),
                    "X-Channels": "1",
                    "X-Latency-Ms": str(elapsed_ms),
                },
            )
        else:
            buf = io.BytesIO()
            sf.write(buf, full_audio, KOKORO_SAMPLE_RATE, format="WAV", subtype="PCM_16")
            return Response(
                content=buf.getvalue(),
                media_type="audio/wav",
                headers={"X-Latency-Ms": str(elapsed_ms)},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


if __name__ == "__main__":
    import signal
    import sys
    import uvicorn

    def sigint_handler(signum, frame):
        logger.info("\nShutdown signal received. Exiting...")
        os._exit(0)

    try:
        signal.signal(signal.SIGINT, sigint_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, sigint_handler)
    except Exception:
        pass

    logger.info(f"Kokoro TTS API starting on http://{TTS_HOST}:{TTS_PORT}")
    logger.info(f"  Streaming endpoint: POST http://{TTS_HOST}:{TTS_PORT}/v1/audio/speech/stream")
    logger.info(f"  Standard endpoint:  POST http://{TTS_HOST}:{TTS_PORT}/v1/audio/speech")

    try:
        config = uvicorn.Config(
            app=app,
            host=TTS_HOST,
            port=TTS_PORT,
            loop="asyncio",
            timeout_graceful_shutdown=1,
            log_level="info",
        )
        server = uvicorn.Server(config)
        server.run()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Kokoro TTS server stopped.")
        os._exit(0)

