"""
System Health, Diagnostic Testing & Knowledge REST API Router.
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional

import aiohttp
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf
import torch

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    GROQ_STT_MODEL,
    KOKORO_MODE,
    KOKORO_TTS_SPEED,
    KOKORO_TTS_URL,
    KOKORO_TTS_VOICE,
    LLM_BACKEND,
    ROOT_DIR,
    SHERPA_MODEL_DIR,
    SHERPA_NUM_THREADS,
    STT_BACKEND,
)
from backend.knowledge import INDIAWALLS_PRESETS, INDIAWALLS_SYSTEM_PROMPT
from backend.services.llm import gemini_rotator, get_gemini_client, get_groq_client
from backend.services.tts import get_kokoro_pipeline, get_voice_tensor, normalize_text_for_hindi_tts
import backend.database as db

logger = logging.getLogger("hindi-ai-calling.api.system")

router = APIRouter(tags=["System & Diagnostics"])


@router.get("/greeting.mp3")
async def get_greeting_mp3():
    """Serve the IndiaWalls Kokoro TTS greeting audio in MP3 format for Exotel."""
    candidates = [
        ROOT_DIR / "assets" / "audio" / "greeting_indiawalls.mp3",
        ROOT_DIR / "greeting_indiawalls.mp3",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        raise HTTPException(status_code=404, detail="Greeting MP3 file not found")
    return FileResponse(str(path), media_type="audio/mpeg", filename="greeting_indiawalls.mp3")


@router.get("/greeting.wav")
async def get_greeting_wav():
    """Serve the IndiaWalls Kokoro TTS greeting audio in 16kHz WAV format for Exotel."""
    candidates = [
        ROOT_DIR / "assets" / "audio" / "greeting_indiawalls_16k.wav",
        ROOT_DIR / "greeting_indiawalls_16k.wav",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        raise HTTPException(status_code=404, detail="Greeting WAV file not found")
    return FileResponse(str(path), media_type="audio/wav", filename="greeting_indiawalls_16k.wav")


@router.get("/api/health")
async def health_check():
    """Check health of backend, STT, Groq API configuration, and Kokoro TTS."""
    kokoro_ok = False
    kokoro_details = {
        "mode": KOKORO_MODE,
        "voice": KOKORO_TTS_VOICE,
        "language": "hi",
    }

    if KOKORO_MODE == "inproc":
        kokoro_ok = True
        kokoro_details.update({
            "in_process": True,
            "status": "ready (in-process framework)",
        })
    else:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2.0)) as session:
                async with session.get(f"{KOKORO_TTS_URL}/health") as resp:
                    if resp.status == 200:
                        kokoro_ok = True
                        kokoro_details.update(await resp.json())
        except Exception as e:
            kokoro_details["error"] = str(e)

    # Check local STT model status
    local_stt_status = {"backend": STT_BACKEND}
    if STT_BACKEND == "local":
        model_files = ["model.int8.onnx", "tokens.txt"]
        all_exist = all(os.path.exists(os.path.join(SHERPA_MODEL_DIR, f)) for f in model_files)
        local_stt_status.update({
            "model_dir": SHERPA_MODEL_DIR,
            "model": "IndicConformer (Hindi CTC)",
            "files_present": all_exist,
            "num_threads": SHERPA_NUM_THREADS,
            "rate_limited": False,
        })
    else:
        local_stt_status.update({
            "model": GROQ_STT_MODEL,
            "rate_limited": True,
            "rate_limit": "20 RPM",
        })

    # LLM status
    gemini_ok = bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your-"))
    groq_ok = bool(GROQ_API_KEY and not GROQ_API_KEY.startswith("your-"))
    active_llm_ok = gemini_ok if LLM_BACKEND == "gemini" else groq_ok
    active_llm_model = GEMINI_MODEL if LLM_BACKEND == "gemini" else GROQ_LLM_MODEL

    return {
        "status": "healthy",
        "stt": local_stt_status,
        "llm_backend": LLM_BACKEND,
        "llm_configured": active_llm_ok,
        "llm_model": active_llm_model,
        "gemini_configured": gemini_ok,
        "gemini_model": GEMINI_MODEL,
        "groq_configured": groq_ok,
        "groq_llm_model": GROQ_LLM_MODEL,
        "kokoro_tts": {
            "mode": KOKORO_MODE,
            "connected": kokoro_ok,
            "voice": KOKORO_TTS_VOICE,
            "details": kokoro_details,
        },
        "database": {
            "engine": "Supabase Cloud PostgreSQL",
            "ready": db.check_supabase_schema(),
        },
    }


@router.get("/api/voices")
async def get_voices():
    """Return supported Kokoro Hindi voices."""
    return [
        {"id": "hf_alpha", "name": "Alpha (Female 1 - Default)", "gender": "female", "language": "hi"},
        {"id": "hf_beta",  "name": "Beta (Female 2)",           "gender": "female", "language": "hi"},
        {"id": "hm_omega", "name": "Omega (Male 1)",            "gender": "male",   "language": "hi"},
        {"id": "hm_psi",   "name": "Psi (Male 2)",              "gender": "male",   "language": "hi"},
    ]


@router.get("/api/firm-info")
async def get_firm_info():
    """Return IndiaWalls firm metadata and interactive test presets."""
    return {
        "company_name": "Indiawalls Infratech Private Limited",
        "tagline": "Precast Concrete Walls & Fencing Solutions",
        "phone": "7820879777",
        "whatsapp": "9653545525",
        "office": "Khasra No. 251, 252, Tehsil Tapukara, Daganheri, Alwar, Rajasthan - 301707",
        "presets": INDIAWALLS_PRESETS,
    }


class PipelineTestRequest(BaseModel):
    query: str = Field(..., description="User query text in Hindi")
    voice: Optional[str] = Field(default="hf_alpha", description="Kokoro TTS voice")


@router.post("/api/test-pipeline")
async def test_pipeline_query(req: PipelineTestRequest):
    """
    Test single query response with detailed per-stage latency profiling.
    Executes: LLM (Gemini / Groq) -> TTS (Kokoro-82M in-process) and returns audio + latency metrics.
    """
    user_text = req.query.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")

    voice = req.voice or KOKORO_TTS_VOICE
    t_start = time.time()

    # 1. LLM Generation
    t_llm_start = time.time()
    try:
        loop = asyncio.get_running_loop()
        if LLM_BACKEND == "gemini":
            gemini_client, active_key = get_gemini_client()
            if not gemini_client:
                raise HTTPException(
                    status_code=400,
                    detail="GEMINI_API_KEY is not configured in .env file. Please add your Gemini key.",
                )
            from google.genai import types

            def _generate_gemini():
                try:
                    resp = gemini_client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=user_text,
                        config=types.GenerateContentConfig(
                            system_instruction=INDIAWALLS_SYSTEM_PROMPT,
                            temperature=0.2,
                            max_output_tokens=140,
                        ),
                    )
                    gemini_rotator.record_usage(active_key)
                    return resp.text.strip()
                except Exception as ex:
                    err_str = str(ex).lower()
                    if "429" in err_str or "resourceexhausted" in err_str or "quota" in err_str:
                        gemini_rotator.record_429(active_key)
                    raise

            reply_text = await loop.run_in_executor(None, _generate_gemini)
        else:
            client = get_groq_client()
            if not client:
                raise HTTPException(
                    status_code=400,
                    detail="GROQ_API_KEY is not configured in .env file.",
                )
            completion = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GROQ_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": INDIAWALLS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    max_completion_tokens=140,
                    temperature=0.2,
                ),
            )
            reply_text = completion.choices[0].message.content.strip()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM error during test ({LLM_BACKEND}): {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")
    t_llm_end = time.time()
    llm_latency_ms = round((t_llm_end - t_llm_start) * 1000)

    # 2. TTS Synthesis
    t_tts_start = time.time()
    audio_base64 = ""

    if KOKORO_MODE == "inproc":
        try:
            pipeline = get_kokoro_pipeline()
            voice_data = get_voice_tensor(pipeline, voice)
            loop = asyncio.get_running_loop()

            def _synthesize_wav():
                chunks = []
                synth_text = normalize_text_for_hindi_tts(reply_text)
                with torch.inference_mode():
                    for _gs, _ps, audio in pipeline(
                        synth_text,
                        voice=voice_data,
                        speed=KOKORO_TTS_SPEED,
                        split_pattern=r"[,!?;:।\n]+",
                    ):
                        if audio is not None and len(audio) > 0:
                            a = (
                                audio.detach().cpu().numpy()
                                if isinstance(audio, torch.Tensor)
                                else np.asarray(audio)
                            )
                            chunks.append(a)
                if not chunks:
                    raise RuntimeError("No audio generated by Kokoro")
                full_audio = np.concatenate(chunks)
                buf = io.BytesIO()
                sf.write(buf, full_audio, 24000, format="WAV", subtype="PCM_16")
                return buf.getvalue()

            audio_bytes = await loop.run_in_executor(None, _synthesize_wav)
            audio_base64 = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"In-process Kokoro TTS error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Kokoro TTS generation failed: {str(e)}")
    else:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session:
                async with session.post(
                    f"{KOKORO_TTS_URL}/v1/audio/speech",
                    json={"input": reply_text, "voice": voice, "response_format": "wav"},
                ) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        raise HTTPException(status_code=500, detail=f"Kokoro TTS server error: {err}")
                    audio_bytes = await resp.read()
                    audio_base64 = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("utf-8")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Kokoro TTS server error: {str(e)}")

    t_tts_end = time.time()
    tts_latency_ms = round((t_tts_end - t_tts_start) * 1000)
    total_latency_ms = round((time.time() - t_start) * 1000)

    # Save test interaction into Call History so conversation is visible
    try:
        test_call_id = f"web_{int(time.time()*1000)}"
        db.create_call(
            call_id=test_call_id,
            channel="web",
            from_number="Web Console",
        )
        db.add_call_message(call_id=test_call_id, role="user", content=user_text, latency_ms=0)
        db.add_call_message(call_id=test_call_id, role="assistant", content=reply_text, latency_ms=total_latency_ms)
        db.end_call(call_id=test_call_id, status="completed")
    except Exception as e:
        logger.warning(f"Notice: Could not log test query to history: {e}")

    return {
        "reply_text": reply_text,
        "audio_base64": audio_base64,
        "latency_ms": {
            "llm": llm_latency_ms,
            "tts": tts_latency_ms,
            "total": total_latency_ms,
        },
        "model": GEMINI_MODEL if LLM_BACKEND == "gemini" else GROQ_LLM_MODEL,
        "voice": voice,
        "backend": LLM_BACKEND,
    }


@router.post("/api/test-stt")
async def test_stt(file: UploadFile = File(...)):
    """Test STT transcription latency with an uploaded WAV/MP3 file."""
    audio_content = await file.read()
    if not audio_content:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded")

    t0 = time.time()
    try:
        client = get_groq_client()
        if not client:
            raise HTTPException(status_code=400, detail="GROQ_API_KEY is not configured")
        loop = asyncio.get_running_loop()
        transcription = await loop.run_in_executor(
            None,
            lambda: client.audio.transcriptions.create(
                file=(file.filename or "audio.wav", audio_content),
                model=GROQ_STT_MODEL,
                language="hi",
                response_format="text",
            ),
        )
        t1 = time.time()
        return {
            "transcript": str(transcription).strip(),
            "latency_ms": round((t1 - t0) * 1000),
            "model": GROQ_STT_MODEL,
        }
    except Exception as e:
        logger.error(f"STT test error: {e}")
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {str(e)}")
