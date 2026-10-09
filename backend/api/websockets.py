"""
WebSocket Streaming Routers for Web Microphone and Exotel Telephony.
"""

import json
import logging
import time
from typing import Optional
import uuid

from fastapi import APIRouter, Request, Response, WebSocket
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.exotel import ExotelFrameSerializer as PipecatExotelSerializer
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from backend.config import (
    GROQ_API_KEY,
    GROQ_STT_MODEL,
    KOKORO_MODE,
    KOKORO_TTS_URL,
    KOKORO_TTS_VOICE,
    LLM_BACKEND,
    SHERPA_MODEL_DIR,
    SHERPA_NUM_THREADS,
    STT_BACKEND,
    WS_HOST,
    WS_PORT,
)
import backend.database as db
from backend.knowledge import INDIAWALLS_SYSTEM_PROMPT
from backend.services.llm import create_pipecat_llm_service
from backend.services.stt import SherpaOnnxSTTService
from backend.services.tts import KokoroTTSService
from backend.telephony import (
    ExotelFrameSerializer,
    RawAudioFrameSerializer,
)

logger = logging.getLogger("hindi-ai-calling.api.ws")

router = APIRouter(tags=["WebSockets & Telephony"])


# ---------------------------------------------------------------------------
# 1. Web Browser Microphone WebSocket Call (/ws)
# ---------------------------------------------------------------------------
@router.websocket("/ws")
async def websocket_browser_endpoint(websocket: WebSocket):
    """Handle continuous real-time speech-to-speech call session from browser mic."""
    await websocket.accept()
    logger.info("Browser client connected to voice stream WebSocket")

    call_id = f"web_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    caller_name = (
        websocket.query_params.get("user")
        or websocket.query_params.get("caller")
        or "Web Visitor"
    )
    db.create_call(
        call_id=call_id,
        channel="web",
        from_number=caller_name,
    )
    logger.info(f"Started web call session {call_id} for {caller_name}")

    # Transport (16kHz in, 24kHz out)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            serializer=RawAudioFrameSerializer(sample_rate=16000),
        ),
    )

    # Silero VAD – Balanced for browser mic (normal speaking voice)
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=16000,
        params=VADParams(
            start_secs=0.20,   # 200ms sustained speech needed to trigger
            stop_secs=0.30,    # Allow natural mid-sentence pauses
            confidence=0.60,   # Balanced: rejects noise, catches normal speech
            min_volume=0.50,   # Normal conversational volume threshold
        ),
    )
    vad = VADProcessor(vad_analyzer=vad_analyzer)

    # STT
    if STT_BACKEND == "local":
        stt = SherpaOnnxSTTService(
            model_dir=SHERPA_MODEL_DIR,
            num_threads=SHERPA_NUM_THREADS,
            sample_rate=16000,
        )
    else:
        stt = GroqSTTService(
            api_key=GROQ_API_KEY,
            settings=GroqSTTService.Settings(
                model=GROQ_STT_MODEL,
                language=Language.HI,
            ),
        )

    # LLM & TTS (low temperature for focused responses, 100 tokens for 20-40 word answers)
    llm = create_pipecat_llm_service(INDIAWALLS_SYSTEM_PROMPT, temperature=0.3, max_tokens=100)
    tts = KokoroTTSService(
        mode=KOKORO_MODE,
        base_url=KOKORO_TTS_URL,
        voice=KOKORO_TTS_VOICE,
        sample_rate=24000,
    )

    # Context & Aggregators
    context = LLMContext()
    if LLM_BACKEND != "gemini":
        context.add_message({"role": "system", "content": INDIAWALLS_SYSTEM_PROMPT})

    # VAD barge-in: when user speaks loud enough (passes min_volume threshold),
    # immediately stop AI response and prepare a new one from user's input.
    user_turn_strategies = UserTurnStrategies(
        start=[
            VADUserTurnStartStrategy(enable_interruptions=True),
            TranscriptionUserTurnStartStrategy(enable_interruptions=True),
        ],
        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.80)],
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=None,
            user_turn_strategies=user_turn_strategies,
            user_turn_stop_timeout=0.80,
        ),
    )

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        logger.info("User started speaking -> interrupting AI TTS")
        try:
            await websocket.send_text(json.dumps({"type": "interruption"}))
        except Exception:
            pass

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        logger.info(f"User transcript: '{message.content}'")
        if message.content and message.content.strip():
            db.add_call_message(call_id=call_id, role="user", content=message.content.strip())
        try:
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "user_text": message.content,
            }))
        except Exception:
            pass

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        logger.info(f"AI transcript: '{message.content}'")
        if message.content and message.content.strip():
            db.add_call_message(call_id=call_id, role="assistant", content=message.content.strip())
        try:
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "reply_text": message.content,
            }))
        except Exception:
            pass

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    try:
        runner = PipelineRunner()
        await runner.run(task)
    except Exception as e:
        logger.error(f"Web call session error: {e}")
    finally:
        db.end_call(call_id=call_id, status="completed")
        logger.info(f"Web call session completed & saved: {call_id}")


# ---------------------------------------------------------------------------
# 2. Exotel Telephony WebSocket Call (/exotel-ws)
# ---------------------------------------------------------------------------
@router.websocket("/exotel-ws")
async def exotel_websocket_endpoint(websocket: WebSocket):
    """
    Exotel Voicebot / AudioStream WebSocket endpoint.
    Handles real-time phone call audio streams via Exotel Media Stream protocol (16-bit Linear PCM).
    """
    await websocket.accept()
    logger.info("Exotel telephony caller connected to WebSocket")

    call_id = f"exo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    query_from = (
        websocket.query_params.get("From")
        or websocket.query_params.get("from")
        or websocket.query_params.get("Caller")
        or websocket.query_params.get("caller")
        or "Exotel Phone Caller"
    )
    query_call_sid = (
        websocket.query_params.get("CallSid")
        or websocket.query_params.get("call_sid")
        or websocket.query_params.get("CallId")
        or ""
    )
    db.create_call(
        call_id=call_id,
        channel="exotel",
        from_number=query_from,
        call_sid=query_call_sid,
    )
    logger.info(f"Started Exotel call session {call_id} for {query_from} (sid={query_call_sid})")

    def on_exotel_start_event(call_sid, from_number, start_payload):
        actual_caller = from_number or query_from
        if actual_caller or call_sid:
            db.update_call_caller_id(
                call_id=call_id,
                caller_number=actual_caller or "Exotel Phone Caller",
                call_sid=call_sid or query_call_sid,
            )
            logger.info(f"Updated call {call_id} with verified caller ID: {actual_caller}")

    serializer = ExotelFrameSerializer(
        stream_sid="exotel_stream",
        params=PipecatExotelSerializer.InputParams(exotel_sample_rate=8000, sample_rate=16000),
        on_start_callback=on_exotel_start_event,
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=2,
            serializer=serializer,
        ),
    )

    # Silero VAD – Balanced for phone audio (8kHz codec, slightly lower volume)
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=16000,
        params=VADParams(
            start_secs=0.20,   # 200ms sustained speech needed to trigger
            stop_secs=0.30,    # Natural pause tolerance
            confidence=0.60,   # Balanced: rejects codec noise, catches normal speech
            min_volume=0.45,   # Mid-range: normal phone voice (not shouting, not whisper)
        ),
    )
    vad = VADProcessor(vad_analyzer=vad_analyzer)

    if STT_BACKEND == "local":
        stt = SherpaOnnxSTTService(
            model_dir=SHERPA_MODEL_DIR,
            num_threads=SHERPA_NUM_THREADS,
            sample_rate=16000,
        )
    else:
        stt = GroqSTTService(
            api_key=GROQ_API_KEY,
            settings=GroqSTTService.Settings(
                model=GROQ_STT_MODEL,
                language=Language.HI,
            ),
        )

    llm = create_pipecat_llm_service(INDIAWALLS_SYSTEM_PROMPT, temperature=0.3, max_tokens=100)
    tts = KokoroTTSService(
        mode=KOKORO_MODE,
        base_url=KOKORO_TTS_URL,
        voice=KOKORO_TTS_VOICE,
        sample_rate=16000,
    )

    context = LLMContext()
    if LLM_BACKEND != "gemini":
        context.add_message({"role": "system", "content": INDIAWALLS_SYSTEM_PROMPT})

    # VAD barge-in: when user speaks loud enough (passes min_volume threshold),
    # immediately stop AI response and prepare a new one from user's input.
    user_turn_strategies = UserTurnStrategies(
        start=[
            VADUserTurnStartStrategy(enable_interruptions=True),
            TranscriptionUserTurnStartStrategy(enable_interruptions=True),
        ],
        stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.80)],
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=None,
            user_turn_strategies=user_turn_strategies,
            user_turn_stop_timeout=0.80,
        ),
    )

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_exotel_user_turn_started(aggregator, strategy):
        logger.info(f"Exotel caller started speaking ({strategy.__class__.__name__}) -> barge-in triggered")

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_exotel_user_transcript(aggregator, strategy, message):
        logger.info(f"Exotel Phone Caller: '{message.content}'")
        if message.content and message.content.strip():
            db.add_call_message(call_id=call_id, role="user", content=message.content.strip())

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_exotel_ai_transcript(aggregator, message):
        logger.info(f"Exotel AI Speech: '{message.content}'")
        if message.content and message.content.strip():
            db.add_call_message(call_id=call_id, role="assistant", content=message.content.strip())

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    try:
        runner = PipelineRunner()
        await runner.run(task)
    except Exception as e:
        logger.error(f"Exotel phone call session exception: {e}")
    finally:
        db.end_call(call_id=call_id, status="completed")
        logger.info(f"Exotel phone call completed & saved: {call_id} (stream_sid={serializer._stream_sid})")


# ---------------------------------------------------------------------------
# 3. Exotel Webhook & Info Endpoints
# ---------------------------------------------------------------------------
@router.api_route("/exotel-call", methods=["GET", "POST"])
async def exotel_call_webhook(request: Request):
    """
    Exotel Applet Webhook: Returns XML instructing Exotel to stream
    phone audio bidirectionally to /exotel-ws.
    """
    host = request.headers.get("host", f"{WS_HOST}:{WS_PORT}")
    ws_proto = "wss" if ("https" in str(request.base_url) or request.headers.get("x-forwarded-proto") == "https") else "ws"
    stream_url = f"{ws_proto}://{host}/exotel-ws"
    logger.info(f"Exotel call initiated, routing to stream URL: {stream_url}")

    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
    return Response(content=xml_response, media_type="application/xml")


@router.get("/exotel-ws")
async def exotel_ws_http_info(request: Request):
    """Informational HTTP endpoint for browser checks."""
    host = request.headers.get("host", f"{WS_HOST}:{WS_PORT}")
    ws_proto = "wss" if ("https" in str(request.base_url) or request.headers.get("x-forwarded-proto") == "https") else "ws"
    return {
        "status": "online",
        "service": "Exotel Telephony WebSocket Stream",
        "websocket_url": f"{ws_proto}://{host}/exotel-ws",
        "audio_format": "16-bit Linear PCM (16000Hz)",
    }
