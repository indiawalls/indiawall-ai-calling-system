"""
Kokoro-82M Local Hindi TTS Service for Pipecat.
Supports in-process streaming and remote HTTP API modes with volume boost.
"""

import asyncio
import logging
import os
import re
import threading
import time
from typing import AsyncGenerator, Optional

import aiohttp
import numpy as np
import torch

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService, TTSSettings
from pipecat.transcriptions.language import Language

from backend.config import KOKORO_MODE, KOKORO_TTS_SPEED, KOKORO_TTS_URL, KOKORO_TTS_VOICE
from backend.telephony.filler import get_filler_audio_16k, get_filler_audio_24k
from .normalizer import normalize_text_for_hindi_tts

logger = logging.getLogger("hindi-ai-calling.tts")

_KOKORO_PIPELINE = None
_VOICE_CACHE = {}


def get_voice_tensor(pipeline, voice_name: str):
    """Cache loaded voice tensors to avoid repeated HuggingFace lookups."""
    global _VOICE_CACHE
    if voice_name not in _VOICE_CACHE:
        try:
            logger.info(f"Pre-caching Kokoro voice tensor: {voice_name}")
            _VOICE_CACHE[voice_name] = pipeline.load_voice(voice_name)
        except Exception as e:
            logger.warning(f"Could not preload voice {voice_name}: {e}")
            return voice_name
    return _VOICE_CACHE.get(voice_name, voice_name)


def get_kokoro_pipeline():
    """Lazily load or return the in-process Kokoro KPipeline for Hindi."""
    global _KOKORO_PIPELINE
    if _KOKORO_PIPELINE is None:
        logger.info("Initializing in-process Kokoro TTS pipeline (lang='h')...")
        from kokoro import KPipeline

        _KOKORO_PIPELINE = KPipeline(lang_code="h")
        get_voice_tensor(_KOKORO_PIPELINE, KOKORO_TTS_VOICE)
        logger.info(f"Kokoro-82M TTS in-process pipeline ready (default voice={KOKORO_TTS_VOICE})")
    return _KOKORO_PIPELINE


class KokoroTTSService(TTSService):
    """
    Pipecat TTS service running Kokoro-82M directly in-process or via HTTP API.
    Streams synthesized PCM audio chunks asynchronously into the Pipecat pipeline
    with ultra-low Time-to-First-Audio.
    """

    def __init__(
        self,
        *,
        mode: str = KOKORO_MODE,
        base_url: str = KOKORO_TTS_URL,
        voice: str = KOKORO_TTS_VOICE,
        sample_rate: int = 24000,
        settings: Optional[TTSSettings] = None,
        echo_shield = None,
        **kwargs,
    ):
        default_settings = TTSSettings(
            model="kokoro-82m",
            voice=voice,
            language=Language.HI,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )
        self._mode = mode
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._sample_rate = sample_rate
        self._sent_filler_for_turn = False
        self._echo_shield = echo_shield

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # A new LLM response turn starts: re-enable filler for the very first sentence
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame, CancelFrame)):
            self._sent_filler_for_turn = False
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        clean_text = normalize_text_for_hindi_tts(text.strip())
        if not clean_text:
            return

        # Strip redundant leading acknowledgment from synthesized text
        stripped = re.sub(
            r"^(जी\s*सर[,।]?\s*|जी\s*मैम[,।]?\s*|जी\s*हाँ[,।]?\s*|जी[,।]?\s*)",
            "",
            clean_text,
            flags=re.IGNORECASE,
        ).strip()
        if stripped:
            clean_text = stripped

        # Send instant filler ONLY ONCE per turn (strictly on the very first sentence of the response)
        is_first_sentence_of_turn = not self._sent_filler_for_turn
        is_exit_or_greeting = bool(re.search(r"^(नमस्ते|नमस्कार|धन्यवाद|अलविदा|बाय)", clean_text))

        if is_first_sentence_of_turn:
            self._sent_filler_for_turn = True
            if not is_exit_or_greeting:
                filler_audio = get_filler_audio_16k() if self._sample_rate == 16000 else get_filler_audio_24k()
                if filler_audio:
                    if self._echo_shield:
                        self._echo_shield.record_playback(len(filler_audio) / (self._sample_rate * 2))
                    logger.info(f"KokoroTTS: Emitting opening instant acknowledgment 'जी सर,' ({len(filler_audio)} bytes)")
                    yield TTSAudioRawFrame(
                        audio=filler_audio,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

        logger.info(f"KokoroTTS synthesis ({self._mode}): '{clean_text[:40]}' (voice={self._voice})")

        # In-Process Mode (Direct inside framework, zero network latency)
        if self._mode == "inproc":
            stop_event = threading.Event()
            try:
                pipeline = get_kokoro_pipeline()
                voice_data = get_voice_tensor(pipeline, self._voice)
                loop = asyncio.get_running_loop()
                q: asyncio.Queue = asyncio.Queue()

                def _worker():
                    try:
                        with torch.inference_mode():
                            for _gs, _ps, audio in pipeline(
                                clean_text,
                                voice=voice_data,
                                speed=KOKORO_TTS_SPEED,
                                split_pattern=r"[।!?\n]+",
                            ):
                                if stop_event.is_set():
                                    break
                                if audio is not None and len(audio) > 0:
                                    audio_np = (
                                        audio.detach().cpu().numpy()
                                        if isinstance(audio, torch.Tensor)
                                        else np.asarray(audio)
                                    )
                                    if self._sample_rate != 24000:
                                        try:
                                            import soxr
                                            audio_np = soxr.resample(audio_np, 24000, self._sample_rate)
                                        except Exception:
                                            ratio = self._sample_rate / 24000.0
                                            new_len = int(len(audio_np) * ratio)
                                            old_x = np.arange(len(audio_np))
                                            new_x = np.linspace(0, len(audio_np) - 1, new_len)
                                            audio_np = np.interp(new_x, old_x, audio_np).astype(np.float32)

                                    # Peak-safe audio gain without digital clipping/distortion
                                    max_peak = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
                                    if max_peak > 0.01:
                                        gain = min(1.25, 0.94 / max_peak)
                                        clean_audio = audio_np * gain
                                    else:
                                        clean_audio = audio_np

                                    pcm_data = (
                                        (clean_audio * 32767)
                                        .astype(np.int16)
                                        .tobytes()
                                    )
                                    loop.call_soon_threadsafe(q.put_nowait, pcm_data)
                    except Exception as ex:
                        if not stop_event.is_set():
                            logger.error(f"Kokoro worker error: {ex}")
                            loop.call_soon_threadsafe(q.put_nowait, ex)
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, None)

                t = threading.Thread(target=_worker, daemon=True)
                t.start()

                try:
                    while True:
                        item = await q.get()
                        if item is None or stop_event.is_set():
                            break
                        if isinstance(item, Exception):
                            yield ErrorFrame(error=f"Kokoro TTS error: {item}")
                            break
                        if self._echo_shield and isinstance(item, (bytes, bytearray)):
                            self._echo_shield.record_playback(len(item) / (self._sample_rate * 2))
                        yield TTSAudioRawFrame(
                            audio=item,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                finally:
                    stop_event.set()
            except asyncio.CancelledError:
                stop_event.set()
                raise
            except Exception as e:
                stop_event.set()
                logger.error(f"Kokoro in-process TTS failed: {e}")
                yield ErrorFrame(error=f"Kokoro in-process TTS failed: {e}")
            return

        # Remote HTTP Mode
        url = f"{self._base_url}/v1/audio/speech/stream"
        payload = {
            "input": text,
            "voice": self._voice,
            "response_format": "pcm",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        logger.error(f"Kokoro TTS server error: {resp.status} - {err}")
                        yield ErrorFrame(error=f"Kokoro TTS error: {resp.status}")
                        return

                    buffer = bytearray()
                    while True:
                        chunk = await resp.content.read(4096)
                        if not chunk:
                            break
                        buffer.extend(chunk)
                        usable_len = (len(buffer) // 2) * 2
                        if usable_len > 0:
                            audio_frame = bytes(buffer[:usable_len])
                            del buffer[:usable_len]
                            if self._echo_shield:
                                self._echo_shield.record_playback(len(audio_frame) / (self._sample_rate * 2))
                            yield TTSAudioRawFrame(
                                audio=audio_frame,
                                sample_rate=self._sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )

                    if len(buffer) >= 2:
                        usable_len = (len(buffer) // 2) * 2
                        audio_frame = bytes(buffer[:usable_len])
                        if self._echo_shield:
                            self._echo_shield.record_playback(len(audio_frame) / (self._sample_rate * 2))
                        yield TTSAudioRawFrame(
                            audio=audio_frame,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
        except Exception as e:
            logger.error(f"Kokoro TTS request failed: {e}")
            yield ErrorFrame(error=f"Kokoro TTS request failed: {e}")
