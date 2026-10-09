"""
Sherpa-ONNX IndicConformer STT Service for Pipecat.

A custom Pipecat SegmentedSTTService that uses sherpa-onnx's OfflineRecognizer
with the AI4Bharat IndicConformer NeMo CTC model for local Hindi speech-to-text.

Model: parismitaglobalsolutions/indicconformer-sherpa-onnx (NeMo CTC, INT8)
No external API calls — runs entirely on CPU, ~250-300 MB RAM.
"""

import asyncio
import io
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
import wave

import numpy as np

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.settings import STTSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

logger = logging.getLogger("sherpa-stt")

# Global singleton cache for IndicConformer recognizer
_SHARED_RECOGNIZER = None


def get_shared_recognizer(model_dir: str, num_threads: int = 2):
    """Load and cache the IndicConformer recognizer once in memory across all calls."""
    global _SHARED_RECOGNIZER
    if _SHARED_RECOGNIZER is None:
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError(
                "sherpa-onnx is not installed. Install it with:\n"
                "  pip install sherpa-onnx"
            )

        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        for path, name in [
            (model_path, "model.int8.onnx"),
            (tokens_path, "tokens.txt"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Run 'python download_models.py' to download the model files."
                )

        logger.info(f"Loading shared IndicConformer CTC model from {os.path.abspath(model_dir)}")
        _SHARED_RECOGNIZER = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            decoding_method="greedy_search",
        )
        logger.info("Shared IndicConformer STT model ready (NeMo CTC, INT8)")
    return _SHARED_RECOGNIZER


@dataclass
class SherpaOnnxSTTSettings(STTSettings):
    """Settings for the Sherpa-ONNX STT service."""
    pass


class SherpaOnnxSTTService(SegmentedSTTService):
    """Local Hindi STT service using sherpa-onnx IndicConformer (NeMo CTC)."""

    Settings = SherpaOnnxSTTSettings

    def __init__(
        self,
        *,
        model_dir: str = "models/indicconformer-hi",
        num_threads: int = 2,
        sample_rate: int = 16000,
        **kwargs,
    ):
        settings = self.Settings(
            model="indicconformer-hi",
            language=Language.HI,
        )

        super().__init__(
            sample_rate=sample_rate,
            settings=settings,
            ttfs_p99_latency=0.25,
            **kwargs,
        )

        self._model_dir = model_dir
        self._num_threads = num_threads
        self._target_sample_rate = sample_rate
        self._recognizer = None
        self._init_recognizer()

    def _init_recognizer(self):
        """Initialize or reuse the cached sherpa-onnx OfflineRecognizer."""
        self._recognizer = get_shared_recognizer(self._model_dir, self._num_threads)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        if language == Language.HI:
            return "hi"
        return None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data using sherpa-onnx IndicConformer."""
        if self._recognizer is None:
            yield ErrorFrame(error="Sherpa-ONNX recognizer not initialized")
            return

        try:
            t0 = time.time()
            await self.start_processing_metrics()

            samples = self._decode_wav_audio(audio)
            if samples is None or len(samples) == 0:
                logger.warning("Empty audio received for STT")
                await self.stop_processing_metrics()
                return

            audio_duration = round(len(samples) / 16000.0, 2)
            loop = asyncio.get_running_loop()

            def _decode():
                stream = self._recognizer.create_stream()
                stream.accept_waveform(16000, samples)
                self._recognizer.decode_stream(stream)
                return stream.result.text.strip() if stream.result.text else ""

            text = await loop.run_in_executor(None, _decode)
            stt_elapsed = round((time.time() - t0) * 1000)
            await self.stop_processing_metrics()

            # Filter single breath noise artifacts, solitary hesitation fillers and micro-fragments
            is_valid_speech = False
            if text:
                clean_t = text.strip()
                words = clean_t.split()
                filler_tokens = {"ह", "अ", "आ", "ए", "उ", "हम्म", "हं", "hm", "hmm", "uh", "um", "ये", "जी", "ओह", "हाँ", "हा", "न"}
                if len(clean_t) >= 3 and not (all(w in filler_tokens for w in words) and len(words) <= 2):
                    is_valid_speech = True

            if is_valid_speech:
                logger.info(f"STT (IndicConformer) finished in {stt_elapsed}ms ({audio_duration}s audio): [{text}]")
                yield TranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                )
            else:
                if text:
                    logger.info(f"STT (IndicConformer) ignored noise artifact in {stt_elapsed}ms: [{text}]")
                else:
                    logger.debug(f"STT (IndicConformer) finished in {stt_elapsed}ms ({audio_duration}s audio): [empty]")

        except Exception as e:
            logger.error(f"Sherpa-ONNX STT error: {e}", exc_info=True)
            yield ErrorFrame(error=f"Sherpa-ONNX STT error: {e}")

    def _decode_wav_audio(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Decode WAV audio bytes to float32 numpy array and resample to 16kHz."""
        try:
            content = io.BytesIO(audio_bytes)
            with wave.open(content, "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

            if sample_width == 2:  # 16-bit PCM
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:  # 32-bit PCM
                samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                logger.error(f"Unsupported sample width: {sample_width}")
                return None

            if sample_rate != 16000:
                try:
                    import soxr
                    samples = soxr.resample(samples, sample_rate, 16000)
                except Exception:
                    ratio = 16000.0 / sample_rate
                    new_length = int(len(samples) * ratio)
                    old_indices = np.arange(len(samples))
                    new_indices = np.linspace(0, len(samples) - 1, new_length)
                    samples = np.interp(new_indices, old_indices, samples).astype(np.float32)

            return samples

        except Exception as e:
            logger.error(f"Error decoding WAV audio: {e}")
            return None
