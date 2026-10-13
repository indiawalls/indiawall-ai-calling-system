"""
Instant acknowledgment filler manager for IndiaWalls AI Voice Calling.
Provides pre-cached, zero-latency "जी सर," audio to mask LLM generation latency.
"""

from pathlib import Path
import logging

logger = logging.getLogger("hindi-ai-calling.filler")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FILLER_16K_PATH = ROOT_DIR / "assets" / "audio" / "filler_ji_sir_16k.pcm"
FILLER_24K_PATH = ROOT_DIR / "assets" / "audio" / "filler_ji_sir_24k.pcm"

_FILLER_16K: bytes = b""
_FILLER_24K: bytes = b""

try:
    if FILLER_16K_PATH.exists():
        _FILLER_16K = FILLER_16K_PATH.read_bytes()
        logger.info(f"Loaded instant 16kHz filler audio: {len(_FILLER_16K)} bytes (~{round(len(_FILLER_16K)/32000, 2)}s)")
    if FILLER_24K_PATH.exists():
        _FILLER_24K = FILLER_24K_PATH.read_bytes()
        logger.info(f"Loaded instant 24kHz filler audio: {len(_FILLER_24K)} bytes (~{round(len(_FILLER_24K)/48000, 2)}s)")
except Exception as e:
    logger.warning(f"Could not load pre-cached filler audio: {e}")


def get_filler_audio_16k() -> bytes:
    """Return pre-rendered 16kHz PCM audio for 'जी सर,' (telephony / Exotel)."""
    return _FILLER_16K


def get_filler_audio_24k() -> bytes:
    """Return pre-rendered 24kHz PCM audio for 'जी सर,' (web browser)."""
    return _FILLER_24K
