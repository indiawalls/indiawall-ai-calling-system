"""Text-to-Speech services module."""

from .kokoro_service import KokoroTTSService, get_kokoro_pipeline, get_voice_tensor
from .normalizer import normalize_text_for_hindi_tts

__all__ = [
    "KokoroTTSService",
    "get_kokoro_pipeline",
    "get_voice_tensor",
    "normalize_text_for_hindi_tts",
]
