"""Speech-to-Text services module."""

from .sherpa_service import SherpaOnnxSTTService, get_shared_recognizer

__all__ = ["SherpaOnnxSTTService", "get_shared_recognizer"]
