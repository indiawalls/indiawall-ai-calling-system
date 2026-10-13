"""
Custom VAD Processor with volume debug logging.

Wraps the standard SileroVADAnalyzer to periodically log actual audio volume
levels during calls. This helps calibrate min_volume thresholds for different
audio sources (browser mic vs phone codec).
"""

import logging
import time

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState

logger = logging.getLogger("hindi-ai-calling.vad")


class DebugSileroVADAnalyzer(SileroVADAnalyzer):
    """SileroVADAnalyzer with periodic volume/confidence debug logging.

    Logs volume and confidence every ~2 seconds so you can see
    what levels normal speech produces and calibrate min_volume.
    """

    def __init__(self, *, sample_rate: int, params: VADParams | None = None,
                 log_interval: float = 2.0, label: str = ""):
        super().__init__(sample_rate=sample_rate, params=params)
        self.set_sample_rate(sample_rate)
        self._log_interval = log_interval
        self._last_log_time = 0.0
        self._label = label or "VAD"
        self._peak_volume = 0.0
        self._frame_count = 0
        self._last_confidence = 0.0
        self._prev_state = VADState.QUIET

    def voice_confidence(self, buffer: bytes) -> float:
        conf = super().voice_confidence(buffer)
        try:
            self._last_confidence = float(conf)
        except Exception:
            self._last_confidence = 0.0
        return conf

    def _run_analyzer(self, buffer: bytes) -> VADState:
        """Override to inject volume and confidence logging."""
        # Run parent logic
        state = super()._run_analyzer(buffer)

        now = time.monotonic()
        vol = float(self._prev_volume) if self._prev_volume is not None else 0.0
        self._peak_volume = max(self._peak_volume, vol)
        self._frame_count += 1
        conf_val = float(self._last_confidence)

        if now - self._last_log_time >= self._log_interval:
            state_name = state.name
            params = self._params
            logger.info(
                f"[{self._label}] Audio active: vol={vol:.3f} peak={self._peak_volume:.3f} "
                f"conf={conf_val:.2f} frames={self._frame_count} state={state_name} "
                f"(thresholds: min_vol={params.min_volume}, conf={params.confidence})"
            )
            self._peak_volume = 0.0
            self._last_log_time = now

        # Log real state transitions only
        if state != self._prev_state:
            if state == VADState.SPEAKING:
                logger.info(f"[{self._label}] User speech DETECTED! (vol={vol:.3f}, conf={conf_val:.2f})")
            elif state == VADState.QUIET and self._prev_state in (VADState.SPEAKING, VADState.STOPPING):
                logger.info(f"[{self._label}] User speech ENDED (vol={vol:.3f}) -> finalizing turn")
            self._prev_state = state

        return state
