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
        self._log_interval = log_interval
        self._last_log_time = 0.0
        self._label = label or "VAD"
        self._peak_volume = 0.0
        self._frame_count = 0

    def _run_analyzer(self, buffer: bytes) -> VADState:
        """Override to inject volume logging."""
        # Run parent logic
        state = super()._run_analyzer(buffer)

        now = time.monotonic()
        vol = self._prev_volume
        self._peak_volume = max(self._peak_volume, vol)
        self._frame_count += 1

        if now - self._last_log_time >= self._log_interval:
            state_name = state.name
            params = self._params
            logger.debug(
                f"[{self._label}] vol={vol:.3f} peak={self._peak_volume:.3f} "
                f"state={state_name} "
                f"(thresholds: min_vol={params.min_volume}, conf={params.confidence})"
            )
            self._peak_volume = 0.0
            self._last_log_time = now

        # Log transitions at INFO level for easier debugging
        if state == VADState.SPEAKING and self._vad_starting_count == 0:
            logger.info(f"[{self._label}] Voice DETECTED (vol={vol:.3f})")
        elif state == VADState.QUIET and self._vad_stopping_count == 0:
            if self._frame_count > 5:  # Skip initial quiet
                logger.info(f"[{self._label}] Voice ENDED (vol={vol:.3f})")

        return state
