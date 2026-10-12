"""
Smart Noise Suppression & Echo Filter for Pipecat.
- Eliminates microphone background noise (fans, hum, breath) with a dynamic noise gate.
- Suppresses speaker acoustic echo while AI assistant is speaking to prevent broken responses.
- Allows HIGH-VOLUME commands (e.g., loud 'रुको', 'सुनो') to intentionally interrupt the AI.
"""

import logging
import time
import numpy as np

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger("hindi-ai-calling.telephony.noise_filter")


class SmartNoiseAndEchoFilter(FrameProcessor):
    """
    Intelligent audio pre-filter for VAD & STT:
    1. Noise Gate: Silences low-level background noise (hiss, fan, breath) below noise_gate_threshold.
    2. Echo Cancellation: When bot is speaking, blocks incoming echo so AI voice never breaks itself.
    3. High-Volume Barge-In: If user speaks firmly/loudly (above interrupt_threshold), passes audio
       through so intentional commands ('रुको', 'सुनो') interrupt the bot cleanly.
    """

    def __init__(
        self,
        noise_gate_threshold: float = 0.025,
        interrupt_threshold: float = 0.32,
        echo_tail_secs: float = 0.30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._noise_gate_threshold = noise_gate_threshold
        self._interrupt_threshold = interrupt_threshold
        self._echo_tail_secs = echo_tail_secs
        self._bot_speaking = False
        self._bot_stopped_time = 0.0

    def set_bot_speaking(self, speaking: bool):
        """Allow external event handlers to directly notify when bot starts/stops speech."""
        self._bot_speaking = speaking
        if not speaking:
            self._bot_stopped_time = time.time()
        logger.debug(f"SmartFilter: Bot speaking set to {speaking} directly")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (BotStartedSpeakingFrame, TTSStartedFrame)):
            self._bot_speaking = True
            logger.debug("SmartFilter: Bot speaking -> activating echo shield")
            await self.push_frame(frame, direction)

        elif isinstance(frame, (BotStoppedSpeakingFrame, TTSStoppedFrame)):
            self._bot_speaking = False
            self._bot_stopped_time = time.time()
            logger.debug("SmartFilter: Bot stopped speaking -> echo hangover active")
            await self.push_frame(frame, direction)

        elif isinstance(frame, InputAudioRawFrame):
            if not frame.audio or len(frame.audio) == 0:
                await self.push_frame(frame, direction)
                return

            # Compute RMS volume level (0.0 to 1.0)
            samples = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

            is_echo_window = self._bot_speaking or (time.time() - self._bot_stopped_time < self._echo_tail_secs)

            if is_echo_window:
                # Bot is speaking: only pass if user speaks with HIGH volume (intentional interruption)
                if rms >= self._interrupt_threshold:
                    logger.info(f"SmartFilter: High-volume command detected during playback (vol={rms:.2f}) -> allowing barge-in")
                    await self.push_frame(frame, direction)
                else:
                    # Echo / ambient noise: silence frame to keep AI voice playing unbroken
                    silent_audio = b"\x00" * len(frame.audio)
                    await self.push_frame(
                        InputAudioRawFrame(
                            audio=silent_audio,
                            num_channels=frame.num_channels,
                            sample_rate=frame.sample_rate,
                        ),
                        direction,
                    )
            else:
                # Bot is quiet: apply noise gate to filter out ambient room hiss / breath
                if rms < self._noise_gate_threshold:
                    silent_audio = b"\x00" * len(frame.audio)
                    await self.push_frame(
                        InputAudioRawFrame(
                            audio=silent_audio,
                            num_channels=frame.num_channels,
                            sample_rate=frame.sample_rate,
                        ),
                        direction,
                    )
                else:
                    await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
