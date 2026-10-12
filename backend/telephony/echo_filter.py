"""
Smart Noise Suppression & Echo Filter for Pipecat.
- Eliminates microphone background noise (fans, hum, breath) with a dynamic noise gate.
- Suppresses speaker acoustic echo while AI assistant is speaking to prevent broken responses.
- Allows HIGH-VOLUME commands (e.g., loud 'रुको', 'सुनो') to intentionally interrupt the AI.
- Automatically resets echo shield on interruption/cancellation so bot never gets stuck deaf.
"""

import logging
import time
import numpy as np

from pipecat.frames.frames import (
    BotInterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartInterruptionFrame,
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
    4. Auto-Reset: Immediately disarms echo shield upon any interruption or user turn start.
    """

    def __init__(
        self,
        noise_gate_threshold: float = 0.006,
        interrupt_threshold: float = 0.10,
        echo_tail_secs: float = 0.20,
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
            logger.debug("SmartFilter: Bot started speaking -> echo shield armed")
            await self.push_frame(frame, direction)

        elif isinstance(
            frame,
            (
                BotStoppedSpeakingFrame,
                TTSStoppedFrame,
                BotInterruptionFrame,
                InterruptionFrame,
                StartInterruptionFrame,
                CancelFrame,
            ),
        ):
            self._bot_speaking = False
            self._bot_stopped_time = time.time()
            logger.debug("SmartFilter: Bot stopped/interrupted -> echo shield disarmed")
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
                # Bot is currently speaking:
                # Only pass through if user speaks firmly/loudly (barge-in command like "रुको" / "सुनो")
                if rms >= self._interrupt_threshold:
                    logger.info(f"SmartFilter: High-volume command detected during playback (RMS={rms:.2f} >= {self._interrupt_threshold:.2f}) -> allowing barge-in")
                    await self.push_frame(frame, direction)
                else:
                    # Echo / ambient noise: replace with silence to prevent bot from interrupting itself
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
                # Bot is quiet: pass normal user speech through to VAD & STT
                if rms < self._noise_gate_threshold:
                    # Only complete room silence/mic hiss is muted
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
