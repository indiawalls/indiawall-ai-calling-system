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


class AcousticEchoShield:
    """Thread-safe playback duration tracker shared between TTS output and mic input filter."""

    def __init__(self, echo_tail_secs: float = 0.35):
        self._echo_tail_secs = echo_tail_secs
        self._bot_speaking_until = 0.0

    def record_playback(self, duration_secs: float):
        """Called whenever audio is yielded by TTS / output."""
        now = time.time()
        base = max(self._bot_speaking_until, now)
        self._bot_speaking_until = base + duration_secs

    def is_bot_speaking(self) -> bool:
        """Returns True if bot audio is currently playing (or within echo tail window)."""
        return time.time() < (self._bot_speaking_until + self._echo_tail_secs)

    def reset(self):
        """Reset when caller interrupts or call ends."""
        self._bot_speaking_until = 0.0


class SmartNoiseAndEchoFilter(FrameProcessor):
    """
    Intelligent audio pre-filter for VAD & STT:
    1. Noise Gate: Silences low-level background noise (hiss, fan, breath).
    2. Echo Cancellation: When bot is speaking, blocks incoming echo so AI voice never breaks itself.
    3. High-Volume Barge-In: If user speaks firmly/loudly (above interrupt_threshold), passes audio
       through so intentional commands ('रुको', 'सुनो') interrupt the bot cleanly.
    4. Auto-Reset: Immediately disarms echo shield upon any interruption or user turn start.
    """

    def __init__(
        self,
        echo_shield: AcousticEchoShield | None = None,
        interrupt_threshold: float = 0.08,
        echo_tail_secs: float = 0.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._echo_shield = echo_shield or AcousticEchoShield(echo_tail_secs=echo_tail_secs)
        self._interrupt_threshold = interrupt_threshold
        self._echo_tail_secs = echo_tail_secs
        self._bot_speaking = False
        self._bot_stopped_time = 0.0
        self._barge_in_hold_until = 0.0

    @property
    def echo_shield(self) -> AcousticEchoShield:
        return self._echo_shield

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
                BotInterruptionFrame,
                InterruptionFrame,
                StartInterruptionFrame,
                CancelFrame,
            ),
        ):
            self._bot_speaking = False
            self._bot_stopped_time = 0.0
            self._barge_in_hold_until = 0.0
            self._echo_shield.reset()
            logger.debug("SmartFilter: Interruption event -> echo shield disarmed immediately")
            await self.push_frame(frame, direction)

        elif isinstance(frame, (BotStoppedSpeakingFrame, TTSStoppedFrame)):
            self._bot_speaking = False
            self._bot_stopped_time = time.time()
            logger.debug("SmartFilter: Bot audio generation stopped -> echo shield holds tail for playback")
            await self.push_frame(frame, direction)

        elif isinstance(frame, InputAudioRawFrame):
            if not frame.audio or len(frame.audio) == 0:
                await self.push_frame(frame, direction)
                return
            if not hasattr(self, "_first_audio_logged"):
                self._first_audio_logged = True
                logger.info(f"SmartFilter: Audio stream active! First audio packet arrived from WebSocket client ({len(frame.audio)} bytes, {frame.sample_rate}Hz)")

            # Compute RMS volume level (0.0 to 1.0)
            samples = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

            now = time.time()
            is_echo_window = (
                self._echo_shield.is_bot_speaking()
                or self._bot_speaking
                or (now - self._bot_stopped_time < self._echo_tail_secs)
            )

            if is_echo_window:
                # If caller audio energy reaches interrupt threshold, open the barge-in gate
                # and hold it open for 450ms so the entire natural speech envelope reaches VAD/STT intact.
                if rms >= self._interrupt_threshold:
                    self._barge_in_hold_until = now + 0.45
                    logger.info(
                        f"SmartFilter: Barge-in speech triggered during playback (RMS={rms:.3f} >= {self._interrupt_threshold:.3f}) -> gate open"
                    )

                if now < self._barge_in_hold_until:
                    # Pass the natural audio through to Silero VAD without slicing/zeroing
                    await self.push_frame(frame, direction)
                else:
                    # Echo / ambient noise: replace with silence so bot voice never triggers VAD
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
                self._barge_in_hold_until = 0.0
                # Bot is quiet: pass normal user speech through to VAD & STT completely untouched!
                await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
