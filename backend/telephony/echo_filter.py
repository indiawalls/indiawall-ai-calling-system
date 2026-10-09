"""
Telephony Speaker Acoustic Echo Filter for Pipecat.
Suppresses telephone microphone bleed while AI assistant is speaking.
"""

import logging
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger("hindi-ai-calling.telephony.echo")


class TelephonyEchoFilter(FrameProcessor):
    """Filters out incoming telephone audio while the AI bot is speaking to prevent
    phone speaker acoustic echo from triggering VAD and false STT transcriptions."""

    def __init__(self, echo_tail_secs: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self._echo_tail_secs = echo_tail_secs
        self._bot_speaking = False
        self._bot_stopped_time = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            logger.info("TelephonyEchoFilter: Bot started speaking -> silencing incoming audio to prevent echo")
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_time = time.time()
            logger.info("TelephonyEchoFilter: Bot stopped speaking -> starting 350ms echo hangover timer")
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame):
            is_echo_window = self._bot_speaking or (time.time() - self._bot_stopped_time < self._echo_tail_secs)
            if is_echo_window:
                # Replace with silence to prevent VAD from triggering on phone speaker echo
                silent_frame = InputAudioRawFrame(
                    audio=b"\x00" * len(frame.audio),
                    num_channels=frame.num_channels,
                    sample_rate=frame.sample_rate,
                )
                await self.push_frame(silent_frame, direction)
            else:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
