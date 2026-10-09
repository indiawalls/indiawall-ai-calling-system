"""
Exotel Media Stream Frame Serializer for Telephony Phone Calls.
Handles bidirectional 16-bit Linear PCM audio, resampling, DTMF, barge-in clear events,
and caller phone number detection from the start JSON event.
"""

import audioop
import base64
import json
import logging
from typing import Any, Callable, Optional

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
)
from pipecat.serializers.exotel import ExotelFrameSerializer as PipecatExotelSerializer

logger = logging.getLogger("hindi-ai-calling.telephony.exotel")


class ExotelFrameSerializer(PipecatExotelSerializer):
    """
    Exotel serializer with enhanced start event handling, caller phone extraction,
    and zero-latency frame-by-frame Linear PCM resampling (audioop).
    """

    def __init__(
        self,
        stream_sid: str = "exotel_stream",
        call_sid: str | None = None,
        params: Optional[PipecatExotelSerializer.InputParams] = None,
        on_start_callback: Optional[Callable] = None,
        **kwargs,
    ):
        if params is None:
            params = PipecatExotelSerializer.InputParams(exotel_sample_rate=8000, sample_rate=16000)
        super().__init__(stream_sid=stream_sid, call_sid=call_sid, params=params, **kwargs)
        self._first_media_logged = False
        self._on_start_callback = on_start_callback
        self._in_audioop_state = None
        self._out_audioop_state = None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Handles Exotel start/stop lifecycle events and media streaming."""
        if not data:
            return None
        try:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            msg = json.loads(data)
            event_type = msg.get("event")

            if event_type == "start":
                start_obj = msg.get("start", {})
                logger.info(f"Exotel RAW start event: {json.dumps(msg, indent=2, default=str)}")

                sid = (
                    start_obj.get("streamSid")
                    or start_obj.get("stream_sid")
                    or start_obj.get("stream_id")
                    or msg.get("streamSid")
                    or msg.get("stream_sid")
                    or msg.get("stream_id")
                )
                if sid:
                    self._stream_sid = sid
                call_id = (
                    start_obj.get("callSid")
                    or start_obj.get("call_sid")
                    or start_obj.get("call_id")
                    or msg.get("callSid")
                    or msg.get("call_sid")
                )
                if call_id:
                    self._call_sid = call_id

                from_num = (
                    start_obj.get("from")
                    or start_obj.get("caller")
                    or start_obj.get("From")
                    or start_obj.get("Caller")
                    or start_obj.get("customField")
                    or msg.get("from")
                    or msg.get("caller")
                    or msg.get("From")
                    or ""
                )

                if self._on_start_callback:
                    try:
                        self._on_start_callback(self._call_sid, from_num, start_obj)
                    except Exception as cb_err:
                        logger.warning(f"Error in Exotel on_start_callback: {cb_err}")

                media_format = start_obj.get("mediaFormat", {}) or start_obj.get("media_format", {})
                detected_sr = (
                    media_format.get("sampleRate")
                    or media_format.get("sample_rate")
                    or start_obj.get("sampleRate")
                    or start_obj.get("sample_rate")
                )
                if detected_sr:
                    new_sr = int(detected_sr)
                    if new_sr != self._exotel_sample_rate:
                        self._exotel_sample_rate = new_sr
                        self._in_audioop_state = None
                        self._out_audioop_state = None

                logger.info(
                    f"Exotel Stream Start: streamSid={self._stream_sid}, "
                    f"callSid={self._call_sid}, "
                    f"caller={from_num or 'Unknown'}, "
                    f"exotel_sample_rate={self._exotel_sample_rate}, "
                    f"pipeline_sample_rate={self._sample_rate}"
                )
                return None

            elif event_type == "media":
                media_obj = msg.get("media", {})
                payload_b64 = media_obj.get("payload", "")
                if not payload_b64:
                    return None
                raw_bytes = base64.b64decode(payload_b64)
                if not raw_bytes:
                    return None

                if not self._first_media_logged:
                    self._first_media_logged = True
                    logger.info(
                        f"Exotel FIRST media chunk: "
                        f"payload_bytes={len(raw_bytes)}, "
                        f"exotel_sr={self._exotel_sample_rate}, "
                        f"pipeline_sr={self._sample_rate or 16000}"
                    )

                # Real-time frame-by-frame resampling (avoids multi-frame buffering delays)
                in_sr = self._exotel_sample_rate or 8000
                out_sr = self._sample_rate or 16000
                if in_sr != out_sr:
                    resampled_audio, self._in_audioop_state = audioop.ratecv(
                        raw_bytes, 2, 1, in_sr, out_sr, self._in_audioop_state
                    )
                else:
                    resampled_audio = raw_bytes

                if not resampled_audio:
                    return None

                return InputAudioRawFrame(
                    audio=resampled_audio,
                    num_channels=1,
                    sample_rate=out_sr,
                )

            elif event_type == "dtmf":
                digit = msg.get("dtmf", {}).get("digit")
                try:
                    return InputDTMFFrame(KeypadEntry(digit))
                except (ValueError, TypeError):
                    logger.info(f"Invalid DTMF digit: {digit}")
                    return None

            elif event_type == "stop":
                logger.info(f"Exotel Stream Stop: streamSid={self._stream_sid}")
                return EndFrame()

        except Exception as ex:
            logger.error(f"Error deserializing Exotel frame: {ex}")

        return None

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes frames to Exotel JSON protocol with barge-in clear support."""
        if isinstance(frame, InterruptionFrame):
            answer = {
                "event": "clear",
                "streamSid": self._stream_sid,
                "stream_sid": self._stream_sid,
            }
            logger.info(f"Exotel Barge-in: Sent clear event for stream {self._stream_sid}")
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio
            in_sr = frame.sample_rate
            out_sr = self._exotel_sample_rate or 8000
            if in_sr != out_sr:
                resampled_data, self._out_audioop_state = audioop.ratecv(
                    data, 2, 1, in_sr, out_sr, self._out_audioop_state
                )
            else:
                resampled_data = data

            if resampled_data is None or len(resampled_data) == 0:
                return None

            payload = base64.b64encode(resampled_data).decode("ascii")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "stream_sid": self._stream_sid,
                "media": {"payload": payload},
            }
            return json.dumps(answer)

        return await super().serialize(frame)
