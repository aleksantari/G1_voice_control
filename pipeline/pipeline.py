"""Main language control pipeline.

Wires together STT, LLM parsing, regex fallback, and validation
into a single entry point for voice-controlled robot commands.
"""

import logging
import time

from config.settings import _load_yaml
from parser.llm_parser import LLMCommandParser
from parser.regex_fallback import RegexFallbackParser
from pipeline.fallback import CommandValidator, FallbackManager
from stt.audio_recorder import AudioRecorder
from stt.stt_whisper import WhisperSTT

logger = logging.getLogger(__name__)


class LanguageControlPipeline:
    """End-to-end pipeline: audio/text -> validated RobotCommand."""

    def __init__(self, config_path: str = "config/config.yaml"):
        config = _load_yaml()

        # STT
        whisper_cfg = config["whisper"]
        self.stt = WhisperSTT(
            model_size=whisper_cfg["model_size"],
            device=whisper_cfg.get("device"),
        )
        self.recorder = AudioRecorder()

        # Parsers
        self.llm_parser = LLMCommandParser()
        self.regex_parser = RegexFallbackParser()

        # Validation + fallback
        pipeline_cfg = config["pipeline"]
        self.validator = CommandValidator(
            confidence_threshold=pipeline_cfg["confidence_threshold"]
        )
        self.fallback = FallbackManager(
            self.llm_parser, self.regex_parser, self.validator
        )

        logger.info("Pipeline initialized.")

    def process_text(self, text: str) -> dict:
        """Parse a text command (skip STT).

        Args:
            text: The spoken command as a string.

        Returns:
            Result dict with command, source, validity, and latency.
        """
        start = time.perf_counter()
        cmd, source = self.fallback.parse_with_fallback(text)
        latency_parse_ms = (time.perf_counter() - start) * 1000

        valid, message = self.validator.validate(cmd)

        return {
            "text": text,
            "command": cmd,
            "source": source,
            "valid": valid,
            "message": message,
            "latency_stt_ms": 0.0,
            "latency_parse_ms": latency_parse_ms,
        }

    def process_audio_file(self, audio_path: str) -> dict:
        """Full pipeline: audio file -> transcription -> command.

        Args:
            audio_path: Path to a WAV/MP3 audio file.

        Returns:
            Result dict with command, source, validity, and latency.
        """
        start_stt = time.perf_counter()
        result = self.stt.transcribe_file(audio_path)
        latency_stt_ms = (time.perf_counter() - start_stt) * 1000
        text = result["text"]

        start_parse = time.perf_counter()
        cmd, source = self.fallback.parse_with_fallback(text)
        latency_parse_ms = (time.perf_counter() - start_parse) * 1000

        valid, message = self.validator.validate(cmd)

        return {
            "text": text,
            "command": cmd,
            "source": source,
            "valid": valid,
            "message": message,
            "latency_stt_ms": latency_stt_ms,
            "latency_parse_ms": latency_parse_ms,
        }

    def process_microphone(self) -> dict:
        """Full pipeline: push-to-talk -> transcription -> command.

        Returns:
            Result dict with command, source, validity, and latency.
        """
        audio = self.recorder.record_push_to_talk()

        start_stt = time.perf_counter()
        result = self.stt.transcribe_array(audio)
        latency_stt_ms = (time.perf_counter() - start_stt) * 1000
        text = result["text"]

        start_parse = time.perf_counter()
        cmd, source = self.fallback.parse_with_fallback(text)
        latency_parse_ms = (time.perf_counter() - start_parse) * 1000

        valid, message = self.validator.validate(cmd)

        return {
            "text": text,
            "command": cmd,
            "source": source,
            "valid": valid,
            "message": message,
            "latency_stt_ms": latency_stt_ms,
            "latency_parse_ms": latency_parse_ms,
        }
