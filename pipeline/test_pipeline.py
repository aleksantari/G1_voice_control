"""Tests for the language control pipeline."""

import json

import pytest
from unittest.mock import patch, MagicMock

from schema.command_schema import Action, Magnitude, RobotCommand
from pipeline.fallback import CommandValidator, FallbackManager
from parser.regex_fallback import RegexFallbackParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(action, magnitude, confidence):
    """Build a fake OpenAI chat completion response."""
    data = {"action": action, "confidence": confidence}
    if magnitude is not None:
        data["magnitude"] = magnitude
    content = json.dumps(data)
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _build_pipeline():
    """Build a pipeline with mocked STT and OpenAI client."""
    with patch("stt.stt_whisper.whisper"), \
         patch("stt.stt_whisper.torch") as mock_torch, \
         patch("parser.llm_parser.OpenAI") as mock_openai_cls, \
         patch("parser.llm_parser.load_dotenv"):

        mock_torch.cuda.is_available.return_value = False

        from pipeline.pipeline import LanguageControlPipeline
        pipe = LanguageControlPipeline()

    return pipe, mock_openai_cls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessTextHappyPath:
    def test_move_up_small(self):
        pipe, mock_openai_cls = _build_pipeline()
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "MOVE_UP", "SMALL", 0.95
        )

        result = pipe.process_text("move up a little")

        assert result["text"] == "move up a little"
        assert result["command"].action == Action.MOVE_UP
        assert result["command"].magnitude == Magnitude.SMALL
        assert result["source"] == "llm"
        assert result["valid"] is True
        assert result["message"] == "ok"


class TestFallbackOnLLMException:
    def test_regex_kicks_in(self):
        pipe, mock_openai_cls = _build_pipeline()
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        result = pipe.process_text("retract")

        assert result["command"].action == Action.RETRACT
        assert result["source"] == "regex"


class TestTotalFailure:
    def test_returns_safe_stop(self):
        pipe, mock_openai_cls = _build_pipeline()
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        result = pipe.process_text("how are you today")

        assert result["command"].action == Action.STOP
        assert result["command"].confidence == 0.0
        assert result["source"] == "failed"


class TestStopBypassesThreshold:
    def test_low_confidence_stop_still_valid(self):
        validator = CommandValidator(confidence_threshold=0.7)
        stop_cmd = RobotCommand(
            action=Action.STOP, magnitude=None, confidence=0.1, raw_text="stop"
        )
        valid, message = validator.validate(stop_cmd)

        assert valid is True
        assert message == "ok"


class TestLatencyFields:
    def test_latency_present_and_positive(self):
        pipe, mock_openai_cls = _build_pipeline()
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "MOVE_LEFT", "MID", 0.9
        )

        result = pipe.process_text("go left")

        assert "latency_stt_ms" in result
        assert "latency_parse_ms" in result
        assert result["latency_stt_ms"] == 0.0  # no STT for process_text
        assert result["latency_parse_ms"] > 0
