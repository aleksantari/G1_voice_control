"""Tests for the LLM command parser (mocked — no real API calls)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from parser.llm_parser import LLMCommandParser
from schema.command_schema import Action, Magnitude


def _mock_response(content: dict) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = json.dumps(content)
    return mock


@pytest.fixture
def parser():
    """Create an LLMCommandParser with a mocked OpenAI client."""
    with patch("parser.llm_parser.OpenAI"):
        p = LLMCommandParser()
    return p


class TestMoveUpSmall:
    """'move up a little' maps to MOVE_UP / SMALL."""

    def test_action_and_magnitude(self, parser):
        parser.client.chat.completions.create.return_value = _mock_response(
            {"action": "MOVE_UP", "magnitude": "SMALL", "confidence": 0.95, "frame": "CAMERA"}
        )
        cmd = parser.parse("move up a little")
        assert cmd.action == Action.MOVE_UP
        assert cmd.magnitude == Magnitude.SMALL
        assert cmd.value_mm == 2.0
        assert cmd.raw_text == "move up a little"


class TestStopCommand:
    """'stop' maps to STOP with None magnitude."""

    def test_stop_parse(self, parser):
        parser.client.chat.completions.create.return_value = _mock_response(
            {"action": "STOP", "magnitude": None, "confidence": 0.99, "frame": "CAMERA"}
        )
        cmd = parser.parse("stop")
        assert cmd.action == Action.STOP
        assert cmd.magnitude is None
        assert cmd.value_mm is None


class TestInvalidActionFallback:
    """Unknown action in JSON leads to graceful STOP fallback."""

    def test_bad_action_returns_stop(self, parser):
        parser.client.chat.completions.create.return_value = _mock_response(
            {"action": "FLY_AWAY", "magnitude": "MID", "confidence": 0.8, "frame": "CAMERA"}
        )
        cmd = parser.parse("fly away")
        assert cmd.action == Action.STOP
        assert cmd.confidence == 0.0


class TestResponseFormat:
    """Verify response_format={"type": "json_object"} is passed."""

    def test_json_object_format_requested(self, parser):
        parser.client.chat.completions.create.return_value = _mock_response(
            {"action": "MOVE_LEFT", "magnitude": "MID", "confidence": 0.9, "frame": "CAMERA"}
        )
        parser.parse("go left")
        call_kwargs = parser.client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}
