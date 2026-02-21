"""Tests for the regex fallback parser."""

import pytest

from parser.regex_fallback import RegexFallbackParser
from schema.command_schema import Action, Magnitude


@pytest.fixture
def parser():
    return RegexFallbackParser()


class TestDirectionalCommands:
    def test_move_up_small(self, parser):
        cmd = parser.parse("move up a little")
        assert cmd.action == Action.MOVE_UP
        assert cmd.magnitude == Magnitude.SMALL

    def test_retract_default_mid(self, parser):
        cmd = parser.parse("retract")
        assert cmd.action == Action.RETRACT
        assert cmd.magnitude == Magnitude.MID

    def test_advance_scope(self, parser):
        cmd = parser.parse("advance the scope")
        assert cmd.action == Action.MOVE_FORWARD
        assert cmd.magnitude == Magnitude.MID


class TestRotationPriority:
    def test_rotate_left_not_move_left(self, parser):
        cmd = parser.parse("rotate left")
        assert cmd.action == Action.ROTATE_LEFT

    def test_rotate_right(self, parser):
        cmd = parser.parse("rotate right")
        assert cmd.action == Action.ROTATE_RIGHT


class TestStopVariants:
    def test_stop(self, parser):
        cmd = parser.parse("stop")
        assert cmd.action == Action.STOP

    def test_freeze(self, parser):
        cmd = parser.parse("freeze")
        assert cmd.action == Action.STOP

    def test_hold_it(self, parser):
        cmd = parser.parse("hold it")
        assert cmd.action == Action.STOP

    def test_dont_move(self, parser):
        cmd = parser.parse("don't move")
        assert cmd.action == Action.STOP


class TestMagnitude:
    def test_big_magnitude(self, parser):
        cmd = parser.parse("go way up")
        assert cmd.action == Action.MOVE_UP
        assert cmd.magnitude == Magnitude.BIG

    def test_small_magnitude(self, parser):
        cmd = parser.parse("nudge left")
        assert cmd.action == Action.MOVE_LEFT
        assert cmd.magnitude == Magnitude.SMALL


class TestNoMatch:
    def test_unrelated_text(self, parser):
        result = parser.parse("how are you")
        assert result is None


class TestConfidence:
    def test_all_commands_have_correct_confidence(self, parser):
        commands = [
            "move up", "move down", "go left", "go right",
            "advance", "retract", "rotate left", "rotate right", "stop",
        ]
        for text in commands:
            cmd = parser.parse(text)
            assert cmd is not None, f"No match for '{text}'"
            assert cmd.confidence == 0.6, f"Wrong confidence for '{text}'"


class TestNoisyInput:
    def test_filler_words(self, parser):
        cmd = parser.parse("uh go left I think")
        assert cmd.action == Action.MOVE_LEFT
