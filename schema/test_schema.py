"""Tests for the RobotCommand schema."""

import pytest
from pydantic import ValidationError

from schema.command_schema import Action, Magnitude, MAGNITUDE_MM, RobotCommand


class TestValidCommandCreation:
    """Valid command creation for each action type."""

    @pytest.mark.parametrize("action", list(Action))
    def test_all_actions_create_successfully(self, action: Action):
        cmd = RobotCommand(action=action, confidence=0.9, raw_text="test")
        assert cmd.action == action
        assert cmd.frame == "CAMERA"

    def test_move_forward(self):
        cmd = RobotCommand(action="MOVE_FORWARD", confidence=0.9, raw_text="go forward")
        assert cmd.action == Action.MOVE_FORWARD

    def test_retract(self):
        cmd = RobotCommand(action="RETRACT", confidence=0.9, raw_text="pull back")
        assert cmd.action == Action.RETRACT

    def test_rotate_right(self):
        cmd = RobotCommand(action="ROTATE_RIGHT", confidence=0.85, raw_text="turn right")
        assert cmd.action == Action.ROTATE_RIGHT


class TestDefaultMagnitude:
    """Default magnitude behavior."""

    def test_defaults_to_mid(self):
        cmd = RobotCommand(action="MOVE_UP", confidence=0.9, raw_text="go up")
        assert cmd.magnitude == Magnitude.MID

    def test_explicit_small(self):
        cmd = RobotCommand(
            action="MOVE_UP", magnitude="SMALL", confidence=0.9, raw_text="go up a little"
        )
        assert cmd.magnitude == Magnitude.SMALL

    def test_explicit_big(self):
        cmd = RobotCommand(
            action="MOVE_UP", magnitude="BIG", confidence=0.9, raw_text="go up a lot"
        )
        assert cmd.magnitude == Magnitude.BIG


class TestStopCommand:
    """STOP command has no magnitude or value_mm."""

    def test_stop_clears_magnitude(self):
        cmd = RobotCommand(action="STOP", confidence=1.0, raw_text="stop")
        assert cmd.magnitude is None
        assert cmd.value_mm is None

    def test_stop_ignores_explicit_magnitude(self):
        cmd = RobotCommand(
            action="STOP", magnitude="BIG", confidence=1.0, raw_text="stop"
        )
        assert cmd.magnitude is None
        assert cmd.value_mm is None


class TestValueMmAutoPopulation:
    """value_mm auto-population from magnitude."""

    def test_small_value(self):
        cmd = RobotCommand(
            action="MOVE_LEFT", magnitude="SMALL", confidence=0.8, raw_text="left small"
        )
        assert cmd.value_mm == 2.0

    def test_mid_value(self):
        cmd = RobotCommand(action="MOVE_LEFT", confidence=0.8, raw_text="left")
        assert cmd.value_mm == 4.0

    def test_big_value(self):
        cmd = RobotCommand(
            action="MOVE_LEFT", magnitude="BIG", confidence=0.8, raw_text="left big"
        )
        assert cmd.value_mm == 6.0

    def test_magnitude_mm_dict_complete(self):
        for mag in Magnitude:
            assert mag in MAGNITUDE_MM
            assert isinstance(MAGNITUDE_MM[mag], float)


class TestConfidenceValidation:
    """Confidence boundary validation."""

    def test_zero_confidence_valid(self):
        cmd = RobotCommand(action="MOVE_UP", confidence=0.0, raw_text="test")
        assert cmd.confidence == 0.0

    def test_one_confidence_valid(self):
        cmd = RobotCommand(action="MOVE_UP", confidence=1.0, raw_text="test")
        assert cmd.confidence == 1.0

    def test_negative_confidence_invalid(self):
        with pytest.raises(ValidationError):
            RobotCommand(action="MOVE_UP", confidence=-0.1, raw_text="test")

    def test_above_one_confidence_invalid(self):
        with pytest.raises(ValidationError):
            RobotCommand(action="MOVE_UP", confidence=1.1, raw_text="test")


class TestJsonRoundTrip:
    """JSON serialization round-trip."""

    def test_round_trip_move(self):
        original = RobotCommand(
            action="MOVE_FORWARD", magnitude="BIG", confidence=0.95, raw_text="go forward big"
        )
        data = original.model_dump()
        restored = RobotCommand(**data)
        assert restored == original

    def test_round_trip_stop(self):
        original = RobotCommand.create_stop("stop now")
        data = original.model_dump()
        restored = RobotCommand(**data)
        assert restored == original

    def test_dump_contains_expected_keys(self):
        cmd = RobotCommand(action="MOVE_UP", confidence=0.9, raw_text="go up")
        data = cmd.model_dump()
        assert set(data.keys()) == {
            "action", "magnitude", "frame", "confidence", "value_mm", "raw_text"
        }


class TestCreateStop:
    """The create_stop classmethod."""

    def test_creates_stop_action(self):
        cmd = RobotCommand.create_stop("halt")
        assert cmd.action == Action.STOP

    def test_confidence_is_one(self):
        cmd = RobotCommand.create_stop("halt")
        assert cmd.confidence == 1.0

    def test_no_magnitude(self):
        cmd = RobotCommand.create_stop("halt")
        assert cmd.magnitude is None
        assert cmd.value_mm is None

    def test_preserves_raw_text(self):
        cmd = RobotCommand.create_stop("please stop moving")
        assert cmd.raw_text == "please stop moving"

    def test_frame_is_camera(self):
        cmd = RobotCommand.create_stop("stop")
        assert cmd.frame == "CAMERA"


class TestIsValid:
    """The is_valid helper at boundary."""

    def test_below_threshold_returns_false(self):
        cmd = RobotCommand(action="MOVE_UP", confidence=0.69, raw_text="test")
        assert cmd.is_valid() is False

    def test_at_threshold_returns_true(self):
        cmd = RobotCommand(action="MOVE_UP", confidence=0.7, raw_text="test")
        assert cmd.is_valid() is True

    def test_above_threshold_returns_true(self):
        cmd = RobotCommand(action="MOVE_UP", confidence=0.95, raw_text="test")
        assert cmd.is_valid() is True
