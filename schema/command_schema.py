"""Canonical command schema for the G1 voice control pipeline.

Defines the structured JSON format that all upstream modules (LLM parser,
regex fallback) produce and all downstream modules (validation, robot bridge)
consume.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Action(str, Enum):
    """Robot actions supported by the endoscope assistant."""

    MOVE_FORWARD = "MOVE_FORWARD"
    RETRACT = "RETRACT"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    ROTATE_LEFT = "ROTATE_LEFT"
    ROTATE_RIGHT = "ROTATE_RIGHT"
    STOP = "STOP"


class Magnitude(str, Enum):
    """Movement magnitude levels with corresponding millimeter values."""

    SMALL = "SMALL"
    MID = "MID"
    BIG = "BIG"


MAGNITUDE_MM: dict[Magnitude, float] = {
    Magnitude.SMALL: 2.0,
    Magnitude.MID: 4.0,
    Magnitude.BIG: 6.0,
}


class RobotCommand(BaseModel):
    """A single validated robot command.

    Produced by the LLM parser or regex fallback, consumed by the
    robot bridge to generate DDS messages for the Unitree G1.
    """

    action: Action
    magnitude: Optional[Magnitude] = Magnitude.MID
    frame: str = "CAMERA"
    confidence: float = Field(ge=0.0, le=1.0)
    value_mm: Optional[float] = None
    raw_text: str

    @model_validator(mode="after")
    def _validate_stop_and_populate_value(self) -> "RobotCommand":
        """Enforce STOP has no magnitude/value_mm and auto-populate value_mm."""
        if self.action == Action.STOP:
            self.magnitude = None
            self.value_mm = None
        elif self.magnitude is not None and self.value_mm is None:
            self.value_mm = MAGNITUDE_MM[self.magnitude]
        return self

    def is_valid(self) -> bool:
        """Return True if confidence meets the minimum threshold (0.7)."""
        return self.confidence >= 0.7

    @classmethod
    def create_stop(cls, raw_text: str) -> "RobotCommand":
        """Create a STOP command with full confidence.

        Args:
            raw_text: The original spoken text that triggered this command.
        """
        return cls(
            action=Action.STOP,
            magnitude=None,
            confidence=1.0,
            raw_text=raw_text,
        )
