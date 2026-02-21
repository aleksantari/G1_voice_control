"""Regex-based fallback command parser.

Provides instant, offline command parsing using pattern matching.
Used as a fallback when the LLM parser is unavailable or too slow.
"""

import re

from schema.command_schema import Action, Magnitude, RobotCommand


class RegexFallbackParser:
    """Parses spoken surgical commands using regex pattern matching.

    Checks patterns in safety-critical order:
    1. STOP (highest priority)
    2. Rotation (before simple left/right)
    3. Directional movement
    """

    CONFIDENCE = 0.6

    # STOP patterns - checked first (safety critical)
    _STOP_RE = re.compile(
        r"\b(stop|halt|freeze|hold|don'?t\s+move)\b", re.IGNORECASE
    )

    # Rotation patterns - checked before simple left/right
    _ROTATE_LEFT_RE = re.compile(
        r"\b(rotate\s+left|twist\s+left|turn\s+left|counter[- ]?clockwise)\b",
        re.IGNORECASE,
    )
    _ROTATE_RIGHT_RE = re.compile(
        r"\b(rotate\s+right|twist\s+right|turn\s+right|clockwise)\b",
        re.IGNORECASE,
    )

    # Direction patterns
    _DIRECTION_PATTERNS = [
        (re.compile(r"\b(up|raise|higher)\b", re.IGNORECASE), Action.MOVE_UP),
        (re.compile(r"\b(down|lower)\b", re.IGNORECASE), Action.MOVE_DOWN),
        (re.compile(r"\b(left)\b", re.IGNORECASE), Action.MOVE_LEFT),
        (re.compile(r"\b(right)\b", re.IGNORECASE), Action.MOVE_RIGHT),
        (
            re.compile(r"\b(forward|advance|push|deeper)\b", re.IGNORECASE),
            Action.MOVE_FORWARD,
        ),
        (
            re.compile(r"\b(back|retract|pull|withdraw)\b", re.IGNORECASE),
            Action.RETRACT,
        ),
    ]

    # Magnitude patterns
    _SMALL_RE = re.compile(
        r"\b(a\s+little|slightly|tiny|nudge|bit|smidge)\b", re.IGNORECASE
    )
    _BIG_RE = re.compile(
        r"\b(a\s+lot|big|far|much|significantly|way)\b", re.IGNORECASE
    )

    def parse(self, text: str) -> RobotCommand | None:
        """Parse a spoken command using regex patterns.

        Args:
            text: The transcribed spoken command.

        Returns:
            A RobotCommand with confidence=0.6, or None if no pattern matches.
        """
        lower = text.lower()

        # 1. STOP (safety critical - check first)
        if self._STOP_RE.search(lower):
            return RobotCommand(
                action=Action.STOP,
                magnitude=None,
                confidence=self.CONFIDENCE,
                raw_text=text,
            )

        # 2. Rotation (check before simple left/right)
        if self._ROTATE_LEFT_RE.search(lower):
            return RobotCommand(
                action=Action.ROTATE_LEFT,
                magnitude=self._get_magnitude(lower),
                confidence=self.CONFIDENCE,
                raw_text=text,
            )
        if self._ROTATE_RIGHT_RE.search(lower):
            return RobotCommand(
                action=Action.ROTATE_RIGHT,
                magnitude=self._get_magnitude(lower),
                confidence=self.CONFIDENCE,
                raw_text=text,
            )

        # 3. Directional movement
        for pattern, action in self._DIRECTION_PATTERNS:
            if pattern.search(lower):
                return RobotCommand(
                    action=action,
                    magnitude=self._get_magnitude(lower),
                    confidence=self.CONFIDENCE,
                    raw_text=text,
                )

        return None

    def _get_magnitude(self, lower_text: str) -> Magnitude:
        """Determine magnitude from text, defaulting to MID."""
        if self._SMALL_RE.search(lower_text):
            return Magnitude.SMALL
        if self._BIG_RE.search(lower_text):
            return Magnitude.BIG
        return Magnitude.MID
