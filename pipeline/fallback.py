"""Command validation and fallback parsing logic.

Provides a two-tier parsing strategy: try LLM first, fall back to regex
if the LLM fails or returns low-confidence results.
"""

import logging

from schema.command_schema import Action, RobotCommand

logger = logging.getLogger(__name__)


class CommandValidator:
    """Validates parsed commands against a confidence threshold."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def validate(self, command: RobotCommand) -> tuple[bool, str]:
        """Validate a parsed command.

        STOP commands always pass (safety-critical).
        Other commands must meet the confidence threshold.

        Returns:
            (True, "ok") if valid, (False, reason) if not.
        """
        if command.action == Action.STOP:
            return True, "ok"

        if command.confidence < self.confidence_threshold:
            return False, (
                f"confidence {command.confidence:.2f} "
                f"< {self.confidence_threshold:.2f}"
            )

        return True, "ok"


class FallbackManager:
    """Manages LLM-first parsing with regex fallback.

    Never raises exceptions — always returns a RobotCommand.
    """

    FALLBACK_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, llm_parser, regex_parser, validator: CommandValidator):
        self.llm_parser = llm_parser
        self.regex_parser = regex_parser
        self.validator = validator

    def parse_with_fallback(self, text: str) -> tuple[RobotCommand, str]:
        """Parse text using LLM first, falling back to regex.

        Args:
            text: Transcribed spoken command.

        Returns:
            (RobotCommand, source) where source is "llm", "regex", or "failed".
        """
        # Try LLM first
        try:
            cmd = self.llm_parser.parse(text)
            if cmd.confidence >= self.FALLBACK_CONFIDENCE_THRESHOLD:
                logger.info("LLM parsed '%s' -> %s (conf=%.2f)", text, cmd.action.value, cmd.confidence)
                return cmd, "llm"
            logger.warning(
                "LLM confidence %.2f < %.2f for '%s', trying regex",
                cmd.confidence, self.FALLBACK_CONFIDENCE_THRESHOLD, text,
            )
        except Exception as exc:
            logger.warning("LLM failed for '%s': %s, trying regex", text, exc)

        # Try regex fallback
        try:
            cmd = self.regex_parser.parse(text)
            if cmd is not None:
                logger.info("Regex parsed '%s' -> %s", text, cmd.action.value)
                return cmd, "regex"
            logger.warning("Regex returned None for '%s'", text)
        except Exception as exc:
            logger.warning("Regex failed for '%s': %s", text, exc)

        # Total failure — safe STOP
        logger.error("All parsers failed for '%s', returning safe STOP", text)
        return RobotCommand(
            action=Action.STOP, magnitude=None, confidence=0.0, raw_text=text
        ), "failed"
