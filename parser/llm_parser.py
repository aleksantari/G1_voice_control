"""LLM-based command parser using OpenAI's Chat Completions API.

Converts natural language surgical commands into validated RobotCommand
objects using GPT-4o-mini with structured JSON output.
"""

import json
import logging
import time

from openai import OpenAI
from pydantic import ValidationError

from config.settings import _load_yaml, _PROJECT_ROOT
from dotenv import load_dotenv
from parser.prompt_templates import SYSTEM_PROMPT, USER_TEMPLATE
from schema.command_schema import RobotCommand

logger = logging.getLogger(__name__)


class LLMCommandParser:
    """Parses spoken surgical commands into RobotCommand via OpenAI API.

    Uses GPT-4o-mini with response_format=json_object to guarantee
    valid JSON output, then validates against the RobotCommand schema.
    """

    def __init__(self, model: str | None = None):
        config = _load_yaml()
        llm_config = config["llm"]
        self.model = model or llm_config["model"]
        self.temperature = llm_config["temperature"]
        self.max_tokens = llm_config["max_tokens"]
        load_dotenv(_PROJECT_ROOT / ".env")
        self.client = OpenAI()

    def parse(self, text: str) -> RobotCommand:
        """Parse a spoken command string into a validated RobotCommand.

        Args:
            text: The transcribed spoken command (e.g. "move up a little").

        Returns:
            A validated RobotCommand. On parse failure, returns a low-confidence
            STOP command as a safe fallback.
        """
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(text=text)},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            data["raw_text"] = text
            cmd = RobotCommand(**data)
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Parse failed for '%s' (%.0fms): %s", text, elapsed_ms, exc
            )
            return RobotCommand(
                action="STOP", magnitude=None, confidence=0.0, raw_text=text
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Parsed '%s' -> %s (%.0fms)", text, cmd.action.value, elapsed_ms
        )
        return cmd
