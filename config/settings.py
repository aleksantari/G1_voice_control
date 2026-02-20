"""Central configuration loader.

Loads secrets from .env and non-secret settings from config.yaml.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"


def _load_yaml() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_openai_api_key() -> str:
    """Return the OpenAI API key from the environment.

    Raises ValueError if the key is not set.
    """
    load_dotenv(_PROJECT_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key or key == "your-api-key-here":
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return key


def get_settings() -> dict:
    """Return the full settings dict with config.yaml values and the API key."""
    config = _load_yaml()
    config["openai_api_key"] = get_openai_api_key()
    return config
