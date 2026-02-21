"""Whisper-based speech-to-text transcription.

Uses OpenAI's Whisper model locally for real-time transcription
of spoken surgical commands.
"""

import logging
import time

import numpy as np
import torch
import whisper

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Local speech-to-text using OpenAI Whisper.

    Loads a Whisper model on init and provides methods to transcribe
    audio from files or numpy arrays.
    """

    def __init__(self, model_size: str = "base.en", device: str | None = None):
        """Load a Whisper model.

        Args:
            model_size: Whisper model variant (e.g. "tiny.en", "base.en").
            device: "cuda" or "cpu". Auto-detects if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_size = model_size
        self.fp16 = device == "cuda"

        logger.info("Loading Whisper model '%s' on %s...", model_size, device)
        self.model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded.")

    def transcribe_file(self, audio_path: str) -> dict:
        """Transcribe an audio file.

        Args:
            audio_path: Path to a WAV/MP3/etc. audio file.

        Returns:
            {"text": str, "language": str, "duration": float}
            where duration is the Whisper processing time in seconds.
        """
        start = time.perf_counter()
        result = self.model.transcribe(
            audio_path, language="en", fp16=self.fp16
        )
        duration = time.perf_counter() - start

        text = result["text"].strip()
        logger.info("Transcribed file in %.2fs: '%s'", duration, text)
        return {"text": text, "language": "en", "duration": duration}

    def transcribe_array(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> dict:
        """Transcribe audio from a numpy array.

        Args:
            audio_array: Float32 audio samples, mono, at the given sample rate.
            sample_rate: Sample rate in Hz (Whisper expects 16000).

        Returns:
            {"text": str, "language": str, "duration": float}
            where duration is the Whisper processing time in seconds.
        """
        audio = whisper.pad_or_trim(audio_array.astype(np.float32))

        start = time.perf_counter()
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        options = whisper.DecodingOptions(language="en", fp16=self.fp16)
        result = whisper.decode(self.model, mel, options)
        duration = time.perf_counter() - start

        text = result.text.strip()
        logger.info("Transcribed array in %.2fs: '%s'", duration, text)
        return {"text": text, "language": "en", "duration": duration}
