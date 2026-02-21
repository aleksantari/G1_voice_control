"""Audio recording utilities for voice command capture.

Provides push-to-talk and fixed-duration recording using sounddevice,
and WAV file saving via soundfile.
"""

import logging
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio from the default microphone.

    Supports push-to-talk (ENTER to start/stop) and fixed-duration
    recording modes. Audio is captured as float32 mono at 16 kHz.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_push_to_talk(self) -> np.ndarray:
        """Record audio with push-to-talk: ENTER to start, ENTER to stop.

        Returns:
            Float32 numpy array of audio samples in [-1, 1].
        """
        chunks: list[np.ndarray] = []
        recording = threading.Event()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio callback status: %s", status)
            if recording.is_set():
                chunks.append(indata.copy())

        input("Press ENTER to start recording...")
        recording.set()

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        with stream:
            input("Recording... Press ENTER to stop.")
            recording.clear()

        audio = np.concatenate(chunks, axis=0).flatten()
        logger.info(
            "Recorded %.2fs of audio (%d samples)",
            len(audio) / self.sample_rate,
            len(audio),
        )
        return audio

    def record_fixed_duration(self, duration_seconds: float = 5.0) -> np.ndarray:
        """Record audio for a fixed duration.

        Args:
            duration_seconds: How long to record in seconds.

        Returns:
            Float32 numpy array of audio samples in [-1, 1].
        """
        num_samples = int(self.sample_rate * duration_seconds)
        logger.info("Recording %.1fs of audio...", duration_seconds)
        audio = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()
        logger.info("Recorded %d samples.", len(audio))
        return audio

    def save_wav(
        self, audio: np.ndarray, path: str, sample_rate: int = 16000
    ) -> None:
        """Save audio to a WAV file.

        Args:
            audio: Float32 numpy array of audio samples.
            path: Output file path.
            sample_rate: Sample rate in Hz.
        """
        sf.write(path, audio, sample_rate)
        logger.info("Saved WAV to %s", path)
