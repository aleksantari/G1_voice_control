"""Audio recording utilities for voice command capture.

Records via PulseAudio (parec) so it works through PipeWire and can
access USB microphones. Saves WAV files via soundfile.
"""

import logging
import subprocess
import time

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _find_source() -> str | None:
    """Auto-detect a PulseAudio input source, preferring USB mics."""
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"], text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    sources = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and ".monitor" not in parts[1]:
            sources.append(parts[1])

    # Prefer USB sources
    for s in sources:
        if "usb" in s.lower():
            return s
    # Fall back to first non-monitor input
    return sources[0] if sources else None


class AudioRecorder:
    """Records audio from the microphone via PulseAudio.

    Uses parec (PulseAudio record) to capture audio through PipeWire,
    which has access to all audio devices including USB mics.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.source = _find_source()
        if self.source:
            logger.info("AudioRecorder: using source '%s'", self.source)
        else:
            logger.warning("AudioRecorder: no PulseAudio source found")

    def _parec_cmd(self) -> list[str]:
        """Build the parec command."""
        cmd = [
            "parec",
            "--format=float32le",
            f"--channels={self.channels}",
            f"--rate={self.sample_rate}",
        ]
        if self.source:
            cmd.append(f"--device={self.source}")
        return cmd

    def record_push_to_talk(self) -> np.ndarray:
        """Record audio with push-to-talk: ENTER to start, ENTER to stop.

        Returns:
            Float32 numpy array of audio samples in [-1, 1].
        """
        input("Press ENTER to start recording...")

        proc = subprocess.Popen(
            self._parec_cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        input("Recording... Press ENTER to stop.")
        proc.terminate()
        raw = proc.stdout.read()
        proc.wait()

        audio = np.frombuffer(raw, dtype=np.float32)
        logger.info(
            "Recorded %.2fs of audio (%d samples at %dHz)",
            len(audio) / self.sample_rate,
            len(audio),
            self.sample_rate,
        )
        return audio

    def record_fixed_duration(self, duration_seconds: float = 5.0) -> np.ndarray:
        """Record audio for a fixed duration.

        Args:
            duration_seconds: How long to record in seconds.

        Returns:
            Float32 numpy array of audio samples in [-1, 1].
        """
        logger.info("Recording %.1fs of audio...", duration_seconds)

        proc = subprocess.Popen(
            self._parec_cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(duration_seconds)
        proc.terminate()
        raw = proc.stdout.read()
        proc.wait()

        audio = np.frombuffer(raw, dtype=np.float32)
        logger.info("Recorded %d samples at %dHz.", len(audio), self.sample_rate)
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
