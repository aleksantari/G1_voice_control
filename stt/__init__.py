"""Speech-to-text module using OpenAI Whisper."""

__all__ = ["WhisperSTT", "AudioRecorder"]


def __getattr__(name):
    if name == "WhisperSTT":
        from stt.stt_whisper import WhisperSTT
        return WhisperSTT
    if name == "AudioRecorder":
        from stt.audio_recorder import AudioRecorder
        return AudioRecorder
    raise AttributeError(f"module 'stt' has no attribute {name!r}")
