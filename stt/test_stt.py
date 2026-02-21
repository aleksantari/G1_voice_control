"""Tests for the STT module."""

import os
import tempfile

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from stt.stt_whisper import WhisperSTT
from stt.audio_recorder import AudioRecorder


class TestWhisperSTTInit:
    @patch("stt.stt_whisper.whisper")
    @patch("stt.stt_whisper.torch")
    def test_loads_model_with_correct_args(self, mock_torch, mock_whisper):
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = MagicMock()

        stt = WhisperSTT(model_size="tiny.en")

        mock_whisper.load_model.assert_called_once_with("tiny.en", device="cpu")
        assert stt.model_size == "tiny.en"
        assert stt.device == "cpu"
        assert stt.fp16 is False

    @patch("stt.stt_whisper.whisper")
    @patch("stt.stt_whisper.torch")
    def test_cuda_sets_fp16_true(self, mock_torch, mock_whisper):
        mock_torch.cuda.is_available.return_value = True
        mock_whisper.load_model.return_value = MagicMock()

        stt = WhisperSTT(model_size="base.en")

        mock_whisper.load_model.assert_called_once_with("base.en", device="cuda")
        assert stt.fp16 is True

    @patch("stt.stt_whisper.whisper")
    @patch("stt.stt_whisper.torch")
    def test_explicit_device_overrides_auto(self, mock_torch, mock_whisper):
        mock_whisper.load_model.return_value = MagicMock()

        stt = WhisperSTT(model_size="tiny.en", device="cpu")

        mock_whisper.load_model.assert_called_once_with("tiny.en", device="cpu")
        # torch.cuda.is_available should not be called when device is explicit
        assert stt.device == "cpu"


class TestTranscribeArray:
    @patch("stt.stt_whisper.whisper")
    @patch("stt.stt_whisper.torch")
    def test_returns_correct_dict_structure(self, mock_torch, mock_whisper):
        mock_torch.cuda.is_available.return_value = False
        mock_whisper.load_model.return_value = MagicMock()

        # Mock the transcription pipeline
        mock_result = MagicMock()
        mock_result.text = " move up "
        mock_whisper.decode.return_value = mock_result
        mock_whisper.pad_or_trim.return_value = np.zeros(16000 * 30, dtype=np.float32)
        mock_mel = MagicMock()
        mock_mel.to.return_value = mock_mel
        mock_whisper.log_mel_spectrogram.return_value = mock_mel
        mock_whisper.DecodingOptions.return_value = MagicMock()

        stt = WhisperSTT(model_size="tiny.en")

        # Generate a dummy sine wave
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        sine_wave = np.sin(2 * np.pi * 440 * t)

        result = stt.transcribe_array(sine_wave)

        assert "text" in result
        assert "language" in result
        assert "duration" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["duration"], float)
        assert result["language"] == "en"
        assert result["text"] == "move up"  # should be stripped


class TestSaveWav:
    def test_creates_valid_wav_file(self):
        recorder = AudioRecorder(sample_rate=16000, channels=1)

        # Generate 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            recorder.save_wav(audio, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            # Read it back and verify
            import soundfile as sf
            data, sr = sf.read(path)
            assert sr == 16000
            assert len(data) == 16000
        finally:
            os.unlink(path)
