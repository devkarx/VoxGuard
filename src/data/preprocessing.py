"""
Mel-Spectrogram Preprocessing for the CRNN Baseline.

Converts raw audio files into normalized Mel-spectrogram tensors
suitable for the DeepfakeCRNN model. Only used by the baseline
architecture — the Wav2Vec2 model operates on raw waveforms directly.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import soundfile as sf
from torchaudio import transforms

from src.config import AudioConfig

logger = logging.getLogger(__name__)

_cfg = AudioConfig()


class AudioPreprocessor:
    """
    Loads audio files and extracts Mel-spectrogram features.

    Pipeline: load → mono → resample → pad/truncate → Mel-spectrogram → dB scale.

    Args:
        target_sample_rate: Output sample rate.
        target_duration:    Fixed duration in seconds.
        n_mels:             Number of Mel bands.
        n_fft:              FFT window size.
        hop_length:         STFT hop size.
    """

    def __init__(
        self,
        target_sample_rate: int = _cfg.sample_rate,
        target_duration: float = _cfg.duration,
        n_mels: int = _cfg.n_mels,
        n_fft: int = _cfg.n_fft,
        hop_length: int = _cfg.hop_length,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.target_length = int(target_sample_rate * target_duration)

        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = transforms.AmplitudeToDB()

    def process(self, file_path: Union[str, Path]) -> Optional[torch.Tensor]:
        """
        Load an audio file and return its Mel-spectrogram.

        Args:
            file_path: Path to a .wav or .flac file.

        Returns:
            Tensor of shape (1, n_mels, time_steps), or None on failure.
        """
        try:
            audio_data, sample_rate = sf.read(str(file_path))

            # Ensure shape is (channels, samples)
            if len(audio_data.shape) == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(audio_data).T.float()

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if the source rate doesn't match
            if sample_rate != self.target_sample_rate:
                resampler = transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            # Pad or truncate to fixed length
            length = waveform.shape[1]
            if length < self.target_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.target_length - length))
            elif length > self.target_length:
                waveform = waveform[:, :self.target_length]

            mel_spec = self.mel_transform(waveform)
            return self.amplitude_to_db(mel_spec)

        except Exception as exc:
            logger.error("Preprocessing failed for %s: %s", file_path, exc)
            return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sr={self.target_sample_rate}, "
            f"length={self.target_length})"
        )
