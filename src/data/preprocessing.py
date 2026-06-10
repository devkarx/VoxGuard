"""
Mel-Spectrogram Preprocessing for the CRNN Baseline.

Converts raw audio files into normalized Mel-spectrogram tensors suitable
for the DeepfakeCRNN model. Used only by the baseline architecture.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import soundfile as sf
from torchaudio import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE = 16000
_DURATION = 4.0       # seconds
_N_MELS = 64
_N_FFT = 1024
_HOP_LENGTH = 512


class AudioPreprocessor:
    """
    Loads audio files and extracts Mel-spectrogram features.

    Pipeline: load → mono → resample → pad/truncate → MelSpectrogram → dB scale.

    Args:
        target_sample_rate: Output sample rate (default: 16 kHz).
        target_duration:    Fixed duration in seconds (default: 4.0).
        n_mels:             Number of Mel bands (default: 64).
        n_fft:              FFT window size (default: 1024).
        hop_length:         STFT hop size (default: 512).
    """

    def __init__(
        self,
        target_sample_rate: int = _SAMPLE_RATE,
        target_duration: float = _DURATION,
        n_mels: int = _N_MELS,
        n_fft: int = _N_FFT,
        hop_length: int = _HOP_LENGTH,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.target_length = int(target_sample_rate * target_duration)
        self.n_mels = n_mels

        self.mel_spectrogram_transform = transforms.MelSpectrogram(
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

            # Convert to (1, samples) tensor
            if len(audio_data.shape) == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(audio_data).T.float()

            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample
            if sample_rate != self.target_sample_rate:
                resampler = transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            # Pad or truncate
            length = waveform.shape[1]
            if length < self.target_length:
                waveform = torch.nn.functional.pad(waveform, (0, self.target_length - length))
            elif length > self.target_length:
                waveform = waveform[:, :self.target_length]

            # Feature extraction
            mel_spec = self.mel_spectrogram_transform(waveform)
            return self.amplitude_to_db(mel_spec)

        except Exception as exc:
            logger.error("Failed to process %s: %s", file_path, exc)
            return None
