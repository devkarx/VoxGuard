"""
RawBoost Audio Augmentation.

Applies realistic signal degradation to raw waveforms during training,
simulating conditions encountered in real-world deployment: background
noise, codec compression, transmission artifacts, and gain variation.

Reference:
    H. Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
    applied to Automatic Speaker Verification Anti-Spoofing" (2022)
    https://arxiv.org/abs/2111.04433
"""

import random

import numpy as np
import torch


# Augmentation parameter ranges
_SNR_RANGE = (15, 40)         # dB — conservative to avoid destroying the signal
_BIT_DEPTHS = (8, 12, 14)     # simulate low-bitrate codecs
_FILTER_COEFF = (-0.95, 0.95) # IIR filter coefficient range
_GAIN_RANGE = (-6, 6)         # dB gain perturbation


class RawBoostAugmentor:
    """
    Applies a random subset of audio degradation transforms.

    Each call randomly selects 1–3 augmentations from:
        - Additive colored noise (white or pink)
        - Codec simulation (quantization + dithering)
        - Random IIR filtering
        - Gain perturbation

    Args:
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._transforms = [
            self._additive_noise,
            self._codec_simulation,
            self._random_filter,
            self._gain_perturbation,
        ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to a waveform.

        Args:
            waveform: 1-D float32 tensor of shape (num_samples,).

        Returns:
            Augmented waveform tensor with the same shape.
        """
        x = waveform.numpy()

        num_augs = random.randint(1, 3)
        chosen = random.sample(self._transforms, num_augs)

        for transform in chosen:
            x = transform(x)

        return torch.from_numpy(x).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sr={self.sample_rate}, n_transforms={len(self._transforms)})"

    # ------------------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------------------
    @staticmethod
    def _additive_noise(x: np.ndarray) -> np.ndarray:
        """Add white or pink (1/f) noise at a random SNR."""
        snr_db = random.uniform(*_SNR_RANGE)
        noise = np.random.randn(len(x))

        # 50/50 chance of pink noise (cumulative sum of white noise)
        if random.random() > 0.5:
            noise = np.cumsum(noise)
            noise -= noise.mean()

        signal_power = np.mean(x ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))

        return (x + scale * noise).astype(np.float32)

    @staticmethod
    def _codec_simulation(x: np.ndarray) -> np.ndarray:
        """Simulate low-bitrate codec via quantization with dither noise."""
        bits = random.choice(_BIT_DEPTHS)
        max_val = 2 ** (bits - 1)
        dither = np.random.uniform(-0.5 / max_val, 0.5 / max_val, len(x))
        return (np.round(x * max_val + dither) / max_val).astype(np.float32)

    @staticmethod
    def _random_filter(x: np.ndarray) -> np.ndarray:
        """Apply a single-pole IIR filter with a random coefficient."""
        coeff = random.uniform(*_FILTER_COEFF)
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = x[i] + coeff * y[i - 1]

        peak = np.max(np.abs(y)) + 1e-10
        return (y / max(peak, 1.0)).astype(np.float32)

    @staticmethod
    def _gain_perturbation(x: np.ndarray) -> np.ndarray:
        """Apply a random gain change in dB."""
        gain_db = random.uniform(*_GAIN_RANGE)
        return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)
