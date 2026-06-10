"""
RawBoost Audio Augmentation.

Applies realistic audio degradation to raw waveforms during training,
simulating conditions encountered in real-world deployment: background noise,
codec compression, transmission artifacts, and gain variation.

Reference:
    H. Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
    applied to Automatic Speaker Verification Anti-Spoofing" (2022)
    https://arxiv.org/abs/2111.04433
"""

import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SNR_RANGE = (15, 40)        # dB — conservative to avoid destroying signal
_BIT_DEPTHS = (8, 12, 14)    # Codec simulation quantization levels
_FILTER_COEFF = (-0.95, 0.95) # IIR filter coefficient range
_GAIN_RANGE = (-6, 6)         # dB gain perturbation
_MIN_AUGS = 1
_MAX_AUGS = 3


class RawBoostAugmentor:
    """
    Applies a random subset of audio degradation transforms to a waveform.

    Each call randomly selects 1–3 augmentations from:
        1. Additive colored noise (white or pink)
        2. Codec simulation (quantization + dithering)
        3. Random IIR filtering
        4. Gain perturbation

    Args:
        sample_rate: Audio sample rate (used for future frequency-aware augmentations).
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._augmentations = [
            self._additive_noise,
            self._codec_simulation,
            self._random_filter,
            self._gain_perturbation,
        ]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Augment a waveform tensor in-place.

        Args:
            waveform: 1-D float32 tensor of shape (num_samples,).

        Returns:
            Augmented waveform tensor (same shape).
        """
        x = waveform.numpy()

        num_augs = random.randint(_MIN_AUGS, _MAX_AUGS)
        chosen = random.sample(self._augmentations, num_augs)

        for aug_fn in chosen:
            x = aug_fn(x)

        return torch.from_numpy(x).float()

    # ------------------------------------------------------------------
    # Individual augmentation methods
    # ------------------------------------------------------------------
    @staticmethod
    def _additive_noise(x: np.ndarray) -> np.ndarray:
        """Add white or pink (1/f) noise at a random SNR."""
        snr_db = random.uniform(*_SNR_RANGE)
        noise = np.random.randn(len(x))

        # 50% chance of converting to pink noise via cumulative sum
        if random.random() > 0.5:
            noise = np.cumsum(noise)
            noise -= noise.mean()

        signal_power = np.mean(x ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))

        return (x + scale * noise).astype(np.float32)

    @staticmethod
    def _codec_simulation(x: np.ndarray) -> np.ndarray:
        """Simulate low-bitrate codec via quantization with dither."""
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

        max_abs = np.max(np.abs(y)) + 1e-10
        return (y / max(max_abs, 1.0)).astype(np.float32)

    @staticmethod
    def _gain_perturbation(x: np.ndarray) -> np.ndarray:
        """Apply random gain change in dB."""
        gain_db = random.uniform(*_GAIN_RANGE)
        return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)
