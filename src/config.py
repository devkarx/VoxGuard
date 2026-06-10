"""
Centralized configuration for training, inference, and audio processing.

All hyperparameters and audio processing constants live here so they can
be imported from a single place instead of being scattered across modules.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class AudioConfig:
    """Audio I/O and feature extraction settings."""

    sample_rate: int = 16_000
    duration: float = 4.0       # seconds — clips are padded/cropped to this
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512

    @property
    def target_length(self) -> int:
        """Number of samples in a fixed-length clip."""
        return int(self.sample_rate * self.duration)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture-level knobs for both models."""

    # Wav2Vec2
    wav2vec_name: str = "facebook/wav2vec2-base"
    wav2vec_hidden_dim: int = 768
    wav2vec_num_hidden_states: int = 13   # 1 CNN + 12 transformer layers
    wav2vec_frozen_layers: int = 8        # freeze bottom N transformer layers
    attention_bottleneck: int = 128
    classifier_hidden: int = 256
    classifier_dropout: float = 0.3

    # CRNN baseline
    crnn_hidden_size: int = 128
    crnn_lstm_layers: int = 2
    crnn_channels: tuple = (1, 16, 32, 64)
    crnn_cnn_dropout: float = 0.2
    crnn_lstm_dropout: float = 0.3
    crnn_classifier_dropout: float = 0.5


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    epochs: int = 20
    batch_size: int = 16
    gradient_accumulation_steps: int = 2

    # Optimizer
    backbone_lr: float = 1e-5      # lower LR for pre-trained layers
    head_lr: float = 1e-4          # higher LR for the classifier head
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler — linear warmup then cosine decay
    warmup_fraction: float = 0.1   # fraction of total steps for warmup
    min_lr: float = 1e-7

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: str = "weights"

    # Logging
    log_dir: str = "runs"

    # Data
    num_workers: int = 4
    class_mapping: Dict[str, int] = field(default_factory=lambda: {
        "genuine": 0, "bonafide": 0, "real": 0,
        "spoof": 1, "deepfake": 1, "fake": 1,
    })
