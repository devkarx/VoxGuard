"""
CRNN Baseline for Deepfake Audio Detection.

A Convolutional Recurrent Neural Network operating on 64-band
Mel-spectrograms. Serves as a baseline comparison against the
Wav2Vec2 primary model to demonstrate the advantage of learning
directly from raw waveforms.

Architecture:
    Mel-spectrogram (1, 64, T)
    → 3× [Conv2D → BatchNorm → ReLU → MaxPool → Dropout2D]
    → Reshape to (B, T', features)
    → Bidirectional LSTM (2 layers)
    → Max-pool over time
    → Dropout → Linear → logit
"""

import torch
import torch.nn as nn

from src.config import AudioConfig, ModelConfig

_cfg = ModelConfig()
_audio = AudioConfig()


class DeepfakeCRNN(nn.Module):
    """
    CNN-RNN hybrid for deepfake audio detection.

    Processes Mel-spectrograms of shape (batch, 1, n_mels, time_steps)
    and outputs a single logit per sample.

    Args:
        n_mels:      Number of Mel frequency bands.
        hidden_size: LSTM hidden dimension.
        num_layers:  Number of stacked LSTM layers.
    """

    def __init__(
        self,
        n_mels: int = _audio.n_mels,
        hidden_size: int = _cfg.crnn_hidden_size,
        num_layers: int = _cfg.crnn_lstm_layers,
    ) -> None:
        super().__init__()

        channels = _cfg.crnn_channels

        # CNN feature extractor
        self.cnn = nn.Sequential(
            self._conv_block(channels[0], channels[1]),
            self._conv_block(channels[1], channels[2]),
            self._conv_block(channels[2], channels[3]),
        )

        # After 3× MaxPool(2,2) the frequency axis shrinks to n_mels // 8
        cnn_output_dim = channels[3] * (n_mels // 8)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=_cfg.crnn_lstm_dropout if num_layers > 1 else 0.0,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=_cfg.crnn_classifier_dropout),
            nn.Linear(hidden_size * 2, 1),  # *2 for bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel-spectrogram tensor of shape (batch, 1, n_mels, time_steps).

        Returns:
            Logits of shape (batch,).
        """
        x = self.cnn(x)

        # (B, C, F, T) → (B, T, C*F) for the LSTM
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * freq)

        lstm_out, _ = self.lstm(x)

        # Max-pool over the time dimension
        x, _ = torch.max(lstm_out, dim=1)

        logits = self.classifier(x)
        return logits.squeeze(-1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Conv → BN → ReLU → Pool → Dropout."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=_cfg.crnn_cnn_dropout),
        )
