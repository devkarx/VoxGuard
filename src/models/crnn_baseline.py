"""
CRNN Baseline Model for Deepfake Audio Detection.

A Convolutional Recurrent Neural Network operating on 64-band Mel-spectrograms.
Included as a baseline comparison against the Wav2Vec2 primary model.

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
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_N_MELS = 64
_HIDDEN_SIZE = 128
_NUM_LSTM_LAYERS = 2
_CNN_CHANNELS = [1, 16, 32, 64]
_CNN_DROPOUT = 0.2
_LSTM_DROPOUT = 0.3
_CLASSIFIER_DROPOUT = 0.5


class DeepfakeCRNN(nn.Module):
    """
    CNN-RNN hybrid model for deepfake audio detection.

    Processes Mel-spectrograms of shape (batch, 1, n_mels, time_steps)
    and outputs a single logit per sample.

    Args:
        n_mels:      Number of Mel frequency bands (default: 64).
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers:  Number of stacked LSTM layers (default: 2).
    """

    def __init__(
        self,
        n_mels: int = _N_MELS,
        hidden_size: int = _HIDDEN_SIZE,
        num_layers: int = _NUM_LSTM_LAYERS,
    ) -> None:
        super().__init__()

        # --- CNN feature extractor ---
        self.cnn = nn.Sequential(
            self._conv_block(_CNN_CHANNELS[0], _CNN_CHANNELS[1]),
            self._conv_block(_CNN_CHANNELS[1], _CNN_CHANNELS[2]),
            self._conv_block(_CNN_CHANNELS[2], _CNN_CHANNELS[3]),
        )

        # After 3 rounds of MaxPool(2,2): n_mels // 8
        cnn_output_dim = _CNN_CHANNELS[3] * (n_mels // 8)

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=_LSTM_DROPOUT,
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=_CLASSIFIER_DROPOUT),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel-spectrogram tensor of shape (batch, 1, n_mels, time_steps).

        Returns:
            Logits of shape (batch,).
        """
        # CNN feature extraction
        x = self.cnn(x)

        # Reshape: (B, C, F, T) → (B, T, C*F) for LSTM
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Temporal modeling
        lstm_out, _ = self.lstm(x)

        # Max-pool over time dimension
        x, _ = torch.max(lstm_out, dim=1)

        # Classification
        logits = self.classifier(x)
        return logits.squeeze(-1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Single CNN block: Conv → BN → ReLU → Pool → Dropout."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=_CNN_DROPOUT),
        )
