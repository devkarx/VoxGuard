"""
Wav2Vec2-Based Deepfake Audio Classifier.

Uses facebook/wav2vec2-base as a frozen feature extractor with partial
fine-tuning of the top transformer layers. Captures phase coherence,
prosody, and micro-timing patterns that Mel-spectrograms discard.

Architecture:
    Raw 16kHz waveform
    → Wav2Vec2 CNN encoder (frozen)
    → 12-layer Transformer (top 4 fine-tuned)
    → Weighted layer aggregation (learnable)
    → Attentive statistics pooling
    → MLP classifier → logit

Reference:
    This approach mirrors ASVspoof 2021/2024 winning solutions.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME = "facebook/wav2vec2-base"
_HIDDEN_DIM = 768          # Wav2Vec2-Base hidden size
_NUM_HIDDEN_STATES = 13    # 1 CNN output + 12 transformer layers
_DEFAULT_FROZEN = 8        # Freeze bottom 8 of 12 transformer layers


class Wav2Vec2DeepfakeDetector(nn.Module):
    """
    Binary classifier built on top of Wav2Vec2 self-supervised representations.

    Accepts raw 16 kHz mono waveforms and outputs a single logit where
    values > 0 indicate deepfake and values < 0 indicate genuine speech.

    Args:
        num_frozen_layers: Number of bottom transformer layers to freeze.
            Wav2Vec2-Base has 12 transformer layers. Default freezes the
            bottom 8 and fine-tunes the top 4.
    """

    def __init__(self, num_frozen_layers: int = _DEFAULT_FROZEN) -> None:
        super().__init__()

        # --- Pre-trained backbone ---
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(_MODEL_NAME)
        self._freeze_backbone(num_frozen_layers)

        # --- Learnable weighted sum across all hidden states ---
        self.layer_weights = nn.Parameter(
            torch.ones(_NUM_HIDDEN_STATES) / _NUM_HIDDEN_STATES
        )

        # --- Attentive statistics pooling ---
        self.attention = nn.Sequential(
            nn.Linear(_HIDDEN_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(_HIDDEN_DIM * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, num_samples) — raw 16 kHz float32 audio.

        Returns:
            Logits of shape (batch,).
        """
        # Extract all 13 hidden states from the backbone
        outputs = self.wav2vec2(waveform, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, T, 768)

        # Weighted aggregation across layers
        stacked = torch.stack(hidden_states, dim=0)          # (13, B, T, 768)
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (B, T, 768)

        # Attentive statistics pooling
        attn_scores = self.attention(weighted)                # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)     # (B, T, 1)

        mean = (weighted * attn_weights).sum(dim=1)           # (B, 768)
        variance = ((weighted - mean.unsqueeze(1)) ** 2 * attn_weights).sum(dim=1)
        std = torch.sqrt(variance + 1e-8)                    # (B, 768)

        pooled = torch.cat([mean, std], dim=1)                # (B, 1536)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _freeze_backbone(self, num_frozen_layers: int) -> None:
        """Freeze the CNN encoder and bottom transformer layers."""
        self.wav2vec2.feature_extractor._freeze_parameters()

        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def get_layer_weights(self) -> torch.Tensor:
        """Return normalized layer contribution weights for visualization."""
        return torch.softmax(self.layer_weights, dim=0).detach().cpu()

    def count_parameters(self) -> dict:
        """Return trainable vs. frozen parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
