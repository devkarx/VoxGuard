"""
Wav2Vec2-Based Deepfake Audio Classifier.

Uses facebook/wav2vec2-base as a frozen feature extractor with partial
fine-tuning of the top transformer layers. Captures phase coherence,
prosody, and micro-timing patterns that Mel-spectrograms discard.

Architecture:
    Raw 16 kHz waveform
    → Wav2Vec2 CNN encoder (frozen)
    → 12-layer Transformer (top 4 fine-tuned)
    → Weighted layer aggregation (learnable)
    → Attentive statistics pooling
    → MLP classifier → logit

Based on approaches from ASVspoof 2021/2024 top-performing systems.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from src.config import ModelConfig

_cfg = ModelConfig()


class Wav2Vec2DeepfakeDetector(nn.Module):
    """
    Binary classifier built on Wav2Vec2 self-supervised representations.

    Accepts raw 16 kHz mono waveforms and outputs a single logit where
    values > 0 indicate deepfake and values < 0 indicate genuine speech.

    Args:
        num_frozen_layers: Number of bottom transformer layers to freeze.
            Wav2Vec2-Base has 12 transformer layers; the default freezes
            the bottom 8 and fine-tunes the top 4.
    """

    def __init__(self, num_frozen_layers: int = _cfg.wav2vec_frozen_layers) -> None:
        super().__init__()

        # Pre-trained backbone
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(_cfg.wav2vec_name)

        # Freeze CNN feature extractor and bottom transformer layers
        self._freeze_backbone(num_frozen_layers)

        # Learnable weighted sum across all hidden states.
        # Initialized uniformly — the model learns which layers matter most
        # for detecting deepfakes during training.
        self.layer_weights = nn.Parameter(
            torch.ones(_cfg.wav2vec_num_hidden_states) / _cfg.wav2vec_num_hidden_states
        )

        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Linear(_cfg.wav2vec_hidden_dim, _cfg.attention_bottleneck),
            nn.Tanh(),
            nn.Linear(_cfg.attention_bottleneck, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(_cfg.wav2vec_hidden_dim * 2, _cfg.classifier_hidden),
            nn.BatchNorm1d(_cfg.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(_cfg.classifier_dropout),
            nn.Linear(_cfg.classifier_hidden, 1),
        )

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

        # Attentive statistics pooling — learn which time frames matter
        attn_scores = self.attention(weighted)                # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)     # (B, T, 1)

        mean = (weighted * attn_weights).sum(dim=1)           # (B, 768)
        variance = ((weighted - mean.unsqueeze(1)) ** 2 * attn_weights).sum(dim=1)
        std = torch.sqrt(variance + 1e-8)                    # (B, 768)

        pooled = torch.cat([mean, std], dim=1)                # (B, 1536)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)

    def _freeze_backbone(self, num_frozen_layers: int) -> None:
        """Freeze the CNN encoder and bottom transformer layers."""
        self.wav2vec2.feature_extractor._freeze_parameters()

        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def get_layer_weights(self) -> torch.Tensor:
        """Return softmax-normalized layer contribution weights."""
        return torch.softmax(self.layer_weights, dim=0).detach().cpu()

    def count_parameters(self) -> dict:
        """Return trainable vs. frozen parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable_params": trainable, "frozen_params": frozen, "total": trainable + frozen}
