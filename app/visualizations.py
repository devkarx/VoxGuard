"""
Audio visualization helpers for the Streamlit front-end.

All plots share a consistent dark theme that matches the app's
glassmorphism aesthetic. Renders at 120 DPI for crisp output
on high-resolution displays.
"""

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torchaudio import transforms

matplotlib.use("Agg")

# Shared theme palette
_BG = "#0e1117"
_ACCENT = "#818cf8"
_ACCENT_LIGHT = "#a78bfa"
_GRID = "#1e2030"
_TEXT = "#94a3b8"

# Global rendering settings
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "lines.antialiased": True,
    "text.antialiased": True,
})


def plot_waveform(audio: np.ndarray, sample_rate: int) -> plt.Figure:
    """
    Render an amplitude-over-time waveform.

    Args:
        audio:       1-D numpy array of audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        matplotlib Figure ready for st.pyplot().
    """
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    t = np.linspace(0, len(audio) / sample_rate, len(audio))
    ax.plot(t, audio, color=_ACCENT, linewidth=0.35, alpha=0.9)
    ax.fill_between(t, audio, alpha=0.10, color=_ACCENT_LIGHT)

    ax.set_xlabel("Time (s)", color=_TEXT, fontsize=9)
    ax.set_ylabel("Amplitude", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.set_xlim(0, t[-1] if len(t) > 0 else 1)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)

    plt.tight_layout()
    return fig


def plot_mel_spectrogram(audio: np.ndarray, sample_rate: int) -> plt.Figure:
    """
    Render a Mel-spectrogram heatmap.

    Args:
        audio:       1-D numpy array of audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        matplotlib Figure ready for st.pyplot().
    """
    waveform = torch.from_numpy(audio).float().unsqueeze(0)

    mel = transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64,
    )
    db = transforms.AmplitudeToDB()
    spec = db(mel(waveform))

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    img = ax.imshow(
        spec[0].numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="bilinear",
    )

    ax.set_xlabel("Time Frames", color=_TEXT, fontsize=9)
    ax.set_ylabel("Mel Bands", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors=_TEXT, labelsize=8)
    cbar.set_label("dB", color=_TEXT, fontsize=9)

    plt.tight_layout()
    return fig


def plot_layer_weights(weights: list) -> plt.Figure:
    """
    Render Wav2Vec2 layer contribution weights as a horizontal bar chart.

    Args:
        weights: List of 13 floats (softmax-normalized layer weights).

    Returns:
        matplotlib Figure ready for st.pyplot().
    """
    labels = ["CNN"] + [f"T-{i}" for i in range(1, 13)]
    values = np.array(weights)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(values)))
    ax.barh(labels, values, color=colors, height=0.6, edgecolor="none")

    ax.set_xlabel("Contribution Weight", color=_TEXT, fontsize=9)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.grid(True, color=_GRID, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig
