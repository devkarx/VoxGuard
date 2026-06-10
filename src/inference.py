"""
Unified Inference Engine for Deepfake Audio Detection.

Automatically selects the best available model architecture:
    1. Wav2Vec2DeepfakeDetector  →  if weights/best_wav2vec_model.pth exists
    2. DeepfakeCRNN (fallback)   →  if weights/best_model.pth exists
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
from torchaudio import transforms as T

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
_WAV2VEC_WEIGHTS = _WEIGHTS_DIR / "best_wav2vec_model.pth"
_CRNN_WEIGHTS = _WEIGHTS_DIR / "best_model.pth"
_SAMPLE_RATE = 16000
_DURATION = 4.0
_TARGET_LENGTH = int(_SAMPLE_RATE * _DURATION)


def predict_audio(file_path: str, model_path: Optional[str] = None) -> dict:
    """
    Run deepfake detection on a single audio file.

    Automatically selects the best available model if ``model_path`` is None.

    Args:
        file_path:  Path to a .wav or .flac audio file.
        model_path: Explicit path to model weights (auto-detects if None).

    Returns:
        dict with keys:
            - label (str):       "Deepfake (AI-Generated)" or "Genuine (Human)"
            - confidence (float): Percentage confidence in the prediction
            - raw_prob (float):   Raw sigmoid probability (0 = genuine, 1 = fake)
            - model_type (str):  "wav2vec2" or "crnn"
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve model path
    resolved_path, is_wav2vec = _resolve_model_path(model_path)

    # Load model and run inference
    if is_wav2vec:
        model = _load_wav2vec(resolved_path, device)
        features = _preprocess_raw(file_path).unsqueeze(0).to(device)
        model_type = "wav2vec2"
    else:
        model = _load_crnn(resolved_path, device)
        features = _preprocess_melspec(file_path)
        if features is None:
            raise ValueError("Failed to preprocess audio file.")
        features = features.unsqueeze(0).to(device)
        model_type = "crnn"

    with torch.no_grad():
        logits = model(features)
        prob = torch.sigmoid(logits).item()

    is_deepfake = prob > 0.5
    return {
        "label": "Deepfake (AI-Generated)" if is_deepfake else "Genuine (Human)",
        "confidence": (prob if is_deepfake else 1.0 - prob) * 100.0,
        "raw_prob": prob,
        "model_type": model_type,
    }


def get_layer_weights() -> Optional[list]:
    """
    Return Wav2Vec2 layer contribution weights for visualization.

    Returns None if no Wav2Vec2 model is loaded.
    """
    if not _WAV2VEC_WEIGHTS.exists():
        return None

    device = torch.device("cpu")
    model = _load_wav2vec(str(_WAV2VEC_WEIGHTS), device)
    weights = model.get_layer_weights().numpy().tolist()
    return weights


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_model_path(model_path: Optional[str]) -> tuple:
    """Return (resolved_path, is_wav2vec) tuple."""
    if model_path is not None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        return str(p), "wav2vec" in p.stem.lower()

    if _WAV2VEC_WEIGHTS.exists():
        return str(_WAV2VEC_WEIGHTS), True
    if _CRNN_WEIGHTS.exists():
        return str(_CRNN_WEIGHTS), False

    raise FileNotFoundError(
        f"No model weights found. Place weights in {_WEIGHTS_DIR}/."
    )


def _load_wav2vec(path: str, device: torch.device):
    """Load Wav2Vec2DeepfakeDetector from a checkpoint."""
    from src.models import Wav2Vec2DeepfakeDetector

    model = Wav2Vec2DeepfakeDetector().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _load_crnn(path: str, device: torch.device):
    """Load DeepfakeCRNN from a checkpoint."""
    from src.models import DeepfakeCRNN

    model = DeepfakeCRNN().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def _preprocess_raw(file_path: str) -> torch.Tensor:
    """Load audio as a raw 16 kHz waveform tensor."""
    audio, sr = sf.read(str(file_path))
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio).float()

    if sr != _SAMPLE_RATE:
        waveform = T.Resample(sr, _SAMPLE_RATE)(waveform)

    n = waveform.shape[0]
    if n < _TARGET_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, _TARGET_LENGTH - n))
    elif n > _TARGET_LENGTH:
        start = (n - _TARGET_LENGTH) // 2
        waveform = waveform[start : start + _TARGET_LENGTH]

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    return waveform


def _preprocess_melspec(file_path: str):
    """Load audio and return a Mel-spectrogram tensor."""
    from src.data import AudioPreprocessor
    return AudioPreprocessor().process(file_path)
