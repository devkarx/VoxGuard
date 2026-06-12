"""
Inference API for deepfake audio detection.

Automatically selects the best available model:
    1. Wav2Vec2DeepfakeDetector  →  weights/best_wav2vec_model.pth
    2. DeepfakeCRNN (fallback)   →  weights/best_model.pth

Loaded models are cached in memory so repeated calls don't
re-deserialize the checkpoint every time.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
from torchaudio import transforms as T

from src.config import AudioConfig

logger = logging.getLogger(__name__)

_cfg = AudioConfig()

# ---- Paths ---------------------------------------------------------------
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
_WAV2VEC_WEIGHTS = _WEIGHTS_DIR / "best_wav2vec_model.pth"
_CRNN_WEIGHTS = _WEIGHTS_DIR / "best_model.pth"

# ---- Model cache ----------------------------------------------------------
# Keeps loaded models in memory between calls so we don't hit disk
# and re-initialize the full Wav2Vec2 backbone on every prediction.
_model_cache: dict = {}

# Shared resampler — avoids re-creating the transform per call
_resampler_cache: dict = {}


def predict_audio(file_path: str, model_path: Optional[str] = None) -> dict:
    """
    Run deepfake detection on a single audio file.

    Automatically selects the best available model if *model_path* is None.

    Args:
        file_path:  Path to a .wav or .flac audio file.
        model_path: Explicit path to model weights (auto-detects if None).

    Returns:
        dict with keys:
            label      – "Deepfake (AI-Generated)" or "Genuine (Human)"
            confidence – percentage confidence in the prediction
            raw_prob   – raw sigmoid probability (0 → genuine, 1 → deepfake)
            model_type – "wav2vec2" or "crnn"
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_path, is_wav2vec = _resolve_model_path(model_path)
    logger.info("Using model: %s (device=%s)", resolved_path, device)

    if is_wav2vec:
        model = _load_wav2vec(resolved_path, device)
        features = _preprocess_raw(file_path).unsqueeze(0).to(device)
        model_type = "wav2vec2"
    else:
        model = _load_crnn(resolved_path, device)
        features = _preprocess_melspec(file_path)
        if features is None:
            raise ValueError(
                "Failed to extract Mel-spectrogram from the audio file.")
        features = features.unsqueeze(0).to(device)
        model_type = "crnn"

    with torch.no_grad():
        logits = model(features)
        prob = torch.sigmoid(logits).item()

    is_deepfake = prob > 0.5
    confidence = (prob if is_deepfake else 1.0 - prob) * 100.0

    logger.info("Prediction: %s (confidence=%.2f%%)",
                "DEEPFAKE" if is_deepfake else "GENUINE", confidence)

    return {
        "label": "Deepfake (AI-Generated)" if is_deepfake else "Genuine (Human)",
        "confidence": confidence,
        "raw_prob": prob,
        "model_type": model_type,
    }


def get_layer_weights() -> Optional[list]:
    """
    Return Wav2Vec2 layer contribution weights for visualization.

    Returns None if no Wav2Vec2 checkpoint is available.
    """
    if not _WAV2VEC_WEIGHTS.exists():
        return None

    model = _load_wav2vec(str(_WAV2VEC_WEIGHTS), torch.device("cpu"))
    return model.get_layer_weights().numpy().tolist()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_model_path(model_path: Optional[str]) -> tuple:
    """Determine which checkpoint to load and whether it's a Wav2Vec2 model."""
    if model_path is not None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        return str(p), "wav2vec" in p.stem.lower()

    # Auto-detect: prefer Wav2Vec2 over CRNN
    if _WAV2VEC_WEIGHTS.exists():
        return str(_WAV2VEC_WEIGHTS), True
    if _CRNN_WEIGHTS.exists():
        return str(_CRNN_WEIGHTS), False

    raise FileNotFoundError(
        f"No model weights found. Place checkpoints in {_WEIGHTS_DIR}/."
    )


def _load_wav2vec(path: str, device: torch.device):
    """Load Wav2Vec2DeepfakeDetector, returning from cache if available."""
    cache_key = f"wav2vec:{path}:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from src.models import Wav2Vec2DeepfakeDetector

    logger.info("Loading Wav2Vec2 checkpoint from %s", path)
    model = Wav2Vec2DeepfakeDetector().to(device)

    # weights_only=False is required here because the checkpoint was saved
    # with torch.save() and contains the full state dict + optimizer state.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    _model_cache[cache_key] = model
    return model


def _load_crnn(path: str, device: torch.device):
    """Load DeepfakeCRNN, returning from cache if available."""
    cache_key = f"crnn:{path}:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from src.models import DeepfakeCRNN

    logger.info("Loading CRNN checkpoint from %s", path)
    model = DeepfakeCRNN().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    _model_cache[cache_key] = model
    return model


def _preprocess_raw(file_path: str) -> torch.Tensor:
    """Load audio as a peak-normalized 16 kHz waveform tensor."""
    audio, sr = sf.read(str(file_path))
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio).float()

    # Resample if needed, reusing the cached transform
    if sr != _cfg.sample_rate:
        if sr not in _resampler_cache:
            _resampler_cache[sr] = T.Resample(sr, _cfg.sample_rate)
        waveform = _resampler_cache[sr](waveform)

    # Pad or center-crop to fixed length
    n = waveform.shape[0]
    target = _cfg.target_length
    if n < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - n))
    elif n > target:
        start = (n - target) // 2
        waveform = waveform[start: start + target]

    # Peak-normalize
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    return waveform


def _preprocess_melspec(file_path: str):
    """Load audio and return a Mel-spectrogram tensor for the CRNN model."""
    from src.data import AudioPreprocessor
    return AudioPreprocessor().process(file_path)
