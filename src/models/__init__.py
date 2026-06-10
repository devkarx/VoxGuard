"""Model architectures for deepfake audio detection."""

from src.models.wav2vec_classifier import Wav2Vec2DeepfakeDetector
from src.models.crnn_baseline import DeepfakeCRNN

__all__ = ["Wav2Vec2DeepfakeDetector", "DeepfakeCRNN"]
