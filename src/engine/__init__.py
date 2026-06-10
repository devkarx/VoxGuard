"""Training and evaluation engine."""

from src.engine.metrics import compute_eer, compute_accuracy, compute_min_dcf
from src.engine.trainer import Trainer

__all__ = ["Trainer", "compute_eer", "compute_accuracy", "compute_min_dcf"]
