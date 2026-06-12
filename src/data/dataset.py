"""
Raw Waveform Dataset for Wav2Vec2 training and evaluation.

Loads audio as raw 16 kHz float32 waveforms with no spectrogram
conversion. Supports both directory-based label discovery and
ASVspoof-style protocol files. Optionally applies RawBoost
augmentation during training.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import soundfile as sf
from torch.utils.data import Dataset
from torchaudio import transforms as T

from src.config import AudioConfig, TrainingConfig
from src.data.augmentation import RawBoostAugmentor

logger = logging.getLogger(__name__)

_audio_cfg = AudioConfig()
_train_cfg = TrainingConfig()


class RawWaveformDataset(Dataset):
    """
    PyTorch Dataset that serves fixed-length raw waveforms.

    Discovers audio files by scanning subdirectory names (e.g. ``real/``,
    ``fake/``) or by reading an ASVspoof-style protocol file.

    Args:
        data_dir:       Root directory containing audio files.
        metadata_file:  Optional ASVspoof protocol file with per-file labels.
        target_sr:      Sample rate to resample to.
        target_duration: Duration in seconds to pad/crop to.
        class_mapping:  Custom string → int label map.
        is_training:    If True, apply RawBoost augmentation and random crop.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        target_sr: int = _audio_cfg.sample_rate,
        target_duration: float = _audio_cfg.duration,
        class_mapping: Optional[Dict[str, int]] = None,
        is_training: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.target_sr = target_sr
        self.target_length = int(target_sr * target_duration)
        self.is_training = is_training
        self.class_mapping = class_mapping or _train_cfg.class_mapping

        self.augmentor = RawBoostAugmentor(
            sample_rate=target_sr) if is_training else None
        self._resampler_cache: dict = {}

        self.file_paths: List[Path] = []
        self.labels: List[int] = []
        self._discover_files()

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def _discover_files(self) -> None:
        """Populate file_paths and labels from directory structure or metadata."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}")

        if self.metadata_file and self.metadata_file.exists():
            self._load_from_protocol()
        else:
            self._load_from_directories()

        logger.info(
            "Loaded %d files (genuine=%d, spoof=%d)",
            len(self.file_paths), self.labels.count(0), self.labels.count(1),
        )

    def _load_from_protocol(self) -> None:
        """Parse ASVspoof-style protocol: SPEAKER_ID UTTI_ID - SYSTEM LABEL."""
        with open(self.metadata_file, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                filename = parts[1]
                label_str = parts[-1].lower()
                if label_str not in self.class_mapping:
                    continue

                # Try .flac first (ASVspoof default), fall back to .wav
                path = self.data_dir / f"{filename}.flac"
                if not path.exists():
                    path = self.data_dir / f"{filename}.wav"
                if path.exists():
                    self.file_paths.append(path)
                    self.labels.append(self.class_mapping[label_str])

    def _load_from_directories(self) -> None:
        """Infer labels from subdirectory names (e.g. real/, fake/)."""
        for ext in ("*.wav", "*.flac"):
            for path in self.data_dir.rglob(ext):
                parent = path.parent.name.lower()
                if parent in self.class_mapping:
                    self.file_paths.append(path)
                    self.labels.append(self.class_mapping[parent])

    # ------------------------------------------------------------------
    # Audio loading & normalization
    # ------------------------------------------------------------------
    def _load_audio(self, file_path: Path) -> torch.Tensor:
        """Load, resample, and normalize a single audio file to a fixed length."""
        audio, sr = sf.read(str(file_path))

        # Convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio).float()

        # Resample to target rate if needed
        if sr != self.target_sr:
            if sr not in self._resampler_cache:
                self._resampler_cache[sr] = T.Resample(sr, self.target_sr)
            waveform = self._resampler_cache[sr](waveform)

        # Pad short clips or crop long ones
        n = waveform.shape[0]
        if n < self.target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.target_length - n))
        elif n > self.target_length:
            if self.is_training:
                start = random.randint(0, n - self.target_length)
            else:
                start = (n - self.target_length) // 2
            waveform = waveform[start: start + self.target_length]

        # Peak-normalize to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (waveform, label) where waveform has shape (target_length,)."""
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform = self._load_audio(path)
            if self.is_training and self.augmentor is not None:
                waveform = self.augmentor(waveform)
            return waveform, label

        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return torch.zeros(self.target_length), label

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_files={len(self)}, "
            f"sr={self.target_sr}, "
            f"duration={self.target_length / self.target_sr:.1f}s, "
            f"training={self.is_training})"
        )
