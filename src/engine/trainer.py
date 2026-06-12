"""
Training loop for the Wav2Vec2 deepfake detector.

Handles mixed-precision training, gradient accumulation, layer-wise
learning rate decay, and cosine-annealing scheduling. Designed to run
on a single consumer GPU (e.g. T4 with 16 GB VRAM).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from src.config import TrainingConfig
from src.engine.metrics import compute_eer, compute_accuracy

logger = logging.getLogger(__name__)


class Trainer:
    """
    End-to-end training manager for binary deepfake detection.

    Usage:
        model = Wav2Vec2DeepfakeDetector()
        train_loader = DataLoader(...)
        val_loader   = DataLoader(...)

        trainer = Trainer(model, train_loader, val_loader)
        trainer.fit()

    Args:
        model:        The detection model (must output a single logit).
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        config:       Training hyperparameters.
        device:       Device string or torch.device (auto-detects GPU).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.use_amp)
        self.writer = SummaryWriter(
            log_dir=self.config.log_dir) if SummaryWriter else None

        self.best_eer = float("inf")

    def fit(self) -> None:
        """Run the full training loop."""
        logger.info(
            "Starting training — %d epochs, device=%s, AMP=%s",
            self.config.epochs, self.device, self.config.use_amp,
        )
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss, eer, accuracy = self._validate(epoch)

            self.scheduler.step()

            # Log to TensorBoard (if available)
            if self.writer:
                self.writer.add_scalars(
                    "loss", {"train": train_loss, "val": val_loss}, epoch)
                self.writer.add_scalar("metrics/eer", eer, epoch)
                self.writer.add_scalar("metrics/accuracy", accuracy, epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch)

            logger.info(
                "Epoch %02d  train_loss=%.4f  val_loss=%.4f  EER=%.4f  acc=%.4f",
                epoch, train_loss, val_loss, eer, accuracy,
            )

            # Save best checkpoint
            if eer < self.best_eer:
                self.best_eer = eer
                self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)

        if self.writer:
            self.writer.close()
        logger.info("Training complete. Best EER: %.4f", self.best_eer)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for step, (waveforms, labels) in enumerate(self.train_loader, start=1):
            waveforms = waveforms.to(self.device)
            labels = labels.float().to(self.device)

            with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                logits = self.model(waveforms)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if step % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        return running_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple:
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_scores = []

        for waveforms, labels in self.val_loader:
            waveforms = waveforms.to(self.device)
            labels_dev = labels.float().to(self.device)

            logits = self.model(waveforms)
            loss = self.criterion(logits, labels_dev)
            running_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

        avg_loss = running_loss / max(len(self.val_loader), 1)
        all_labels = np.concatenate(all_labels)
        all_scores = np.concatenate(all_scores)

        eer, _ = compute_eer(all_labels, all_scores)
        predictions = (all_scores > 0.5).astype(int)
        accuracy = compute_accuracy(all_labels, predictions)

        return avg_loss, eer, accuracy

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build AdamW with layer-wise learning rate decay.

        The pre-trained backbone gets a smaller LR than the randomly
        initialized classifier head — standard practice for fine-tuning.
        """
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "wav2vec2" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": self.config.backbone_lr},
            {"params": head_params, "lr": self.config.head_lr},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Cosine annealing with linear warmup."""
        total_steps = self.config.epochs
        warmup_steps = max(1, int(total_steps * self.config.warmup_fraction))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            progress = (current_step - warmup_steps) / \
                max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        filename = "best_wav2vec_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        path = ckpt_dir / filename

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eer": self.best_eer,
        }
        torch.save(state, path)
        logger.info("Saved checkpoint → %s", path)
