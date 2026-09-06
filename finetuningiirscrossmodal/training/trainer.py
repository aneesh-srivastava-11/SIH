"""
Base Trainer Infrastructure for Matcher Fine-Tuning.

Provides training loop management, optimizer configuration, learning rate scheduling,
checkpoint saving, and validation loss tracking with GPU/CPU support.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseMatcherTrainer:
    """
    Base Trainer class for fine-tuning feature matching models.

    Args:
        model: PyTorch module/matcher to fine-tune.
        train_loader: PyTorch DataLoader for training split.
        val_loader: PyTorch DataLoader for validation split.
        checkpoint_dir: Directory path to save trained checkpoints.
        model_name: Identifier for model ('eloftr' or 'roma').
        lr: Learning rate (default: 1e-4).
        weight_decay: Weight decay coefficient (default: 1e-4).
        device: Target computation device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "eloftr",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

        # Filter parameters requiring gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

        self.best_val_loss = float("inf")

    def train_epoch(self, loss_fn: Callable[[Any, Any, str], Tuple[torch.Tensor, Dict[str, float]]]) -> float:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Move tensors to target device
            for k in ["image0", "image1", "homography"]:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            loss, metrics = loss_fn(self.model, batch, self.device)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / max(num_batches, 1)

    def validate(self, loss_fn: Callable[[Any, Any, str], Tuple[torch.Tensor, Dict[str, float]]]) -> float:
        """Runs validation evaluation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                for k in ["image0", "image1", "homography"]:
                    if k in batch and isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                loss, _ = loss_fn(self.model, batch, self.device)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(
        self,
        epochs: int = 50,
        loss_fn: Optional[Callable] = None,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes full training pipeline across specified epochs with early stopping.
        """
        if len(self.train_loader.dataset) == 0:
            print(f"[{self.model_name.upper()} Trainer] WARNING: Training dataset is empty.")
            print(f"Please populate rendered pairs in 'data/rendered_pairs/' with reference/target PNGs and pair JSONs.")
            return {"status": "skipped", "reason": "empty_dataset"}

        history = {"train_loss": [], "val_loss": []}
        patience_counter = 0

        print(f"--- Starting {self.model_name.upper()} Fine-Tuning ({epochs} epochs, device: {self.device}) ---")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(loss_fn)
            val_loss = self.validate(loss_fn)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save checkpoint per epoch
            ckpt_path = self.checkpoint_dir / f"finetuned_{self.model_name}_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

            # Update best checkpoint if improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / f"finetuned_{self.model_name}_best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_loss": val_loss,
                }, best_path)
                print(f"  --> Saved new best checkpoint to {best_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        return {"status": "completed", "history": history, "best_val_loss": self.best_val_loss}
