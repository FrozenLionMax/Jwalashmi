"""
Solar Flare Early Warning System — Training Pipeline
Supports pre-training on GOES data and fine-tuning on Aditya-L1 data.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.model.architecture import FlareForecaster, FlareForecasterLoss


def create_dataloaders(X: np.ndarray, y: np.ndarray,
                       lead_times: Optional[np.ndarray] = None,
                       val_split: float = 0.15,
                       batch_size: int = cfg.BATCH_SIZE,
                       shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders from numpy arrays.
    
    Args:
        X: (N, T, C) feature windows
        y: (N,) integer class labels
        lead_times: (N,) lead times in minutes (None = zeros)
        val_split: fraction for validation
    """
    if lead_times is None:
        lead_times = np.zeros(len(y), dtype=np.float32)

    # Chronological split (no data leakage)
    n_val = int(len(X) * val_split)
    n_train = len(X) - n_val

    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.long)
    lt_train = torch.tensor(lead_times[:n_train], dtype=torch.float32)

    X_val = torch.tensor(X[n_train:], dtype=torch.float32)
    y_val = torch.tensor(y[n_train:], dtype=torch.long)
    lt_val = torch.tensor(lead_times[n_train:], dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train, lt_train)
    val_ds = TensorDataset(X_val, y_val, lt_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    return train_loader, val_loader


def train_epoch(model: FlareForecaster, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: FlareForecasterLoss,
                device: torch.device) -> Dict[str, float]:
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    total_losses = {"total": 0, "classification": 0, "lead_time": 0}
    correct = 0
    total = 0

    for X_batch, y_batch, lt_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        lt_batch = lt_batch.to(device)

        optimizer.zero_grad()
        class_logits, lead_pred, _ = model(X_batch)
        losses = criterion(class_logits, lead_pred, y_batch, lt_batch)
        losses["total"].backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = X_batch.size(0)
        total_losses["total"] += losses["total"].item() * batch_size
        total_losses["classification"] += losses["classification"].item() * batch_size
        total_losses["lead_time"] += losses["lead_time"].item() * batch_size

        preds = class_logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += batch_size

    n = total
    return {
        "loss": total_losses["total"] / n,
        "cls_loss": total_losses["classification"] / n,
        "lt_loss": total_losses["lead_time"] / n,
        "accuracy": correct / n if n > 0 else 0,
    }


@torch.no_grad()
def evaluate(model: FlareForecaster, loader: DataLoader,
             criterion: FlareForecasterLoss,
             device: torch.device) -> Dict[str, float]:
    """Evaluate on validation set. Returns dict of metrics."""
    model.eval()
    total_losses = {"total": 0, "classification": 0, "lead_time": 0}
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch, lt_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        lt_batch = lt_batch.to(device)

        class_logits, lead_pred, _ = model(X_batch)
        losses = criterion(class_logits, lead_pred, y_batch, lt_batch)

        batch_size = X_batch.size(0)
        total_losses["total"] += losses["total"].item() * batch_size
        total_losses["classification"] += losses["classification"].item() * batch_size
        total_losses["lead_time"] += losses["lead_time"].item() * batch_size

        preds = class_logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += batch_size

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    n = total
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Per-class accuracy
    per_class_acc = {}
    for cls in range(cfg.N_CLASSES):
        mask = all_targets == cls
        if mask.sum() > 0:
            per_class_acc[cfg.CLASS_NAMES[cls]] = (all_preds[mask] == cls).mean()

    return {
        "loss": total_losses["total"] / n if n > 0 else 0,
        "cls_loss": total_losses["classification"] / n if n > 0 else 0,
        "lt_loss": total_losses["lead_time"] / n if n > 0 else 0,
        "accuracy": correct / n if n > 0 else 0,
        "per_class_accuracy": per_class_acc,
    }


def train_model(X: np.ndarray, y: np.ndarray,
                lead_times: Optional[np.ndarray] = None,
                n_features: Optional[int] = None,
                mode: str = "pretrain",
                pretrained_path: Optional[str] = None,
                epochs: Optional[int] = None,
                lr: Optional[float] = None,
                save_dir: Optional[str] = None) -> FlareForecaster:
    """
    Full training pipeline.
    
    Args:
        X: (N, T, C) feature windows
        y: (N,) labels
        lead_times: (N,) lead times in minutes
        n_features: number of input features (inferred from X if None)
        mode: 'pretrain' (GOES data) or 'finetune' (Aditya-L1 data)
        pretrained_path: path to pre-trained model weights (for finetune mode)
        epochs: override number of epochs
        lr: override learning rate
        save_dir: directory to save checkpoints
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining device: {device}")

    n_features = n_features or X.shape[2]
    save_dir = save_dir or str(cfg.MODEL_DIR)
    os.makedirs(save_dir, exist_ok=True)

    # Determine hyperparameters based on mode
    if mode == "pretrain":
        lr = lr or cfg.LEARNING_RATE_PRETRAIN
        epochs = epochs or cfg.EPOCHS_PRETRAIN
    else:
        lr = lr or cfg.LEARNING_RATE_FINETUNE
        epochs = epochs or cfg.EPOCHS_FINETUNE

    # Create model
    model = FlareForecaster(n_input_channels=n_features).to(device)

    # Load pre-trained weights if fine-tuning
    if mode == "finetune" and pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.freeze_cnn()

    # Create data loaders
    train_loader, val_loader = create_dataloaders(X, y, lead_times)

    # Loss and optimizer
    criterion = FlareForecasterLoss().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    params = model.count_parameters()
    print(f"\nModel: {params['trainable']:,} trainable / {params['total']:,} total params")
    print(f"Training: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Config: lr={lr}, epochs={epochs}, batch_size={cfg.BATCH_SIZE}")
    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>11} "
          f"{'Train Acc':>10} {'Val Acc':>10} {'LR':>10} {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"{epoch:5d} {train_metrics['loss']:11.4f} {val_metrics['loss']:11.4f} "
              f"{train_metrics['accuracy']:10.4f} {val_metrics['accuracy']:10.4f} "
              f"{current_lr:10.6f} {dt:5.1f}s")

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])

        # Early stopping & checkpointing
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            checkpoint_path = os.path.join(save_dir, f"best_{mode}_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= cfg.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={cfg.PATIENCE})")
            break

    # Load best model
    best_path = os.path.join(save_dir, f"best_{mode}_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        print(f"\nLoaded best model (val_loss={best_val_loss:.4f})")

    # Final evaluation
    print("\nFinal Validation Results:")
    final_metrics = evaluate(model, val_loader, criterion, device)
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    if final_metrics.get("per_class_accuracy"):
        for cls, acc in final_metrics["per_class_accuracy"].items():
            print(f"  {cls:>5s} accuracy: {acc:.4f}")

    return model
