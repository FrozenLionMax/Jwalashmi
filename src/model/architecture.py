"""
Solar Flare Early Warning System — Neural Network Architecture
1D Convolutional Neural Network with Multi-Head Self-Attention
for solar flare forecasting.

Architecture Overview:
    Input (B, T, C) → CNN Feature Extractor → Attention → Classification + Lead Time
    
    Where:
        B = batch size
        T = time steps (3600 = 60 min at 1s cadence)
        C = input channels (number of features)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


class ConvBlock(nn.Module):
    """1D Convolution → BatchNorm → ReLU → MaxPool"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, pool_size: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class TemporalAttention(nn.Module):
    """
    Multi-Head Self-Attention over temporal features.
    
    This is the key innovation: attention learns WHICH time steps
    in the input window are most predictive of upcoming flares.
    The attention weights are interpretable — they show the model
    is "looking at" precursor signatures.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) — batch of temporal feature sequences
        
        Returns:
            output: (B, T, D) — attention-refined features
            weights: (B, T, T) — attention weight matrix (for visualization)
        """
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = self.norm(x + self.dropout(attn_out))
        return x, attn_weights


class FlareForecaster(nn.Module):
    """
    Solar Flare Forecasting Model
    
    Architecture:
        1. CNN Feature Extractor: 3 Conv1D blocks extract local temporal patterns
           (e.g., rapid flux increases, spectral hardening)
        2. Temporal Attention: Multi-head attention over CNN features learns
           long-range dependencies (e.g., precursor heating 30 min before flare)
        3. Dual Output Heads:
           a. Classification: predicts flare class [none, B, C, M, X]
           b. Lead Time: predicts minutes until flare peak
    """

    def __init__(self,
                 n_input_channels: int = 6,
                 n_classes: int = cfg.N_CLASSES,
                 cnn_channels: list = None,
                 cnn_kernels: list = None,
                 n_heads: int = cfg.ATTENTION_HEADS,
                 hidden_dim: int = cfg.HIDDEN_DIM,
                 dropout: float = cfg.DROPOUT):
        super().__init__()

        cnn_channels = cnn_channels or cfg.CNN_CHANNELS
        cnn_kernels = cnn_kernels or cfg.CNN_KERNELS

        # ── CNN Feature Extractor ──────────────────────────
        layers = []
        in_ch = n_input_channels
        for out_ch, kernel in zip(cnn_channels, cnn_kernels):
            layers.append(ConvBlock(in_ch, out_ch, kernel, pool_size=2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Feature dimension after CNN
        self.d_model = cnn_channels[-1]

        # ── Temporal Attention ─────────────────────────────
        self.attention = TemporalAttention(
            d_model=self.d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # ── Classification Head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # ── Lead Time Regression Head ──────────────────────
        self.lead_time_head = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # lead time is always non-negative
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, T, C) — batch of input windows
                B = batch size
                T = time steps (e.g., 3600)
                C = input channels (features)
        
        Returns:
            class_logits: (B, n_classes) — raw logits for flare class
            lead_time: (B, 1) — predicted lead time in minutes
            attn_weights: (B, T', T') — attention weights for visualization
        """
        # CNN expects (B, C, T), so transpose
        x = x.transpose(1, 2)  # (B, C, T)

        # Extract temporal features
        x = self.cnn(x)  # (B, D, T')

        # Transpose back for attention: (B, T', D)
        x = x.transpose(1, 2)

        # Apply temporal attention
        x_attn, attn_weights = self.attention(x)  # (B, T', D)

        # Global average pooling over time
        x_pooled = x_attn.mean(dim=1)  # (B, D)

        # Dual output heads
        class_logits = self.classifier(x_pooled)    # (B, n_classes)
        lead_time = self.lead_time_head(x_pooled)   # (B, 1)

        return class_logits, lead_time, attn_weights

    def freeze_cnn(self):
        """Freeze CNN layers for fine-tuning (only train attention + heads)."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("CNN layers frozen. Only attention and output heads will be trained.")

    def unfreeze_cnn(self):
        """Unfreeze CNN layers."""
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("All layers unfrozen.")

    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract attention weights for visualization/explainability.
        
        Args:
            x: single sample (1, T, C)
        
        Returns:
            attention map as numpy array (T', T')
        """
        self.eval()
        with torch.no_grad():
            _, _, attn = self.forward(x)
        return attn[0].cpu().numpy()

    def count_parameters(self) -> dict:
        """Count trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {"trainable": trainable, "frozen": frozen, "total": total}


class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance (rare M/X flares).
    
    FL(p) = -alpha * (1-p)^gamma * log(p)
    
    When gamma > 0, the loss down-weights easy examples and focuses
    on hard-to-classify samples (like rare M/X-class flares).
    This is dramatically better than standard CrossEntropy for imbalanced data.
    """

    def __init__(self, class_weights: list = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        weights = torch.tensor(class_weights or cfg.CLASS_WEIGHTS, dtype=torch.float32)
        self.register_buffer("weight", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)  # probability of correct class
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class FlareForecasterLoss(nn.Module):
    """
    Combined loss for flare forecasting:
        L = alpha * FocalLoss(class) + beta * MSE(lead_time) + gamma * EarlyWarningBonus
    
    Uses Focal Loss instead of CrossEntropy for much better
    performance on rare M/X-class flares.
    The EarlyWarningBonus rewards the model for predicting flares
    with MORE lead time. This directly optimizes for what ISRO
    evaluates: "how early can you warn?"
    """

    def __init__(self,
                 class_weights: list = None,
                 alpha: float = 1.0,
                 beta: float = 0.1,
                 gamma: float = 0.15):
        super().__init__()
        self.focal_loss = FocalLoss(class_weights, gamma=2.0)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, class_logits: torch.Tensor, lead_time_pred: torch.Tensor,
                class_target: torch.Tensor, lead_time_target: torch.Tensor) -> dict:
        """
        Args:
            class_logits: (B, n_classes)
            lead_time_pred: (B, 1)
            class_target: (B,) -- integer class labels
            lead_time_target: (B,) -- true lead time in minutes
        
        Returns:
            dict with 'total', 'classification', 'lead_time', 'early_warning' losses
        """
        # Classification loss (Focal Loss)
        loss_cls = self.focal_loss(class_logits, class_target)

        # Lead time regression loss (only for samples with a flare)
        flare_mask = class_target > 0
        if flare_mask.any():
            lt_pred = lead_time_pred[flare_mask].squeeze(-1)
            lt_true = lead_time_target[flare_mask]
            loss_lt = F.mse_loss(lt_pred, lt_true)

            # Early warning bonus: reward larger lead times for correct predictions
            pred_class = class_logits[flare_mask].argmax(dim=1)
            correct = pred_class == class_target[flare_mask]
            if correct.any():
                # Bonus proportional to predicted lead time (capped at 60 min)
                bonus = torch.clamp(lt_pred[correct], 0, 60).mean()
                loss_ew = -bonus  # negative because we want to MAXIMIZE lead time
            else:
                loss_ew = torch.tensor(0.0, device=class_logits.device)
        else:
            loss_lt = torch.tensor(0.0, device=class_logits.device)
            loss_ew = torch.tensor(0.0, device=class_logits.device)

        total = self.alpha * loss_cls + self.beta * loss_lt + self.gamma * loss_ew

        return {
            "total": total,
            "classification": loss_cls,
            "lead_time": loss_lt,
            "early_warning": loss_ew,
        }


# ═══════════════════════════════════════════════════════════════
#  Tier 1: Strategic Forecaster (5-10 hour warnings)
# ═══════════════════════════════════════════════════════════════

class StrategicForecaster(nn.Module):
    """
    Tier 1: Strategic Solar Flare Forecaster (5-10 hours ahead)
    
    Simpler architecture than the tactical model. Uses downsampled
    6-hour windows at 1-min cadence (360 time steps) to predict
    the probability of each flare class in the next 10 hours.
    
    Architecture:
        1. 2-layer CNN for feature extraction
        2. Self-attention for long-range temporal patterns
        3. Probabilistic output (soft predictions with confidence)
    """

    def __init__(self,
                 n_input_channels: int = 9,
                 n_classes: int = cfg.N_CLASSES,
                 cnn_channels: list = None,
                 cnn_kernels: list = None,
                 hidden_dim: int = None,
                 dropout: float = cfg.DROPOUT):
        super().__init__()

        cnn_channels = cnn_channels or cfg.STRATEGIC_CNN_CHANNELS
        cnn_kernels = cnn_kernels or cfg.STRATEGIC_CNN_KERNELS
        hidden_dim = hidden_dim or cfg.STRATEGIC_HIDDEN_DIM

        # CNN Feature Extractor (lighter than tactical)
        layers = []
        in_ch = n_input_channels
        for out_ch, kernel in zip(cnn_channels, cnn_kernels):
            layers.append(ConvBlock(in_ch, out_ch, kernel, pool_size=2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.d_model = cnn_channels[-1]

        # Self-attention
        self.attention = TemporalAttention(
            d_model=self.d_model,
            n_heads=4,
            dropout=dropout,
        )

        # Classification head (probabilistic output)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) -- T=360 for 6-hour window at 1-min cadence
        
        Returns:
            class_logits: (B, n_classes)
            attn_weights: (B, T', T')
        """
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x_attn, attn_weights = self.attention(x)
        x_pooled = x_attn.mean(dim=1)
        class_logits = self.classifier(x_pooled)
        return class_logits, attn_weights

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


class StrategicLoss(nn.Module):
    """
    Focal Loss for strategic forecaster.
    No lead time component -- just classification probability.
    """

    def __init__(self, class_weights: list = None):
        super().__init__()
        self.focal = FocalLoss(class_weights, gamma=2.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        loss = self.focal(logits, targets)
        return {"total": loss, "classification": loss}


# ═══════════════════════════════════════════════════════════════
#  Model Summary
# ═══════════════════════════════════════════════════════════════
def print_model_summary(n_features: int = 9, window_size: int = cfg.WINDOW_SIZE):
    """Print a summary of the model architecture."""
    model = FlareForecaster(n_input_channels=n_features)
    params = model.count_parameters()

    print("=" * 60)
    print("  Solar Flare Forecaster - Architecture Summary")
    print("=" * 60)
    print(f"  Input shape:  (batch, {window_size}, {n_features})")
    print(f"  CNN channels: {cfg.CNN_CHANNELS}")
    print(f"  CNN kernels:  {cfg.CNN_KERNELS}")
    print(f"  Attention:    {cfg.ATTENTION_HEADS} heads, d_model={model.d_model}")
    print(f"  Hidden dim:   {cfg.HIDDEN_DIM}")
    print(f"  Output:       {cfg.N_CLASSES} classes + lead time")
    print(f"  Parameters:   {params['total']:,} total")
    print(f"                {params['trainable']:,} trainable")
    print("=" * 60)

    # Test forward pass
    x = torch.randn(2, window_size, n_features)
    logits, lt, attn = model(x)
    print(f"\n  Test forward pass:")
    print(f"    Input:          {x.shape}")
    print(f"    Class logits:   {logits.shape}")
    print(f"    Lead time:      {lt.shape}")
    print(f"    Attention map:  {attn.shape}")
    print(f"    Predicted class: {cfg.CLASS_NAMES[logits[0].argmax().item()]}")
    print(f"    Predicted lead:  {lt[0].item():.1f} min")

    return model


if __name__ == "__main__":
    print_model_summary()
