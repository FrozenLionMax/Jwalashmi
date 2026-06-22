"""
JWALASHMI - Actual Feature Importance + Brier Score
=====================================================
Computes REAL gradient x input attribution and exact Brier score
from model softmax outputs. Run on Colab T4 GPU.

Expected runtime: ~5 minutes
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os


# ============================================================================
# MODEL (exact match to src/model/architecture.py)
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


class TemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = self.norm(x + self.dropout(attn_out))
        return x, attn_weights


class FlareForecaster(nn.Module):
    def __init__(self, n_input_channels=9, n_classes=5,
                 cnn_channels=None, cnn_kernels=None,
                 n_heads=4, hidden_dim=64, dropout=0.3):
        super().__init__()
        cnn_channels = cnn_channels or [32, 64, 128]
        cnn_kernels = cnn_kernels or [7, 5, 3]
        layers = []
        in_ch = n_input_channels
        for out_ch, kernel in zip(cnn_channels, cnn_kernels):
            layers.append(ConvBlock(in_ch, out_ch, kernel, pool_size=2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.d_model = cnn_channels[-1]
        self.attention = TemporalAttention(self.d_model, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
        self.lead_time_head = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.ReLU())

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x_attn, attn_weights = self.attention(x)
        x_pool = x_attn.mean(dim=1)
        return self.classifier(x_pool), self.lead_time_head(x_pool), attn_weights


# ============================================================================
# GRADIENT x INPUT ATTRIBUTION
# ============================================================================
def compute_gradient_attribution(models, X, device='cuda', batch_size=32):
    """
    Compute mean |gradient x input| per feature across ensemble.
    This is the standard saliency-based feature importance method.
    """
    n_features = X.shape[2]
    feature_importance = np.zeros(n_features)
    n_samples = 0

    for model_idx, model in enumerate(models):
        model.eval()
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            batch.requires_grad_(True)

            logits, _, _ = model(batch)
            # Use predicted class probability as target
            probs = torch.softmax(logits, dim=1)
            pred_classes = probs.argmax(dim=1)
            # Sum probabilities of predicted classes
            target = probs[range(len(pred_classes)), pred_classes].sum()

            model.zero_grad()
            target.backward()

            # |gradient x input| averaged over time dimension
            grad_input = (batch.grad * batch).abs()  # (B, T, C)
            # Average over time, sum over batch
            attr = grad_input.mean(dim=1).detach().cpu().numpy()  # (B, C)
            feature_importance += attr.sum(axis=0)
            n_samples += len(batch)

            batch.requires_grad_(False)

        print(f"  Model {model_idx}: done ({n_samples} samples processed)")

    # Normalize to sum to 1
    feature_importance /= n_samples
    feature_importance /= feature_importance.sum()
    return feature_importance


# ============================================================================
# ACTUAL BRIER SCORE FROM SOFTMAX
# ============================================================================
def compute_brier_from_softmax(models, X, y, device='cuda', batch_size=32):
    """Compute exact Brier score from ensemble softmax probabilities."""
    X_tensor = torch.FloatTensor(X).to(device)
    all_probs = []

    for model in models:
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i+batch_size]
                logits, _, _ = model(batch)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs_list))

    avg_probs = np.mean(all_probs, axis=0)  # (N, 5)

    # Binary Brier (M+X)
    p_mx = avg_probs[:, 3] + avg_probs[:, 4]
    o_mx = (y >= 3).astype(float)
    brier_binary = float(np.mean((p_mx - o_mx) ** 2))

    # Multi-class Brier
    one_hot = np.zeros_like(avg_probs)
    one_hot[np.arange(len(y)), y] = 1
    brier_multi = float(np.mean(np.sum((avg_probs - one_hot) ** 2, axis=1)))

    return brier_binary, brier_multi, avg_probs


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  JWALASHMI - REAL FEATURE IMPORTANCE + BRIER SCORE")
    print("=" * 60)

    MODEL_DIR = 'models/v6_1_ensemble'
    DATA_X = 'data/processed/X_tactical.npy'
    DATA_Y = 'data/processed/y_tactical.npy'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    if not os.path.exists(MODEL_DIR):
        print(f"  ERROR: {MODEL_DIR} not found")
        exit(1)
    if not os.path.exists(DATA_X):
        print(f"  ERROR: {DATA_X} not found")
        exit(1)

    # Load data
    X = np.load(DATA_X)
    y = np.load(DATA_Y)
    n_features = X.shape[2]
    print(f"  Data: {X.shape} ({n_features} features)")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Detect model input channels from first checkpoint
    sample_path = os.path.join(MODEL_DIR, 'model_0.pt')
    sample_state = torch.load(sample_path, map_location=device)
    if isinstance(sample_state, dict) and 'model_state_dict' in sample_state:
        sample_state = sample_state['model_state_dict']
    # First conv weight shape: (out_channels, in_channels, kernel_size)
    first_conv_key = [k for k in sample_state.keys() if 'conv.weight' in k][0]
    model_n_features = sample_state[first_conv_key].shape[1]
    print(f"  Models trained with {model_n_features} features, data has {n_features}")

    # Trim data to match model input
    if n_features > model_n_features:
        print(f"  Trimming data from {n_features} to {model_n_features} features")
        X = X[:, :, :model_n_features]
        n_features = model_n_features

    # Load models
    print(f"\n--- Loading Models ---")
    models = []
    for i in range(10):
        path = os.path.join(MODEL_DIR, f'model_{i}.pt')
        if os.path.exists(path):
            model = FlareForecaster(n_input_channels=n_features).to(device)
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            model.eval()
            models.append(model)
    print(f"  Loaded {len(models)} models")

    # --- Feature Importance ---
    print(f"\n--- Computing Gradient x Input Attribution ---")
    # Use a subset for speed (500 samples)
    subset_idx = np.random.RandomState(42).choice(len(X), min(500, len(X)), replace=False)
    X_subset = X[subset_idx]

    importance = compute_gradient_attribution(models, X_subset, device=device)

    FEATURE_NAMES = [
        'derivative', 'rolling_max_ratio', 'norm_flux', 'acceleration',
        'energy_integral', 'hard_soft_ratio', 'neupert', 'bg_slope',
        'spectral_hardness', 'long_slope', 'qpp_power', 'long_ratio'
    ]
    # If fewer features than names, truncate
    names = FEATURE_NAMES[:n_features]

    print(f"\n{'=' * 60}")
    print(f"  REAL FEATURE IMPORTANCE (gradient x input)")
    print(f"{'=' * 60}")
    ranked = sorted(zip(names, importance), key=lambda x: -x[1])
    for rank, (name, imp) in enumerate(ranked, 1):
        bar = '#' * int(imp * 100)
        print(f"  {rank:>2}. {name:<20s} {imp:.4f}  {bar}")

    hel1os_feats = ['hard_soft_ratio', 'neupert', 'spectral_hardness']
    hel1os_total = sum(imp for name, imp in zip(names, importance) if name in hel1os_feats)
    print(f"\n  HEL1OS contribution: {hel1os_total:.1%} (from 3/{n_features} features)")

    # --- Brier Score ---
    print(f"\n--- Computing Exact Brier Score ---")
    brier_bin, brier_multi, probs = compute_brier_from_softmax(models, X, y, device=device)
    print(f"\n  Binary Brier (M+X): {brier_bin:.4f}")
    print(f"  Multi-class Brier:  {brier_multi:.4f}")

    # --- Save ---
    results = {
        'feature_importance': {name: round(float(imp), 4)
                               for name, imp in zip(names, importance)},
        'feature_ranking': [name for name, _ in ranked],
        'hel1os_contribution': round(float(hel1os_total), 4),
        'brier_binary_mx': round(brier_bin, 4),
        'brier_multiclass': round(brier_multi, 4),
        'n_samples_attribution': len(X_subset),
        'n_models': len(models),
    }

    with open('actual_feature_importance.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  [SAVED] actual_feature_importance.json")
    print(f"\n  Copy these values into Section 5.5 of RESEARCH_PAPER.md")
    print(f"  Done!")
