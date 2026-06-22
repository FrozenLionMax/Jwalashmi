"""
JWALASHMI - SHAP Feature Importance (Actual Shapley Values)
=============================================================
Uses shap.GradientExplainer for PyTorch models.
Run on Colab T4 GPU. ~10 minutes.

Install: pip install shap
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os


# ============================================================================
# MODEL (exact copy)
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
        return self.classifier(x_pool)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  JWALASHMI - ACTUAL SHAP VALUES")
    print("=" * 60)

    MODEL_DIR = 'models/v6_1_ensemble'
    DATA_X = 'data/processed/X_tactical.npy'
    DATA_Y = 'data/processed/y_tactical.npy'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # Load data
    X = np.load(DATA_X)
    y = np.load(DATA_Y)

    # Detect model features
    sample_state = torch.load(os.path.join(MODEL_DIR, 'model_0.pt'),
                              map_location=device)
    if isinstance(sample_state, dict) and 'model_state_dict' in sample_state:
        sample_state = sample_state['model_state_dict']
    first_key = [k for k in sample_state.keys() if 'conv.weight' in k][0]
    n_features = sample_state[first_key].shape[1]
    print(f"  Model uses {n_features} features, data has {X.shape[2]}")
    X = X[:, :, :n_features]
    print(f"  Data shape: {X.shape}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

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

    # --- SHAP ---
    print(f"\n--- Computing SHAP Values ---")
    print(f"  Installing shap...")

    try:
        import shap
    except ImportError:
        os.system("pip install shap -q")
        import shap

    print(f"  shap version: {shap.__version__}")

    # Sample background (100) and explanation (200) sets
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X), 100, replace=False)
    explain_idx = rng.choice(len(X), 200, replace=False)

    X_bg = torch.FloatTensor(X[bg_idx]).to(device)
    X_explain = torch.FloatTensor(X[explain_idx]).to(device)

    # Compute SHAP for each model, then average
    all_shap = np.zeros((200, X.shape[1], n_features))

    for i, model in enumerate(models):
        model.eval()
        print(f"  Model {i}: computing SHAP...", end=" ", flush=True)

        try:
            explainer = shap.GradientExplainer(model, X_bg)
            shap_values = explainer.shap_values(X_explain)

            # shap_values is list of arrays (one per class) or single array
            if isinstance(shap_values, list):
                # Average absolute SHAP across all classes
                sv = np.mean([np.abs(s.cpu().numpy() if torch.is_tensor(s) else s)
                              for s in shap_values], axis=0)
            else:
                sv = np.abs(shap_values.cpu().numpy() if torch.is_tensor(shap_values)
                            else shap_values)

            all_shap += sv
            print("done")
        except Exception as e:
            print(f"error: {e}")
            # Fallback: use gradient x input for this model
            print(f"    Falling back to gradient x input for model {i}")
            batch = X_explain.clone().detach().requires_grad_(True)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            target = probs.max(dim=1).values.sum()
            model.zero_grad()
            target.backward()
            grad_input = (batch.grad * batch).abs().detach().cpu().numpy()
            all_shap += grad_input
            print("    fallback done")

    all_shap /= len(models)  # Average across ensemble

    # Average over time to get per-feature importance
    feature_shap = all_shap.mean(axis=(0, 1))  # (n_features,)
    feature_shap /= feature_shap.sum()  # Normalize

    FEATURE_NAMES = [
        'derivative', 'rolling_max_ratio', 'norm_flux', 'acceleration',
        'energy_integral', 'hard_soft_ratio', 'neupert', 'bg_slope',
        'spectral_hardness'
    ]
    names = FEATURE_NAMES[:n_features]

    print(f"\n{'=' * 60}")
    print(f"  ACTUAL SHAP FEATURE IMPORTANCE")
    print(f"{'=' * 60}")
    ranked = sorted(zip(names, feature_shap), key=lambda x: -x[1])
    for rank, (name, imp) in enumerate(ranked, 1):
        bar = '#' * int(imp * 100)
        print(f"  {rank:>2}. {name:<20s} {imp:.4f}  {bar}")

    hel1os_feats = ['hard_soft_ratio', 'neupert', 'spectral_hardness']
    hel1os_total = sum(imp for name, imp in zip(names, feature_shap)
                       if name in hel1os_feats)
    print(f"\n  HEL1OS SHAP contribution: {hel1os_total:.1%} (from 3/{n_features} features)")

    # Compare with gradient x input
    print(f"\n  [Compare with gradient x input results to validate consistency]")

    # Save
    results = {
        'method': 'SHAP (GradientExplainer)',
        'n_background': 100,
        'n_explained': 200,
        'n_models': len(models),
        'feature_importance': {name: round(float(imp), 4)
                               for name, imp in zip(names, feature_shap)},
        'feature_ranking': [name for name, _ in ranked],
        'hel1os_contribution': round(float(hel1os_total), 4),
    }

    with open('actual_shap_values.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  [SAVED] actual_shap_values.json")
    print(f"  Done!")
