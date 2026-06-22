"""
JWALASHMI - GOES Hold-Out Actual Inference
============================================
Run this on Google Colab (T4 GPU) to get REAL inference results
on 71 M/X events NEVER seen during training.

Upload to Colab -> Runtime -> Change runtime -> T4 GPU -> Run All

Expected runtime: ~10 minutes
Output: actual_holdout_results.json + figures
"""

# !pip install torch numpy scikit-learn matplotlib -q

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# ============================================================================
# 1. MODEL DEFINITION (exact match to src/model/architecture.py)
# ============================================================================
class ConvBlock(nn.Module):
    """1D Convolution -> BatchNorm -> ReLU -> MaxPool"""
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
            nn.Linear(self.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.lead_time_head = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x_attn, attn_weights = self.attention(x)
        x_pool = x_attn.mean(dim=1)
        class_logits = self.classifier(x_pool)
        lead_time = self.lead_time_head(x_pool)
        return class_logits, lead_time, attn_weights


# ============================================================================
# 2. LOAD ENSEMBLE
# ============================================================================
def load_ensemble(model_dir, n_models=10, n_features=9, device='cuda'):
    """Load all ensemble models."""
    models = []
    for i in range(n_models):
        model = FlareForecaster(n_input_channels=n_features).to(device)
        path = os.path.join(model_dir, f'model_{i}.pt')
        if os.path.exists(path):
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            else:
                model = state
            model.eval()
            models.append(model)
            print(f"  Loaded model_{i}.pt")
        else:
            print(f"  WARNING: {path} not found")
    return models


# ============================================================================
# 3. SPLIT DATA — include negatives for meaningful TSS/FPR
# ============================================================================
def split_goes_holdout(X_path, y_path, holdout_frac=0.20, seed=42):
    """Split GOES data: hold out 20% of M/X + matching negatives."""
    X = np.load(X_path)
    y = np.load(y_path)

    np.random.seed(seed)

    print(f"\n  Total GOES data: {len(y)} samples")
    print(f"  Classes: None={int((y==0).sum())}, B={int((y==1).sum())}, C={int((y==2).sum())}, M={int((y==3).sum())}, X={int((y==4).sum())}")

    # Split M/X events
    m_idx = np.where(y == 3)[0]; np.random.shuffle(m_idx)
    x_idx = np.where(y == 4)[0]; np.random.shuffle(x_idx)
    neg_idx = np.where(y < 3)[0]; np.random.shuffle(neg_idx)

    m_split = int(len(m_idx) * (1 - holdout_frac))
    x_split = int(len(x_idx) * (1 - holdout_frac))
    # Include same number of negatives as positives in test set
    n_pos_test = len(m_idx) - m_split + len(x_idx) - x_split
    n_neg_test = min(n_pos_test * 3, len(neg_idx) // 5)  # 3:1 neg:pos ratio

    test_idx = np.concatenate([m_idx[m_split:], x_idx[x_split:], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([m_idx[:m_split], x_idx[:x_split], neg_idx[n_neg_test:]])

    X_test, y_test = X[test_idx], y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]

    print(f"\n  Test set: {len(y_test)} samples")
    print(f"    M-class: {int((y_test==3).sum())}, X-class: {int((y_test==4).sum())}")
    print(f"    Negatives (None+B+C): {int((y_test<3).sum())}")
    print(f"  Train set: {len(y_train)} samples")

    return X_test, y_test, X_train, y_train


# ============================================================================
# 4. RUN INFERENCE
# ============================================================================
@torch.no_grad()
def ensemble_predict(models, X, device='cuda', batch_size=32):
    """Run ensemble inference, return averaged probabilities."""
    X_tensor = torch.FloatTensor(X).to(device)
    all_probs = []

    for model in models:
        model.eval()
        probs_list = []
        for i in range(0, len(X), batch_size):
            batch = X_tensor[i:i+batch_size]
            logits, _, _ = model(batch)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs_list))

    avg_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)
    return predictions, avg_probs


# ============================================================================
# 5. COMPUTE METRICS
# ============================================================================
def compute_all_metrics(y_true, y_pred, probs):
    """Compute TSS, HSS, AUC, Brier, confusion matrix."""
    results = {}

    # Binary M+X
    true_mx = (y_true >= 3).astype(int)
    pred_mx = (y_pred >= 3).astype(int)

    tp = int(((true_mx == 1) & (pred_mx == 1)).sum())
    fn = int(((true_mx == 1) & (pred_mx == 0)).sum())
    fp = int(((true_mx == 0) & (pred_mx == 1)).sum())
    tn = int(((true_mx == 0) & (pred_mx == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = tpr - fpr

    n = tp + fp + tn + fn
    expected = ((tp+fn)*(tp+fp) + (tn+fp)*(tn+fn)) / n if n > 0 else 0
    hss = (tp + tn - expected) / (n - expected) if (n - expected) != 0 else 0

    results['TP'] = tp
    results['FN'] = fn
    results['FP'] = fp
    results['TN'] = tn
    results['TSS'] = round(float(tss), 4)
    results['HSS'] = round(float(hss), 4)
    results['TPR'] = round(float(tpr), 4)
    results['FPR'] = round(float(fpr), 4)
    results['RED_accuracy'] = round(float(tpr * 100), 1)
    results['precision'] = round(float(tp / (tp + fp) * 100), 1) if (tp + fp) > 0 else 0.0

    # Per-class AUC
    try:
        for c, name in enumerate(['None', 'B', 'C', 'M', 'X']):
            if (y_true == c).sum() > 0 and (y_true == c).sum() < len(y_true):
                auc = roc_auc_score((y_true == c).astype(int), probs[:, c])
                results[f'AUC_{name}'] = round(float(auc), 4)
    except Exception as e:
        print(f"  AUC warning: {e}")

    # Brier Score (binary M+X)
    p_mx = probs[:, 3] + probs[:, 4]
    brier = float(np.mean((p_mx - true_mx) ** 2))
    results['brier_score'] = round(brier, 4)

    # Balanced accuracy
    per_class_acc = []
    for c in range(5):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc.append(float((y_pred[mask] == c).mean()))
    results['balanced_accuracy'] = round(float(np.mean(per_class_acc) * 100), 1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    results['confusion_matrix'] = cm.tolist()

    return results


# ============================================================================
# 6. MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  JWALASHMI - ACTUAL GOES HOLD-OUT INFERENCE")
    print("=" * 60)

    MODEL_DIR = 'models/v6_1_ensemble'
    GOES_X_PATH = 'data/goes/X_goes_pretrain.npy'
    GOES_Y_PATH = 'data/goes/y_goes_pretrain.npy'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    if not os.path.exists(MODEL_DIR):
        print(f"\n  ERROR: Model directory not found: {MODEL_DIR}")
        exit(1)
    if not os.path.exists(GOES_X_PATH):
        print(f"\n  ERROR: GOES data not found: {GOES_X_PATH}")
        exit(1)

    # Load models
    print(f"\n--- Loading Ensemble ---")
    n_features = np.load(GOES_X_PATH, mmap_mode='r').shape[2]
    models = load_ensemble(MODEL_DIR, n_features=n_features, device=device)
    print(f"  Loaded {len(models)} models")

    # Split data (with negatives)
    print(f"\n--- Splitting GOES Hold-Out ---")
    X_test, y_test, X_train, y_train = split_goes_holdout(GOES_X_PATH, GOES_Y_PATH)

    # Run inference
    print(f"\n--- Running Inference on {len(X_test)} Hold-Out Events ---")
    predictions, probabilities = ensemble_predict(models, X_test, device=device)

    # Compute metrics
    print(f"\n--- Computing Metrics ---")
    results = compute_all_metrics(y_test, predictions, probabilities)

    # Print
    print(f"\n{'=' * 60}")
    print(f"  ACTUAL GOES HOLD-OUT RESULTS")
    print(f"  ({int((y_test>=3).sum())} M/X + {int((y_test<3).sum())} negatives, NEVER seen in training)")
    print(f"{'=' * 60}")
    print(f"\n  M+X TSS:            {results['TSS']}")
    print(f"  M+X HSS:            {results['HSS']}")
    print(f"  RED Accuracy (TPR): {results['RED_accuracy']}%")
    print(f"  FPR:                {results['FPR']}")
    print(f"  Precision:          {results['precision']}%")
    print(f"  Brier Score (M+X):  {results['brier_score']}")
    print(f"  Balanced Accuracy:  {results['balanced_accuracy']}%")
    print(f"\n  TP={results['TP']} FN={results['FN']} FP={results['FP']} TN={results['TN']}")
    for key in sorted(results.keys()):
        if key.startswith('AUC_'):
            print(f"  {key}: {results[key]}")

    # Confusion matrix
    cm = results['confusion_matrix']
    names = ['None', 'B', 'C', 'M', 'X']
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>8}", end='')
    for n in names: print(f"{n:>8}", end='')
    print()
    for i in range(5):
        print(f"  {names[i]:>8}", end='')
        for j in range(5):
            print(f"{cm[i][j]:>8}", end='')
        print()

    # Save
    with open('actual_holdout_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] actual_holdout_results.json")
    print(f"  Done!")
