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
# 1. MODEL DEFINITION (must match training architecture)
# ============================================================================
class FlareForecaster(nn.Module):
    def __init__(self, n_features=9, n_classes=5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.MaxPool1d(4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(32))
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5))
        self.classifier = nn.Linear(128, n_classes)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)
        attn_out, weights = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x.mean(dim=1)
        features = self.fc(x)
        logits = self.classifier(features)
        lead = self.regressor(features)
        return logits, lead, weights


# ============================================================================
# 2. LOAD ENSEMBLE
# ============================================================================
def load_ensemble(model_dir, n_models=10, n_features=9, device='cuda'):
    """Load all ensemble models."""
    models = []
    for i in range(n_models):
        model = FlareForecaster(n_features=n_features).to(device)
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
# 3. SPLIT DATA INTO TRAIN/HOLDOUT
# ============================================================================
def split_goes_holdout(X_path, y_path, holdout_frac=0.20, seed=42):
    """Split GOES data into train and hold-out sets."""
    X = np.load(X_path)
    y = np.load(y_path)

    # Select only M (3) and X (4) class events
    mx_mask = y >= 3
    X_mx = X[mx_mask]
    y_mx = y[mx_mask]

    print(f"\n  Total GOES M/X events: {len(y_mx)}")
    print(f"    M-class: {(y_mx == 3).sum()}")
    print(f"    X-class: {(y_mx == 4).sum()}")

    # Stratified split
    np.random.seed(seed)
    m_indices = np.where(y_mx == 3)[0]
    x_indices = np.where(y_mx == 4)[0]

    np.random.shuffle(m_indices)
    np.random.shuffle(x_indices)

    m_split = int(len(m_indices) * (1 - holdout_frac))
    x_split = int(len(x_indices) * (1 - holdout_frac))

    test_idx = np.concatenate([m_indices[m_split:], x_indices[x_split:]])
    train_idx = np.concatenate([m_indices[:m_split], x_indices[:x_split]])

    print(f"\n  Train: {len(train_idx)} ({(y_mx[train_idx]==3).sum()} M, {(y_mx[train_idx]==4).sum()} X)")
    print(f"  Test:  {len(test_idx)} ({(y_mx[test_idx]==3).sum()} M, {(y_mx[test_idx]==4).sum()} X)")

    return X_mx[test_idx], y_mx[test_idx], X_mx[train_idx], y_mx[train_idx]


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

    # Average across ensemble
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

    tp = ((true_mx == 1) & (pred_mx == 1)).sum()
    fn = ((true_mx == 1) & (pred_mx == 0)).sum()
    fp = ((true_mx == 0) & (pred_mx == 1)).sum()
    tn = ((true_mx == 0) & (pred_mx == 0)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = tpr - fpr

    n = tp + fp + tn + fn
    expected = ((tp+fn)*(tp+fp) + (tn+fp)*(tn+fn)) / n if n > 0 else 0
    hss = (tp + tn - expected) / (n - expected) if (n - expected) != 0 else 0

    results['TSS'] = round(tss, 4)
    results['HSS'] = round(hss, 4)
    results['TPR'] = round(tpr, 4)
    results['FPR'] = round(fpr, 4)
    results['RED_accuracy'] = round(tpr * 100, 1)
    results['precision'] = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0

    # Per-class AUC
    try:
        for c, name in enumerate(['None', 'B', 'C', 'M', 'X']):
            if (y_true == c).sum() > 0 and (y_true == c).sum() < len(y_true):
                auc = roc_auc_score((y_true == c).astype(int), probs[:, c])
                results[f'AUC_{name}'] = round(auc, 4)
    except Exception as e:
        print(f"  AUC computation warning: {e}")

    # Brier Score
    n_classes = probs.shape[1]
    brier = 0
    for i in range(len(y_true)):
        for c in range(n_classes):
            target = 1.0 if y_true[i] == c else 0.0
            brier += (probs[i, c] - target) ** 2
    brier /= len(y_true)
    results['brier_score'] = round(brier, 4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    results['confusion_matrix'] = cm.tolist()

    return results


# ============================================================================
# 6. MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  JWALASHMI - ACTUAL GOES HOLD-OUT INFERENCE")
    print("=" * 60)

    # --- CONFIGURE PATHS ---
    # These paths match the repo structure after: git clone ... && cd Jwalashmi
    MODEL_DIR = 'models/v6_1_ensemble'              # model_0.pt ... model_9.pt
    GOES_X_PATH = 'data/goes/X_goes_pretrain.npy'   # (2271, 3600, 9) GOES windows
    GOES_Y_PATH = 'data/goes/y_goes_pretrain.npy'   # (2271,) labels [0-4]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # Check if files exist
    if not os.path.exists(MODEL_DIR):
        print(f"\n  ERROR: Model directory not found: {MODEL_DIR}")
        print(f"  Upload your models to Colab first.")
        print(f"  Expected files: model_0.pt through model_9.pt")
        exit(1)

    if not os.path.exists(GOES_X_PATH):
        print(f"\n  ERROR: GOES data not found: {GOES_X_PATH}")
        print(f"  Upload X_goes.npy and y_goes.npy to Colab.")
        exit(1)

    # Load models
    print(f"\n--- Loading Ensemble ---")
    n_features = np.load(GOES_X_PATH, mmap_mode='r').shape[2]
    models = load_ensemble(MODEL_DIR, n_features=n_features, device=device)
    print(f"  Loaded {len(models)} models")

    # Split data
    print(f"\n--- Splitting GOES Hold-Out ---")
    X_test, y_test, X_train, y_train = split_goes_holdout(GOES_X_PATH, GOES_Y_PATH)

    # Run inference
    print(f"\n--- Running Inference on {len(X_test)} Hold-Out Events ---")
    predictions, probabilities = ensemble_predict(models, X_test, device=device)

    # Compute metrics
    print(f"\n--- Computing Metrics ---")
    results = compute_all_metrics(y_test, predictions, probabilities)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  ACTUAL GOES HOLD-OUT RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  M+X TSS:          {results['TSS']}")
    print(f"  M+X HSS:          {results['HSS']}")
    print(f"  RED Accuracy:     {results['RED_accuracy']}%")
    print(f"  Precision (M+X):  {results['precision']}%")
    print(f"  Brier Score:      {results['brier_score']}")
    for key in sorted(results.keys()):
        if key.startswith('AUC_'):
            print(f"  {key}:          {results[key]}")

    # Save
    with open('actual_holdout_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] actual_holdout_results.json")
    print(f"\n  Copy these numbers into Section 5.7b of RESEARCH_PAPER.md")
    print(f"  Done!")
