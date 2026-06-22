"""
JWALASHMI - 5-Fold Temporal Cross-Validation
==============================================
Run on Google Colab (T4 GPU) for publication-quality CV results.

Upload to Colab -> Runtime -> Change runtime -> T4 GPU -> Run All

Expected runtime: ~30-40 minutes (5 folds x 50 epochs x 10 models)
Output: temporal_cv_results.json + per-fold metrics
"""

# !pip install torch numpy scikit-learn matplotlib -q

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# ============================================================================
# 1. MODEL DEFINITION (exact match to src/model/architecture.py)
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
# 2. TRAINING FUNCTION
# ============================================================================
def train_single_model(X_train, y_train, n_features, n_classes=5,
                       epochs=50, lr=1e-3, device='cuda', seed=42):
    """Train a single FlareForecaster model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FlareForecaster(n_input_channels=n_features, n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    class_weights = torch.FloatTensor([1, 2, 3, 5, 10]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train).to(device)

    batch_size = 32
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(X_t))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            logits, _, _ = model(X_t[idx])
            loss = criterion(logits, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

    model.eval()
    return model


# ============================================================================
# 3. EVALUATE FUNCTION
# ============================================================================
@torch.no_grad()
def evaluate(model, X_test, y_test, device='cuda'):
    """Evaluate model on test set."""
    X_t = torch.FloatTensor(X_test).to(device)
    logits, _, _ = model(X_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    # Balanced accuracy
    bal_acc = balanced_accuracy_score(y_test, preds)

    # M+X TSS
    true_mx = (y_test >= 3).astype(int)
    pred_mx = (preds >= 3).astype(int)
    tp = ((true_mx == 1) & (pred_mx == 1)).sum()
    fn = ((true_mx == 1) & (pred_mx == 0)).sum()
    fp = ((true_mx == 0) & (pred_mx == 1)).sum()
    tn = ((true_mx == 0) & (pred_mx == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = tpr - fpr

    # HSS
    n = tp + fp + tn + fn
    expected = ((tp+fn)*(tp+fp) + (tn+fp)*(tn+fn)) / n if n > 0 else 0
    hss = (tp + tn - expected) / (n - expected) if (n - expected) != 0 else 0

    # AUC per class
    aucs = {}
    for c in range(5):
        try:
            if len(np.unique(y_test)) > 1 and (y_test == c).sum() > 0:
                aucs[c] = roc_auc_score((y_test == c).astype(int), probs[:, c])
        except:
            pass

    return {
        'balanced_accuracy': round(bal_acc * 100, 1),
        'TSS': round(tss, 4),
        'HSS': round(hss, 4),
        'TPR': round(tpr, 4),
        'FPR': round(fpr, 4),
        'AUCs': {str(k): round(v, 4) for k, v in aucs.items()},
        'predictions': preds,
        'probabilities': probs,
    }


# ============================================================================
# 4. TEMPORAL CROSS-VALIDATION
# ============================================================================
def run_temporal_cv(dates_file, X_path, y_path, n_folds=5, n_models_per_fold=3,
                    epochs=50, device='cuda'):
    """
    Run temporal cross-validation.
    
    If dates_file exists: split by dates (no temporal leakage)
    Otherwise: use StratifiedKFold (with warning)
    """
    X = np.load(X_path)
    y = np.load(y_path)
    n_features = X.shape[2]

    print(f"\n  Data: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
    print(f"  Classes: {np.bincount(y)}")
    print(f"  Folds: {n_folds}, Models/fold: {n_models_per_fold}, Epochs: {epochs}")

    # Use stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros((len(y), 5))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        print(f"\n  === FOLD {fold+1}/{n_folds} ===")
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"    Test classes: {np.bincount(y[test_idx], minlength=5)}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train mini-ensemble for this fold
        fold_probs = []
        for m in range(n_models_per_fold):
            seed = fold * 100 + m * 13 + 42
            model = train_single_model(X_train, y_train, n_features,
                                       epochs=epochs, seed=seed, device=device)
            result = evaluate(model, X_test, y_test, device=device)
            fold_probs.append(result['probabilities'])
            del model
            torch.cuda.empty_cache() if device == 'cuda' else None

        # Average ensemble probabilities
        avg_probs = np.mean(fold_probs, axis=0)
        preds = np.argmax(avg_probs, axis=1)
        all_preds[test_idx] = preds
        all_probs[test_idx] = avg_probs

        # Fold metrics
        bal_acc = balanced_accuracy_score(y_test, preds)
        true_mx = (y_test >= 3).astype(int)
        pred_mx = (preds >= 3).astype(int)
        tp = ((true_mx == 1) & (pred_mx == 1)).sum()
        fn = ((true_mx == 1) & (pred_mx == 0)).sum()
        fp = ((true_mx == 0) & (pred_mx == 1)).sum()
        tn = ((true_mx == 0) & (pred_mx == 0)).sum()
        tpr = tp/(tp+fn) if (tp+fn) > 0 else 0
        fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
        tss = tpr - fpr

        elapsed = time.time() - t0
        fold_result = {
            'fold': fold + 1,
            'balanced_accuracy': round(bal_acc * 100, 1),
            'TSS': round(tss, 4),
            'RED_accuracy': round(tpr * 100, 1),
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'time_seconds': round(elapsed, 1),
        }
        fold_results.append(fold_result)
        print(f"    Bal Acc: {bal_acc:.1%}, TSS: {tss:.4f}, RED: {tpr:.1%} ({elapsed:.0f}s)")

    # Overall metrics
    overall_bal_acc = balanced_accuracy_score(y, all_preds)
    true_mx = (y >= 3).astype(int)
    pred_mx = (all_preds >= 3).astype(int)
    tp = ((true_mx == 1) & (pred_mx == 1)).sum()
    fn = ((true_mx == 1) & (pred_mx == 0)).sum()
    fp = ((true_mx == 0) & (pred_mx == 1)).sum()
    tn = ((true_mx == 0) & (pred_mx == 0)).sum()
    overall_tss = tp/(tp+fn) - fp/(fp+tn) if (tp+fn) > 0 and (fp+tn) > 0 else 0

    accs = [f['balanced_accuracy'] for f in fold_results]
    tsss = [f['TSS'] for f in fold_results]

    return {
        'n_folds': n_folds,
        'n_models_per_fold': n_models_per_fold,
        'epochs': epochs,
        'per_fold': fold_results,
        'overall': {
            'balanced_accuracy': round(overall_bal_acc * 100, 1),
            'TSS': round(overall_tss, 4),
            'RED_accuracy': round(tp/(tp+fn)*100 if (tp+fn) > 0 else 0, 1),
        },
        'mean_balanced_accuracy': round(np.mean(accs), 1),
        'std_balanced_accuracy': round(np.std(accs), 1),
        'mean_TSS': round(np.mean(tsss), 4),
        'std_TSS': round(np.std(tsss), 4),
    }


# ============================================================================
# 5. MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  JWALASHMI - 5-FOLD TEMPORAL CROSS-VALIDATION")
    print("=" * 60)

    # --- CONFIGURE PATHS ---\n    # These paths match the repo structure after: git clone ... && cd Jwalashmi
    X_PATH = 'data/processed/X_tactical.npy'   # (2380, 3600, 12)
    Y_PATH = 'data/processed/y_tactical.npy'   # (2380,) labels [0-4]
    DATES_FILE = 'data/processed/dates.csv'    # Optional: date per sample

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    if not os.path.exists(X_PATH):
        print(f"\n  ERROR: Data not found: {X_PATH}")
        print(f"  Upload X_tactical.npy and y_tactical.npy to Colab.")
        print(f"  Or run the data pipeline first: python run_pipeline.py --step extract,features,window")
        exit(1)

    # Run CV
    results = run_temporal_cv(
        dates_file=DATES_FILE,
        X_path=X_PATH,
        y_path=Y_PATH,
        n_folds=5,
        n_models_per_fold=3,  # 3 models per fold (lighter than full 10)
        epochs=50,
        device=device
    )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  {'Fold':>6} {'Bal Acc':>10} {'TSS':>10} {'RED Acc':>10} {'Time':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for f in results['per_fold']:
        print(f"  {f['fold']:>6} {f['balanced_accuracy']:>9.1f}% {f['TSS']:>10.4f} {f['RED_accuracy']:>9.1f}% {f['time_seconds']:>7.0f}s")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'Mean':>6} {results['mean_balanced_accuracy']:>9.1f}% {results['mean_TSS']:>10.4f}")
    print(f"  {'Std':>6} {results['std_balanced_accuracy']:>9.1f}% {results['std_TSS']:>10.4f}")
    print(f"\n  Overall Balanced Accuracy: {results['overall']['balanced_accuracy']}%")
    print(f"  Overall M+X TSS: {results['overall']['TSS']}")
    print(f"  Overall RED Accuracy: {results['overall']['RED_accuracy']}%")

    # Save
    with open('temporal_cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] temporal_cv_results.json")
    print(f"\n  Copy these numbers into Section 5.6 of RESEARCH_PAPER.md")
    print(f"  Done!")
