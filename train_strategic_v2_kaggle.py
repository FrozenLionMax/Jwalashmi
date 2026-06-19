"""
JWALASHMI Strategic V2 - Kaggle Training
==========================================
12-hour early warning | 5-model ensemble | 12 features (SoLEXS + HEL1OS)
Paste this entire code into ONE Kaggle cell and run.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

# ===== PATHS (Kaggle) =====
DATA_DIR = Path("/kaggle/working/ISRO/data/processed")
MODEL_DIR = Path("/kaggle/working/ISRO/models/strategic_v2_ensemble")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ['None', 'B', 'C', 'M', 'X']
N_CLASSES = 5

print("=" * 60)
print("  JWALASHMI STRATEGIC V2")
print("  12-Hour Early Warning | 5-Model Ensemble")
print("  12 Features (SoLEXS + HEL1OS)")
print("=" * 60)

# ===== LOAD DATA =====
X = np.load(str(DATA_DIR / "X_strategic_v2.npy"))
y = np.load(str(DATA_DIR / "y_strategic_v2.npy"))
lt = np.load(str(DATA_DIR / "lt_strategic_v2.npy"))

print(f"X: {X.shape}  y: {y.shape}")
print(f"Window: {X.shape[1]} min = {X.shape[1]/60:.0f} hours")
print(f"Features: {X.shape[2]}")
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n}: {(y==i).sum()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===== STRATEGIC MODEL ARCHITECTURE =====
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)
    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))

class StrategicV2(nn.Module):
    """
    Enhanced Strategic Forecaster for 12-hour prediction.
    3-layer CNN + 8-head attention + dual output heads.
    """
    def __init__(self, n_features=12, n_classes=5):
        super().__init__()
        # Deeper CNN than V1 (3 layers vs 2)
        self.cnn = nn.Sequential(
            ConvBlock(n_features, 64, kernel_size=7, pool=2),
            nn.Dropout(0.2),
            ConvBlock(64, 128, kernel_size=5, pool=2),
            nn.Dropout(0.2),
            ConvBlock(128, 256, kernel_size=3, pool=2),
            nn.Dropout(0.3),
        )
        # Multi-head attention (8 heads like tactical)
        self.attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(256)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )
        # Lead time head (hours)
        self.lead_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.cnn(x)        # (B, 256, T')
        x = x.transpose(1, 2)  # (B, T', 256)
        x_attn, attn_w = self.attn(x, x, x)
        x = self.attn_norm(x + x_attn)  # residual
        x_pool = x.mean(dim=1)  # global average pool
        class_logits = self.classifier(x_pool)
        lead_pred = self.lead_head(x_pool)
        return class_logits, lead_pred, attn_w

print(f"\nModel parameters: {sum(p.numel() for p in StrategicV2().parameters()):,}")

# ===== AUGMENTATION =====
def augment_strategic(X, y, lt, multiplier=3):
    rng = np.random.default_rng()
    Xs, ys, ls = [X], [y], [lt]
    for _ in range(multiplier - 1):
        X_aug = X.copy()
        # Gaussian noise
        noise = rng.normal(0, 0.03, X_aug.shape).astype(np.float32)
        X_aug += noise
        # Amplitude scale
        scale = rng.uniform(0.85, 1.15, (len(X_aug), 1, X_aug.shape[2])).astype(np.float32)
        X_aug *= scale
        # Time shift (roll by up to 30 min)
        for i in range(len(X_aug)):
            shift = rng.integers(-30, 30)
            X_aug[i] = np.roll(X_aug[i], shift, axis=0)
        Xs.append(X_aug)
        ys.append(y.copy())
        ls.append(lt.copy())
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ls)

# ===== TRAIN 5-MODEL ENSEMBLE =====
n_features = X.shape[2]
class_weights = torch.tensor([1.0, 2.0, 3.0, 6.0, 12.0], dtype=torch.float32).to(device)
scaler = torch.amp.GradScaler("cuda") if use_amp else None

n_models = 5
models = []
all_val_accs = []

print("\n" + "=" * 60)
print("  Training 5-Model Ensemble (60 epochs each)")
print("=" * 60)

for model_idx in range(n_models):
    if model_idx > 0 and device.type == "cuda":
        torch.cuda.empty_cache()
        time.sleep(8)

    seed = model_idx * 77 + 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n--- Model {model_idx+1}/{n_models} (seed={seed}) ---")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(int(len(X) * 0.15), 10)

    X_train_raw = X[perm[:-n_val]]
    y_train_raw = y[perm[:-n_val]]
    lt_train_raw = lt[perm[:-n_val]]
    X_val = torch.tensor(X[perm[-n_val:]], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[perm[-n_val:]], dtype=torch.long).to(device)

    model = StrategicV2(n_features=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    best_val_acc = 0
    best_state = None
    batch_size = 32

    for epoch in range(60):
        # Fresh augmentation
        X_aug, y_aug, lt_aug = augment_strategic(X_train_raw, y_train_raw, lt_train_raw, multiplier=3)
        X_t = torch.tensor(X_aug, dtype=torch.float32)
        y_t = torch.tensor(y_aug, dtype=torch.long)
        lt_t = torch.tensor(lt_aug / 60.0, dtype=torch.float32)  # convert to hours

        model.train()
        epoch_loss, n_b = 0, 0
        perm_e = torch.randperm(len(X_t))

        for i in range(0, len(X_t), batch_size):
            idx = perm_e[i:i+batch_size]
            xb = X_t[idx].to(device)
            yb = y_t[idx].to(device)
            lb = lt_t[idx].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    cl, lp, _ = model(xb)
                    loss_cls = F.cross_entropy(cl, yb, weight=class_weights)
                    loss_reg = F.mse_loss(lp.squeeze(-1), lb)
                    loss = loss_cls + 0.05 * loss_reg
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                cl, lp, _ = model(xb)
                loss_cls = F.cross_entropy(cl, yb, weight=class_weights)
                loss_reg = F.mse_loss(lp.squeeze(-1), lb)
                loss = loss_cls + 0.05 * loss_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_b += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    vl, _, _ = model(X_val)
            else:
                vl, _, _ = model(X_val)
            v_acc = (vl.argmax(1) == y_val).float().mean().item()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n_b:.4f} val={v_acc*100:.1f}% best={best_val_acc*100:.1f}%")

    model.load_state_dict(best_state)
    models.append(model)
    all_val_accs.append(best_val_acc)
    print(f"  Best: {best_val_acc*100:.1f}%")

print(f"\nEnsemble mean: {np.mean(all_val_accs)*100:.1f}% +/- {np.std(all_val_accs)*100:.1f}%")

# ===== EVALUATE =====
print("\n" + "=" * 60)
print("  Evaluation")
print("=" * 60)

from sklearn.metrics import classification_report, balanced_accuracy_score

X_all = torch.tensor(X, dtype=torch.float32).to(device)
all_probs = []
for m in models:
    m.eval()
    with torch.no_grad():
        ps = []
        for i in range(0, len(X_all), 64):
            b = X_all[i:i+64]
            if use_amp:
                with torch.amp.autocast("cuda"):
                    l, _, _ = m(b)
            else:
                l, _, _ = m(b)
            ps.append(F.softmax(l, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(ps))

probs = np.mean(all_probs, axis=0)
preds = probs.argmax(1)
bal = balanced_accuracy_score(y, preds)

print(f"\nBalanced Accuracy: {bal*100:.1f}%")
print(classification_report(y, preds, target_names=CLASS_NAMES, digits=3))

# ===== SAVE =====
print("Saving models...")
for i, m in enumerate(models):
    torch.save(m.state_dict(), str(MODEL_DIR / f"strategic_v2_model_{i}.pt"))

best_i = int(np.argmax(all_val_accs))
torch.save(models[best_i].state_dict(), str(MODEL_DIR.parent / "strategic_v2_best.pt"))

meta = {
    "version": "Strategic_V2",
    "n_features": n_features,
    "n_models": n_models,
    "window_minutes": int(X.shape[1]),
    "window_hours": int(X.shape[1] / 60),
    "lead_time_hours": "5-12",
    "balanced_accuracy": float(bal),
    "val_accs": [float(a) for a in all_val_accs],
    "class_names": CLASS_NAMES,
}
with open(str(MODEL_DIR / "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved {n_models} models + metadata")

# Zip for download
import shutil
shutil.make_archive("/kaggle/working/strategic_v2_models", "zip", str(MODEL_DIR))
print("\nDownload strategic_v2_models.zip from Output tab!")

print("\n" + "=" * 60)
print("  STRATEGIC V2 COMPLETE!")
print(f"  Window: {X.shape[1]/60:.0f} hours | Features: {n_features}")
print(f"  Balanced Accuracy: {bal*100:.1f}%")
print(f"  Lead Time: 5-12 hours")
print("=" * 60)
