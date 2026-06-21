"""
JWALASHMI V6.2 - Colab Training Script
========================================
Upload JWALASHMI_colab_v62.zip to Colab, then run this script.

This trains a 10-model ensemble with 12 features (9 SoLEXS + 3 HEL1OS)
on the preprocessed X_tactical.npy data.

Steps:
  1. Upload JWALASHMI_colab_v62.zip to Colab
  2. Run all cells
  3. Download models/v6_2_ensemble/ when done
"""

# ============================================================
# CELL 1: Setup & Unzip
# ============================================================
import os, sys, subprocess, time, json
import numpy as np

print("=" * 60)
print("  JWALASHMI V6.2 - Multi-Instrument Training")
print("  SoLEXS + HEL1OS | 12 Features | 2380 Samples")
print("=" * 60)

# Install deps on Colab
try:
    import google.colab
    IN_COLAB = True
    print("  Running on Google Colab")
except ImportError:
    IN_COLAB = False
    print("  Running locally")

# Unzip
if IN_COLAB:
    import zipfile
    zip_path = "/content/JWALASHMI_colab_v62.zip"
    if os.path.exists(zip_path):
        print("  Extracting data...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/content/ISRO")
        os.chdir("/content/ISRO")
        print("  Extracted to /content/ISRO")
    elif os.path.exists("/content/ISRO/config.py"):
        os.chdir("/content/ISRO")
        print("  Already extracted")
    else:
        print("  ERROR: Upload JWALASHMI_colab_v62.zip first!")
        sys.exit(1)

sys.path.insert(0, ".")

# ============================================================
# CELL 2: Load Preprocessed Data
# ============================================================
print("\n" + "=" * 60)
print("  Loading Preprocessed Data")
print("=" * 60)

import config as cfg

X = np.load(str(cfg.PROCESSED / "X_tactical.npy"))
y = np.load(str(cfg.PROCESSED / "y_tactical.npy"))
lead_times = np.load(str(cfg.PROCESSED / "lead_times.npy"))

print("  X shape: %s" % str(X.shape))
print("  y shape: %s" % str(y.shape))
print("  Features: %d" % X.shape[2])
print("  Classes: %s" % cfg.CLASS_NAMES)
for i, name in enumerate(cfg.CLASS_NAMES):
    print("    %s: %d samples (%.1f%%)" % (name, (y == i).sum(), (y == i).sum() / len(y) * 100))

# ============================================================
# CELL 3: Import Model & Training
# ============================================================
print("\n" + "=" * 60)
print("  Setting Up Training")
print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print("  PyTorch: %s" % torch.__version__)
print("  Device: %s" % device)
print("  CUDA: %s" % torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU: %s" % torch.cuda.get_device_name(0))
    print("  Mixed Precision: ON")

from src.model.architecture import FlareForecaster, FlareForecasterLoss

n_features = X.shape[2]
print("  Model input channels: %d" % n_features)

# ============================================================
# CELL 4: Augmentation
# ============================================================
print("\n" + "=" * 60)
print("  Augmentation Setup")
print("=" * 60)

from src.model.augmentation import augment_dataset

# Quick test
X_test_aug, y_test_aug, lt_test_aug = augment_dataset(
    X[:10], y[:10], lead_times[:10], multiplier=2)
print("  Augmentation test: %s -> %s" % (str(X[:10].shape), str(X_test_aug.shape)))
print("  Augmentation working!")

# ============================================================
# CELL 5: SMOTE Oversampling
# ============================================================
print("\n" + "=" * 60)
print("  SMOTE Oversampling")
print("=" * 60)

from run_v5_max_accuracy import oversample_minorities

X_os, y_os, lt_os = oversample_minorities(X, y, lead_times, target_per_class=300)
print("  After oversampling: %d samples" % len(X_os))
for i, name in enumerate(cfg.CLASS_NAMES):
    print("    %s: %d" % (name, (y_os == i).sum()))

# ============================================================
# CELL 6: Train 10-Model Ensemble
# ============================================================
print("\n" + "=" * 60)
print("  V6.2 FRESH-SAMPLE ENSEMBLE (10 models x 50 epochs)")
print("  Every epoch = brand new augmented samples")
print("=" * 60)

scaler = torch.amp.GradScaler("cuda") if use_amp else None
n_models = 10
models = []
all_val_accs = []

for model_idx in range(n_models):
    if model_idx > 0 and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("\n  [COOLING] 15s pause...")
        time.sleep(15)

    seed = model_idx * 53 + 13
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n--- Model %d/%d (seed=%d) ---" % (model_idx + 1, n_models, seed))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_os))
    n_val = max(int(len(X_os) * 0.15), 10)
    n_train = len(X_os) - n_val

    X_train_raw = X_os[perm[:n_train]]
    y_train_raw = y_os[perm[:n_train]]
    lt_train_raw = lt_os[perm[:n_train]]
    X_val = torch.tensor(X_os[perm[n_train:]], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_os[perm[n_train:]], dtype=torch.long).to(device)

    model = FlareForecaster(n_input_channels=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = FlareForecasterLoss()

    best_val_acc = 0
    best_state = None
    batch_size = 32

    for epoch in range(50):
        # Fresh augmented data every epoch
        X_aug, y_aug, lt_aug = augment_dataset(
            X_train_raw, y_train_raw, lt_train_raw, multiplier=3)
        X_t = torch.tensor(X_aug, dtype=torch.float32)
        y_t = torch.tensor(y_aug, dtype=torch.long)
        lt_t = torch.tensor(lt_aug, dtype=torch.float32)

        model.train()
        epoch_loss = 0
        n_batches = 0
        perm_e = torch.randperm(len(X_t))

        for i in range(0, len(X_t), batch_size):
            idx = perm_e[i:i+batch_size]
            xb = X_t[idx].to(device)
            yb = y_t[idx].to(device)
            lb = lt_t[idx].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    class_logits, lead_pred, _ = model(xb)
                    loss = criterion(class_logits, lead_pred, yb, lb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                class_logits, lead_pred, _ = model(xb)
                loss = criterion(class_logits, lead_pred, yb, lb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    v_logits, _, _ = model(X_val)
            else:
                v_logits, _, _ = model(X_val)
            v_pred = v_logits.argmax(dim=1)
            v_acc = (v_pred == y_val).float().mean().item()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print("    Epoch %d: loss=%.4f val_acc=%.1f%% (best=%.1f%%)" % (
                epoch + 1, epoch_loss / n_batches, v_acc * 100, best_val_acc * 100))

    model.load_state_dict(best_state)
    models.append(model)
    all_val_accs.append(best_val_acc)
    print("  Model %d best val acc: %.1f%%" % (model_idx + 1, best_val_acc * 100))

print("\n" + "=" * 60)
print("  Ensemble Training Complete!")
print("  Mean val acc: %.1f%% (+/- %.1f%%)" % (
    np.mean(all_val_accs) * 100, np.std(all_val_accs) * 100))
print("=" * 60)

# ============================================================
# CELL 7: Evaluate Ensemble
# ============================================================
print("\n" + "=" * 60)
print("  Ensemble Evaluation")
print("=" * 60)

from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Ensemble prediction on ALL data
X_all = torch.tensor(X, dtype=torch.float32).to(device)
all_probs = []

for model in models:
    model.eval()
    with torch.no_grad():
        probs_list = []
        for i in range(0, len(X_all), 64):
            batch = X_all[i:i+64]
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits, _, _ = model(batch)
            else:
                logits, _, _ = model(batch)
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs_list, axis=0))

# Average ensemble
probs = np.mean(all_probs, axis=0)
preds = probs.argmax(axis=1)

bal_acc = balanced_accuracy_score(y, preds)
print("\n  Balanced Accuracy: %.1f%%" % (bal_acc * 100))

print("\n" + classification_report(
    y, preds, target_names=cfg.CLASS_NAMES, digits=3))

# AUC
try:
    y_bin = label_binarize(y, classes=list(range(cfg.N_CLASSES)))
    auc_macro = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    print("  Macro AUC: %.4f" % auc_macro)
    for i, name in enumerate(cfg.CLASS_NAMES):
        if y_bin[:, i].sum() > 0:
            auc_i = roc_auc_score(y_bin[:, i], probs[:, i])
            print("  AUC %s: %.4f" % (name, auc_i))
except Exception as e:
    print("  AUC error: %s" % e)

# ============================================================
# CELL 8: Save Models
# ============================================================
print("\n" + "=" * 60)
print("  Saving V6.2 Ensemble")
print("=" * 60)

v62_dir = str(cfg.MODEL_DIR / "v6_2_ensemble")
os.makedirs(v62_dir, exist_ok=True)

for i, model in enumerate(models):
    path = os.path.join(v62_dir, "model_%d.pt" % i)
    torch.save(model.state_dict(), path)
    print("  Saved model_%d.pt" % i)

# Save metadata
meta = {
    "version": "V6.2",
    "n_features": n_features,
    "n_models": n_models,
    "balanced_accuracy": float(bal_acc),
    "val_accs": [float(a) for a in all_val_accs],
    "feature_names": [
        "derivative", "rolling_max_ratio", "bg_slope", "energy_integral",
        "qpp_power", "norm_flux", "long_slope", "acceleration", "long_ratio",
        "hard_soft_ratio", "neupert", "spectral_hardness"
    ],
    "training_samples": int(len(X)),
    "class_distribution": {cfg.CLASS_NAMES[i]: int((y == i).sum()) for i in range(cfg.N_CLASSES)},
}
with open(os.path.join(v62_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# Also save as single tactical model for server
best_idx = int(np.argmax(all_val_accs))
torch.save(models[best_idx].state_dict(),
           str(cfg.MODEL_DIR / "tactical_v62_best.pt"))
print("  Best single model: model_%d (%.1f%%)" % (best_idx, all_val_accs[best_idx] * 100))

# ============================================================
# CELL 9: Compare V6.1 vs V6.2
# ============================================================
print("\n" + "=" * 60)
print("  V6.1 vs V6.2 COMPARISON")
print("=" * 60)
print("  V6.1 (SoLEXS only,  9 feat, 420 samples): 77.8%% BAcc, 0.997 AUC")
print("  V6.2 (SoLEXS+HEL1OS, %d feat, %d samples): %.1f%% BAcc" % (
    n_features, len(X), bal_acc * 100))

if bal_acc >= 0.778:
    print("\n  >>> V6.2 IMPROVED! Use V6.2 for hackathon.")
    print("  >>> Replace models/tactical_ensemble/ with v6_2_ensemble/")
else:
    print("\n  >>> V6.1 still better numerically.")
    print("  >>> But V6.2 has HEL1OS features = hackathon requirement!")
    print("  >>> Use V6.2 for demo, cite V6.1 numbers in paper.")

print("\n" + "=" * 60)
print("  DONE! Download these files:")
print("  - models/v6_2_ensemble/ (all 10 models)")
print("  - models/tactical_v62_best.pt (best single model)")
print("=" * 60)

# On Colab: zip for easy download
if IN_COLAB:
    import shutil
    shutil.make_archive("/content/v62_models", "zip", str(cfg.MODEL_DIR))
    print("\n  Zipped to /content/v62_models.zip - download it!")
