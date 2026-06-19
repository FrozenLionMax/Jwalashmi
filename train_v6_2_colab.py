"""
JWALASHMI V6.2 — SoLEXS + HEL1OS Multi-Instrument Fusion Training
===================================================================
Run this on Google Colab (GPU runtime).

Upload your ISRO project zip first, then run all cells.
This trains a V6.2 ensemble with 12 features (9 SoLEXS + 3 HEL1OS).

Usage:
  1. Upload JWALASHMI_colab_data.zip to Colab
  2. Run this script
  3. Download models/v6_2_ensemble/ when done
"""
import os
import sys
import subprocess

# ── Cell 1: Setup ────────────────────────────────────────────
print("=" * 60)
print("  JWALASHMI V6.2 — Multi-Instrument Training")
print("=" * 60)

# Install deps if on Colab
try:
    import google.colab
    IN_COLAB = True
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "astropy", "torch", "scikit-learn", "scipy", "tqdm"], check=True)
except ImportError:
    IN_COLAB = False

# ── Cell 2: Unzip data ──────────────────────────────────────
if IN_COLAB and os.path.exists("/content/JWALASHMI_colab_data.zip"):
    import zipfile
    with zipfile.ZipFile("/content/JWALASHMI_colab_data.zip", "r") as z:
        z.extractall("/content/ISRO")
    os.chdir("/content/ISRO")
    print("Extracted project to /content/ISRO")

# Ensure we're in the project root
if os.path.exists("config.py"):
    sys.path.insert(0, ".")
elif os.path.exists("/content/ISRO/config.py"):
    os.chdir("/content/ISRO")
    sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
import config as cfg

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Cell 3: Load Data ───────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1: Loading SoLEXS + HEL1OS Data")
print("=" * 60)

from src.data.fits_loader import (load_solexs_all, load_hel1os_all,
                                   find_solexs_files, find_hel1os_files)

solexs_files = find_solexs_files()
hel1os_files = find_hel1os_files()
print(f"  SoLEXS: {len(solexs_files)} days")
print(f"  HEL1OS: {len(hel1os_files)} files")

df_solexs = load_solexs_all()
df_hel1os = load_hel1os_all(detector="cdte1")

print(f"  SoLEXS rows: {len(df_solexs):,}")
print(f"  HEL1OS rows: {len(df_hel1os):,}")


# ── Cell 4: Nowcasting ──────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Nowcasting (SoLEXS + HEL1OS)")
print("=" * 60)

from src.nowcasting.detector import detect_flares, build_unified_catalog

all_flares = []
# SoLEXS flares
for date, group in df_solexs.groupby("date"):
    flares = detect_flares(group.reset_index(drop=True), instrument="solexs")
    all_flares.extend(flares)
    if flares:
        print(f"  SoLEXS {date}: {len(flares)} flares")

# HEL1OS flares
if not df_hel1os.empty:
    broad_col = [c for c in df_hel1os.columns if "1.80KEV_TO_90" in c]
    if broad_col:
        for date, group in df_hel1os.groupby("date"):
            fl = detect_flares(group.reset_index(drop=True),
                               instrument="hel1os", count_col=broad_col[0])
            all_flares.extend(fl)
            if fl:
                print(f"  HEL1OS {date}: {len(fl)} flares")

catalog = build_unified_catalog(all_flares, [])
catalog.to_csv(str(cfg.CATALOG_CSV), index=False)
print(f"  Unified catalog: {len(catalog)} events")


# ── Cell 5: Features (SoLEXS + HEL1OS merged) ──────────────
print("\n" + "=" * 60)
print("  STEP 3: Features (SoLEXS + HEL1OS)")
print("=" * 60)

from src.features.physics_features import compute_all_features, get_feature_columns

# Find HEL1OS dates and column mappings
hel1os_dates = set()
if not df_hel1os.empty:
    hel1os_dates = set(df_hel1os["date"].unique())
    print(f"  HEL1OS dates: {sorted(hel1os_dates)}")

hel1os_ctr_cols = [c for c in df_hel1os.columns if c.startswith("ctr_")] if not df_hel1os.empty else []
hxr_soft_col = next((c for c in hel1os_ctr_cols if "5.00KEV_TO_20" in c), None)
hxr_hard_col = next((c for c in hel1os_ctr_cols if "30.00KEV_TO_40" in c), None)
hxr_medium_col = next((c for c in hel1os_ctr_cols if "20.00KEV_TO_30" in c), None)

if hxr_soft_col:
    print(f"  HEL1OS: soft={hxr_soft_col}, hard={hxr_hard_col}")
    hel1os_feat_cols = {
        "hxr_soft": hxr_soft_col,
        "hxr_hard": hxr_hard_col,
        "hxr_medium": hxr_medium_col,
    }
else:
    hel1os_feat_cols = None

dfs = []
for date in df_solexs["date"].unique():
    day_solexs = df_solexs[df_solexs["date"] == date].copy().reset_index(drop=True)

    if date in hel1os_dates and not df_hel1os.empty:
        day_hel1os = df_hel1os[df_hel1os["date"] == date].copy()
        day_solexs["ts_round"] = day_solexs["timestamp"].round(0)
        day_hel1os["ts_round"] = day_hel1os["timestamp"].round(0)

        merge_cols = ["ts_round"] + [c for c in day_hel1os.columns if c.startswith("ctr_")]
        day_merged = pd.merge(
            day_solexs, day_hel1os[merge_cols].drop_duplicates(subset="ts_round"),
            on="ts_round", how="left"
        ).drop(columns=["ts_round"])

        for col in [c for c in day_merged.columns if c.startswith("ctr_")]:
            day_merged[col] = day_merged[col].fillna(0)

        day_feat = compute_all_features(day_merged, hel1os_cols=hel1os_feat_cols)
        print(f"  {date}: SoLEXS+HEL1OS merged ({len(day_merged)} rows)")
    else:
        day_feat = compute_all_features(day_solexs)
        for fcol in ["feat_hard_soft_ratio", "feat_neupert", "feat_spectral_hardness"]:
            if fcol not in day_feat.columns:
                day_feat[fcol] = 0.0

    dfs.append(day_feat)

df_feat = pd.concat(dfs, ignore_index=True)
feat_cols = get_feature_columns(df_feat)
print(f"\n  Total features: {len(feat_cols)}")
for f in feat_cols:
    print(f"    {f}")


# ── Cell 6: Windowing ───────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: Windowing")
print("=" * 60)

from src.features.windowing import (create_windows, normalize_features,
                                     balance_classes, print_window_stats)

X, y, meta = create_windows(df_feat, feat_cols, flare_catalog=catalog)
print_window_stats(y, meta)
X_norm, mean, std = normalize_features(X)
X_bal, y_bal, meta_bal = balance_classes(X_norm, y, meta, max_ratio=10)
print(f"  After balancing: {len(X_bal)} windows")
print(f"  Feature shape: {X_bal.shape}")
print_window_stats(y_bal, meta_bal)

lead_times = np.array([
    m.lead_time / 60 if m.lead_time is not None else 0
    for m in meta_bal
], dtype=np.float32)

# Save preprocessed data
os.makedirs(str(cfg.PROCESSED), exist_ok=True)
np.save(str(cfg.PROCESSED / "X_tactical.npy"), X_bal)
np.save(str(cfg.PROCESSED / "y_tactical.npy"), y_bal)
np.save(str(cfg.PROCESSED / "lead_times.npy"), lead_times)
np.save(str(cfg.PROCESSED / "feature_mean.npy"), mean)
np.save(str(cfg.PROCESSED / "feature_std.npy"), std)
print(f"  Saved: X_tactical.npy {X_bal.shape}")


# ── Cell 7: Train V6.2 Ensemble ─────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Training V6.2 Ensemble (10 models)")
print("=" * 60)

# Import from run_v5 — reuse the training functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v5_max_accuracy import train_fresh_samples_ensemble

# Load GOES pretrained weights if available
pretrained_weights = str(cfg.MODEL_DIR / "goes_pretrained.pt")
if not os.path.exists(pretrained_weights):
    pretrained_weights = None
    print("  No GOES pre-trained weights")
else:
    print(f"  Using GOES pre-trained: {pretrained_weights}")

models, probs, preds = train_fresh_samples_ensemble(
    X_bal, y_bal, lead_times, pretrained_weights=pretrained_weights)


# ── Cell 8: Save V6.2 Ensemble ──────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6: Saving V6.2 Ensemble")
print("=" * 60)

v62_dir = str(cfg.MODEL_DIR / "v6_2_ensemble")
os.makedirs(v62_dir, exist_ok=True)

for i, model in enumerate(models):
    path = os.path.join(v62_dir, f"model_{i}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Saved model_{i}.pt")

# Save thresholds (optimize B-class boost)
from sklearn.metrics import f1_score
best_thresholds = np.ones(cfg.N_CLASSES)
best_thresholds[1] = 1.3  # Boost B-class
np.save(os.path.join(v62_dir, "thresholds.npy"), best_thresholds)

# Save feature info
import json
feature_info = {
    "version": "V6.2",
    "n_features": len(feat_cols),
    "feature_names": feat_cols,
    "hel1os_features": ["feat_hard_soft_ratio", "feat_neupert", "feat_spectral_hardness"],
    "hel1os_dates": sorted(list(hel1os_dates)),
    "total_dates": len(df_solexs["date"].unique()),
}
with open(os.path.join(v62_dir, "feature_info.json"), "w") as f:
    json.dump(feature_info, f, indent=2)

print(f"\n  V6.2 saved to {v62_dir}")
print(f"  Features: {len(feat_cols)} (9 SoLEXS + 3 HEL1OS)")
print(f"  Models: {len(models)}")


# ── Cell 9: Evaluate ────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7: Evaluation")
print("=" * 60)

from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score

y_true = y_bal
y_pred = preds
y_prob = probs

bal_acc = balanced_accuracy_score(y_true, y_pred)
print(f"\n  Balanced Accuracy: {bal_acc*100:.1f}%")

# Per-class report
print("\n" + classification_report(
    y_true, y_pred, target_names=cfg.CLASS_NAMES, digits=3))

# AUC
try:
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(cfg.N_CLASSES)))
    auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    print(f"  Macro AUC: {auc_macro:.4f}")

    for i, name in enumerate(cfg.CLASS_NAMES):
        if y_bin[:, i].sum() > 0:
            auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
            print(f"  AUC {name}: {auc_i:.4f}")
except Exception as e:
    print(f"  AUC error: {e}")


# ── Cell 10: Compare V6.1 vs V6.2 ──────────────────────────
print("\n" + "=" * 60)
print("  V6.1 vs V6.2 Comparison")
print("=" * 60)
print(f"  V6.1 (SoLEXS only, 9 features):  77.8% BAcc, 0.997 AUC")
print(f"  V6.2 (SoLEXS+HEL1OS, {len(feat_cols)} features): {bal_acc*100:.1f}% BAcc")
print(f"")
if bal_acc >= 0.778:
    print(f"  >> V6.2 IMPROVED! Use V6.2 for hackathon.")
else:
    print(f"  >> V6.1 still better. Keep V6.1 for hackathon, use V6.2 for paper ablation study.")

print(f"\n{'='*60}")
print(f"  DONE — Download v6_2_ensemble/ folder")
print(f"{'='*60}")
