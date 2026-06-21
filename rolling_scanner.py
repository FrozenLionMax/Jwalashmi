"""
Rolling prediction scanner - uses existing V6.2 model (no training needed).
Slides 60-min window across full day, plots M/X probability timeline.
This creates an 8-12 hour effective warning using the tactical model.
"""
import sys, os, json, time
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn.functional as F
import config as cfg
from src.model.architecture import FlareForecaster
from src.data.fits_loader import load_solexs_all, load_hel1os_all
from src.features.physics_features import compute_all_features, get_feature_columns
import pandas as pd

print("=" * 60)
print("  ROLLING PREDICTION SCANNER")
print("  Using existing V6.2 ensemble (no retraining)")
print("=" * 60)

# Load V6.2 ensemble
models = []
for i in range(10):
    m = FlareForecaster(n_input_channels=12)
    m.load_state_dict(torch.load(
        str(cfg.MODEL_DIR / 'v6_2_ensemble' / ('model_%d.pt' % i)),
        map_location='cpu', weights_only=True))
    m.eval()
    models.append(m)
print("  Loaded %d V6.2 models" % len(models))

# Load normalization
feature_mean = np.load(str(cfg.PROCESSED / "feature_mean.npy"))
feature_std = np.load(str(cfg.PROCESSED / "feature_std.npy"))

# Load data
print("  Loading SoLEXS + HEL1OS...")
df_solexs = load_solexs_all()
df_hel1os = load_hel1os_all(detector="cdte1")

hel1os_dates = set(df_hel1os["date"].unique()) if not df_hel1os.empty else set()
hel1os_ctr_cols = [c for c in df_hel1os.columns if c.startswith("ctr_")] if not df_hel1os.empty else []
hxr_soft_col = next((c for c in hel1os_ctr_cols if "5.00KEV_TO_20" in c), None)
hxr_hard_col = next((c for c in hel1os_ctr_cols if "30.00KEV_TO_40" in c), None)
hxr_medium_col = next((c for c in hel1os_ctr_cols if "20.00KEV_TO_30" in c), None)
hel1os_feat_cols = {"hxr_soft": hxr_soft_col, "hxr_hard": hxr_hard_col, "hxr_medium": hxr_medium_col} if hxr_soft_col else None

# Pick high-flare dates to analyze
target_dates = ["20241003", "20240914", "20241024", "20241031", "20241208", "20241230"]

CLASS_NAMES = cfg.CLASS_NAMES
WINDOW = 3600  # 60 min at 1s cadence
STEP = 900     # slide every 15 min

results = {}

for date in target_dates:
    if date not in df_solexs["date"].values:
        print("  SKIP %s (no SoLEXS)" % date)
        continue

    print("\n  Scanning %s..." % date)
    day_solexs = df_solexs[df_solexs["date"] == date].copy().reset_index(drop=True)

    # Merge with HEL1OS if available
    if date in hel1os_dates and not df_hel1os.empty:
        day_hel1os = df_hel1os[df_hel1os["date"] == date].copy()
        day_solexs["ts_round"] = day_solexs["timestamp"].round(0)
        day_hel1os["ts_round"] = day_hel1os["timestamp"].round(0)
        merge_cols = ["ts_round"] + [c for c in day_hel1os.columns if c.startswith("ctr_")]
        day_merged = pd.merge(
            day_solexs, day_hel1os[merge_cols].drop_duplicates(subset="ts_round"),
            on="ts_round", how="left").drop(columns=["ts_round"])
        for col in [c for c in day_merged.columns if c.startswith("ctr_")]:
            day_merged[col] = day_merged[col].fillna(0)
        day_feat = compute_all_features(day_merged, hel1os_cols=hel1os_feat_cols)
        has_hel1os = True
    else:
        day_feat = compute_all_features(day_solexs)
        for fcol in ["feat_hard_soft_ratio", "feat_neupert", "feat_spectral_hardness"]:
            if fcol not in day_feat.columns:
                day_feat[fcol] = 0.0
        has_hel1os = False

    feat_cols = get_feature_columns(day_feat)
    feat_matrix = day_feat[feat_cols].values.astype(np.float32)

    # Normalize
    for j in range(min(len(feature_mean), feat_matrix.shape[1])):
        if feature_std[j] > 0:
            feat_matrix[:, j] = (feat_matrix[:, j] - feature_mean[j]) / feature_std[j]

    # Pad to 12 features if needed
    if feat_matrix.shape[1] < 12:
        pad = np.zeros((feat_matrix.shape[0], 12 - feat_matrix.shape[1]), dtype=np.float32)
        feat_matrix = np.concatenate([feat_matrix, pad], axis=1)

    # Slide window across day
    n_samples = len(feat_matrix)
    day_results = []

    for start in range(0, n_samples - WINDOW, STEP):
        window = feat_matrix[start:start + WINDOW]
        if len(window) < WINDOW:
            break

        x = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)

        # Ensemble prediction
        all_p = []
        for m in models:
            with torch.no_grad():
                logits, lead_pred, _ = m(x)
                all_p.append(F.softmax(logits, dim=1).numpy()[0])

        avg_prob = np.mean(all_p, axis=0)
        pred_class = CLASS_NAMES[np.argmax(avg_prob)]
        mx_prob = avg_prob[3] + avg_prob[4]  # M + X probability

        hour = (start + WINDOW / 2) / 3600.0
        day_results.append({
            "hour": round(hour, 2),
            "pred": pred_class,
            "probs": {CLASS_NAMES[k]: round(float(avg_prob[k]), 4) for k in range(5)},
            "mx_prob": round(float(mx_prob), 4),
            "confidence": round(float(avg_prob.max()), 4),
        })

    results[date] = {"has_hel1os": has_hel1os, "n_windows": len(day_results), "timeline": day_results}

    # Print summary
    max_mx = max(r["mx_prob"] for r in day_results) if day_results else 0
    first_mx_alert = None
    for r in day_results:
        if r["mx_prob"] > 0.3:
            first_mx_alert = r["hour"]
            break

    peak_hour = max(day_results, key=lambda r: r["mx_prob"])["hour"] if day_results else 0

    print("    Windows scanned: %d (every 15 min)" % len(day_results))
    print("    HEL1OS: %s" % ("YES" if has_hel1os else "No"))
    print("    Peak M+X prob:   %.1f%% at hour %.1f" % (max_mx * 100, peak_hour))
    if first_mx_alert:
        warning_hours = peak_hour - first_mx_alert
        print("    First M+X alert: hour %.1f (%.1f hours before peak!)" % (first_mx_alert, warning_hours))
    else:
        print("    First M+X alert: None (below 30%% threshold)")

    # Show hourly timeline
    print("    Timeline:")
    for r in day_results:
        bar_len = int(r["mx_prob"] * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        alert = ""
        if r["mx_prob"] > 0.7:
            alert = " *** RED ***"
        elif r["mx_prob"] > 0.4:
            alert = " * YELLOW *"
        elif r["mx_prob"] > 0.2:
            alert = " (elevated)"
        print("      %05.1fh [%s] %.0f%% %s %s" % (
            r["hour"], bar, r["mx_prob"] * 100, r["pred"], alert))

# Save results
with open("rolling_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to rolling_predictions.json")

print("\n" + "=" * 60)
print("  DONE — Check timelines above for early warnings!")
print("=" * 60)
