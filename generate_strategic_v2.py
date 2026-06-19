"""
Generate Strategic V2 data: 12-hour windows at 1-min cadence with 12 features.
Fixed: uses FlareEvent.peak_time (unix timestamp) for proper labeling.
"""
import sys, os, time
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import config as cfg
from src.data.fits_loader import load_solexs_all, load_hel1os_all
from src.features.physics_features import compute_all_features, get_feature_columns
from src.nowcasting.detector import detect_flares

t0 = time.time()
print("=" * 60)
print("  STRATEGIC V2 DATA GENERATOR (FIXED)")
print("  12-hour windows | 1-min cadence | 12 features")
print("=" * 60)

# Load all data
print("\n[1/5] Loading data...")
df_solexs = load_solexs_all()
df_hel1os = load_hel1os_all(detector="cdte1")
print("  SoLEXS: %d rows, HEL1OS: %d rows" % (len(df_solexs), len(df_hel1os)))

hel1os_dates = set(df_hel1os["date"].unique()) if not df_hel1os.empty else set()
hel1os_ctr_cols = [c for c in df_hel1os.columns if c.startswith("ctr_")] if not df_hel1os.empty else []
hxr_soft_col = next((c for c in hel1os_ctr_cols if "5.00KEV_TO_20" in c), None)
hxr_hard_col = next((c for c in hel1os_ctr_cols if "30.00KEV_TO_40" in c), None)
hxr_medium_col = next((c for c in hel1os_ctr_cols if "20.00KEV_TO_30" in c), None)
hel1os_feat_cols = {"hxr_soft": hxr_soft_col, "hxr_hard": hxr_hard_col, "hxr_medium": hxr_medium_col} if hxr_soft_col else None

# Detect flares
print("\n[2/5] Detecting flares...")
flare_events = {}  # date -> list of (peak_timestamp, class)
for date, group in df_solexs.groupby("date"):
    day_flares = detect_flares(group.reset_index(drop=True), instrument="solexs")
    if day_flares:
        flare_events[date] = [(f.peak_time, f.estimated_class) for f in day_flares]

total_flares = sum(len(v) for v in flare_events.values())
print("  Found %d flares across %d dates" % (total_flares, len(flare_events)))
# Count by class
class_counts = {"B": 0, "C": 0, "M": 0, "X": 0}
for date, flares in flare_events.items():
    for _, cls in flares:
        if cls in class_counts:
            class_counts[cls] += 1
print("  Classes:", class_counts)

# Process each day
print("\n[3/5] Computing features and creating windows...")
WINDOW_MIN = 720   # 12 hours
STEP_MIN = 30      # slide every 30 min for more samples

all_windows = []
all_labels = []
all_lead_times = []

for date in sorted(df_solexs["date"].unique()):
    day_solexs = df_solexs[df_solexs["date"] == date].copy().reset_index(drop=True)
    
    # Get timestamps for this day
    day_timestamps = day_solexs["timestamp"].values
    day_start_ts = day_timestamps[0]
    
    # Merge HEL1OS
    if date in hel1os_dates and not df_hel1os.empty:
        day_hel1os = df_hel1os[df_hel1os["date"] == date].copy()
        day_solexs_m = day_solexs.copy()
        day_solexs_m["ts_round"] = day_solexs_m["timestamp"].round(0)
        day_hel1os["ts_round"] = day_hel1os["timestamp"].round(0)
        merge_cols = ["ts_round"] + [c for c in day_hel1os.columns if c.startswith("ctr_")]
        day_merged = pd.merge(
            day_solexs_m, day_hel1os[merge_cols].drop_duplicates(subset="ts_round"),
            on="ts_round", how="left").drop(columns=["ts_round"])
        for col in [c for c in day_merged.columns if c.startswith("ctr_")]:
            day_merged[col] = day_merged[col].fillna(0)
        day_feat = compute_all_features(day_merged, hel1os_cols=hel1os_feat_cols)
    else:
        day_feat = compute_all_features(day_solexs)
        for fcol in ["feat_hard_soft_ratio", "feat_neupert", "feat_spectral_hardness"]:
            if fcol not in day_feat.columns:
                day_feat[fcol] = 0.0

    feat_cols = get_feature_columns(day_feat)
    feat_matrix = day_feat[feat_cols].values.astype(np.float32)
    
    # Pad to 12 features
    if feat_matrix.shape[1] < 12:
        pad = np.zeros((feat_matrix.shape[0], 12 - feat_matrix.shape[1]), dtype=np.float32)
        feat_matrix = np.concatenate([feat_matrix, pad], axis=1)
    
    # Downsample to 1-min (every 60th sample)
    n_sec = len(feat_matrix)
    idx_1min = np.arange(0, n_sec, 60)
    feat_1min = feat_matrix[idx_1min]
    ts_1min = day_timestamps[idx_1min] if len(day_timestamps) >= len(idx_1min) else None
    
    n_min = len(feat_1min)
    if n_min < WINDOW_MIN:
        continue
    
    # Get flares for this day
    day_flare_list = flare_events.get(date, [])
    
    # Slide windows
    day_count = 0
    for start in range(0, n_min - WINDOW_MIN, STEP_MIN):
        window = feat_1min[start:start + WINDOW_MIN]
        if len(window) < WINDOW_MIN:
            break
        
        # Window end timestamp
        if ts_1min is not None and (start + WINDOW_MIN) < len(ts_1min):
            window_end_ts = ts_1min[start + WINDOW_MIN]
        else:
            window_end_ts = day_start_ts + (start + WINDOW_MIN) * 60
        
        # Label: max flare class in NEXT 12 hours after window end
        label = 0
        best_lead_min = 0
        class_map = {"B": 1, "C": 2, "M": 3, "X": 4}
        
        for peak_ts, flare_cls in day_flare_list:
            time_to_flare = peak_ts - window_end_ts  # seconds
            # Flare must be AFTER window end and within 12 hours (43200 sec)
            if 0 < time_to_flare < 43200:
                fc = class_map.get(flare_cls, 0)
                if fc > label:
                    label = fc
                    best_lead_min = time_to_flare / 60.0
        
        all_windows.append(window)
        all_labels.append(label)
        all_lead_times.append(best_lead_min)
        day_count += 1
    
    flare_str = " | ".join(["%s(%s)" % (c, t) for t, c in day_flare_list[:3]]) if day_flare_list else "quiet"
    n_flare_windows = sum(1 for l in all_labels[-day_count:] if l > 0)
    print("  %s: %d windows (%d flare) [%s]" % (date, day_count, n_flare_windows, flare_str))

X = np.array(all_windows, dtype=np.float32)
y = np.array(all_labels, dtype=np.int64)
lt = np.array(all_lead_times, dtype=np.float32)

print("\n[4/5] Raw dataset:")
print("  X: %s  (%.0f-hour windows at 1-min cadence)" % (str(X.shape), X.shape[1]/60))
print("  Distribution:")
for i, name in enumerate(cfg.CLASS_NAMES):
    print("    %s: %d (%.1f%%)" % (name, (y==i).sum(), (y==i).mean()*100))

# Normalize
print("\n[5/5] Normalizing and saving...")
mean = X.reshape(-1, X.shape[2]).mean(axis=0)
std = X.reshape(-1, X.shape[2]).std(axis=0)
std[std == 0] = 1
X = (X - mean) / std

# Save (don't oversample here — let Kaggle script handle it)
os.makedirs(str(cfg.PROCESSED), exist_ok=True)
np.save(str(cfg.PROCESSED / "X_strategic_v2.npy"), X)
np.save(str(cfg.PROCESSED / "y_strategic_v2.npy"), y)
np.save(str(cfg.PROCESSED / "lt_strategic_v2.npy"), lt)
np.save(str(cfg.PROCESSED / "strategic_v2_mean.npy"), mean)
np.save(str(cfg.PROCESSED / "strategic_v2_std.npy"), std)

# Stats
print("\nSaved:")
print("  X_strategic_v2.npy: %s (%.1f MB)" % (str(X.shape), X.nbytes/1e6))
print("  Lead times: min=%.0f max=%.0f avg=%.0f min" % (
    lt[lt>0].min() if (lt>0).any() else 0, lt.max(), lt[lt>0].mean() if (lt>0).any() else 0))
print("  Time: %.1f min" % ((time.time()-t0)/60))
print("\n" + "=" * 60)
print("  DONE! Ready for Kaggle")
print("=" * 60)
