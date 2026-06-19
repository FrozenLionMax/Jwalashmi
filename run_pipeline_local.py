"""Run data pipeline steps 1-4 locally to generate X_tactical.npy with HEL1OS."""
import sys, time, os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
import config as cfg

t0 = time.time()

# Step 1: Load
print("=" * 60)
print("  STEP 1: Loading Data")
print("=" * 60)
from src.data.fits_loader import (load_solexs_all, load_hel1os_all,
                                   find_solexs_files, find_hel1os_files)
print("  SoLEXS: %d days" % len(find_solexs_files()))
print("  HEL1OS: %d files" % len(find_hel1os_files()))
df_solexs = load_solexs_all()
df_hel1os = load_hel1os_all(detector="cdte1")
print("  SoLEXS rows: %d" % len(df_solexs))
print("  HEL1OS rows: %d" % len(df_hel1os))
print("  Time: %.1fs" % (time.time()-t0))

# Step 2: Nowcast
print("\n" + "=" * 60)
print("  STEP 2: Nowcasting")
print("=" * 60)
from src.nowcasting.detector import detect_flares, build_unified_catalog
all_flares = []
for date, group in df_solexs.groupby("date"):
    flares = detect_flares(group.reset_index(drop=True), instrument="solexs")
    all_flares.extend(flares)
    if flares:
        print("  SoLEXS %s: %d flares" % (date, len(flares)))
if not df_hel1os.empty:
    broad_col = [c for c in df_hel1os.columns if "1.80KEV_TO_90" in c]
    if broad_col:
        for date, group in df_hel1os.groupby("date"):
            fl = detect_flares(group.reset_index(drop=True),
                               instrument="hel1os", count_col=broad_col[0])
            all_flares.extend(fl)
            if fl:
                print("  HEL1OS %s: %d flares" % (date, len(fl)))
catalog = build_unified_catalog(all_flares, [])
catalog.to_csv(str(cfg.CATALOG_CSV), index=False)
print("  Catalog: %d events" % len(catalog))
dist = {}
for c in catalog["estimated_class"]:
    dist[c] = dist.get(c, 0) + 1
print("  Distribution: %s" % dist)

# Step 3: Features (SoLEXS + HEL1OS)
print("\n" + "=" * 60)
print("  STEP 3: Features (SoLEXS + HEL1OS)")
print("=" * 60)
from src.features.physics_features import compute_all_features, get_feature_columns

hel1os_dates = set()
if not df_hel1os.empty:
    hel1os_dates = set(df_hel1os["date"].unique())
    print("  HEL1OS dates: %d" % len(hel1os_dates))

hel1os_ctr_cols = [c for c in df_hel1os.columns if c.startswith("ctr_")] if not df_hel1os.empty else []
hxr_soft_col = next((c for c in hel1os_ctr_cols if "5.00KEV_TO_20" in c), None)
hxr_hard_col = next((c for c in hel1os_ctr_cols if "30.00KEV_TO_40" in c), None)
hxr_medium_col = next((c for c in hel1os_ctr_cols if "20.00KEV_TO_30" in c), None)

if hxr_soft_col:
    print("  HEL1OS bands found: soft, hard, medium")
    hel1os_feat_cols = {
        "hxr_soft": hxr_soft_col,
        "hxr_hard": hxr_hard_col,
        "hxr_medium": hxr_medium_col,
    }
else:
    hel1os_feat_cols = None
    print("  HEL1OS: No matching energy bands")

dfs = []
merged_count = 0
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
        merged_count += 1
        print("  %s: SoLEXS+HEL1OS merged (%d rows)" % (date, len(day_merged)))
    else:
        day_feat = compute_all_features(day_solexs)
        for fcol in ["feat_hard_soft_ratio", "feat_neupert", "feat_spectral_hardness"]:
            if fcol not in day_feat.columns:
                day_feat[fcol] = 0.0
    dfs.append(day_feat)

df_feat = pd.concat(dfs, ignore_index=True)
feat_cols = get_feature_columns(df_feat)
print("  Merged dates: %d" % merged_count)
print("  Features: %d - %s" % (len(feat_cols), feat_cols))

# Step 4: Windowing
print("\n" + "=" * 60)
print("  STEP 4: Windowing")
print("=" * 60)
from src.features.windowing import (create_windows, normalize_features,
                                     balance_classes, print_window_stats)
X, y, meta = create_windows(df_feat, feat_cols, flare_catalog=catalog)
print_window_stats(y, meta)
X_norm, mean, std = normalize_features(X)
X_bal, y_bal, meta_bal = balance_classes(X_norm, y, meta, max_ratio=10)
print("  After balancing: %d windows" % len(X_bal))
print_window_stats(y_bal, meta_bal)

lead_times = np.array([
    m.lead_time / 60 if m.lead_time is not None else 0
    for m in meta_bal
], dtype=np.float32)

# Save
os.makedirs(str(cfg.PROCESSED), exist_ok=True)
np.save(str(cfg.PROCESSED / "X_tactical.npy"), X_bal)
np.save(str(cfg.PROCESSED / "y_tactical.npy"), y_bal)
np.save(str(cfg.PROCESSED / "lead_times.npy"), lead_times)
np.save(str(cfg.PROCESSED / "feature_mean.npy"), mean)
np.save(str(cfg.PROCESSED / "feature_std.npy"), std)

print("\n" + "=" * 60)
print("  DONE!")
print("=" * 60)
print("  X_tactical shape: %s" % str(X_bal.shape))
print("  Classes: %s" % str(cfg.CLASS_NAMES))
for i, n in enumerate(cfg.CLASS_NAMES):
    print("    %s: %d" % (n, (y_bal == i).sum()))
print("  Total time: %.1f min" % ((time.time()-t0)/60))
