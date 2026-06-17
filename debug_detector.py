"""Test nowcasting detection across all days."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fits_loader import find_solexs_files, load_solexs_lightcurve
from src.nowcasting.detector import detect_flares

files = find_solexs_files()
total_flares = 0

print("Nowcasting Detection Results (sigma=3, bg_window=3600):")
print("-" * 70)

for entry in files:
    df = load_solexs_lightcurve(entry["lc_path"])
    flares = detect_flares(df, instrument="solexs")
    total_flares += len(flares)

    if flares:
        classes = [f.estimated_class for f in flares]
        max_peak = max(f.peak_counts for f in flares)
        class_str = ", ".join(f"{c}:{classes.count(c)}" for c in sorted(set(classes)))
        print(f"  {entry['date']}: {len(flares):2d} flares | {class_str:20s} | peak={max_peak:.0f} cts/s")
    else:
        max_cts = df["counts"].max()
        print(f"  {entry['date']}:  0 flares |                      | max_flux={max_cts:.0f} cts/s")

print("-" * 70)
print(f"Total: {total_flares} flares detected across {len(files)} days")
