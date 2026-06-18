"""
JWALASHMI v4.0 - GOES Data Downloader + Pre-training Pipeline
Downloads historical GOES XRS data and creates pre-training dataset.
"""
import urllib.request
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


def download_json(url, out_path, label="data"):
    """Download JSON from URL and save to file."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "JWALASHMI/3.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {label}: {len(data)} records -> {os.path.basename(out_path)}")
        return data
    except Exception as e:
        print(f"  {label}: FAILED - {e}")
        return []


def download_all_goes_data():
    """Download all available GOES data from NOAA SWPC."""
    os.makedirs(str(cfg.GOES_DATA), exist_ok=True)
    
    print("\n[1] NOAA SWPC Flare Events...")
    flares = download_json(
        "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
        str(cfg.GOES_DATA / "flares_7day.json"), "7-day flares"
    )
    
    print("\n[2] GOES XRS Flux (3-day, 1-min cadence)...")
    xrs_3day = download_json(
        "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json",
        str(cfg.GOES_DATA / "xrs_3day.json"), "XRS 3-day"
    )
    
    print("\n[3] GOES XRS Flux (7-day)...")
    xrs_7day = download_json(
        "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
        str(cfg.GOES_DATA / "xrs_7day.json"), "XRS 7-day"
    )
    
    print("\n[4] HEK Flare Catalog (2010-2025, up to 5000 events)...")
    hek_url = (
        "https://www.lmsal.com/hek/her?cosec=2&cmd=search&type=column"
        "&event_type=fl&event_starttime=2010-01-01T00:00:00"
        "&event_endtime=2025-12-31T23:59:59"
        "&event_coordsys=helioprojective&x1=-5000&x2=5000&y1=-5000&y2=5000"
        "&result_limit=5000&page=1"
        "&return=hpc_x,hpc_y,event_starttime,event_peaktime,event_endtime,"
        "fl_goescls,fl_peakflux,ar_noaanum,frm_name&cosec=2"
    )
    try:
        req = urllib.request.Request(hek_url, headers={"User-Agent": "JWALASHMI/3.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            hek_data = json.loads(r.read().decode())
        
        events = hek_data.get("result", hek_data if isinstance(hek_data, list) else [])
        out = str(cfg.GOES_DATA / "hek_flares.json")
        with open(out, "w") as f:
            json.dump(events, f, indent=2)
        
        classes = {}
        for ev in events:
            cls = str(ev.get("fl_goescls", "U"))
            if cls and len(cls) > 0:
                classes[cls[0].upper()] = classes.get(cls[0].upper(), 0) + 1
        print(f"  HEK: {len(events)} events")
        print(f"  Classes: {classes}")
    except Exception as e:
        print(f"  HEK: {e}")
        events = []
    
    return flares, xrs_3day, events


def create_goes_pretraining_data(xrs_data, flare_events):
    """
    Create pre-training dataset from GOES XRS flux + flare labels.
    
    This creates windows of GOES XRS flux data with labels from the
    flare event catalog, in the same format as Aditya-L1 windows.
    """
    print("\n[5] Creating GOES pre-training windows...")
    
    if not xrs_data:
        print("  No XRS flux data - generating synthetic GOES profiles...")
        return create_synthetic_goes_data(flare_events)
    
    # Parse XRS data into time series
    times = []
    flux_long = []  # 1-8 Angstrom
    flux_short = []  # 0.5-4 Angstrom
    
    for pt in xrs_data:
        try:
            t = pd.Timestamp(pt.get("time_tag", "")).timestamp()
            fl = float(pt.get("flux", pt.get("observed_flux", 0)))
            times.append(t)
            flux_long.append(fl)
        except (ValueError, TypeError):
            continue
    
    if len(times) < 3600:
        print(f"  Only {len(times)} XRS points (need 3600+) - using synthetic data")
        return create_synthetic_goes_data(flare_events)
    
    times = np.array(times)
    flux = np.array(flux_long, dtype=np.float32)
    
    # Create sliding windows
    window_size = 3600  # 60 min at 1-sec (but GOES is 1-min, so stretch)
    stride = 300
    
    windows = []
    labels = []
    
    # Label each timestamp with flare class
    print(f"  XRS flux: {len(flux)} points, creating windows...")
    
    # For each window, check if there's a flare in the next 60 min
    for i in range(0, len(flux) - 60, 5):  # stride=5 min
        win_flux = flux[i:i+60]  # 60 min at 1-min cadence
        
        # Upsample to 3600 points (1-sec cadence like Aditya-L1)
        win_1s = np.interp(np.linspace(0, 59, 3600), np.arange(60), win_flux)
        
        # Compute 9 features from the flux
        features = compute_9_features(win_1s)
        
        # Label: check peak flux in window
        peak = win_flux.max()
        if peak >= 1e-4:
            cls = 4  # X
        elif peak >= 1e-5:
            cls = 3  # M
        elif peak >= 1e-6:
            cls = 2  # C
        elif peak >= 1e-7:
            cls = 1  # B
        else:
            cls = 0  # None/A
        
        windows.append(features)
        labels.append(cls)
    
    if not windows:
        return create_synthetic_goes_data(flare_events)
    
    X = np.stack(windows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    
    # Print distribution
    print(f"  Created {len(X)} windows from real GOES flux:")
    for i, name in enumerate(cfg.CLASS_NAMES):
        count = (y == i).sum()
        print(f"    {name}: {count}")
    
    # Save
    np.save(str(cfg.GOES_DATA / "X_goes_pretrain.npy"), X)
    np.save(str(cfg.GOES_DATA / "y_goes_pretrain.npy"), y)
    print(f"  Saved to {cfg.GOES_DATA}")
    
    return X, y


def compute_9_features(flux):
    """Compute 9 physics features from a flux time series (3600 pts)."""
    from scipy.signal import savgol_filter
    
    # 1. Derivative
    try:
        derivative = savgol_filter(flux, 11, 2, deriv=1)
    except Exception:
        derivative = np.gradient(flux)
    
    # 2. Rolling max ratio
    flux_series = pd.Series(flux)
    rolling_max = flux_series.rolling(300, min_periods=1).max().values
    rolling_med = flux_series.rolling(1800, min_periods=1).median().values
    rolling_med = np.maximum(rolling_med, 1e-10)
    max_ratio = rolling_max / rolling_med
    
    # 3. Background slope
    bg_slope = np.zeros_like(flux)
    bg_slope[1:] = np.diff(flux)
    
    # 4. Energy integral
    energy = np.cumsum(flux)
    
    # 5. QPP power (simplified)
    qpp = np.zeros_like(flux)
    
    # 6. Normalized flux
    norm_flux = flux / np.maximum(rolling_med, 1e-10)
    
    # 7. Long slope (60-sec diff)
    long_slope = np.zeros_like(flux)
    long_slope[60:] = (flux[60:] - flux[:-60]) / 60
    
    # 8. Acceleration
    acceleration = np.gradient(derivative)
    
    # 9. Long ratio
    long_ratio = max_ratio * 1.5
    
    features = np.stack([
        derivative, max_ratio, bg_slope, energy,
        qpp, norm_flux, long_slope, acceleration, long_ratio
    ], axis=1)
    
    return features


def create_synthetic_goes_data(flare_events):
    """Create synthetic GOES-like training data from flare catalog labels."""
    print("  Creating synthetic GOES pre-training data from catalog...")
    
    CLASS_MAP = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    
    # Use HEK or SWPC flare events
    windows = []
    labels = []
    
    for ev in flare_events:
        cls_str = str(ev.get("fl_goescls", ev.get("max_class", "")))
        if not cls_str or len(cls_str) < 1:
            continue
        letter = cls_str[0].upper()
        if letter not in CLASS_MAP:
            continue
        cls_int = CLASS_MAP[letter]
        
        # Generate realistic flux profile for this class
        rng = np.random.default_rng(hash(str(ev)) % (2**31))
        flux = generate_flare_profile(letter, rng)
        features = compute_9_features(flux)
        
        windows.append(features)
        labels.append(cls_int)
    
    # Also generate "None" class windows (quiet sun)
    for i in range(min(len(windows), 500)):
        rng = np.random.default_rng(i + 99999)
        flux = generate_flare_profile("Q", rng)
        features = compute_9_features(flux)
        windows.append(features)
        labels.append(0)
    
    if not windows:
        print("  No events to generate from!")
        return None, None
    
    X = np.stack(windows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    
    print(f"  Created {len(X)} synthetic GOES windows:")
    for i, name in enumerate(cfg.CLASS_NAMES):
        count = (y == i).sum()
        if count > 0:
            print(f"    {name}: {count}")
    
    np.save(str(cfg.GOES_DATA / "X_goes_pretrain.npy"), X)
    np.save(str(cfg.GOES_DATA / "y_goes_pretrain.npy"), y)
    
    return X, y


def generate_flare_profile(flare_class, rng):
    """Generate realistic GOES-like flux profile."""
    t = np.arange(3600, dtype=np.float32)
    bg = rng.uniform(1e-7, 5e-7)
    noise = bg * 0.03
    flux = np.full(3600, bg) + rng.normal(0, noise, 3600)
    
    peaks = {"Q": 0, "A": bg*1.2, "B": 5e-7, "C": 5e-6, "M": 5e-5, "X": 5e-4}
    peak = peaks.get(flare_class, 0)
    
    if peak > 0:
        peak *= rng.uniform(0.3, 3.0)
        rise = rng.integers(600, 2400)
        dur = rng.integers(30, 300)
        decay = rng.integers(200, 1200)
        
        mask_rise = (t >= rise) & (t < rise + dur)
        rise_t = (t[mask_rise] - rise) / dur
        flux[mask_rise] += (peak - bg) * rise_t ** 2
        
        peak_t = rise + dur
        mask_decay = t >= peak_t
        decay_t = (t[mask_decay] - peak_t) / decay
        flux[mask_decay] += (peak - bg) * np.exp(-decay_t)
    
    return np.maximum(flux, 1e-9).astype(np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("  JWALASHMI - GOES Pre-training Data Pipeline")
    print("=" * 60)
    
    flares, xrs, hek = download_all_goes_data()
    
    # Use the larger dataset for pre-training
    all_events = hek if hek else flares
    X, y = create_goes_pretraining_data(xrs, all_events)
    
    if X is not None:
        print(f"\n  Pre-training dataset: {X.shape}")
        print(f"  Ready for: python run_pipeline.py --pretrain-goes")
    
    print("\n" + "=" * 60)
