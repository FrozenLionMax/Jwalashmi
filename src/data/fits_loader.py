"""
Solar Flare Early Warning System — FITS Data Loader
Reads SoLEXS and HEL1OS Level-1 FITS files into pandas DataFrames.
"""
import glob
import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


# ═══════════════════════════════════════════════════════════════
#  SoLEXS Loader
# ═══════════════════════════════════════════════════════════════

def load_solexs_lightcurve(fits_path: str) -> pd.DataFrame:
    """
    Load a single SoLEXS Level-1 light curve (.lc.gz) file.

    Returns DataFrame with columns:
        - timestamp : float   (Unix time)
        - datetime  : pd.Timestamp (UTC)
        - counts    : float   (counts/sec in 2-22 keV)
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        time_unix = data["TIME"].astype(np.float64)
        counts = data["COUNTS"].astype(np.float64)

    df = pd.DataFrame({
        "timestamp": time_unix,
        "datetime": pd.to_datetime(time_unix, unit="s", utc=True),
        "counts": counts,
    })
    df["instrument"] = "solexs"
    df["energy_band"] = "2-22keV"
    return df


def load_solexs_gti(gti_path: str) -> List[tuple]:
    """Load Good Time Intervals from a SoLEXS GTI file."""
    with fits.open(gti_path) as hdul:
        if len(hdul) > 1 and hdul[1].data is not None:
            starts = hdul[1].data["START"]
            stops = hdul[1].data["STOP"]
            return list(zip(starts, stops))
    return []


def find_solexs_files(data_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Find all SoLEXS Level-1 data directories and their files.

    Returns list of dicts with keys: date, lc_path, pi_path, gti_path
    """
    data_dir = data_dir or str(cfg.SOLEXS_RAW)
    results = []

    for day_dir in sorted(glob.glob(os.path.join(data_dir, "AL1_SLX_L1_*"))):
        date_str = os.path.basename(day_dir).split("_")[3]  # YYYYMMDD
        sdd2_dir = os.path.join(day_dir, "SDD2")
        if not os.path.isdir(sdd2_dir):
            continue

        entry = {"date": date_str, "dir": sdd2_dir}
        for f in os.listdir(sdd2_dir):
            full = os.path.join(sdd2_dir, f)
            if f.endswith(".lc.gz"):
                entry["lc_path"] = full
            elif f.endswith(".pi.gz"):
                entry["pi_path"] = full
            elif f.endswith(".gti.gz"):
                entry["gti_path"] = full
        if "lc_path" in entry:
            results.append(entry)

    return results


def load_solexs_all(data_dir: Optional[str] = None,
                    apply_gti: bool = True) -> pd.DataFrame:
    """
    Load ALL SoLEXS light curves into a single DataFrame.
    Optionally masks data outside Good Time Intervals as NaN.
    """
    files = find_solexs_files(data_dir)
    dfs = []

    for entry in tqdm(files, desc="Loading SoLEXS"):
        df = load_solexs_lightcurve(entry["lc_path"])
        df["date"] = entry["date"]

        if apply_gti and "gti_path" in entry:
            gti = load_solexs_gti(entry["gti_path"])
            if gti:
                mask = np.zeros(len(df), dtype=bool)
                for start, stop in gti:
                    mask |= (df["timestamp"] >= start) & (df["timestamp"] <= stop)
                df.loc[~mask, "counts"] = np.nan

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# ═══════════════════════════════════════════════════════════════
#  HEL1OS Loader
# ═══════════════════════════════════════════════════════════════

def load_hel1os_lightcurve(fits_path: str) -> pd.DataFrame:
    """
    Load a single HEL1OS lightcurve FITS file (e.g. lightcurve_cdte1.fits).

    Returns DataFrame with columns:
        - timestamp  : float (Unix time, converted from MJD)
        - datetime   : pd.Timestamp (UTC)
        - ctr_BAND   : float (count rate for each energy band)
        - err_BAND   : float (statistical error for each band)
    """
    with fits.open(fits_path) as hdul:
        # Detect detector type from filename
        fname = os.path.basename(fits_path).lower()
        if "cdte1" in fname:
            det = "cdte1"
        elif "cdte2" in fname:
            det = "cdte2"
        elif "czt1" in fname:
            det = "czt1"
        elif "czt2" in fname:
            det = "czt2"
        else:
            det = "unknown"

        all_data = {}
        ref_time = None

        for i in range(1, len(hdul)):
            hdu = hdul[i]
            band_name = hdu.name  # e.g. CDTE1_LC_BAND_5.00KEV_TO_20.00KEV
            data = hdu.data

            # Extract band identifier (e.g. "5.00KEV_TO_20.00KEV")
            parts = band_name.split("BAND_")
            band_id = parts[1] if len(parts) > 1 else band_name

            mjd = data["MJD"].astype(np.float64)
            ctr = data["CTR"].astype(np.float64)
            err = data["STAT_ERR"].astype(np.float64)

            # Convert MJD to Unix timestamp
            # MJD = JD - 2400000.5; Unix epoch (1970-01-01) = JD 2440587.5 = MJD 40587.0
            unix_time = (mjd - 40587.0) * 86400.0

            band_df = pd.DataFrame({
                "timestamp": unix_time,
                f"ctr_{band_id}": ctr,
                f"err_{band_id}": err,
            })

            if ref_time is None:
                ref_time = band_df[["timestamp"]].copy()
                all_data["timestamp"] = unix_time
            # Merge on closest timestamp (they may differ slightly between bands)
            all_data[f"ctr_{band_id}"] = np.interp(
                all_data["timestamp"], unix_time, ctr
            )
            all_data[f"err_{band_id}"] = np.interp(
                all_data["timestamp"], unix_time, err
            )

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["instrument"] = "hel1os"
    df["detector"] = det
    return df


def find_hel1os_files(data_dir: Optional[str] = None) -> List[Dict]:
    """
    Find all extracted HEL1OS lightcurve files.

    Returns list of dicts with keys: date, detector, path
    """
    data_dir = data_dir or str(cfg.HEL1OS_RAW)
    results = []

    for lc_path in sorted(glob.glob(
        os.path.join(data_dir, "**", "lightcurve_*.fits"), recursive=True
    )):
        fname = os.path.basename(lc_path)
        det = fname.replace("lightcurve_", "").replace(".fits", "")

        # Extract date from path (e.g. .../2024/10/03/HLS_.../cdte/...)
        parts = Path(lc_path).parts
        try:
            # Find year-like part
            for j, p in enumerate(parts):
                if p.isdigit() and len(p) == 4 and int(p) > 2000:
                    date_str = f"{parts[j]}{parts[j+1]}{parts[j+2]}"
                    break
            else:
                date_str = "unknown"
        except (IndexError, ValueError):
            date_str = "unknown"

        results.append({
            "date": date_str,
            "detector": det,
            "path": lc_path,
        })

    return results


def load_hel1os_all(data_dir: Optional[str] = None,
                    detector: str = "cdte1") -> pd.DataFrame:
    """
    Load ALL HEL1OS light curves for a specific detector into one DataFrame.

    Args:
        detector: one of 'cdte1', 'cdte2', 'czt1', 'czt2'
    """
    files = find_hel1os_files(data_dir)
    files = [f for f in files if f["detector"] == detector]
    dfs = []

    for entry in tqdm(files, desc=f"Loading HEL1OS {detector}"):
        try:
            df = load_hel1os_lightcurve(entry["path"])
            df["date"] = entry["date"]
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {entry['path']}: {e}")

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# ═══════════════════════════════════════════════════════════════
#  Unified Loader (merges both instruments by time)
# ═══════════════════════════════════════════════════════════════

def load_unified(solexs_dir: Optional[str] = None,
                 hel1os_dir: Optional[str] = None,
                 hel1os_detector: str = "cdte1") -> pd.DataFrame:
    """
    Load and merge SoLEXS + HEL1OS data by timestamp.
    For timestamps where only one instrument has data, the other columns are NaN.
    """
    df_solexs = load_solexs_all(solexs_dir)
    df_hel1os = load_hel1os_all(hel1os_dir, detector=hel1os_detector)

    if df_hel1os.empty:
        return df_solexs

    # Rename SoLEXS counts to avoid collision
    df_solexs = df_solexs.rename(columns={"counts": "solexs_counts"})

    # Round timestamps to nearest second for merging
    df_solexs["ts_round"] = df_solexs["timestamp"].round(0)
    df_hel1os["ts_round"] = df_hel1os["timestamp"].round(0)

    # Select HEL1OS columns to merge
    hel1os_cols = ["ts_round"] + [c for c in df_hel1os.columns
                                   if c.startswith("ctr_") or c.startswith("err_")]

    merged = pd.merge(
        df_solexs,
        df_hel1os[hel1os_cols].drop_duplicates(subset="ts_round"),
        on="ts_round",
        how="outer",
    )
    merged.sort_values("ts_round", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Fill timestamp/datetime from ts_round where missing
    merged["timestamp"] = merged["timestamp"].fillna(merged["ts_round"])
    merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True)

    return merged.drop(columns=["ts_round"])


# ═══════════════════════════════════════════════════════════════
#  Quick Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("SoLEXS Files Found:")
    for f in find_solexs_files():
        print(f"  {f['date']} → {os.path.basename(f['lc_path'])}")

    print("\nHEL1OS Files Found:")
    for f in find_hel1os_files():
        print(f"  {f['date']} / {f['detector']} → {os.path.basename(f['path'])}")

    print("\n" + "=" * 60)
    print("Loading first SoLEXS day...")
    files = find_solexs_files()
    if files:
        df = load_solexs_lightcurve(files[0]["lc_path"])
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Time range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
        print(f"  Counts: min={df['counts'].min():.0f}, max={df['counts'].max():.0f}")

    print("\nLoading first HEL1OS file...")
    hfiles = find_hel1os_files()
    if hfiles:
        df = load_hel1os_lightcurve(hfiles[0]["path"])
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Detector: {df['detector'].iloc[0]}")
