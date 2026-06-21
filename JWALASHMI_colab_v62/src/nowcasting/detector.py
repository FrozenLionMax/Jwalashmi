"""
Solar Flare Nowcasting — Flare Detection Engine
Detects flares in X-ray light curves using peak detection with dynamic thresholds.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


@dataclass
class FlareEvent:
    """A detected solar flare event."""
    start_time: float          # Unix timestamp
    peak_time: float
    stop_time: float
    peak_counts: float         # counts/sec at peak
    background: float          # estimated background at flare time
    duration: float            # seconds
    estimated_class: str       # B, C, M, or X
    confidence: float          # 0-1
    instrument: str = "solexs"
    start_dt: Optional[str] = None
    peak_dt: Optional[str] = None
    stop_dt: Optional[str] = None

    def __post_init__(self):
        self.duration = self.stop_time - self.start_time
        self.start_dt = str(pd.Timestamp(self.start_time, unit="s", tz="UTC"))
        self.peak_dt = str(pd.Timestamp(self.peak_time, unit="s", tz="UTC"))
        self.stop_dt = str(pd.Timestamp(self.stop_time, unit="s", tz="UTC"))


def estimate_background(counts: np.ndarray,
                        window: int = 3600) -> np.ndarray:
    """
    Estimate background level using rolling median.
    Uses a large window to avoid being influenced by flares.
    """
    # Pad to handle edges
    padded = np.pad(counts, window // 2, mode="edge")
    # Rolling median via sorted window approximation (fast)
    bg = pd.Series(padded).rolling(window, center=True, min_periods=1).median().values
    bg = bg[window // 2: window // 2 + len(counts)]
    return bg


def estimate_noise(counts: np.ndarray, background: np.ndarray,
                   window: int = 3600) -> np.ndarray:
    """Estimate noise level as rolling standard deviation of (counts - background)."""
    residual = counts - background
    padded = np.pad(residual, window // 2, mode="edge")
    noise = pd.Series(padded).rolling(window, center=True, min_periods=1).std().values
    noise = noise[window // 2: window // 2 + len(counts)]
    # Floor to avoid division by zero
    noise = np.maximum(noise, 1.0)
    return noise


def classify_flare(peak_counts: float, background: float,
                   instrument: str = "solexs") -> tuple:
    """
    Classify a flare based on peak counts above background.

    Returns: (class_str, confidence)
    """
    net_counts = peak_counts - background
    thresholds = (cfg.SOLEXS_CLASS_THRESHOLDS if instrument == "solexs"
                  else cfg.HELOS_CLASS_THRESHOLDS)

    if net_counts >= thresholds["X"]:
        cls = "X"
        conf = min(1.0, net_counts / (thresholds["X"] * 2))
    elif net_counts >= thresholds["M"]:
        cls = "M"
        conf = 0.7 + 0.3 * (net_counts - thresholds["M"]) / (thresholds["X"] - thresholds["M"])
    elif net_counts >= thresholds["C"]:
        cls = "C"
        conf = 0.5 + 0.2 * (net_counts - thresholds["C"]) / (thresholds["M"] - thresholds["C"])
    elif net_counts >= thresholds["B"]:
        cls = "B"
        conf = 0.3 + 0.2 * (net_counts - thresholds["B"]) / (thresholds["C"] - thresholds["B"])
    else:
        cls = "B"
        conf = 0.1

    return cls, float(np.clip(conf, 0, 1))


def find_flare_boundaries(counts: np.ndarray, timestamps: np.ndarray,
                          peak_idx: int, background: np.ndarray,
                          noise: np.ndarray, sigma: float = 2.0) -> tuple:
    """
    Find the start and stop indices of a flare around a detected peak.
    Start: where flux first exceeds background + sigma*noise before peak.
    Stop: where flux drops below background + sigma*noise after peak.
    """
    threshold = background + sigma * noise

    # Search backward for start
    start_idx = peak_idx
    for i in range(peak_idx - 1, max(0, peak_idx - 7200), -1):  # max 2 hours back
        if counts[i] <= threshold[i]:
            start_idx = i + 1
            break
    else:
        start_idx = max(0, peak_idx - 7200)

    # Search forward for stop
    stop_idx = peak_idx
    for i in range(peak_idx + 1, min(len(counts), peak_idx + 7200)):  # max 2 hours forward
        if counts[i] <= threshold[i]:
            stop_idx = i
            break
    else:
        stop_idx = min(len(counts) - 1, peak_idx + 7200)

    return start_idx, stop_idx


def detect_flares(df: pd.DataFrame,
                  instrument: str = "solexs",
                  count_col: str = "counts",
                  time_col: str = "timestamp",
                  sigma: float = 3.0,
                  min_distance: int = cfg.MIN_PEAK_DISTANCE,
                  min_duration: int = cfg.MIN_FLARE_DURATION) -> List[FlareEvent]:
    """
    Detect solar flares in a light curve DataFrame.

    Args:
        df: DataFrame with at least count_col and time_col columns
        instrument: 'solexs' or 'hel1os'
        count_col: column name with count rates
        time_col: column name with timestamps
        sigma: detection threshold in units of background noise
        min_distance: minimum samples between peaks
        min_duration: minimum flare duration in seconds

    Returns:
        List of FlareEvent objects sorted by peak time
    """
    counts = df[count_col].values.astype(np.float64)
    timestamps = df[time_col].values.astype(np.float64)

    # Replace NaN with 0 for peak detection
    counts_clean = np.nan_to_num(counts, nan=0.0)

    # Smooth lightly to reduce noise spikes (3-sec boxcar)
    counts_smooth = uniform_filter1d(counts_clean, size=3)

    # Estimate background and noise
    bg = estimate_background(counts_clean)
    noise = estimate_noise(counts_clean, bg)

    # Dynamic threshold
    threshold = bg + sigma * noise

    # Find peaks above threshold
    # Use scalar prominence based on median noise to avoid array issues
    median_noise = float(np.median(noise))
    peak_indices, properties = find_peaks(
        counts_smooth,
        height=threshold,
        distance=min_distance,
        prominence=median_noise * 2,
    )

    # Build flare events
    flares = []
    for peak_idx in peak_indices:
        start_idx, stop_idx = find_flare_boundaries(
            counts_clean, timestamps, peak_idx, bg, noise
        )

        duration = timestamps[stop_idx] - timestamps[start_idx]
        if duration < min_duration:
            continue

        peak_counts = counts_clean[peak_idx]
        bg_at_peak = bg[peak_idx]

        cls, conf = classify_flare(peak_counts, bg_at_peak, instrument)

        flare = FlareEvent(
            start_time=timestamps[start_idx],
            peak_time=timestamps[peak_idx],
            stop_time=timestamps[stop_idx],
            peak_counts=peak_counts,
            background=bg_at_peak,
            duration=duration,
            estimated_class=cls,
            confidence=conf,
            instrument=instrument,
        )
        flares.append(flare)

    return sorted(flares, key=lambda f: f.peak_time)


def flares_to_dataframe(flares: List[FlareEvent]) -> pd.DataFrame:
    """Convert a list of FlareEvent objects to a DataFrame."""
    if not flares:
        return pd.DataFrame()

    records = []
    for f in flares:
        records.append({
            "start_time": f.start_time,
            "peak_time": f.peak_time,
            "stop_time": f.stop_time,
            "start_dt": f.start_dt,
            "peak_dt": f.peak_dt,
            "stop_dt": f.stop_dt,
            "peak_counts": f.peak_counts,
            "background": f.background,
            "duration_sec": f.duration,
            "estimated_class": f.estimated_class,
            "confidence": f.confidence,
            "instrument": f.instrument,
        })

    return pd.DataFrame(records)


def build_unified_catalog(solexs_flares: List[FlareEvent],
                          hel1os_flares: List[FlareEvent],
                          match_window: float = 300.0) -> pd.DataFrame:
    """
    Merge flare detections from both instruments into a unified catalog.
    Flares within match_window seconds of each other are considered the same event.
    """
    df_s = flares_to_dataframe(solexs_flares)
    df_h = flares_to_dataframe(hel1os_flares)

    if df_s.empty and df_h.empty:
        return pd.DataFrame()
    if df_h.empty:
        df_s["has_solexs"] = True
        df_s["has_hel1os"] = False
        return df_s
    if df_s.empty:
        df_h["has_solexs"] = False
        df_h["has_hel1os"] = True
        return df_h

    # Match by peak time proximity
    matched_h = set()
    records = []

    for _, row_s in df_s.iterrows():
        best_match = None
        best_dt = match_window

        for idx_h, row_h in df_h.iterrows():
            if idx_h in matched_h:
                continue
            dt = abs(row_s["peak_time"] - row_h["peak_time"])
            if dt < best_dt:
                best_dt = dt
                best_match = idx_h

        record = row_s.to_dict()
        record["has_solexs"] = True

        if best_match is not None:
            matched_h.add(best_match)
            record["has_hel1os"] = True
            record["hel1os_peak_counts"] = df_h.loc[best_match, "peak_counts"]
            record["hel1os_class"] = df_h.loc[best_match, "estimated_class"]
        else:
            record["has_hel1os"] = False

        records.append(record)

    # Add unmatched HEL1OS flares
    for idx_h, row_h in df_h.iterrows():
        if idx_h not in matched_h:
            record = row_h.to_dict()
            record["has_solexs"] = False
            record["has_hel1os"] = True
            records.append(record)

    catalog = pd.DataFrame(records)
    catalog.sort_values("peak_time", inplace=True)
    catalog.reset_index(drop=True, inplace=True)
    return catalog


# ═══════════════════════════════════════════════════════════════
#  Quick Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.data.fits_loader import find_solexs_files, load_solexs_lightcurve

    files = find_solexs_files()
    if files:
        # Test on first day
        print(f"Testing on {files[0]['date']}...")
        df = load_solexs_lightcurve(files[0]["lc_path"])
        flares = detect_flares(df, instrument="solexs")
        print(f"Detected {len(flares)} flares:")
        for f in flares:
            print(f"  {f.peak_dt} | Class {f.estimated_class} | "
                  f"Peak: {f.peak_counts:.0f} cts/s | "
                  f"Duration: {f.duration:.0f}s | "
                  f"Confidence: {f.confidence:.2f}")
