"""
Solar Flare Early Warning System — Physics-Informed Feature Engineering
Computes domain-specific features that encode solar physics knowledge.
These features are what differentiate us from generic ML approaches.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


def compute_derivative(series: np.ndarray, dt: float = 1.0,
                       smooth_window: int = 11) -> np.ndarray:
    """
    Compute smoothed time derivative of a signal.
    Uses Savitzky-Golay filter for noise-robust differentiation.
    """
    if len(series) < smooth_window:
        smooth_window = max(3, len(series) // 2 * 2 + 1)
    try:
        return savgol_filter(series, smooth_window, polyorder=2, deriv=1, delta=dt)
    except Exception:
        return np.gradient(series, dt)


def compute_rolling_max_ratio(counts: np.ndarray,
                               short_window: int = 300,
                               long_window: int = 1800) -> np.ndarray:
    """
    Ratio of short-term max to long-term median.
    Spikes indicate sudden flux increases → possible flare onset.
    
    Physics: Solar flares show impulsive rise over seconds/minutes
    against a slowly varying background. A high ratio means
    "something just spiked" — classic precursor signature.
    """
    s = pd.Series(counts)
    short_max = s.rolling(short_window, min_periods=1, center=False).max()
    long_median = s.rolling(long_window, min_periods=1, center=False).median()
    long_median = long_median.replace(0, np.nan).fillna(1.0)
    return (short_max / long_median).values


def compute_background_slope(counts: np.ndarray,
                              window: int = 1800) -> np.ndarray:
    """
    Linear regression slope of flux over a trailing window.
    
    Physics: Before major flares, the active region gradually heats up,
    causing a slow background rise (pre-flare thermal buildup).
    A positive slope = thermal energy is building up.
    
    Vectorized using cumsum trick for O(N) instead of O(N*W).
    """
    n = len(counts)
    if n < window:
        return np.zeros(n)

    # Vectorized linear regression slope using cumulative sums
    # slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    # where x = 0,1,...,W-1 for each window
    
    W = window
    x = np.arange(W, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    denom = W * sum_x2 - sum_x ** 2
    
    # Rolling sum of y and sum of x*y using cumsum
    cs_y = np.cumsum(counts)
    # For sum(x_i * y_i) in trailing window: use weighted cumsum
    # x_i in window [i-W, i) = 0,1,...,W-1
    # sum(x * y) = sum(j * y[i-W+j]) for j=0..W-1
    #            = sum(j * y_k) where k = i-W..i-1 and j = k - (i-W)
    # This is equivalent to: sum(k * y_k) - (i-W)*sum(y_k)
    
    k_arr = np.arange(n, dtype=np.float64)
    cs_ky = np.cumsum(k_arr * counts)
    
    slopes = np.zeros(n)
    for_range = np.arange(W, n)
    
    sum_y = cs_y[for_range] - cs_y[for_range - W]
    sum_ky = cs_ky[for_range] - cs_ky[for_range - W]
    # Convert to local x coordinates: sum(x*y) = sum_ky - (i-W)*sum_y
    sum_xy = sum_ky - (for_range - W).astype(np.float64) * sum_y
    
    slopes[for_range] = (W * sum_xy - sum_x * sum_y) / max(denom, 1e-10)
    
    # Fill initial values
    slopes[:window] = slopes[window] if window < n else 0
    return slopes


def compute_energy_integral(counts: np.ndarray) -> np.ndarray:
    """
    Cumulative energy integral (running sum of counts).
    
    Physics: The total energy deposited over time tracks the
    overall thermal energy in the corona. Sudden acceleration
    in the integral indicates rapid energy release.
    """
    # Reset integral at the start of each day (every 86400 points)
    result = np.zeros_like(counts)
    day_size = 86400
    for start in range(0, len(counts), day_size):
        end = min(start + day_size, len(counts))
        segment = np.nan_to_num(counts[start:end], nan=0.0)
        result[start:end] = np.cumsum(segment)
    return result


def compute_hard_soft_ratio(hxr: np.ndarray, sxr: np.ndarray,
                             smooth: int = 30) -> np.ndarray:
    """
    Hard X-ray / Soft X-ray ratio.
    
    Physics: During the impulsive phase of a flare, non-thermal
    electrons produce hard X-rays. The hard/soft ratio spikes
    BEFORE the thermal peak — this is the Neupert effect.
    A rising ratio = non-thermal acceleration is beginning.
    """
    sxr_safe = np.maximum(sxr, 1.0)
    ratio = hxr / sxr_safe
    # Smooth to reduce noise
    s = pd.Series(ratio)
    return s.rolling(smooth, min_periods=1, center=True).mean().values


def compute_neupert_derivative(sxr: np.ndarray, hxr: np.ndarray,
                                smooth: int = 30) -> np.ndarray:
    """
    Neupert effect indicator: correlation between d(SXR)/dt and HXR.
    
    Physics: The Neupert effect states that the time derivative of
    soft X-ray flux is proportional to the hard X-ray flux.
    When this correlation is high, non-thermal energy is being
    converted to thermal energy — a flare is in its impulsive phase.
    
    We compute the rolling correlation between d(SXR)/dt and HXR
    as our feature. High values (>0.7) strongly indicate an active flare.
    """
    dsxr_dt = compute_derivative(sxr)
    
    # Rolling Pearson correlation over smooth window
    df = pd.DataFrame({"dsxr": dsxr_dt, "hxr": hxr})
    corr = df["dsxr"].rolling(smooth * 10, min_periods=smooth).corr(df["hxr"])
    return corr.fillna(0).values


def compute_spectral_hardness(hard_band: np.ndarray,
                               soft_band: np.ndarray,
                               smooth: int = 30) -> np.ndarray:
    """
    Spectral hardness ratio within HEL1OS bands.
    
    Physics: During flare onset, the spectrum hardens (more
    high-energy photons relative to low-energy). A rising
    hardness ratio = electron acceleration is intensifying.
    
    Typically: HEL1OS(30-60keV) / HEL1OS(8-20keV)
    """
    soft_safe = np.maximum(soft_band, 0.1)
    ratio = hard_band / soft_safe
    s = pd.Series(ratio)
    return s.rolling(smooth, min_periods=1, center=True).mean().values


def compute_qpp_power(counts: np.ndarray,
                       window: int = 600,
                       period_range: tuple = (10, 300)) -> np.ndarray:
    """
    Quasi-Periodic Pulsation (QPP) spectral power.
    
    Physics: Solar flares often exhibit quasi-periodic pulsations
    with periods of 10-300 seconds. Detecting QPP power before
    the main flare peak can serve as an early warning. QPPs arise
    from MHD oscillations in coronal loops that are about to
    reconnect and release energy.
    
    We compute the FFT power in the 10-300s period band over
    a rolling window and return the peak spectral power.
    """
    result = np.zeros(len(counts))
    half_win = window // 2

    for i in range(half_win, len(counts) - half_win, 100):  # stride=100 for speed
        segment = counts[i - half_win:i + half_win]
        segment = np.nan_to_num(segment, nan=0.0)

        # Remove trend
        segment = segment - np.mean(segment)

        if np.std(segment) < 1e-6:
            continue

        # FFT
        freqs = rfftfreq(len(segment), d=cfg.SOLEXS_CADENCE)
        power = np.abs(rfft(segment)) ** 2

        # Select QPP frequency band
        freq_lo = 1.0 / period_range[1]  # low freq = long period
        freq_hi = 1.0 / period_range[0]  # high freq = short period
        mask = (freqs >= freq_lo) & (freqs <= freq_hi)

        if np.any(mask):
            qpp_power = np.max(power[mask])
            # Normalize by total power
            total_power = np.sum(power[1:]) + 1e-10
            result[i] = qpp_power / total_power

    # Interpolate between computed points
    nonzero = result > 0
    if np.any(nonzero):
        indices = np.arange(len(result))
        result = np.interp(indices, indices[nonzero], result[nonzero])

    return result


def compute_all_features(df: pd.DataFrame,
                          count_col: str = "counts",
                          hel1os_cols: Optional[dict] = None) -> pd.DataFrame:
    """
    Compute all physics-informed features for a DataFrame.
    
    Args:
        df: DataFrame with at minimum a count_col column
        count_col: name of the primary flux column (SoLEXS counts)
        hel1os_cols: dict mapping feature names to column names, e.g.:
            {
                "hxr_soft": "ctr_5.00KEV_TO_20.00KEV",
                "hxr_hard": "ctr_30.00KEV_TO_40.00KEV",
                "hxr_medium": "ctr_20.00KEV_TO_30.00KEV",
            }
    
    Returns:
        DataFrame with all original columns plus feature columns
    """
    result = df.copy()
    counts = np.nan_to_num(result[count_col].values, nan=0.0).astype(np.float64)

    # ── SoLEXS-only features (always available) ──────────────
    result["feat_derivative"] = compute_derivative(counts)
    result["feat_rolling_max_ratio"] = compute_rolling_max_ratio(counts)
    result["feat_bg_slope"] = compute_background_slope(counts)
    result["feat_energy_integral"] = compute_energy_integral(counts)
    result["feat_qpp_power"] = compute_qpp_power(counts)

    # Normalized flux (relative to rolling background)
    bg = pd.Series(counts).rolling(cfg.BG_WINDOW, min_periods=1).median().values
    bg_safe = np.maximum(bg, 1.0)
    result["feat_norm_flux"] = counts / bg_safe

    # ── Long-range precursor features (Tier 1: hours ahead) ──
    # 2-hour background slope — captures slow thermal buildup before major flares
    result["feat_long_slope"] = compute_background_slope(counts, window=7200)

    # Flux acceleration (2nd derivative) — rate of change of the rise rate
    deriv = result["feat_derivative"].values
    result["feat_acceleration"] = compute_derivative(deriv, smooth_window=31)

    # Long-range flux ratio — 30-min max vs 2-hour median
    result["feat_long_ratio"] = compute_rolling_max_ratio(
        counts, short_window=1800, long_window=7200
    )

    # ── HEL1OS features (when available) ─────────────────────
    if hel1os_cols is not None:
        hxr_soft_col = hel1os_cols.get("hxr_soft")
        hxr_hard_col = hel1os_cols.get("hxr_hard")
        hxr_medium_col = hel1os_cols.get("hxr_medium")

        if hxr_soft_col and hxr_soft_col in result.columns:
            hxr_soft = np.nan_to_num(result[hxr_soft_col].values, nan=0.0)

            # Hard/Soft ratio (HEL1OS soft band vs SoLEXS)
            result["feat_hard_soft_ratio"] = compute_hard_soft_ratio(hxr_soft, counts)

            # Neupert derivative correlation
            result["feat_neupert"] = compute_neupert_derivative(counts, hxr_soft)

        if hxr_hard_col and hxr_soft_col:
            if hxr_hard_col in result.columns and hxr_soft_col in result.columns:
                hard = np.nan_to_num(result[hxr_hard_col].values, nan=0.0)
                soft = np.nan_to_num(result[hxr_soft_col].values, nan=0.0)
                result["feat_spectral_hardness"] = compute_spectral_hardness(hard, soft)

    # ── Fill any remaining NaN in features ────────────────────
    feat_cols = [c for c in result.columns if c.startswith("feat_")]
    result[feat_cols] = result[feat_cols].fillna(0)

    return result


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of computed feature column names."""
    return [c for c in df.columns if c.startswith("feat_")]


# ═══════════════════════════════════════════════════════════════
#  Quick Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.data.fits_loader import find_solexs_files, load_solexs_lightcurve

    files = find_solexs_files()
    if files:
        print(f"Computing features for {files[0]['date']}...")
        df = load_solexs_lightcurve(files[0]["lc_path"])
        df_feat = compute_all_features(df)
        feat_cols = get_feature_columns(df_feat)
        print(f"\nComputed {len(feat_cols)} features:")
        for c in feat_cols:
            vals = df_feat[c].values
            print(f"  {c:30s}  min={np.nanmin(vals):12.4f}  "
                  f"max={np.nanmax(vals):12.4f}  mean={np.nanmean(vals):12.4f}")
