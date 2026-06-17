"""
Solar Flare Early Warning System — Sliding Window Generator
Creates labeled training windows for the forecasting model.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


@dataclass
class WindowMetadata:
    """Metadata for a single training window."""
    window_start: float   # Unix timestamp
    window_end: float
    label: int            # 0=none, 1=B, 2=C, 3=M, 4=X
    label_name: str
    flare_time: Optional[float] = None  # time of the labeled flare (if any)
    lead_time: Optional[float] = None   # seconds before flare start


CLASS_TO_INT = {"none": 0, "B": 1, "C": 2, "M": 3, "X": 4}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


def create_windows(df: pd.DataFrame,
                   feature_cols: List[str],
                   flare_catalog: Optional[pd.DataFrame] = None,
                   window_size: int = cfg.WINDOW_SIZE,
                   stride: int = cfg.STRIDE,
                   forecast_horizon: int = cfg.FORECAST_HORIZON,
                   time_col: str = "timestamp") -> Tuple[np.ndarray, np.ndarray, List[WindowMetadata]]:
    """
    Create labeled sliding windows for model training/evaluation.
    
    A window is labeled with the class of the FIRST flare that starts
    within the forecast_horizon after the window ends. If no flare
    starts in that horizon, the label is 0 (no flare).
    
    Args:
        df: DataFrame with features already computed
        feature_cols: list of column names to include as input features
        flare_catalog: DataFrame with 'start_time' and 'estimated_class' columns
        window_size: number of samples per window (e.g. 3600 = 60 min at 1s cadence)
        stride: step between windows (e.g. 300 = 5 min)
        forecast_horizon: how far ahead to look for flares (seconds)
        time_col: column name for timestamps
    
    Returns:
        X: numpy array of shape (N, window_size, n_features)
        y: numpy array of shape (N,) with integer class labels
        metadata: list of WindowMetadata objects
    """
    timestamps = df[time_col].values
    features = df[feature_cols].values.astype(np.float32)
    n_samples, n_features = features.shape

    # Pre-process flare catalog
    flare_times = []
    flare_classes = []
    if flare_catalog is not None and not flare_catalog.empty:
        for _, row in flare_catalog.iterrows():
            t = row.get("start_time", row.get("peak_time", None))
            cls = row.get("estimated_class", "B")
            if t is not None:
                flare_times.append(float(t))
                flare_classes.append(CLASS_TO_INT.get(cls, 1))
        flare_times = np.array(flare_times)
        flare_classes = np.array(flare_classes)

    # Generate windows
    X_list = []
    y_list = []
    meta_list = []

    for start_idx in range(0, n_samples - window_size, stride):
        end_idx = start_idx + window_size
        window_end_time = timestamps[end_idx - 1]
        window_start_time = timestamps[start_idx]

        # Check for flares within the forecast horizon AFTER this window
        horizon_start = window_end_time
        horizon_end = window_end_time + forecast_horizon

        label = 0
        label_name = "none"
        flare_time = None
        lead_time = None

        if len(flare_times) > 0:
            # Find flares starting in the forecast horizon
            mask = (flare_times >= horizon_start) & (flare_times <= horizon_end)
            if np.any(mask):
                # Take the highest-class flare in the horizon
                idx_in_horizon = np.where(mask)[0]
                best_idx = idx_in_horizon[np.argmax(flare_classes[idx_in_horizon])]
                label = int(flare_classes[best_idx])
                label_name = INT_TO_CLASS[label]
                flare_time = float(flare_times[best_idx])
                lead_time = float(flare_time - window_end_time)

        # Extract feature window
        window_features = features[start_idx:end_idx]

        # Skip windows with too many NaN
        nan_frac = np.isnan(window_features).mean()
        if nan_frac > 0.3:
            continue

        # Replace remaining NaN with 0
        window_features = np.nan_to_num(window_features, nan=0.0)

        X_list.append(window_features)
        y_list.append(label)
        meta_list.append(WindowMetadata(
            window_start=float(window_start_time),
            window_end=float(window_end_time + 1),
            label=label,
            label_name=label_name,
            flare_time=flare_time,
            lead_time=lead_time,
        ))

    if not X_list:
        return (np.array([]).reshape(0, window_size, n_features),
                np.array([], dtype=np.int64),
                [])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    return X, y, meta_list


def create_strategic_windows(df: pd.DataFrame,
                             feature_cols: list,
                             flare_catalog: pd.DataFrame = None,
                             window_size: int = None,
                             stride: int = None,
                             forecast_horizon: int = None,
                             downsample: int = None,
                             time_col: str = "timestamp"):
    """
    Create labeled windows for Tier 1 (Strategic) forecasting.
    
    Uses 6-hour windows downsampled to 1-min cadence with 10-hour horizon.
    This produces compact windows (360 time steps) for lightweight prediction.
    
    Args:
        df: DataFrame with features
        feature_cols: columns to use
        flare_catalog: with 'start_time' and 'estimated_class'
        window_size: raw window size in seconds (default: 21600 = 6 hours)
        stride: step between windows in seconds (default: 1800 = 30 min)
        forecast_horizon: look-ahead in seconds (default: 36000 = 10 hours)
        downsample: factor to downsample (default: 60 = 1-min cadence)
    
    Returns:
        X: (N, window_size/downsample, n_features) downsampled windows
        y: (N,) binary labels (0=no flare >= C, 1=B, 2=C, 3=M, 4=X)
        metadata: list of WindowMetadata
    """
    window_size = window_size or cfg.STRATEGIC_WINDOW
    stride = stride or cfg.STRATEGIC_STRIDE
    forecast_horizon = forecast_horizon or cfg.STRATEGIC_HORIZON
    downsample = downsample or cfg.STRATEGIC_DOWNSAMPLE

    timestamps = df[time_col].values
    features = df[feature_cols].values.astype(np.float32)
    n_samples, n_features = features.shape

    # Pre-process flare catalog
    flare_times = []
    flare_classes = []
    if flare_catalog is not None and not flare_catalog.empty:
        for _, row in flare_catalog.iterrows():
            t = row.get("start_time", row.get("peak_time", None))
            cls = row.get("estimated_class", "B")
            if t is not None:
                flare_times.append(float(t))
                flare_classes.append(CLASS_TO_INT.get(cls, 1))
        flare_times = np.array(flare_times)
        flare_classes = np.array(flare_classes)

    X_list = []
    y_list = []
    meta_list = []

    for start_idx in range(0, n_samples - window_size, stride):
        end_idx = start_idx + window_size
        window_end_time = timestamps[end_idx - 1]
        window_start_time = timestamps[start_idx]

        # Check for flares within the strategic horizon
        horizon_start = window_end_time
        horizon_end = window_end_time + forecast_horizon

        label = 0
        label_name = "none"
        flare_time = None
        lead_time = None

        if len(flare_times) > 0:
            mask = (flare_times >= horizon_start) & (flare_times <= horizon_end)
            if np.any(mask):
                idx_in_horizon = np.where(mask)[0]
                best_idx = idx_in_horizon[np.argmax(flare_classes[idx_in_horizon])]
                label = int(flare_classes[best_idx])
                label_name = INT_TO_CLASS[label]
                flare_time = float(flare_times[best_idx])
                lead_time = float(flare_time - window_end_time)

        # Extract and downsample window
        window_raw = features[start_idx:end_idx]
        nan_frac = np.isnan(window_raw).mean()
        if nan_frac > 0.3:
            continue

        window_raw = np.nan_to_num(window_raw, nan=0.0)

        # Downsample by averaging every N samples
        n_out = window_size // downsample
        window_ds = window_raw[:n_out * downsample].reshape(n_out, downsample, n_features)
        window_ds = window_ds.mean(axis=1)  # (n_out, n_features)

        X_list.append(window_ds)
        y_list.append(label)
        meta_list.append(WindowMetadata(
            window_start=float(window_start_time),
            window_end=float(window_end_time + 1),
            label=label,
            label_name=label_name,
            flare_time=flare_time,
            lead_time=lead_time,
        ))

    if not X_list:
        ds_size = window_size // downsample
        return (np.array([]).reshape(0, ds_size, n_features),
                np.array([], dtype=np.int64), [])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    return X, y, meta_list


def normalize_features(X: np.ndarray,
                       mean: Optional[np.ndarray] = None,
                       std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features across the dataset.
    
    Args:
        X: (N, window_size, n_features)
        mean, std: pre-computed stats (for inference mode)
    
    Returns:
        X_normalized, mean, std
    """
    # Compute stats over (N, window_size) for each feature
    if mean is None:
        mean = X.mean(axis=(0, 1))
    if std is None:
        std = X.std(axis=(0, 1))
        std = np.maximum(std, 1e-8)  # prevent division by zero

    X_norm = (X - mean) / std
    return X_norm, mean, std


def balance_classes(X: np.ndarray, y: np.ndarray,
                    metadata: List[WindowMetadata],
                    max_ratio: int = 10) -> Tuple[np.ndarray, np.ndarray, List[WindowMetadata]]:
    """
    Balance class distribution by undersampling the majority class.
    Keeps at most max_ratio * (minority class count) of the majority.
    """
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts[counts > 0].min()
    max_per_class = min_count * max_ratio

    keep_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > max_per_class:
            chosen = np.random.choice(cls_indices, size=max_per_class, replace=False)
            keep_indices.extend(chosen.tolist())
        else:
            keep_indices.extend(cls_indices.tolist())

    keep_indices = sorted(keep_indices)
    return (X[keep_indices], y[keep_indices],
            [metadata[i] for i in keep_indices])


def print_window_stats(y: np.ndarray, metadata: List[WindowMetadata]):
    """Print statistics about the generated windows."""
    print(f"\n{'='*50}")
    print(f"Window Statistics")
    print(f"{'='*50}")
    print(f"Total windows: {len(y)}")

    for cls_int, cls_name in INT_TO_CLASS.items():
        mask = y == cls_int
        count = mask.sum()
        pct = count / len(y) * 100 if len(y) > 0 else 0
        lead_times = [m.lead_time for m, m_mask in zip(metadata, mask) if m_mask and m.lead_time is not None]
        avg_lead = np.mean(lead_times) / 60 if lead_times else 0
        print(f"  {cls_name:>5s}: {count:6d} ({pct:5.1f}%)  "
              f"avg lead time: {avg_lead:.1f} min")
