"""
Solar Flare Early Warning System - Data Augmentation
Multiplies training data by applying physics-preserving transformations.

Techniques:
  1. Gaussian noise injection (sensor noise simulation)
  2. Time shifting (window alignment variation)
  3. Amplitude scaling (instrument sensitivity variation)
  4. Feature dropout (robustness to missing channels)
"""
import numpy as np
from typing import Tuple, Optional


def add_gaussian_noise(X: np.ndarray, noise_factor: float = 0.05,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Add Gaussian noise scaled to each feature's standard deviation.
    Simulates realistic sensor noise in X-ray detectors.
    """
    rng = rng or np.random.default_rng()
    X_aug = X.copy()
    for feat_idx in range(X.shape[2]):
        feat_std = np.std(X[:, :, feat_idx])
        if feat_std > 0:
            noise = rng.normal(0, noise_factor * feat_std, size=X[:, :, feat_idx].shape)
            X_aug[:, :, feat_idx] += noise
    return X_aug


def time_shift(X: np.ndarray, max_shift: int = 300,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Randomly shift each window in time by rolling.
    Simulates slightly different observation windows.
    max_shift: maximum shift in samples (300 = 5 min at 1s cadence)
    """
    rng = rng or np.random.default_rng()
    X_aug = np.zeros_like(X)
    for i in range(len(X)):
        shift = rng.integers(-max_shift, max_shift + 1)
        X_aug[i] = np.roll(X[i], shift, axis=0)
    return X_aug


def amplitude_scale(X: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2),
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Randomly scale amplitude of each sample.
    Simulates instrument sensitivity variations.
    """
    rng = rng or np.random.default_rng()
    X_aug = X.copy()
    for i in range(len(X)):
        scale = rng.uniform(scale_range[0], scale_range[1])
        X_aug[i] = X[i] * scale
    return X_aug


def feature_dropout(X: np.ndarray, drop_prob: float = 0.1,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Randomly zero out entire feature channels.
    Forces model to not over-rely on any single feature.
    """
    rng = rng or np.random.default_rng()
    X_aug = X.copy()
    for i in range(len(X)):
        for f in range(X.shape[2]):
            if rng.random() < drop_prob:
                X_aug[i, :, f] = 0.0
    return X_aug


def augment_dataset(X: np.ndarray, y: np.ndarray,
                    lead_times: np.ndarray,
                    multiplier: int = 10,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main augmentation function. Multiplies dataset by applying
    random combinations of augmentations.
    
    For each original sample, creates (multiplier-1) augmented copies.
    Original samples are always included.
    
    Args:
        X: (N, T, C) feature windows
        y: (N,) class labels
        lead_times: (N,) lead times in minutes
        multiplier: how many times to multiply the dataset
        seed: random seed for reproducibility
    
    Returns:
        X_aug, y_aug, lt_aug with shape (N*multiplier, ...)
    """
    rng = np.random.default_rng(seed)
    
    N = len(X)
    augmented_X = [X.copy()]  # always include originals
    augmented_y = [y.copy()]
    augmented_lt = [lead_times.copy()]
    
    augmentation_fns = [
        lambda x: add_gaussian_noise(x, 0.03, rng),
        lambda x: add_gaussian_noise(x, 0.07, rng),
        lambda x: time_shift(x, 200, rng),
        lambda x: time_shift(x, 500, rng),
        lambda x: amplitude_scale(x, (0.85, 1.15), rng),
        lambda x: amplitude_scale(x, (0.7, 1.3), rng),
        lambda x: feature_dropout(x, 0.1, rng),
    ]
    
    for aug_idx in range(multiplier - 1):
        # Pick 1-3 random augmentations to chain
        n_augs = rng.integers(1, 4)
        chosen = rng.choice(len(augmentation_fns), size=n_augs, replace=False)
        
        X_new = X.copy()
        for fn_idx in chosen:
            X_new = augmentation_fns[fn_idx](X_new)
        
        augmented_X.append(X_new)
        augmented_y.append(y.copy())
        augmented_lt.append(lead_times.copy())
    
    X_out = np.concatenate(augmented_X, axis=0)
    y_out = np.concatenate(augmented_y, axis=0)
    lt_out = np.concatenate(augmented_lt, axis=0)
    
    # Shuffle
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm], lt_out[perm]
