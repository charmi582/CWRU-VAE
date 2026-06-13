"""Dataset balancing, normalisation, and K-fold split utilities."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from config import FAULT_RATIO, K_FOLDS, LABEL_NAMES


def balance_dataset(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_multi: np.ndarray,
    fault_ratio: float = FAULT_RATIO,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate factory-like data distribution:
      - Keep ALL normal segments.
      - Randomly keep *fault_ratio* fraction of fault segments.

    Returns balanced (X, y_binary, y_multi).
    """
    rng = np.random.default_rng(seed)

    normal_idx = np.where(y_binary == 0)[0]
    fault_idx  = np.where(y_binary == 1)[0]

    n_keep = max(1, int(len(fault_idx) * fault_ratio))
    kept_fault = rng.choice(fault_idx, size=n_keep, replace=False)

    idx = np.concatenate([normal_idx, kept_fault])
    rng.shuffle(idx)

    X_b      = X[idx]
    y_bin_b  = y_binary[idx]
    y_multi_b = y_multi[idx]

    print(f"\n  After balancing — factory-like distribution:")
    print(f"    Normal  : {(y_bin_b == 0).sum():,}  ({100*(y_bin_b==0).mean():.1f} %)")
    print(f"    Fault   : {(y_bin_b == 1).sum():,}  ({100*(y_bin_b==1).mean():.1f} %)")
    for cls, name in LABEL_NAMES.items():
        n = (y_multi_b == cls).sum()
        if n:
            print(f"      {name:<22}: {n:,}")

    return X_b, y_bin_b, y_multi_b


def balance_dataset_with_indices(
    y_binary: np.ndarray,
    fault_ratio: float = FAULT_RATIO,
    seed: int = 42,
) -> np.ndarray:
    """Return indices for the factory-like balanced subset."""
    rng = np.random.default_rng(seed)

    normal_idx = np.where(y_binary == 0)[0]
    fault_idx = np.where(y_binary == 1)[0]

    n_keep = max(1, int(len(fault_idx) * fault_ratio))
    kept_fault = rng.choice(fault_idx, size=n_keep, replace=False)

    idx = np.concatenate([normal_idx, kept_fault])
    rng.shuffle(idx)
    return idx


def normalize_per_sample(X: np.ndarray) -> np.ndarray:
    """Z-score normalise each window independently (zero mean, unit std)."""
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mean) / std).astype(np.float32)


def kfold_splits(X: np.ndarray, n_splits: int = K_FOLDS, seed: int = 42):
    """Return list of (train_idx, val_idx) tuples for *X*."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X))


def group_kfold_splits(groups: np.ndarray, n_splits: int = K_FOLDS):
    """Return leakage-resistant splits that keep each group in one fold."""
    n_unique = len(np.unique(groups))
    n_splits = min(n_splits, n_unique)
    if n_splits < 2:
        raise ValueError("Need at least two unique groups for GroupKFold.")
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(groups), 1), dtype=np.float32)
    return list(splitter.split(dummy_x, groups=groups))
