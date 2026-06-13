"""Shared utilities for journal-version strict split experiments."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.optim as optim
from scipy import stats
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import BATCH_SIZE, LEARNING_RATE, SAMPLE_RATE, WINDOW_SIZE
from trainer import _eval_epoch, _train_epoch, get_device, reconstruction_errors


@dataclass
class EvalRow:
    protocol: str
    heldout_group: str
    model: str
    train_normal_count: int
    val_normal_count: int
    test_normal_count: int
    test_fault_count: int
    threshold_strategy: str
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    miss_rate: float
    tn: int
    fp: int
    fn: int
    tp: int
    best_epoch: int = 0
    best_val_loss: float = 0.0
    train_seconds: float = 0.0
    inference_ms_per_window_cpu: float = 0.0
    model_size_mb: float = 0.0
    param_count: int = 0


def normal_train_val_split(
    X_normal: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_normal))
    n_val = max(1, int(round(len(idx) * val_fraction)))
    train_idx = idx[n_val:]
    val_idx = idx[:n_val]
    if len(train_idx) == 0:
        raise ValueError("Not enough normal samples for train/validation split.")
    return X_normal[train_idx], X_normal[val_idx]


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    y_pred = (scores > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0
    mr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": far,
        "miss_rate": mr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def robust_threshold(scores: np.ndarray, k: float = 5.0) -> float:
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    return med + k * 1.4826 * mad


def threshold_candidates(train_scores: np.ndarray, val_scores: np.ndarray) -> dict[str, float]:
    return {
        "train_p95": float(np.percentile(train_scores, 95)),
        "train_p99": float(np.percentile(train_scores, 99)),
        "val_p95": float(np.percentile(val_scores, 95)),
        "val_p99": float(np.percentile(val_scores, 99)),
        "train_mad5": robust_threshold(train_scores, 5.0),
    }


def train_deep_model(
    model_factory: Callable[[], torch.nn.Module],
    X_train: np.ndarray,
    X_val: np.ndarray,
    max_epochs: int,
    patience: int,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], int, float, float]:
    device = device or get_device()
    tr_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_train).unsqueeze(1)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    va_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_val).unsqueeze(1)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    model = model_factory().to(device)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10, factor=0.5)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    no_improv = 0
    start = time.perf_counter()

    for epoch in tqdm(range(1, max_epochs + 1), desc="  train", unit="ep"):
        _train_epoch(model, tr_ld, optimiser, device)
        val_loss, _, _ = _eval_epoch(model, va_ld, device)
        if not np.isfinite(val_loss):
            no_improv += 1
            if no_improv >= patience:
                break
            continue
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
            no_improv = 0
        else:
            no_improv += 1

        if no_improv >= patience:
            break

    train_seconds = time.perf_counter() - start
    if best_state is None:
        best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
        best_val = float("nan")
    model.load_state_dict(best_state)
    return model, best_state, best_epoch, best_val, train_seconds


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(state: dict[str, torch.Tensor]) -> float:
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    return total_bytes / (1024 * 1024)


@torch.no_grad()
def cpu_latency_ms_per_window(model: torch.nn.Module, input_size: int = WINDOW_SIZE, repeats: int = 20) -> float:
    cpu_model = model.to("cpu").eval()
    x = torch.zeros((1, 1, input_size), dtype=torch.float32)
    for _ in range(3):
        cpu_model.anomaly_score(x)
    start = time.perf_counter()
    for _ in range(repeats):
        cpu_model.anomaly_score(x)
    elapsed = time.perf_counter() - start
    return elapsed * 1000 / repeats


def statistical_features(X: np.ndarray) -> np.ndarray:
    """Return compact time/frequency features for traditional baselines."""
    mean = X.mean(axis=1)
    std = X.std(axis=1) + 1e-12
    rms = np.sqrt(np.mean(X**2, axis=1))
    peak = np.max(np.abs(X), axis=1)
    ptp = np.ptp(X, axis=1)
    skew = stats.skew(X, axis=1, bias=False)
    kurt = stats.kurtosis(X, axis=1, fisher=False, bias=False)
    crest = peak / (rms + 1e-12)
    impulse = peak / (np.mean(np.abs(X), axis=1) + 1e-12)
    shape = rms / (np.mean(np.abs(X), axis=1) + 1e-12)

    spectrum = np.abs(rfft(X, axis=1))
    freqs = rfftfreq(X.shape[1], d=1.0 / SAMPLE_RATE)
    spec_sum = spectrum.sum(axis=1) + 1e-12
    centroid = (spectrum * freqs).sum(axis=1) / spec_sum
    bandwidth = np.sqrt((spectrum * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / spec_sum)

    return np.vstack(
        [mean, std, rms, peak, ptp, skew, kurt, crest, impulse, shape, centroid, bandwidth]
    ).T.astype(np.float32)


def isolation_forest_scores(
    X_train_normal: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
) -> tuple[IsolationForest, np.ndarray, np.ndarray]:
    train_features = statistical_features(X_train_normal)
    eval_features = statistical_features(X_eval)
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_features)
    train_scores = -model.score_samples(train_features)
    eval_scores = -model.score_samples(eval_features)
    return model, train_scores, eval_scores


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
