"""Noise robustness study for load-wise CWRU VAE models.

Adds Gaussian noise to each unseen-load test window at fixed SNR levels and
evaluates the saved load-wise VAE models using the original validation-normal
P95 threshold.

Run after:
  python load_wise_experiment.py

Output:
  results/noise_robustness/noise_robustness_metrics.csv
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import BETA, LATENT_DIM, MODEL_DIR, THRESHOLD_PERCENTILE, WINDOW_SIZE
from data_loader import download_and_load_with_metadata
from preprocessor import balance_dataset_with_indices, normalize_per_sample
from trainer import get_device, reconstruction_errors
from vae_model import VAE


OUT_DIR = os.path.join("results", "noise_robustness")
SNR_LEVELS = [None, 30, 20, 10, 5]


@dataclass
class NoiseRow:
    heldout_load: int
    snr_db: str
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


def _load_model(heldout_load: int, device: torch.device) -> VAE:
    path = os.path.join(MODEL_DIR, f"load_wise_heldout_load{heldout_load}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python load_wise_experiment.py` first."
        )
    model = VAE(WINDOW_SIZE, LATENT_DIM, BETA).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _normal_train_val_split(
    X_normal: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_normal))
    n_val = max(1, int(round(len(idx) * val_fraction)))
    return X_normal[idx[n_val:]], X_normal[idx[:n_val]]


def _add_noise_at_snr(X: np.ndarray, snr_db: int, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(np.square(X), axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=X.shape)
    return (X + noise).astype(np.float32)


def _metrics(
    heldout_load: int,
    snr_db: str,
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> NoiseRow:
    y_pred = (scores > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0
    mr = fn / (fn + tp) if (fn + tp) else 0.0
    return NoiseRow(
        heldout_load=heldout_load,
        snr_db=snr_db,
        threshold=threshold,
        roc_auc=roc_auc_score(y_true, scores),
        pr_auc=average_precision_score(y_true, scores),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        false_alarm_rate=far,
        miss_rate=mr,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def _write_rows(rows: list[NoiseRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(NoiseRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def run(seed: int = 42, val_fraction: float = 0.2) -> list[NoiseRow]:
    X_raw, y_binary, _, metadata = download_and_load_with_metadata()
    idx = balance_dataset_with_indices(y_binary, seed=seed)
    X = normalize_per_sample(X_raw[idx])
    y = y_binary[idx]
    loads = metadata["load"][idx].astype(np.int32)

    device = get_device()
    rows: list[NoiseRow] = []

    for heldout_load in sorted(np.unique(loads)):
        print(f"Noise robustness held-out load {heldout_load}")
        model = _load_model(int(heldout_load), device)

        train_normal_mask = (loads != heldout_load) & (y == 0)
        test_mask = loads == heldout_load
        X_train_all = X[train_normal_mask]
        _, X_val = _normal_train_val_split(
            X_train_all, val_fraction=val_fraction, seed=seed + int(heldout_load)
        )

        val_scores = reconstruction_errors(model, X_val, device)
        threshold = float(np.percentile(val_scores, THRESHOLD_PERCENTILE))

        X_test = X[test_mask]
        y_test = y[test_mask]
        for snr in SNR_LEVELS:
            rng = np.random.default_rng(seed + int(heldout_load) * 100 + (snr or 0))
            X_eval = X_test if snr is None else _add_noise_at_snr(X_test, snr, rng)
            scores = reconstruction_errors(model, X_eval, device)
            rows.append(
                _metrics(
                    int(heldout_load),
                    "clean" if snr is None else str(snr),
                    threshold,
                    y_test,
                    scores,
                )
            )

    out_path = os.path.join(OUT_DIR, "noise_robustness_metrics.csv")
    _write_rows(rows, out_path)
    print(f"Wrote {out_path}")
    return rows


if __name__ == "__main__":
    run()
