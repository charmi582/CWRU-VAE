"""Threshold calibration study for load-wise CWRU VAE models.

The load-wise experiment showed that ROC-AUC can remain perfect while a fixed
P95 threshold transfers poorly to some unseen loads. This script compares
several threshold calibration strategies without retraining the VAE models.

Run after:
  python load_wise_experiment.py

Output:
  results/threshold_calibration/threshold_calibration_metrics.csv
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from config import BETA, LATENT_DIM, MODEL_DIR, THRESHOLD_PERCENTILE, WINDOW_SIZE
from data_loader import download_and_load_with_metadata
from preprocessor import balance_dataset_with_indices, normalize_per_sample
from trainer import get_device, reconstruction_errors
from vae_model import VAE


OUT_DIR = os.path.join("results", "threshold_calibration")
TARGET_CAL_FRACTION = 0.10


@dataclass
class CalibrationRow:
    heldout_load: int
    strategy: str
    threshold: float
    eval_normal_count: int
    eval_fault_count: int
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


def _robust_threshold(scores: np.ndarray, k: float) -> float:
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    return med + k * 1.4826 * mad


def _metrics(
    heldout_load: int,
    strategy: str,
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> CalibrationRow:
    y_pred = (scores > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0
    mr = fn / (fn + tp) if (fn + tp) else 0.0
    return CalibrationRow(
        heldout_load=heldout_load,
        strategy=strategy,
        threshold=threshold,
        eval_normal_count=int((y_true == 0).sum()),
        eval_fault_count=int((y_true == 1).sum()),
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


def _write_rows(rows: list[CalibrationRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CalibrationRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def run(seed: int = 42, val_fraction: float = 0.2) -> list[CalibrationRow]:
    X_raw, y_binary, _, metadata = download_and_load_with_metadata()
    idx = balance_dataset_with_indices(y_binary, seed=seed)
    X = normalize_per_sample(X_raw[idx])
    y = y_binary[idx]
    loads = metadata["load"][idx].astype(np.int32)

    device = get_device()
    rows: list[CalibrationRow] = []
    rng = np.random.default_rng(seed)

    for heldout_load in sorted(np.unique(loads)):
        print(f"Calibrating held-out load {heldout_load}")
        model = _load_model(int(heldout_load), device)

        train_normal_mask = (loads != heldout_load) & (y == 0)
        test_mask = loads == heldout_load
        test_normal_mask = test_mask & (y == 0)

        X_train_all = X[train_normal_mask]
        X_train, X_val = _normal_train_val_split(
            X_train_all, val_fraction=val_fraction, seed=seed + int(heldout_load)
        )

        train_scores = reconstruction_errors(model, X_train, device)
        val_scores = reconstruction_errors(model, X_val, device)

        X_test = X[test_mask]
        y_test = y[test_mask]
        test_scores = reconstruction_errors(model, X_test, device)

        strategies = {
            "train_p95": float(np.percentile(train_scores, 95)),
            "train_p99": float(np.percentile(train_scores, 99)),
            "val_p95": float(np.percentile(val_scores, THRESHOLD_PERCENTILE)),
            "val_p99": float(np.percentile(val_scores, 99)),
            "train_median_mad_3": _robust_threshold(train_scores, 3.0),
            "train_median_mad_5": _robust_threshold(train_scores, 5.0),
        }

        for name, threshold in strategies.items():
            rows.append(
                _metrics(
                    int(heldout_load),
                    name,
                    threshold,
                    y_test,
                    test_scores,
                )
            )

        normal_indices = np.where(test_normal_mask)[0]
        n_cal = max(1, int(round(len(normal_indices) * TARGET_CAL_FRACTION)))
        cal_indices = rng.choice(normal_indices, size=n_cal, replace=False)
        eval_mask = test_mask.copy()
        eval_mask[cal_indices] = False

        cal_scores = reconstruction_errors(model, X[cal_indices], device)
        eval_scores = reconstruction_errors(model, X[eval_mask], device)
        y_eval = y[eval_mask]
        for percentile in [95, 99]:
            rows.append(
                _metrics(
                    int(heldout_load),
                    f"target_load_10pct_normal_p{percentile}",
                    float(np.percentile(cal_scores, percentile)),
                    y_eval,
                    eval_scores,
                )
            )

    out_path = os.path.join(OUT_DIR, "threshold_calibration_metrics.csv")
    _write_rows(rows, out_path)
    print(f"Wrote {out_path}")
    return rows


if __name__ == "__main__":
    run()
