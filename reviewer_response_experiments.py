"""Extra reviewer-response analyses for the CWRU VAE study.

This script keeps the original training pipeline unchanged and adds
deployment-oriented metrics requested by reviewers:

- PR-AUC in addition to ROC-AUC
- Precision, recall, F1, false-alarm rate, and miss rate
- Threshold sensitivity at P90/P95/P97.5/P99
- Extreme class-imbalance stress tests at 95:5 and 99:1
- Metadata needed for file-wise and load-wise split protocols

Run after ``python main.py`` so ``results/models/best_vae.pt`` exists.
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

from config import LATENT_DIM, MODEL_DIR, WINDOW_SIZE
from data_loader import download_and_load_with_metadata
from preprocessor import balance_dataset_with_indices, group_kfold_splits, normalize_per_sample
from trainer import get_device, reconstruction_errors
from vae_model import VAE


OUT_DIR = os.path.join("results", "reviewer_response")


@dataclass
class MetricRow:
    scenario: str
    threshold: float
    threshold_percentile: str
    normal_count: int
    fault_count: int
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


def _load_trained_model(device: torch.device) -> VAE:
    model_path = os.path.join(MODEL_DIR, "best_vae.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find {model_path}. Run `python main.py` before this script."
        )

    model = VAE(WINDOW_SIZE, LATENT_DIM).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _metrics_for_scores(
    scenario: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    threshold_percentile: str,
) -> MetricRow:
    y_pred = (scores > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return MetricRow(
        scenario=scenario,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        normal_count=int((y_true == 0).sum()),
        fault_count=int((y_true == 1).sum()),
        roc_auc=roc_auc_score(y_true, scores),
        pr_auc=average_precision_score(y_true, scores),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        false_alarm_rate=false_alarm_rate,
        miss_rate=miss_rate,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def _write_rows(rows: list[MetricRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MetricRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _imbalance_subset(
    y_true: np.ndarray,
    scores: np.ndarray,
    fault_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y_true == 0)[0]
    fault_idx = np.where(y_true == 1)[0]

    n_fault = max(1, int(round(len(normal_idx) * fault_fraction / (1 - fault_fraction))))
    n_fault = min(n_fault, len(fault_idx))
    kept_fault = rng.choice(fault_idx, size=n_fault, replace=False)
    idx = np.concatenate([normal_idx, kept_fault])
    rng.shuffle(idx)
    return y_true[idx], scores[idx]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    X_raw, y_binary, _, metadata = download_and_load_with_metadata()
    idx = balance_dataset_with_indices(y_binary)
    X = normalize_per_sample(X_raw[idx])
    y = y_binary[idx]
    file_groups = metadata["file"][idx]
    load_groups = metadata["load"][idx]

    device = get_device()
    model = _load_trained_model(device)
    scores = reconstruction_errors(model, X, device)
    normal_scores = scores[y == 0]

    rows: list[MetricRow] = []
    for percentile in [90, 95, 97.5, 99]:
        threshold = float(np.percentile(normal_scores, percentile))
        rows.append(
            _metrics_for_scores(
                scenario="threshold_sensitivity_balanced_79.5_20.5",
                y_true=y,
                scores=scores,
                threshold=threshold,
                threshold_percentile=f"P{percentile}",
            )
        )

    p95_threshold = float(np.percentile(normal_scores, 95))
    for scenario, fault_fraction in [("imbalance_95_5", 0.05), ("imbalance_99_1", 0.01)]:
        y_sub, score_sub = _imbalance_subset(y, scores, fault_fraction=fault_fraction)
        rows.append(
            _metrics_for_scores(
                scenario=scenario,
                y_true=y_sub,
                scores=score_sub,
                threshold=p95_threshold,
                threshold_percentile="P95_from_normal_scores",
            )
        )

    metrics_path = os.path.join(OUT_DIR, "reviewer_metrics.csv")
    _write_rows(rows, metrics_path)

    file_splits = group_kfold_splits(file_groups, n_splits=5)
    load_splits = group_kfold_splits(load_groups, n_splits=4)
    protocol_path = os.path.join(OUT_DIR, "strict_split_protocol.txt")
    with open(protocol_path, "w", encoding="utf-8") as f:
        f.write("Leakage-resistant split protocols\n")
        f.write("=================================\n\n")
        f.write(f"Balanced subset windows: {len(y):,}\n")
        f.write(f"Unique source files: {len(np.unique(file_groups))}\n")
        f.write(f"Unique load groups: {len(np.unique(load_groups))}\n\n")
        f.write("File-wise GroupKFold sizes:\n")
        for i, (train_idx, test_idx) in enumerate(file_splits, 1):
            f.write(f"  Fold {i}: train={len(train_idx):,}, test={len(test_idx):,}\n")
        f.write("\nLoad-wise GroupKFold sizes:\n")
        for i, (train_idx, test_idx) in enumerate(load_splits, 1):
            f.write(f"  Fold {i}: train={len(train_idx):,}, test={len(test_idx):,}\n")
        f.write(
            "\nUse these groups for the next full retraining run to avoid mixing "
            "overlapping windows or identical operating conditions across folds.\n"
        )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {protocol_path}")


if __name__ == "__main__":
    main()
