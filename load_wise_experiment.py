"""Load-wise CWRU generalisation experiment.

This experiment addresses the reviewer concern that random window-level
cross-validation may overestimate performance when adjacent windows overlap.
Each fold holds out one load condition completely, trains the VAE only on
normal windows from the remaining loads, and evaluates on all balanced-subset
windows from the unseen load.

Outputs:
  - results/load_wise/load_wise_metrics.csv
  - results/models/load_wise_heldout_load*.pt
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
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

from config import (
    BATCH_SIZE,
    BETA,
    LATENT_DIM,
    LEARNING_RATE,
    MODEL_DIR,
    N_EPOCHS,
    PATIENCE,
    THRESHOLD_PERCENTILE,
    WINDOW_SIZE,
)
from data_loader import download_and_load_with_metadata
from preprocessor import balance_dataset_with_indices, normalize_per_sample
from trainer import _eval_epoch, _train_epoch, get_device, reconstruction_errors
from vae_model import VAE


OUT_DIR = os.path.join("results", "load_wise")


@dataclass
class LoadWiseRow:
    heldout_load: int
    train_normal_count: int
    val_normal_count: int
    test_normal_count: int
    test_fault_count: int
    best_epoch: int
    best_val_loss: float
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


def _normal_train_val_split(
    X_normal: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_normal))
    n_val = max(1, int(round(len(idx) * val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        raise ValueError("Not enough normal windows to create train/validation split.")
    return X_normal[train_idx], X_normal[val_idx]


def _train_one_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> tuple[VAE, dict, int, float]:
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

    model = VAE(WINDOW_SIZE, LATENT_DIM, BETA).to(device)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    no_improv = 0

    for epoch in tqdm(range(1, max_epochs + 1), desc="  train", unit="ep"):
        train_loss, _, _ = _train_epoch(model, tr_ld, optimiser, device)
        val_loss, _, _ = _eval_epoch(model, va_ld, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: p.cpu().clone() for k, p in model.state_dict().items()}
            no_improv = 0
        else:
            no_improv += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"    epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")

        if no_improv >= patience:
            print(f"    early stop @ epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    return model, best_state, best_epoch, best_val


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    y_pred = (scores > threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    miss_rate = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _write_rows(rows: list[LoadWiseRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LoadWiseRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def run_load_wise(
    val_fraction: float = 0.2,
    max_epochs: int = N_EPOCHS,
    patience: int = PATIENCE,
    seed: int = 42,
) -> list[LoadWiseRow]:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_raw, y_binary, _, metadata = download_and_load_with_metadata()
    balanced_idx = balance_dataset_with_indices(y_binary, seed=seed)

    X = normalize_per_sample(X_raw[balanced_idx])
    y = y_binary[balanced_idx]
    loads = metadata["load"][balanced_idx].astype(np.int32)

    device = get_device()
    print(f"\nUsing device: {device}")
    print("Load-wise protocol: train on normal windows from 3 loads; test on held-out load.")

    rows: list[LoadWiseRow] = []
    for heldout_load in sorted(np.unique(loads)):
        print(f"\n{'=' * 64}")
        print(f"Held-out load {heldout_load}")

        train_normal_mask = (loads != heldout_load) & (y == 0)
        test_mask = loads == heldout_load

        X_train_all = X[train_normal_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        X_train, X_val = _normal_train_val_split(
            X_train_all, val_fraction=val_fraction, seed=seed + int(heldout_load)
        )

        print(
            f"  train normal={len(X_train):,} val normal={len(X_val):,} "
            f"test normal={(y_test == 0).sum():,} test fault={(y_test == 1).sum():,}"
        )

        model, state, best_epoch, best_val = _train_one_fold(
            X_train=X_train,
            X_val=X_val,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
        )

        val_scores = reconstruction_errors(model, X_val, device)
        threshold = float(np.percentile(val_scores, THRESHOLD_PERCENTILE))
        test_scores = reconstruction_errors(model, X_test, device)
        metrics = _classification_metrics(y_test, test_scores, threshold)

        model_path = os.path.join(
            MODEL_DIR, f"load_wise_heldout_load{int(heldout_load)}.pt"
        )
        torch.save(state, model_path)

        row = LoadWiseRow(
            heldout_load=int(heldout_load),
            train_normal_count=int(len(X_train)),
            val_normal_count=int(len(X_val)),
            test_normal_count=int((y_test == 0).sum()),
            test_fault_count=int((y_test == 1).sum()),
            best_epoch=int(best_epoch),
            best_val_loss=float(best_val),
            threshold=threshold,
            **metrics,
        )
        rows.append(row)

        print(
            "  "
            f"AUC={row.roc_auc:.4f} PR-AUC={row.pr_auc:.4f} "
            f"F1={row.f1:.4f} FAR={row.false_alarm_rate:.4f} "
            f"MR={row.miss_rate:.4f}"
        )
        print(f"  saved {model_path}")

    out_path = os.path.join(OUT_DIR, "load_wise_metrics.csv")
    _write_rows(rows, out_path)
    print(f"\nWrote {out_path}")

    aucs = np.array([r.roc_auc for r in rows], dtype=np.float64)
    pr_aucs = np.array([r.pr_auc for r in rows], dtype=np.float64)
    f1s = np.array([r.f1 for r in rows], dtype=np.float64)
    print(
        f"Summary: AUC={aucs.mean():.4f}±{aucs.std():.4f}, "
        f"PR-AUC={pr_aucs.mean():.4f}±{pr_aucs.std():.4f}, "
        f"F1={f1s.mean():.4f}±{f1s.std():.4f}"
    )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_load_wise(
        val_fraction=args.val_fraction,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )
