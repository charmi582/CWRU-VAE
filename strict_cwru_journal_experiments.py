"""Strict CWRU experiments for the journal version.

Protocols
---------
load-wise       : hold out one load condition.
fault-size-wise : hold out one fault diameter (007/014/021).
file-wise       : GroupKFold by original .mat file.

Baselines
---------
vae             : convolutional VAE.
cnn-ae          : convolutional deterministic AE.
lstm-ae         : LSTM deterministic AE.
iforest         : Isolation Forest on 12 statistical features.

The output is intentionally long-form so the journal paper can present
threshold calibration as a central contribution instead of an add-on result.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict

import numpy as np
from sklearn.model_selection import GroupKFold

from ae_model import ConvAE
from config import BETA, LATENT_DIM, N_EPOCHS, PATIENCE, WINDOW_SIZE
from data_loader import download_and_load_with_metadata
from journal_experiment_utils import (
    EvalRow,
    classification_metrics,
    cpu_latency_ms_per_window,
    ensure_dir,
    isolation_forest_scores,
    normal_train_val_split,
    parameter_count,
    reconstruction_errors,
    state_dict_size_mb,
    statistical_features,
    threshold_candidates,
    train_deep_model,
)
from lstm_ae_model import LSTMAE
from preprocessor import balance_dataset_with_indices, normalize_per_sample
from trainer import get_device
from vae_model import VAE


OUT_DIR = os.path.join("results", "journal_strict_cwru")


MODEL_FACTORIES = {
    "vae": lambda: VAE(WINDOW_SIZE, LATENT_DIM, BETA),
    "cnn-ae": lambda: ConvAE(WINDOW_SIZE, LATENT_DIM),
    "lstm-ae": lambda: LSTMAE(WINDOW_SIZE, LATENT_DIM),
}


def _write_rows(rows: list[EvalRow], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _normal_test_split(normal_indices: np.ndarray, seed: int, test_fraction: float = 0.2):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(normal_indices)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    return shuffled[n_test:], shuffled[:n_test]


def build_load_splits(y: np.ndarray, metadata: dict[str, np.ndarray]):
    loads = metadata["load"].astype(np.int32)
    for heldout in sorted(np.unique(loads)):
        train_normal_mask = (loads != heldout) & (y == 0)
        test_mask = loads == heldout
        yield "load-wise", str(int(heldout)), train_normal_mask, test_mask


def build_fault_size_splits(y: np.ndarray, metadata: dict[str, np.ndarray], seed: int):
    fault_sizes = metadata["fault_size"].astype(object)
    normal_idx = np.where(y == 0)[0]
    normal_train_pool, normal_test_idx = _normal_test_split(normal_idx, seed=seed)
    for size in ["007", "014", "021"]:
        fault_test_idx = np.where((y == 1) & (fault_sizes == size))[0]
        if len(fault_test_idx) == 0:
            continue
        train_normal_mask = np.zeros_like(y, dtype=bool)
        train_normal_mask[normal_train_pool] = True
        test_mask = np.zeros_like(y, dtype=bool)
        test_mask[normal_test_idx] = True
        test_mask[fault_test_idx] = True
        yield "fault-size-wise", size, train_normal_mask, test_mask


def build_file_splits(y: np.ndarray, metadata: dict[str, np.ndarray], n_splits: int):
    files = metadata["file"].astype(str)
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(files))))
    dummy = np.zeros((len(y), 1), dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy, groups=files), start=1):
        train_normal_mask = np.zeros_like(y, dtype=bool)
        train_normal_mask[train_idx] = y[train_idx] == 0
        test_mask = np.zeros_like(y, dtype=bool)
        test_mask[test_idx] = True
        if (y[test_mask] == 0).sum() == 0 or (y[test_mask] == 1).sum() == 0:
            continue
        yield "file-wise", f"fold{fold}", train_normal_mask, test_mask


def _rows_for_scores(
    protocol: str,
    heldout_group: str,
    model_name: str,
    train_normal_count: int,
    val_normal_count: int,
    y_test: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    best_epoch: int = 0,
    best_val_loss: float = 0.0,
    train_seconds: float = 0.0,
    inference_ms_per_window_cpu: float = 0.0,
    model_size_mb: float = 0.0,
    param_count: int = 0,
) -> list[EvalRow]:
    rows = []
    for strategy, threshold in threshold_candidates(train_scores, val_scores).items():
        metrics = classification_metrics(y_test, test_scores, threshold)
        rows.append(
            EvalRow(
                protocol=protocol,
                heldout_group=heldout_group,
                model=model_name,
                train_normal_count=int(train_normal_count),
                val_normal_count=int(val_normal_count),
                test_normal_count=int((y_test == 0).sum()),
                test_fault_count=int((y_test == 1).sum()),
                threshold_strategy=strategy,
                threshold=float(threshold),
                best_epoch=int(best_epoch),
                best_val_loss=float(best_val_loss),
                train_seconds=float(train_seconds),
                inference_ms_per_window_cpu=float(inference_ms_per_window_cpu),
                model_size_mb=float(model_size_mb),
                param_count=int(param_count),
                **metrics,
            )
        )
    return rows


def evaluate_deep_model(
    protocol: str,
    heldout_group: str,
    model_name: str,
    X_train_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_epochs: int,
    patience: int,
    seed: int,
) -> list[EvalRow]:
    X_train, X_val = normal_train_val_split(X_train_pool, val_fraction=0.2, seed=seed)
    device = get_device()
    model, state, best_epoch, best_val, train_seconds = train_deep_model(
        MODEL_FACTORIES[model_name],
        X_train,
        X_val,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    train_scores = reconstruction_errors(model, X_train, device)
    val_scores = reconstruction_errors(model, X_val, device)
    test_scores = reconstruction_errors(model, X_test, device)
    latency = cpu_latency_ms_per_window(model)

    model_path = os.path.join(
        OUT_DIR,
        "models",
        f"{protocol}_{heldout_group}_{model_name}.pt".replace(os.sep, "_"),
    )
    ensure_dir(os.path.dirname(model_path))
    import torch

    torch.save(state, model_path)

    return _rows_for_scores(
        protocol=protocol,
        heldout_group=heldout_group,
        model_name=model_name,
        train_normal_count=len(X_train),
        val_normal_count=len(X_val),
        y_test=y_test,
        train_scores=train_scores,
        val_scores=val_scores,
        test_scores=test_scores,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        train_seconds=train_seconds,
        inference_ms_per_window_cpu=latency,
        model_size_mb=state_dict_size_mb(state),
        param_count=parameter_count(model),
    )


def evaluate_iforest(
    protocol: str,
    heldout_group: str,
    X_train_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> list[EvalRow]:
    X_train, X_val = normal_train_val_split(X_train_pool, val_fraction=0.2, seed=seed)
    model, train_scores, test_scores = isolation_forest_scores(X_train, X_test, seed=seed)
    val_scores = -model.score_samples(statistical_features(X_val))
    return _rows_for_scores(
        protocol=protocol,
        heldout_group=heldout_group,
        model_name="iforest",
        train_normal_count=len(X_train),
        val_normal_count=len(X_val),
        y_test=y_test,
        train_scores=train_scores,
        val_scores=val_scores,
        test_scores=test_scores,
        param_count=len(model.estimators_),
    )


def run(
    protocols: list[str],
    models: list[str],
    max_epochs: int,
    patience: int,
    seed: int,
    n_file_splits: int,
) -> list[EvalRow]:
    ensure_dir(OUT_DIR)
    X_raw, y_raw, _, metadata_raw = download_and_load_with_metadata()
    idx = balance_dataset_with_indices(y_raw, seed=seed)
    X = normalize_per_sample(X_raw[idx])
    y = y_raw[idx]
    metadata = {k: v[idx] for k, v in metadata_raw.items()}

    split_builders = []
    if "load-wise" in protocols:
        split_builders.extend(build_load_splits(y, metadata))
    if "fault-size-wise" in protocols:
        split_builders.extend(build_fault_size_splits(y, metadata, seed=seed))
    if "file-wise" in protocols:
        split_builders.extend(build_file_splits(y, metadata, n_splits=n_file_splits))

    rows: list[EvalRow] = []
    for protocol, heldout_group, train_normal_mask, test_mask in split_builders:
        X_train_pool = X[train_normal_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        if len(X_train_pool) < 10 or (y_test == 0).sum() == 0 or (y_test == 1).sum() == 0:
            print(f"[SKIP] {protocol}:{heldout_group} has insufficient train/test composition.")
            continue
        print(
            f"\n{protocol}:{heldout_group} "
            f"train_normal={len(X_train_pool)} "
            f"test_normal={(y_test == 0).sum()} test_fault={(y_test == 1).sum()}"
        )
        for model_name in models:
            print(f"  model={model_name}")
            if model_name == "iforest":
                model_rows = evaluate_iforest(
                    protocol, heldout_group, X_train_pool, X_test, y_test, seed=seed
                )
            else:
                model_rows = evaluate_deep_model(
                    protocol,
                    heldout_group,
                    model_name,
                    X_train_pool,
                    X_test,
                    y_test,
                    max_epochs=max_epochs,
                    patience=patience,
                    seed=seed,
                )
            rows.extend(model_rows)

    out_path = os.path.join(OUT_DIR, "strict_cwru_metrics.csv")
    _write_rows(rows, out_path)
    print(f"\nWrote {out_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["fault-size-wise", "file-wise"],
        choices=["load-wise", "fault-size-wise", "file-wise"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vae", "cnn-ae", "iforest"],
        choices=["vae", "cnn-ae", "lstm-ae", "iforest"],
    )
    parser.add_argument("--max-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-file-splits", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        protocols=args.protocols,
        models=args.models,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        n_file_splits=args.n_file_splits,
    )
