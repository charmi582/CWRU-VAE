"""External Paderborn validation experiments for the journal extension.

This script mirrors the strict CWRU experiment design on an external bearing
dataset. It focuses on deployment-relevant protocols:

condition-wise : hold out one Paderborn operating condition.
bearing-wise   : hold out one damaged bearing state with a normal test subset.
file-wise      : hold out complete measurement files.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ae_model import ConvAE
from config import BETA, LATENT_DIM, N_EPOCHS, PATIENCE, STRIDE, WINDOW_SIZE
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
from paderborn_data_loader import (
    DEFAULT_BEARINGS,
    PADERBORN_MAT_DIR,
    download_paderborn_archives,
    extract_paderborn_archives,
    load_paderborn_with_metadata,
)
from preprocessor import normalize_per_sample
from trainer import get_device
from vae_model import VAE


OUT_DIR = os.path.join("results", "journal_external_paderborn")


def model_factory(model_name: str, input_size: int):
    if model_name == "vae":
        return lambda: VAE(input_size, LATENT_DIM, BETA)
    if model_name == "cnn-ae":
        return lambda: ConvAE(input_size, LATENT_DIM)
    if model_name == "lstm-ae":
        return lambda: LSTMAE(input_size, LATENT_DIM)
    raise ValueError(f"Unsupported deep model: {model_name}")


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


def build_condition_splits(y: np.ndarray, metadata: dict[str, np.ndarray]):
    conditions = metadata["condition"].astype(str)
    for heldout in sorted(np.unique(conditions)):
        if heldout == "unknown":
            continue
        train_normal_mask = (conditions != heldout) & (y == 0)
        test_mask = conditions == heldout
        if (y[test_mask] == 0).sum() == 0 or (y[test_mask] == 1).sum() == 0:
            continue
        yield "paderborn-condition-wise", heldout, train_normal_mask, test_mask


def build_bearing_splits(y: np.ndarray, metadata: dict[str, np.ndarray], seed: int):
    bearings = metadata["bearing"].astype(str)
    normal_idx = np.where(y == 0)[0]
    normal_train_pool, normal_test_idx = _normal_test_split(normal_idx, seed=seed)
    for bearing in sorted(np.unique(bearings[y == 1])):
        fault_test_idx = np.where((y == 1) & (bearings == bearing))[0]
        train_normal_mask = np.zeros_like(y, dtype=bool)
        train_normal_mask[normal_train_pool] = True
        test_mask = np.zeros_like(y, dtype=bool)
        test_mask[normal_test_idx] = True
        test_mask[fault_test_idx] = True
        yield "paderborn-bearing-wise", bearing, train_normal_mask, test_mask


def build_file_splits(y: np.ndarray, metadata: dict[str, np.ndarray], n_splits: int, seed: int):
    files = metadata["file"].astype(str)
    unique_files = np.array(sorted(np.unique(files)))
    normal_files = [f for f in unique_files if np.all(y[files == f] == 0)]
    fault_files = [f for f in unique_files if np.any(y[files == f] == 1)]
    if not normal_files or not fault_files:
        return

    rng = np.random.default_rng(seed)
    fault_files = list(rng.permutation(fault_files))
    normal_files = list(rng.permutation(normal_files))
    n_folds = min(n_splits, len(normal_files))
    fault_chunks = np.array_split(fault_files, n_folds)

    for fold in range(n_folds):
        heldout_files = [normal_files[fold]] + list(fault_chunks[fold])
        test_mask = np.zeros_like(y, dtype=bool)
        for fname in heldout_files:
            test_mask |= files == fname
        train_normal_mask = (~test_mask) & (y == 0)
        if (y[test_mask] == 0).sum() == 0 or (y[test_mask] == 1).sum() == 0:
            continue
        yield "paderborn-file-wise", f"fold{fold + 1}", train_normal_mask, test_mask


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
    input_size: int,
) -> list[EvalRow]:
    X_train, X_val = normal_train_val_split(X_train_pool, val_fraction=0.2, seed=seed)
    device = get_device()
    try:
        model, state, best_epoch, best_val, train_seconds = train_deep_model(
            model_factory(model_name, input_size),
            X_train,
            X_val,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )
    except RuntimeError as exc:
        print(f"  [SKIP] {model_name} failed during training: {exc}")
        return []
    train_scores = reconstruction_errors(model, X_train, device)
    val_scores = reconstruction_errors(model, X_val, device)
    test_scores = reconstruction_errors(model, X_test, device)
    if not (
        np.isfinite(train_scores).all()
        and np.isfinite(val_scores).all()
        and np.isfinite(test_scores).all()
    ):
        print(f"  [SKIP] {model_name} produced non-finite anomaly scores.")
        return []
    latency = cpu_latency_ms_per_window(model, input_size=input_size)

    model_path = os.path.join(
        OUT_DIR,
        "models",
        f"{protocol}_{heldout_group}_{model_name}.pt".replace(os.sep, "_").replace(":", "_"),
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
    bearings: tuple[str, ...],
    max_files_per_bearing: int | None,
    max_files_per_condition: int | None,
    max_epochs: int,
    patience: int,
    seed: int,
    n_file_splits: int,
    download: bool,
    extract: bool,
    mat_dir: Path,
    window_size: int,
    stride: int,
    output_name: str = "paderborn_metrics.csv",
) -> list[EvalRow]:
    ensure_dir(OUT_DIR)
    if download:
        download_paderborn_archives(bearings=bearings)
    if extract:
        extract_paderborn_archives()

    X_raw, y_raw, _, metadata = load_paderborn_with_metadata(
        mat_dir=mat_dir,
        bearings=bearings,
        max_files_per_bearing=max_files_per_bearing,
        max_files_per_condition=max_files_per_condition,
        window_size=window_size,
        stride=stride,
    )
    X = normalize_per_sample(X_raw)
    y = y_raw

    split_builders = []
    if "condition-wise" in protocols:
        split_builders.extend(build_condition_splits(y, metadata))
    if "bearing-wise" in protocols:
        split_builders.extend(build_bearing_splits(y, metadata, seed=seed))
    if "file-wise" in protocols:
        split_builders.extend(build_file_splits(y, metadata, n_splits=n_file_splits, seed=seed))

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
                    input_size=window_size,
                )
            rows.extend(model_rows)

    out_path = os.path.join(OUT_DIR, output_name)
    _write_rows(rows, out_path)
    print(f"\nWrote {out_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["condition-wise", "bearing-wise", "file-wise"],
        choices=["condition-wise", "bearing-wise", "file-wise"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vae", "cnn-ae", "iforest"],
        choices=["vae", "cnn-ae", "lstm-ae", "iforest"],
    )
    parser.add_argument("--bearings", nargs="+", default=list(DEFAULT_BEARINGS))
    parser.add_argument("--max-files-per-bearing", type=int, default=None)
    parser.add_argument("--max-files-per-condition", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-file-splits", type=int, default=5)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--mat-dir", type=Path, default=PADERBORN_MAT_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--output-name", default="paderborn_metrics.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        protocols=args.protocols,
        models=args.models,
        bearings=tuple(args.bearings),
        max_files_per_bearing=args.max_files_per_bearing,
        max_files_per_condition=args.max_files_per_condition,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        n_file_splits=args.n_file_splits,
        download=args.download,
        extract=args.extract,
        mat_dir=args.mat_dir,
        window_size=args.window_size,
        stride=args.stride,
        output_name=args.output_name,
    )
