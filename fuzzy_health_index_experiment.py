"""Fuzzy threshold and health-index experiments for journal extension.

The fuzzy layer is intentionally placed after the anomaly-score model. It does
not replace VAE/AE/Isolation Forest training; it converts calibrated anomaly
scores into graded health states so threshold behavior can be studied under
dataset and protocol shifts.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import download_and_load_with_metadata
from journal_experiment_utils import (
    ensure_dir,
    isolation_forest_scores,
    normal_train_val_split,
    statistical_features,
)
from paderborn_data_loader import PADERBORN_MAT_DIR, load_paderborn_with_metadata
from paderborn_external_experiments import (
    build_bearing_splits as build_paderborn_bearing_splits,
    build_condition_splits as build_paderborn_condition_splits,
    build_file_splits as build_paderborn_file_splits,
)
from preprocessor import balance_dataset_with_indices, normalize_per_sample
from strict_cwru_journal_experiments import (
    build_fault_size_splits,
    build_file_splits as build_cwru_file_splits,
    build_load_splits,
)


OUT_DIR = os.path.join("results", "fuzzy_health_index")


@dataclass
class FuzzyRow:
    dataset: str
    protocol: str
    heldout_group: str
    model: str
    decision_policy: str
    train_normal_count: int
    val_normal_count: int
    test_normal_count: int
    test_fault_count: int
    p50: float
    p90: float
    p95: float
    p99: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    miss_rate: float
    uncertain_rate: float
    mean_health_normal: float
    mean_health_fault: float
    normal_state_rate: float
    warning_state_rate: float
    fault_state_rate: float
    tn: int
    fp: int
    fn: int
    tp: int


def _safe_ramp(x: np.ndarray, left: float, right: float) -> np.ndarray:
    width = max(right - left, 1e-12)
    return np.clip((x - left) / width, 0.0, 1.0)


def fuzzy_membership(scores: np.ndarray, p50: float, p90: float, p95: float, p99: float):
    """Return fuzzy memberships and a scalar health index in [0, 1].

    The anchors come from validation-normal scores. P95 marks the warning
    center, while P99 marks high-confidence fault membership.
    """
    normal = 1.0 - _safe_ramp(scores, p90, p95)
    warning_left = _safe_ramp(scores, p50, p95)
    warning_right = 1.0 - _safe_ramp(scores, p95, p99)
    warning = np.minimum(warning_left, warning_right)
    fault = _safe_ramp(scores, p95, p99)
    denom = normal + warning + fault + 1e-12
    health = (0.0 * normal + 0.5 * warning + 1.0 * fault) / denom
    states = np.argmax(np.vstack([normal, warning, fault]), axis=0)
    return normal, warning, fault, health, states


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray):
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


def rows_for_scores(
    dataset: str,
    protocol: str,
    heldout_group: str,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
) -> list[FuzzyRow]:
    p50, p90, p95, p99 = [float(np.percentile(val_scores, p)) for p in (50, 90, 95, 99)]
    _, _, _, health, states = fuzzy_membership(test_scores, p50, p90, p95, p99)

    policies = {
        "hard_val_p95": (test_scores > p95).astype(np.int32),
        "fuzzy_warning_as_alarm_hi50": (health >= 0.50).astype(np.int32),
        "fuzzy_fault_only_hi75": (health >= 0.75).astype(np.int32),
    }
    uncertain = ((health >= 0.25) & (health < 0.75)).astype(np.int32)

    rows: list[FuzzyRow] = []
    for policy, y_pred in policies.items():
        metrics = binary_metrics(y_test, health, y_pred)
        rows.append(
            FuzzyRow(
                dataset=dataset,
                protocol=protocol,
                heldout_group=heldout_group,
                model="iforest",
                decision_policy=policy,
                train_normal_count=len(train_scores),
                val_normal_count=len(val_scores),
                test_normal_count=int((y_test == 0).sum()),
                test_fault_count=int((y_test == 1).sum()),
                p50=p50,
                p90=p90,
                p95=p95,
                p99=p99,
                uncertain_rate=float(uncertain.mean()),
                mean_health_normal=float(health[y_test == 0].mean()) if (y_test == 0).any() else 0.0,
                mean_health_fault=float(health[y_test == 1].mean()) if (y_test == 1).any() else 0.0,
                normal_state_rate=float((states == 0).mean()),
                warning_state_rate=float((states == 1).mean()),
                fault_state_rate=float((states == 2).mean()),
                **metrics,
            )
        )
    return rows


def _write_rows(rows: list[FuzzyRow], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FuzzyRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_split(dataset: str, protocol: str, heldout_group: str, X, y, train_normal_mask, test_mask, seed: int):
    X_train_pool = X[train_normal_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    if len(X_train_pool) < 10 or (y_test == 0).sum() == 0 or (y_test == 1).sum() == 0:
        print(f"[SKIP] {dataset}:{protocol}:{heldout_group}")
        return []
    X_train, X_val = normal_train_val_split(X_train_pool, val_fraction=0.2, seed=seed)
    model, train_scores, test_scores = isolation_forest_scores(X_train, X_test, seed=seed)
    val_scores = -model.score_samples(statistical_features(X_val))
    print(
        f"{dataset}:{protocol}:{heldout_group} "
        f"train_normal={len(X_train)} test_normal={(y_test == 0).sum()} test_fault={(y_test == 1).sum()}"
    )
    return rows_for_scores(dataset, protocol, heldout_group, train_scores, val_scores, test_scores, y_test)


def run_cwru(protocols: list[str], seed: int, n_file_splits: int) -> list[FuzzyRow]:
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
        split_builders.extend(build_cwru_file_splits(y, metadata, n_splits=n_file_splits, seed=seed))

    rows = []
    for protocol, heldout_group, train_normal_mask, test_mask in split_builders:
        rows.extend(run_split("CWRU", protocol, heldout_group, X, y, train_normal_mask, test_mask, seed))
    return rows


def run_paderborn(
    protocols: list[str],
    bearings: tuple[str, ...],
    mat_dir: Path,
    max_files_per_condition: int | None,
    window_size: int,
    stride: int,
    seed: int,
    n_file_splits: int,
) -> list[FuzzyRow]:
    X_raw, y_raw, _, metadata = load_paderborn_with_metadata(
        mat_dir=mat_dir,
        bearings=bearings,
        max_files_per_condition=max_files_per_condition,
        window_size=window_size,
        stride=stride,
    )
    X = normalize_per_sample(X_raw)
    y = y_raw

    split_builders = []
    if "condition-wise" in protocols:
        split_builders.extend(build_paderborn_condition_splits(y, metadata))
    if "bearing-wise" in protocols:
        split_builders.extend(build_paderborn_bearing_splits(y, metadata, seed=seed))
    if "file-wise" in protocols:
        split_builders.extend(
            build_paderborn_file_splits(y, metadata, n_splits=n_file_splits, seed=seed)
        )

    rows = []
    for protocol, heldout_group, train_normal_mask, test_mask in split_builders:
        rows.extend(run_split("Paderborn", protocol, heldout_group, X, y, train_normal_mask, test_mask, seed))
    return rows


def summarize(rows: list[FuzzyRow]) -> None:
    df = pd.DataFrame([asdict(row) for row in rows])
    summary = (
        df.groupby(["dataset", "protocol", "decision_policy"])[
            [
                "roc_auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
                "false_alarm_rate",
                "miss_rate",
                "uncertain_rate",
                "mean_health_normal",
                "mean_health_fault",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary_path = os.path.join(OUT_DIR, "fuzzy_health_index_summary.csv")
    summary.to_csv(summary_path, index=False)

    policy_summary = (
        df.groupby(["dataset", "decision_policy"])[
            ["f1", "false_alarm_rate", "miss_rate", "uncertain_rate", "mean_health_normal", "mean_health_fault"]
        ]
        .mean()
        .reset_index()
    )
    policy_path = os.path.join(OUT_DIR, "fuzzy_policy_summary.csv")
    policy_summary.to_csv(policy_path, index=False)

    md_path = os.path.join(OUT_DIR, "fuzzy_health_index_summary.md")
    lines = [
        "# Fuzzy Health Index Summary\n\n",
        "The fuzzy layer converts calibrated anomaly scores into a graded health index.\n\n",
        "Membership anchors are estimated from validation-normal scores: P50, P90, P95, and P99. ",
        "P95 is treated as the warning center and P99 as high-confidence fault membership.\n\n",
        "## Policy-Level Mean\n\n",
        md_table(policy_summary),
        "\n\n## Protocol-Level Mean\n\n",
        md_table(summary),
        "\n\n## Interpretation\n\n",
        "- `hard_val_p95` is the original calibrated hard-threshold baseline.\n",
        "- `fuzzy_warning_as_alarm_hi50` treats warning and fault states as alarms.\n",
        "- `fuzzy_fault_only_hi75` alarms only on high-confidence fuzzy fault membership.\n",
        "- `uncertain_rate` quantifies the gray zone instead of forcing every window into a binary label.\n",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Wrote {summary_path}")
    print(f"Wrote {policy_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cwru", "paderborn"], choices=["cwru", "paderborn"])
    parser.add_argument(
        "--cwru-protocols",
        nargs="+",
        default=["load-wise", "fault-size-wise", "file-wise"],
        choices=["load-wise", "fault-size-wise", "file-wise"],
    )
    parser.add_argument(
        "--paderborn-protocols",
        nargs="+",
        default=["condition-wise", "bearing-wise", "file-wise"],
        choices=["condition-wise", "bearing-wise", "file-wise"],
    )
    parser.add_argument(
        "--bearings",
        nargs="+",
        default=["K001", "K002", "K003", "K004", "K005", "K006", "KA01", "KA03", "KA05", "KA07", "KI01", "KI03", "KI05", "KI07"],
    )
    parser.add_argument("--mat-dir", type=Path, default=PADERBORN_MAT_DIR)
    parser.add_argument("--max-files-per-condition", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-file-splits", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dir(OUT_DIR)
    all_rows: list[FuzzyRow] = []
    if "cwru" in args.datasets:
        all_rows.extend(run_cwru(args.cwru_protocols, seed=args.seed, n_file_splits=args.n_file_splits))
    if "paderborn" in args.datasets:
        all_rows.extend(
            run_paderborn(
                args.paderborn_protocols,
                bearings=tuple(args.bearings),
                mat_dir=args.mat_dir,
                max_files_per_condition=args.max_files_per_condition,
                window_size=args.window_size,
                stride=args.stride,
                seed=args.seed,
                n_file_splits=args.n_file_splits,
            )
        )
    out_path = os.path.join(OUT_DIR, "fuzzy_health_index_metrics.csv")
    _write_rows(all_rows, out_path)
    print(f"Wrote {out_path}")
    summarize(all_rows)
