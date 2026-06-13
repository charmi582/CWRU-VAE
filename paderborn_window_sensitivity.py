"""Run Paderborn window-size sensitivity experiments.

This script is a diagnostic layer over `paderborn_external_experiments.py`.
It keeps the external dataset subset fixed while changing the segmentation
window so the journal paper can discuss whether the weak Paderborn pilot is
caused by an overly short CWRU-style 1024-point window.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import STRIDE, WINDOW_SIZE
from paderborn_data_loader import DEFAULT_BEARINGS, PADERBORN_MAT_DIR
from paderborn_external_experiments import OUT_DIR, run


def _write_markdown_summary(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall", "false_alarm_rate", "miss_rate"]
    threshold_summary = (
        df.groupby(["window_size", "threshold_strategy"])[metrics]
        .mean()
        .round(4)
        .reset_index()
    )
    best = (
        df.groupby(["window_size", "protocol", "model", "threshold_strategy"])[metrics]
        .mean()
        .reset_index()
        .sort_values(
            ["window_size", "protocol", "model", "f1", "false_alarm_rate"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["window_size", "protocol", "model"])
        .head(1)
        .round(4)
    )

    def md_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        rows = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for col in cols:
                value = row[col]
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    lines = [
        "# Paderborn Window-Size Sensitivity Summary\n",
        "This diagnostic keeps the Paderborn bearing subset fixed and changes only the segmentation window size.",
        "It is intended to test whether the CWRU-style 1024-point window is too short for Paderborn.\n",
        "## Threshold Summary by Window Size\n",
        md_table(threshold_summary),
        "\n## Best Threshold by Window Size, Protocol, and Model\n",
        md_table(best[["window_size", "protocol", "model", "threshold_strategy", "roc_auc", "pr_auc", "f1", "false_alarm_rate", "miss_rate"]]),
        "\n## Interpretation Template\n",
        "- If longer windows improve ROC-AUC/PR-AUC, the paper should report Paderborn as a sampling-rate-sensitive external validation case.",
        "- If longer windows do not improve results, the paper should frame Paderborn as evidence of cross-dataset domain shift rather than a simple segmentation issue.",
        "- In both cases, threshold calibration should remain a core contribution because FAR/Miss trade-offs vary strongly by dataset and protocol.\n",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    parser.add_argument("--stride-ratio", type=float, default=0.5)
    parser.add_argument("--protocols", nargs="+", default=["condition-wise", "bearing-wise", "file-wise"])
    parser.add_argument("--models", nargs="+", default=["iforest"])
    parser.add_argument("--bearings", nargs="+", default=["K001", "K002", "KA01", "KI01"])
    parser.add_argument("--max-files-per-condition", type=int, default=2)
    parser.add_argument("--max-files-per-bearing", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-file-splits", type=int, default=4)
    parser.add_argument("--mat-dir", type=Path, default=PADERBORN_MAT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for window_size in args.window_sizes:
        stride = max(1, int(round(window_size * args.stride_ratio)))
        print(f"\n=== window_size={window_size} stride={stride} ===")
        rows = run(
            protocols=args.protocols,
            models=args.models,
            bearings=tuple(args.bearings),
            max_files_per_bearing=args.max_files_per_bearing,
            max_files_per_condition=args.max_files_per_condition,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
            n_file_splits=args.n_file_splits,
            download=False,
            extract=False,
            mat_dir=args.mat_dir,
            window_size=window_size,
            stride=stride,
            output_name=f"paderborn_metrics_window_{window_size}.csv",
        )
        frame = pd.DataFrame([row.__dict__ for row in rows])
        frame.insert(0, "window_size", window_size)
        frame.insert(1, "stride", stride)
        all_frames.append(frame)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = out_dir / "paderborn_window_sensitivity_metrics.csv"
    combined.to_csv(combined_path, index=False)

    summary_path = out_dir / "paderborn_window_sensitivity_summary.md"
    _write_markdown_summary(combined, summary_path)
    print(f"Wrote {combined_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
