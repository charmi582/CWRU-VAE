"""IMS external validation readiness and pilot experiment.

This script prepares the second external dataset track for the federated journal
paper. It can download/extract the NASA IMS archive when requested, then builds
a normal-only anomaly dataset using early-life windows as normal and late-life
windows as audit faults.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ims_data_loader import (
    IMS_EXTRACTED_DIR,
    IMS_RAW_DIR,
    download_ims_archive,
    extract_ims_archive,
    load_ims_with_metadata,
)
from journal_experiment_utils import ensure_dir


OUT_DIR = os.path.join("results", "external", "ims")


def write_readiness_report(path: str, archive_exists: bool, extracted_exists: bool) -> None:
    lines = [
        "# IMS External Validation Readiness\n\n",
        "The IMS Bearing dataset is selected as the next external validation dataset because it provides run-to-failure bearing experiments. For the present normal-only anomaly monitoring setting, early-life files are treated as normal training candidates and late-life files are reserved as audit fault windows.\n\n",
        "## Data status\n\n",
        f"- Archive present: `{archive_exists}`\n",
        f"- Extracted numeric files present: `{extracted_exists}`\n\n",
        "## Official source\n\n",
        "- NASA PCoE IMS Bearings download: `https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip`\n",
        "- Approximate archive size: 1.07 GB\n\n",
        "## Planned protocol\n\n",
        "- Clients by bearing: each IMS test-bearing channel is treated as a client.\n",
        "- Clients by test condition: each run-to-failure experiment is treated as a condition client.\n",
        "- Normal-only training: earliest 20% of files from each test.\n",
        "- Fault audit: latest 20% of files from each test.\n",
        "- Middle degradation region: excluded from the first external anomaly-monitoring protocol.\n",
        "- Main comparison: local-only, centralized, FedAvg, FedProx, FedBN and personalized variants.\n",
        "- Calibration policies: client-specific, pooled and adaptive drift-aware calibration.\n\n",
        "## Run command\n\n",
        "```bash\n",
        "python ims_external_validation.py --download --extract --summarize\n",
        "```\n",
    ]
    Path(path).write_text("".join(lines), encoding="utf-8")


def summarize_ims(args: argparse.Namespace) -> None:
    X, y, _, metadata = load_ims_with_metadata(
        extracted_dir=args.extracted_dir,
        window_size=args.window_size,
        stride=args.stride,
        early_fraction=args.early_fraction,
        late_fraction=args.late_fraction,
        max_files_per_test=args.max_files_per_test,
    )
    df = pd.DataFrame(
        {
            "label": y,
            "test": metadata["test"],
            "bearing": metadata["bearing"],
            "phase": metadata["phase"],
        }
    )
    summary = (
        df.groupby(["test", "bearing", "phase", "label"])
        .size()
        .reset_index(name="windows")
    )
    summary_path = os.path.join(OUT_DIR, "ims_window_summary.csv")
    summary.to_csv(summary_path, index=False)
    text = [
        "# IMS Window Summary\n\n",
        f"- Windows: {len(X):,}\n",
        f"- Normal windows: {(y == 0).sum():,}\n",
        f"- Fault audit windows: {(y == 1).sum():,}\n",
        f"- Window size: {args.window_size}\n",
        f"- Stride: {args.stride}\n\n",
        summary.to_markdown(index=False),
        "\n",
    ]
    Path(os.path.join(OUT_DIR, "ims_window_summary.md")).write_text("".join(text), encoding="utf-8")
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=IMS_RAW_DIR)
    parser.add_argument("--extracted-dir", type=Path, default=IMS_EXTRACTED_DIR)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--early-fraction", type=float, default=0.20)
    parser.add_argument("--late-fraction", type=float, default=0.20)
    parser.add_argument("--max-files-per-test", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUT_DIR)
    archive = args.raw_dir / "IMS_Bearings.zip"
    if args.download:
        archive = download_ims_archive(args.raw_dir)
    if args.extract:
        extract_ims_archive(archive, args.extracted_dir)

    archive_exists = archive.exists() and archive.stat().st_size > 1024
    extracted_exists = any(args.extracted_dir.rglob("*")) if args.extracted_dir.exists() else False
    report_path = os.path.join(OUT_DIR, "ims_external_validation_readiness.md")
    write_readiness_report(report_path, archive_exists, extracted_exists)
    print(f"Wrote {report_path}")

    if args.summarize:
        summarize_ims(args)


if __name__ == "__main__":
    main()
