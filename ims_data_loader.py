"""NASA IMS bearing dataset loader for external federated validation.

The IMS dataset contains run-to-failure bearing experiments. For normal-only
anomaly monitoring, this loader labels early-life windows as normal and late-life
windows as fault audit windows, leaving the middle degradation region unused.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from config import STRIDE, WINDOW_SIZE


IMS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
IMS_ROOT = Path("results") / "external" / "ims"
IMS_RAW_DIR = IMS_ROOT / "raw"
IMS_EXTRACTED_DIR = IMS_ROOT / "extracted"


def download_ims_archive(raw_dir: Path = IMS_RAW_DIR, overwrite: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "IMS_Bearings.zip"
    if archive.exists() and archive.stat().st_size > 1024 and not overwrite:
        return archive
    with requests.get(IMS_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(archive, "wb") as f, tqdm(
            desc="download IMS_Bearings.zip",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    bar.update(f.write(chunk))
    return archive


def extract_ims_archive(archive: Path, extracted_dir: Path = IMS_EXTRACTED_DIR) -> Path:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    if any(extracted_dir.rglob("*")):
        return extracted_dir
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted_dir)
    return extracted_dir


def _segment(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = [
        signal[start : start + window_size]
        for start in range(0, len(signal) - window_size + 1, stride)
    ]
    return np.asarray(windows, dtype=np.float32) if windows else np.empty((0, window_size), dtype=np.float32)


def _numeric_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and not path.name.lower().endswith((".zip", ".rar", ".pdf", ".doc", ".docx")):
            files.append(path)
    return sorted(files)


def _read_numeric_file(path: Path) -> np.ndarray | None:
    try:
        data = np.loadtxt(path)
    except Exception:
        try:
            data = np.loadtxt(path, delimiter="\t")
        except Exception:
            return None
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] < WINDOW_SIZE:
        return None
    return data.astype(np.float32)


def _test_name(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for part in parts:
        if "1st" in part or "test1" in part:
            return "test1"
        if "2nd" in part or "test2" in part:
            return "test2"
        if "3rd" in part or "test3" in part:
            return "test3"
    return "unknown_test"


def _channel_to_bearing(test_name: str, channel_idx: int, num_channels: int) -> str:
    if num_channels >= 8:
        return f"{test_name}_bearing{channel_idx // 2 + 1}"
    return f"{test_name}_bearing{channel_idx + 1}"


def load_ims_with_metadata(
    extracted_dir: Path = IMS_EXTRACTED_DIR,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    early_fraction: float = 0.20,
    late_fraction: float = 0.20,
    max_files_per_test: int | None = None,
    channels_per_bearing: str = "first",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load IMS run-to-failure files as normal/fault audit windows.

    Parameters
    ----------
    early_fraction:
        Fraction of the earliest files in each test treated as normal.
    late_fraction:
        Fraction of the latest files in each test treated as fault audit data.
    channels_per_bearing:
        `first` keeps one channel per bearing; `all` keeps all channels.
    """
    files = _numeric_files(extracted_dir)
    if not files:
        raise FileNotFoundError(
            f"No IMS numeric files found in {extracted_dir}. Run with --download "
            "or extract the NASA IMS archive into this directory."
        )

    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(_test_name(path), []).append(path)

    segments, y_bin, y_multi = [], [], []
    file_meta, test_meta, bearing_meta, channel_meta, phase_meta = [], [], [], [], []
    for test_name, paths in sorted(grouped.items()):
        paths = sorted(paths)
        if max_files_per_test is not None:
            paths = paths[:max_files_per_test]
        if len(paths) < 5:
            continue
        n = len(paths)
        early_cut = max(1, int(round(n * early_fraction)))
        late_start = min(n - 1, int(round(n * (1.0 - late_fraction))))
        selected = [(path, "normal", 0) for path in paths[:early_cut]]
        selected.extend((path, "fault", 1) for path in paths[late_start:])

        for path, phase, label in selected:
            data = _read_numeric_file(path)
            if data is None:
                continue
            channel_indices = range(data.shape[1])
            if channels_per_bearing == "first" and data.shape[1] >= 2:
                channel_indices = range(0, data.shape[1], 2)
            for channel_idx in channel_indices:
                signal = data[:, channel_idx]
                windows = _segment(signal, window_size=window_size, stride=stride)
                if len(windows) == 0:
                    continue
                bearing = _channel_to_bearing(test_name, channel_idx, data.shape[1])
                segments.append(windows)
                y_bin.append(np.full(len(windows), label, dtype=np.int32))
                y_multi.append(np.full(len(windows), label, dtype=np.int32))
                file_meta.append(np.full(len(windows), str(path), dtype=object))
                test_meta.append(np.full(len(windows), test_name, dtype=object))
                bearing_meta.append(np.full(len(windows), bearing, dtype=object))
                channel_meta.append(np.full(len(windows), str(channel_idx), dtype=object))
                phase_meta.append(np.full(len(windows), phase, dtype=object))

    if not segments:
        raise RuntimeError(f"No usable IMS windows were loaded from {extracted_dir}")

    X = np.concatenate(segments, axis=0)
    y_binary = np.concatenate(y_bin, axis=0)
    y_multiclass = np.concatenate(y_multi, axis=0)
    metadata = {
        "file": np.concatenate(file_meta, axis=0),
        "test": np.concatenate(test_meta, axis=0),
        "bearing": np.concatenate(bearing_meta, axis=0),
        "condition": np.concatenate(test_meta, axis=0),
        "channel": np.concatenate(channel_meta, axis=0),
        "phase": np.concatenate(phase_meta, axis=0),
        "dataset": np.full(len(X), "ims", dtype=object),
    }
    print(f"Loaded IMS: {len(X):,} windows, normal={(y_binary == 0).sum():,}, fault={(y_binary == 1).sum():,}")
    return X, y_binary, y_multiclass, metadata
