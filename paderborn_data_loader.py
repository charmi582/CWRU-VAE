"""Paderborn bearing dataset loader for external validation experiments.

The official Paderborn Bearing DataCenter distributes one archive per bearing
state. This module intentionally keeps raw downloads outside git and converts
the extracted MATLAB files into the same windowed representation used by the
CWRU experiments.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import requests
import scipy.io
from tqdm import tqdm

from config import STRIDE, WINDOW_SIZE


PADERBORN_BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter"
PADERBORN_ROOT = Path("results") / "external" / "paderborn"
PADERBORN_RAW_DIR = PADERBORN_ROOT / "raw"
PADERBORN_MAT_DIR = PADERBORN_ROOT / "mat"

DEFAULT_BEARINGS = ("K001", "K002", "K003", "KA01", "KA03", "KI01", "KI03")
HEALTHY_PREFIX = "K"
FAULT_PREFIXES = ("KA", "KI", "KB")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_paderborn_archives(
    bearings: tuple[str, ...] = DEFAULT_BEARINGS,
    raw_dir: Path = PADERBORN_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Download selected Paderborn bearing archives.

    The full dataset is several GB. For journal experiments, start with a
    representative subset and expand after the pipeline is verified.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for bearing in bearings:
        archive = raw_dir / f"{bearing}.rar"
        paths.append(archive)
        if archive.exists() and archive.stat().st_size > 1024 and not overwrite:
            continue
        url = f"{PADERBORN_BASE_URL}/{bearing}.rar"
        with requests.get(url, stream=True, timeout=60, headers=HEADERS) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(archive, "wb") as f, tqdm(
                desc=f"download {archive.name}",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        bar.update(f.write(chunk))
    return paths


def extract_paderborn_archives(
    raw_dir: Path = PADERBORN_RAW_DIR,
    mat_dir: Path = PADERBORN_MAT_DIR,
) -> None:
    """Extract downloaded archives using 7-Zip or bsdtar when available."""
    mat_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(raw_dir.glob("*.rar"))
    if not archives:
        raise FileNotFoundError(f"No .rar archives found in {raw_dir}")

    seven_zip = shutil.which("7z") or shutil.which("7za")
    tar = shutil.which("tar")
    if not seven_zip and not tar:
        raise RuntimeError(
            "Cannot extract .rar files automatically. Install 7-Zip or extract "
            f"the archives manually into {mat_dir}."
        )

    for archive in archives:
        target = mat_dir / archive.stem
        target.mkdir(parents=True, exist_ok=True)
        if any(target.rglob("*.mat")):
            continue
        if seven_zip:
            cmd = [seven_zip, "x", str(archive), f"-o{target}", "-y"]
        else:
            cmd = [tar, "-xf", str(archive), "-C", str(target)]
        subprocess.run(cmd, check=True)


def _segment(signal: np.ndarray) -> np.ndarray:
    segs = [
        signal[s : s + WINDOW_SIZE]
        for s in range(0, len(signal) - WINDOW_SIZE + 1, STRIDE)
    ]
    return np.array(segs, dtype=np.float32) if segs else np.empty((0, WINDOW_SIZE))


def _is_numeric_vector(value) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and value.size >= WINDOW_SIZE


def _walk_matlab(value):
    """Yield nested MATLAB objects, arrays, dict values, and struct fields."""
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            if not str(key).startswith("__"):
                yield from _walk_matlab(child)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for child in value.flat:
            yield from _walk_matlab(child)
    elif hasattr(value, "_fieldnames"):
        for field in value._fieldnames:
            yield from _walk_matlab(getattr(value, field))


def _name_of_matlab_signal(value) -> str:
    name = getattr(value, "Name", "")
    if isinstance(name, np.ndarray):
        name = "".join(str(x) for x in name.flat)
    return str(name).lower()


def _extract_paderborn_vibration(path: Path) -> np.ndarray | None:
    """Extract the primary vibration channel from a Paderborn MATLAB file."""
    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"[WARN] Cannot load {path}: {exc}")
        return None

    named_candidates = []
    numeric_candidates = []
    for obj in _walk_matlab(mat):
        name = _name_of_matlab_signal(obj)
        if hasattr(obj, "Data") and _is_numeric_vector(getattr(obj, "Data")):
            data = np.asarray(getattr(obj, "Data")).flatten().astype(np.float32)
            if "vibration" in name or "acc" in name:
                named_candidates.append(data)
            else:
                numeric_candidates.append(data)
        elif _is_numeric_vector(obj):
            numeric_candidates.append(np.asarray(obj).flatten().astype(np.float32))

    if named_candidates:
        return max(named_candidates, key=len)
    if numeric_candidates:
        return max(numeric_candidates, key=len)
    return None


def _parse_paderborn_filename(path: Path) -> tuple[str, str, str]:
    """Return operating condition, bearing code, and measurement id."""
    match = re.search(r"(N\d+_M\d+_F\d+)_(K[A-Z]?\d{2,3})_(\d+)", path.stem)
    if match:
        return match.group(1), match.group(2), match.group(3)
    bearing_match = re.search(r"(K[A-Z]?\d{2,3})", path.stem)
    bearing = bearing_match.group(1) if bearing_match else path.parent.name
    return "unknown", bearing, "unknown"


def _binary_label(bearing_code: str) -> int:
    if re.fullmatch(r"K\d{3}", bearing_code):
        return 0
    if bearing_code.startswith(FAULT_PREFIXES):
        return 1
    return 1


def _label_name(bearing_code: str) -> str:
    if _binary_label(bearing_code) == 0:
        return "Healthy"
    if bearing_code.startswith("KI"):
        return "Fault_KI"
    if bearing_code.startswith("KA"):
        return "Fault_KA"
    if bearing_code.startswith("KB"):
        return "Fault_KB"
    return "Fault"


def load_paderborn_with_metadata(
    mat_dir: Path = PADERBORN_MAT_DIR,
    bearings: tuple[str, ...] | None = DEFAULT_BEARINGS,
    max_files_per_bearing: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load extracted Paderborn MATLAB files as windowed anomaly data."""
    mat_paths = sorted(mat_dir.rglob("*.mat"))
    if bearings:
        wanted = set(bearings)
        mat_paths = [
            p for p in mat_paths
            if _parse_paderborn_filename(p)[1] in wanted or p.parent.name in wanted
        ]
    if not mat_paths:
        raise FileNotFoundError(
            f"No Paderborn .mat files found in {mat_dir}. Download/extract archives "
            "or pass --download and --extract to the experiment script."
        )

    per_bearing_seen: dict[str, int] = {}
    segments, y_bin, y_multi = [], [], []
    files, conditions, bearings_seen, measurements, label_names = [], [], [], [], []
    label_to_id = {"Healthy": 0, "Fault_KA": 1, "Fault_KI": 2, "Fault_KB": 3, "Fault": 4}

    for path in mat_paths:
        condition, bearing, measurement = _parse_paderborn_filename(path)
        seen = per_bearing_seen.get(bearing, 0)
        if max_files_per_bearing is not None and seen >= max_files_per_bearing:
            continue
        per_bearing_seen[bearing] = seen + 1

        signal = _extract_paderborn_vibration(path)
        if signal is None or len(signal) < WINDOW_SIZE:
            print(f"[WARN] Skipping {path}: no usable vibration vector")
            continue
        segs = _segment(signal)
        if len(segs) == 0:
            continue

        label_name = _label_name(bearing)
        binary = _binary_label(bearing)
        multi = label_to_id[label_name]
        segments.append(segs)
        y_bin.append(np.full(len(segs), binary, dtype=np.int32))
        y_multi.append(np.full(len(segs), multi, dtype=np.int32))
        files.append(np.full(len(segs), str(path), dtype=object))
        conditions.append(np.full(len(segs), condition, dtype=object))
        bearings_seen.append(np.full(len(segs), bearing, dtype=object))
        measurements.append(np.full(len(segs), measurement, dtype=object))
        label_names.append(np.full(len(segs), label_name, dtype=object))

    if not segments:
        raise RuntimeError(f"No usable Paderborn signals were loaded from {mat_dir}")

    X = np.concatenate(segments, axis=0)
    y_binary = np.concatenate(y_bin, axis=0)
    y_multiclass = np.concatenate(y_multi, axis=0)
    metadata = {
        "file": np.concatenate(files, axis=0),
        "condition": np.concatenate(conditions, axis=0),
        "bearing": np.concatenate(bearings_seen, axis=0),
        "measurement": np.concatenate(measurements, axis=0),
        "label_name": np.concatenate(label_names, axis=0),
        "dataset": np.full(len(X), "paderborn", dtype=object),
    }
    print(f"Loaded Paderborn: {len(X):,} windows, normal={(y_binary == 0).sum():,}, fault={(y_binary == 1).sum():,}")
    return X, y_binary, y_multiclass, metadata
