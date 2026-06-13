"""Download CWRU .mat files and convert them to windowed numpy segments."""
from __future__ import annotations

import os
import re
import numpy as np
import scipy.io
import requests
from tqdm import tqdm

from config import BASE_URL, FILE_CONFIGS, DATA_DIR, WINDOW_SIZE, STRIDE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _download(url: str, path: str, retries: int = 3) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return True

    for attempt in range(1, retries + 1):
        if os.path.exists(path):
            os.remove(path)
        try:
            r = requests.get(url, stream=True, timeout=60, headers=HEADERS)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(path, "wb") as f, tqdm(
                desc=f"  {os.path.basename(path)}",
                total=total, unit="iB", unit_scale=True,
                unit_divisor=1024, leave=False,
            ) as bar:
                for chunk in r.iter_content(8192):
                    bar.update(f.write(chunk))
            # Verify it looks like a MAT file (not an HTML error page)
            if os.path.getsize(path) < 1024:
                raise RuntimeError("downloaded file is unexpectedly small")
            return True
        except Exception as exc:
            print(f"  [WARN] Cannot download {url} (attempt {attempt}/{retries}): {exc}")

    if os.path.exists(path):
        os.remove(path)
    return False


def _extract_de_time(path: str) -> np.ndarray | None:
    """Return the Drive-End time-series from a CWRU .mat file."""
    try:
        mat = scipy.io.loadmat(path)
    except Exception as exc:
        print(f"  [WARN] Cannot load {path}: {exc}")
        return None

    # Primary pattern: keys containing 'DE_time'
    for key in mat:
        if key.startswith("_"):
            continue
        if "DE_time" in key:
            return mat[key].flatten().astype(np.float32)

    # Fallback: first array larger than WINDOW_SIZE
    for key in mat:
        if key.startswith("_"):
            continue
        v = mat[key]
        if isinstance(v, np.ndarray) and v.size >= WINDOW_SIZE:
            return v.flatten().astype(np.float32)

    return None


def _segment(signal: np.ndarray) -> np.ndarray:
    segs = [
        signal[s : s + WINDOW_SIZE]
        for s in range(0, len(signal) - WINDOW_SIZE + 1, STRIDE)
    ]
    return np.array(segs, dtype=np.float32) if segs else np.empty((0, WINDOW_SIZE))


def _fault_size_from_label(label_name: str) -> str:
    """Return CWRU fault-size group from labels such as IR_007 or Ball_014."""
    if label_name == "Normal":
        return "normal"
    match = re.search(r"_(\d{3})$", label_name)
    return match.group(1) if match else "unknown"


def download_and_load_with_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Download CWRU data, segment into windows, and return arrays.

    Returns
    -------
    X          : (N, WINDOW_SIZE) float32
    y_binary   : (N,) int  — 0 = normal, 1 = fault
    y_multi    : (N,) int  — 0/1/2/3 fault-type labels
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print(" Step 1 — Downloading CWRU Bearing Dataset")
    print("=" * 60)

    segments_list, binary_list, multi_list = [], [], []
    file_list, load_list, label_name_list, fault_size_list = [], [], [], []
    skipped = []

    for file_i, (fname, label_name, label_id) in enumerate(tqdm(FILE_CONFIGS, desc="Downloading")):
        url  = BASE_URL + fname
        path = os.path.join(DATA_DIR, fname)

        if not _download(url, path):
            skipped.append(fname)
            continue

        sig = _extract_de_time(path)
        if sig is None or len(sig) < WINDOW_SIZE:
            skipped.append(fname)
            continue

        segs = _segment(sig)
        if len(segs) == 0:
            continue

        binary = 0 if label_id == 0 else 1
        segments_list.append(segs)
        binary_list.append(np.full(len(segs), binary, dtype=np.int32))
        multi_list.append(np.full(len(segs), label_id, dtype=np.int32))
        file_list.append(np.full(len(segs), fname, dtype=object))
        # FILE_CONFIGS is ordered by load condition within each fault family.
        load_list.append(np.full(len(segs), file_i % 4, dtype=np.int32))
        label_name_list.append(np.full(len(segs), label_name, dtype=object))
        fault_size_list.append(
            np.full(len(segs), _fault_size_from_label(label_name), dtype=object)
        )

    if skipped:
        print(f"\n  [WARN] Skipped {len(skipped)} file(s): {skipped}")

    if not segments_list:
        raise RuntimeError(
            "No data was loaded. Check your internet connection or download the "
            ".mat files manually into results/data/ and re-run."
        )

    X       = np.concatenate(segments_list, axis=0)
    y_bin   = np.concatenate(binary_list,   axis=0)
    y_multi = np.concatenate(multi_list,    axis=0)
    metadata = {
        "file": np.concatenate(file_list, axis=0),
        "load": np.concatenate(load_list, axis=0),
        "label_name": np.concatenate(label_name_list, axis=0),
        "fault_size": np.concatenate(fault_size_list, axis=0),
    }

    print(f"\n  Loaded  : {X.shape[0]:,} segments  ×  {X.shape[1]} samples")
    print(f"  Normal  : {(y_bin == 0).sum():,}")
    print(f"  Fault   : {(y_bin == 1).sum():,}")

    return X, y_bin, y_multi, metadata


def download_and_load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible loader used by the original training pipeline."""
    X, y_bin, y_multi, _ = download_and_load_with_metadata()
    return X, y_bin, y_multi
