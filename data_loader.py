"""Download CWRU .mat files and convert them to windowed numpy segments."""
from __future__ import annotations

import os
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


def _download(url: str, path: str) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return True
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
            os.remove(path)
            return False
        return True
    except Exception as exc:
        print(f"  [WARN] Cannot download {url}: {exc}")
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


def download_and_load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    skipped = []

    for fname, label_name, label_id in tqdm(FILE_CONFIGS, desc="Downloading"):
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

    print(f"\n  Loaded  : {X.shape[0]:,} segments  ×  {X.shape[1]} samples")
    print(f"  Normal  : {(y_bin == 0).sum():,}")
    print(f"  Fault   : {(y_bin == 1).sum():,}")

    return X, y_bin, y_multi
