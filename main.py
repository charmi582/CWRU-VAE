"""
CWRU Bearing Anomaly Detection — VAE Pipeline
==============================================
Workflow
--------
1. Download CWRU .mat files from Case Western Reserve University.
2. Segment signals into overlapping windows.
3. Create factory-like imbalanced dataset (all normal + 10 % of faults).
4. Visualise raw signals (time, FFT, spectrogram, distribution, t-SNE).
5. K-fold cross-validation: train VAE on normal data only.
6. Anomaly detection: abnormal signals have high reconstruction error
   because they fall outside the learned normal latent manifold.
7. Visualise results (ROC, error distributions, latent space).
"""

import numpy as np

from data_loader  import download_and_load
from preprocessor import balance_dataset, normalize_per_sample
from trainer      import run_kfold, reconstruction_errors, get_device
from visualizer   import (
    plot_time_domain,
    plot_fft,
    plot_spectrogram,
    plot_class_distribution,
    plot_tsne_features,
    plot_kfold_results,
    plot_training_curves,
    plot_reconstruction_error,
    plot_roc_curve,
    plot_latent_space,
)
from config import THRESHOLD_PERCENTILE


def banner(step: int, title: str):
    print(f"\n{'═'*60}")
    print(f"  Step {step} — {title}")
    print(f"{'═'*60}")


def main():
    print("\n" + "═" * 60)
    print("  CWRU Bearing Anomaly Detection with VAE")
    print("  KL-regularised latent normal manifold")
    print("═" * 60)

    # ── 1  Download & load ────────────────────────────────────────────
    banner(1, "Download & Load CWRU Dataset")
    X_raw, y_bin, y_multi = download_and_load()

    # ── 2  Visualise raw signals ──────────────────────────────────────
    banner(2, "Data Visualisation (raw)")
    print("  Plotting time-domain …")
    plot_time_domain(X_raw, y_multi)
    print("  Plotting FFT spectra …")
    plot_fft(X_raw, y_multi)
    print("  Plotting spectrograms …")
    plot_spectrogram(X_raw, y_multi)

    # ── 3  Balance (factory-like) ─────────────────────────────────────
    banner(3, "Create Factory-Like Dataset (imbalanced)")
    X_bal, y_bin_bal, y_multi_bal = balance_dataset(X_raw, y_bin, y_multi)
    plot_class_distribution(y_bin, y_multi, y_bin_bal, y_multi_bal)

    # ── 4  Normalise ──────────────────────────────────────────────────
    banner(4, "Per-Sample Z-Score Normalisation")
    X_norm = normalize_per_sample(X_bal)
    print(f"  X shape: {X_norm.shape}  dtype: {X_norm.dtype}")

    # ── 5  t-SNE on raw features ──────────────────────────────────────
    banner(5, "t-SNE Feature Visualisation")
    plot_tsne_features(X_norm, y_bin_bal, y_multi_bal)

    # ── 6  Separate normal data for VAE training ──────────────────────
    X_normal = X_norm[y_bin_bal == 0]
    print(f"\n  Normal segments available for VAE training: {X_normal.shape[0]:,}")

    # ── 7  K-Fold VAE training ────────────────────────────────────────
    banner(6, "K-Fold VAE Training (normal data only)")
    model, fold_results, history = run_kfold(X_normal, X_norm, y_bin_bal)

    plot_kfold_results(fold_results)
    plot_training_curves(history)

    # ── 8  Final anomaly evaluation ───────────────────────────────────
    banner(7, "Anomaly Detection Evaluation")
    device = get_device()
    model  = model.to(device)

    all_errors     = reconstruction_errors(model, X_norm, device)
    normal_errors  = all_errors[y_bin_bal == 0]
    fault_errors   = all_errors[y_bin_bal == 1]
    threshold      = np.percentile(normal_errors, THRESHOLD_PERCENTILE)

    print(f"  Normal errors  — mean={normal_errors.mean():.6f}  "
          f"std={normal_errors.std():.6f}")
    print(f"  Fault  errors  — mean={fault_errors.mean():.6f}  "
          f"std={fault_errors.std():.6f}")
    print(f"  Anomaly threshold (P{THRESHOLD_PERCENTILE}): {threshold:.6f}")

    plot_reconstruction_error(normal_errors, fault_errors, threshold)
    final_auc = plot_roc_curve(y_bin_bal, all_errors)

    # ── 9  Latent space ───────────────────────────────────────────────
    banner(8, "Latent Space Visualisation")
    plot_latent_space(model, X_norm, y_bin_bal, y_multi_bal, device)

    # ── Summary ───────────────────────────────────────────────────────
    mean_auc = np.mean([r["auc"] for r in fold_results])
    std_auc  = np.std([r["auc"]  for r in fold_results])

    print(f"\n{'═'*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'═'*60}")
    print(f"  Dataset     : {X_norm.shape[0]:,} segments × {X_norm.shape[1]} samples")
    print(f"  Normal      : {(y_bin_bal==0).sum():,}  "
          f"Fault: {(y_bin_bal==1).sum():,}")
    print(f"  K-Fold AUC  : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Final AUC   : {final_auc:.4f}")
    print(f"  Threshold   : {threshold:.6f}  (P{THRESHOLD_PERCENTILE} of normal)")
    print(f"  Plots       : results/plots/  (10 figures)")
    print(f"  Model       : results/models/best_vae.pt")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
