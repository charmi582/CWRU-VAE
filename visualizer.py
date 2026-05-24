"""All visualisation routines — saves PNG files to results/plots/."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc as sk_auc
import torch

from config import PLOT_DIR, LABEL_NAMES, COLORS, SAMPLE_RATE, WINDOW_SIZE

sns.set_theme(style="whitegrid", font_scale=1.15)
os.makedirs(PLOT_DIR, exist_ok=True)

FIGDPI = 150


def _save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


def _samples_per_class(X, y_multi, n=1):
    out = {}
    for cls in sorted(np.unique(y_multi)):
        idx = np.where(y_multi == cls)[0]
        out[cls] = X[idx[:n]]
    return out


# ── 1  Time Domain ────────────────────────────────────────────────────────────
def plot_time_domain(X, y_multi):
    spc = _samples_per_class(X, y_multi)
    t   = np.arange(WINDOW_SIZE) / SAMPLE_RATE * 1000   # ms
    n   = len(spc)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Time-Domain Vibration Signals", fontsize=15, fontweight="bold")

    for ax, (cls, segs) in zip(axes, spc.items()):
        c = COLORS.get(cls, "gray")
        ax.plot(t, segs[0], color=c, lw=0.8)
        ax.set_title(LABEL_NAMES[cls], color=c, fontweight="bold")
        ax.set_ylabel("Amplitude (g)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    _save(fig, "01_time_domain.png")


# ── 2  FFT Spectrum ───────────────────────────────────────────────────────────
def plot_fft(X, y_multi):
    spc   = _samples_per_class(X, y_multi, n=5)
    freqs = fftfreq(WINDOW_SIZE, 1 / SAMPLE_RATE)
    pos   = freqs >= 0
    n     = len(spc)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("FFT Frequency Spectrum", fontsize=15, fontweight="bold")

    for ax, (cls, segs) in zip(axes, spc.items()):
        c    = COLORS.get(cls, "gray")
        mags = np.mean([np.abs(fft(s)) for s in segs], axis=0)
        ax.plot(freqs[pos] / 1000, mags[pos], color=c, lw=0.9)
        ax.set_title(LABEL_NAMES[cls], color=c, fontweight="bold")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(0, SAMPLE_RATE / 2000)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (kHz)")
    plt.tight_layout()
    _save(fig, "02_fft_spectrum.png")


# ── 3  Spectrogram ────────────────────────────────────────────────────────────
def plot_spectrogram(X, y_multi):
    spc = _samples_per_class(X, y_multi)
    n   = len(spc)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("STFT Spectrograms", fontsize=15, fontweight="bold")

    for ax, (cls, segs) in zip(axes, spc.items()):
        f, t, Zxx = sp_signal.stft(segs[0], fs=SAMPLE_RATE, nperseg=128, noverlap=112)
        im = ax.pcolormesh(t * 1000, f / 1000, np.abs(Zxx),
                           shading="gouraud", cmap="viridis")
        ax.set_title(LABEL_NAMES[cls], fontweight="bold", fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=ax, label="Mag")

    plt.tight_layout()
    _save(fig, "03_spectrogram.png")


# ── 4  Class Distribution ─────────────────────────────────────────────────────
def plot_class_distribution(y_bin_orig, y_multi_orig, y_bin_bal, y_multi_bal):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Class Distribution — Before vs After Balancing",
                 fontsize=15, fontweight="bold")

    # Before — bar chart
    ax = axes[0]
    classes = sorted(np.unique(y_multi_orig))
    counts  = [int((y_multi_orig == c).sum()) for c in classes]
    labels  = [LABEL_NAMES[c] for c in classes]
    clrs    = [COLORS[c] for c in classes]
    bars    = ax.bar(labels, counts, color=clrs, alpha=0.85, edgecolor="white", lw=1.5)
    for b, cnt in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{cnt:,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Original Distribution", fontweight="bold")
    ax.set_ylabel("Segments")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # After — pie chart
    ax = axes[1]
    classes_b = sorted(np.unique(y_multi_bal))
    counts_b  = [int((y_multi_bal == c).sum()) for c in classes_b]
    clrs_b    = [COLORS[c] for c in classes_b]
    lbs_b     = [f"{LABEL_NAMES[c]}\n({cnt:,})" for c, cnt in zip(classes_b, counts_b)]
    wedges, _, pcts = ax.pie(
        counts_b, labels=lbs_b, colors=clrs_b,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for p in pcts:
        p.set_fontweight("bold")
    ax.set_title("Factory-Like Distribution (After Balancing)", fontweight="bold")

    plt.tight_layout()
    _save(fig, "04_class_distribution.png")


# ── 5  t-SNE of Raw Features ──────────────────────────────────────────────────
def plot_tsne_features(X, y_binary, y_multi, max_samples=2000):
    print("  [t-SNE] computing (may take ~1-2 min) …")
    rng = np.random.default_rng(0)
    if len(X) > max_samples:
        idx     = rng.choice(len(X), max_samples, replace=False)
        Xs, yb, ym = X[idx], y_binary[idx], y_multi[idx]
    else:
        Xs, yb, ym = X, y_binary, y_multi

    Z = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000).fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("t-SNE  —  Raw Signal Feature Space", fontsize=15, fontweight="bold")

    for ax, labels, title, cmap in [
        (axes[0], yb, "Normal vs Fault",
         {0: COLORS[0], 1: "#F44336"}),
        (axes[1], ym, "Fault Type",
         COLORS),
    ]:
        for cls in np.unique(labels):
            m = labels == cls
            name = "Normal" if cls == 0 and ax is axes[0] else \
                   ("Fault"  if cls == 1 and ax is axes[0] else LABEL_NAMES.get(cls, str(cls)))
            ax.scatter(Z[m, 0], Z[m, 1], c=cmap.get(cls, "gray"),
                       label=name, alpha=0.55, s=14, linewidths=0)
        ax.set_title(title, fontweight="bold")
        ax.legend(markerscale=2.5, framealpha=0.9, fontsize=10)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    _save(fig, "05_tsne_features.png")


# ── 6  K-Fold Results ────────────────────────────────────────────────────────
def plot_kfold_results(fold_results):
    folds     = [r["fold"]     for r in fold_results]
    aucs      = [r["auc"]      for r in fold_results]
    val_losses = [r["val_loss"] for r in fold_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("K-Fold Cross-Validation Results", fontsize=15, fontweight="bold")
    palette = plt.cm.viridis(np.linspace(0.2, 0.8, len(folds)))

    for ax, vals, ylabel, title, mean_fmt in [
        (axes[0], aucs,       "AUC-ROC",          "AUC-ROC per Fold",        ".3f"),
        (axes[1], val_losses, "Reconstruction Loss", "Val Loss per Fold",    ".5f"),
    ]:
        bars = ax.bar(folds, vals, color=palette, alpha=0.85, edgecolor="white", lw=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.005,
                    f"{v:{mean_fmt}}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        mean_v = np.mean(vals)
        ax.axhline(mean_v, color="crimson", ls="--", lw=2,
                   label=f"Mean: {mean_v:{mean_fmt}} ± {np.std(vals):{mean_fmt}}")
        if "AUC" in title:
            ax.set_ylim(0, 1.08)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Fold"); ax.set_ylabel(ylabel)
        ax.set_xticks(folds)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, "06_kfold_results.png")


# ── 7  Training Curves ────────────────────────────────────────────────────────
def plot_training_curves(history):
    keys   = ["total", "recon", "kl"]
    titles = ["Total Loss", "Reconstruction Loss (MSE)", "KL Divergence"]
    clrs   = ["#2196F3", "#4CAF50", "#FF5722"]
    epochs = range(1, len(history["total"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("VAE Training Curves (Best Fold)", fontsize=15, fontweight="bold")

    for ax, key, title, c in zip(axes, keys, titles, clrs):
        ax.plot(epochs, history[key],         color=c, lw=2,   label="Train")
        ax.plot(epochs, history[f"val_{key}"], color=c, lw=2,
                ls="--", alpha=0.7, label="Validation")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        final = history[key][-1]
        ax.annotate(f"{final:.4f}",
                    xy=(list(epochs)[-1], final),
                    xytext=(-40, 12), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    plt.tight_layout()
    _save(fig, "07_training_curves.png")


# ── 8  Reconstruction Error Distribution ──────────────────────────────────────
def plot_reconstruction_error(normal_err, fault_err, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("VAE Reconstruction Error — Anomaly Detection",
                 fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.hist(normal_err, bins=60, color=COLORS[0], alpha=0.72,
            label="Normal", density=True)
    ax.hist(fault_err,  bins=60, color="#F44336",  alpha=0.72,
            label="Fault",  density=True)
    ax.axvline(threshold, color="black", ls="--", lw=2,
               label=f"Threshold ({threshold:.5f})")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    bp = ax.boxplot([normal_err, fault_err], patch_artist=True,
                    notch=True, labels=["Normal", "Fault"],
                    medianprops=dict(color="black", lw=2))
    for box, c in zip(bp["boxes"], [COLORS[0], "#F44336"]):
        box.set_facecolor(c); box.set_alpha(0.75)
    ax.axhline(threshold, color="black", ls="--", lw=2,
               label=f"Threshold ({threshold:.5f})")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_title("Box-Plot Comparison", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, "08_reconstruction_error.png")


# ── 9  ROC Curve ──────────────────────────────────────────────────────────────
def plot_roc_curve(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc     = sk_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5,
            label=f"VAE Anomaly Detector (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    ax.plot([0, 1], [0, 1], "gray", ls="--", lw=1.5, label="Random")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — VAE Anomaly Detection",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    ax.text(0.58, 0.18, f"AUC = {roc_auc:.4f}", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
    plt.tight_layout()
    _save(fig, "09_roc_curve.png")
    return roc_auc


# ── 10  Latent Space ──────────────────────────────────────────────────────────
def plot_latent_space(model, X, y_binary, y_multi, device, max_samples=2000):
    print("  [Latent t-SNE] computing …")
    model.eval()
    rng = np.random.default_rng(1)

    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        Xs, yb, ym = X[idx], y_binary[idx], y_multi[idx]
    else:
        Xs, yb, ym = X, y_binary, y_multi

    with torch.no_grad():
        Xt = torch.FloatTensor(Xs).unsqueeze(1).to(device)
        mu, _ = model.encoder(Xt)
        Z = mu.cpu().numpy()

    if Z.shape[1] > 2:
        Z2 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Z)
    else:
        Z2 = Z

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("VAE Latent Space  z  (t-SNE projection)",
                 fontsize=15, fontweight="bold")

    for ax, labels, cmap, title in [
        (axes[0], yb, {0: COLORS[0], 1: "#F44336"},
         "Normal vs Fault in Latent Space"),
        (axes[1], ym, COLORS,
         "Fault Type in Latent Space"),
    ]:
        for cls in np.unique(labels):
            m    = labels == cls
            name = LABEL_NAMES.get(cls, str(cls))
            ax.scatter(Z2[m, 0], Z2[m, 1], c=cmap.get(cls, "gray"),
                       label=name, alpha=0.6, s=18, linewidths=0)
        ax.set_title(title, fontweight="bold")
        ax.legend(markerscale=2.5, framealpha=0.9, fontsize=10)
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")

    plt.tight_layout()
    _save(fig, "10_latent_space.png")
