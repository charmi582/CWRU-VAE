"""K-fold VAE training pipeline."""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import (
    BATCH_SIZE, N_EPOCHS, LEARNING_RATE, K_FOLDS,
    WINDOW_SIZE, LATENT_DIM, BETA, PATIENCE, MODEL_DIR,
    THRESHOLD_PERCENTILE,
)
from vae_model import VAE
from preprocessor import kfold_splits


# ── device selection ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── single-epoch helpers ──────────────────────────────────────────────────────
def _train_epoch(model, loader, optimiser, device):
    model.train()
    tot = rec = kl = 0.0
    for (bx,) in loader:
        bx = bx.to(device)
        optimiser.zero_grad()
        x_hat, mu, logvar = model(bx)
        loss, r, k = model.loss_function(bx, x_hat, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        tot += loss.item(); rec += r.item(); kl += k.item()
    n = len(loader)
    return tot / n, rec / n, kl / n


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    tot = rec = kl = 0.0
    for (bx,) in loader:
        bx = bx.to(device)
        x_hat, mu, logvar = model(bx)
        loss, r, k = model.loss_function(bx, x_hat, mu, logvar)
        tot += loss.item(); rec += r.item(); kl += k.item()
    n = len(loader)
    return tot / n, rec / n, kl / n


# ── per-sample reconstruction error ──────────────────────────────────────────
def reconstruction_errors(model, X: np.ndarray, device, bs: int = 512) -> np.ndarray:
    model.eval()
    dl = DataLoader(
        TensorDataset(torch.FloatTensor(X).unsqueeze(1)),
        batch_size=bs, shuffle=False,
    )
    errs = []
    for (bx,) in dl:
        errs.append(model.anomaly_score(bx.to(device)).cpu().numpy())
    return np.concatenate(errs)


# ── K-fold main function ──────────────────────────────────────────────────────
def run_kfold(
    X_normal: np.ndarray,      # normalised normal-only segments
    X_all:    np.ndarray,      # normalised full (balanced) dataset
    y_all:    np.ndarray,      # binary labels for X_all
) -> tuple:
    """
    Train VAE via K-Fold cross-validation on normal data only.
    Evaluate anomaly-detection AUC on the full balanced dataset.

    Returns
    -------
    best_model   : VAE loaded with best weights
    fold_results : list of dicts per fold
    best_history : training history of the best fold
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = get_device()
    print(f"\n  Using device: {device}")

    splits = kfold_splits(X_normal, n_splits=K_FOLDS)

    fold_results = []
    best_auc     = -1.0
    best_state   = None
    best_history = None
    best_fold    = 0

    for fold_i, (tr_idx, va_idx) in enumerate(splits, 1):
        print(f"\n  {'─'*54}")
        print(f"  Fold {fold_i}/{K_FOLDS}  —  "
              f"train {len(tr_idx):,}  val {len(va_idx):,}  (normal only)")

        X_tr = X_normal[tr_idx]
        X_va = X_normal[va_idx]

        tr_ld = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr).unsqueeze(1)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        va_ld = DataLoader(
            TensorDataset(torch.FloatTensor(X_va).unsqueeze(1)),
            batch_size=BATCH_SIZE,
        )

        model     = VAE(WINDOW_SIZE, LATENT_DIM, BETA).to(device)
        optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=10, factor=0.5
        )

        history = {k: [] for k in [
            "total", "recon", "kl",
            "val_total", "val_recon", "val_kl",
        ]}

        best_val  = float("inf")
        best_ckpt = None
        no_improv = 0

        with tqdm(range(1, N_EPOCHS + 1), desc=f"  Fold {fold_i}", unit="ep") as pbar:
            for ep in pbar:
                t, tr, tk = _train_epoch(model, tr_ld, optimiser, device)
                v, vr, vk = _eval_epoch(model, va_ld, device)

                history["total"].append(t);     history["recon"].append(tr)
                history["kl"].append(tk)
                history["val_total"].append(v); history["val_recon"].append(vr)
                history["val_kl"].append(vk)

                scheduler.step(v)

                if v < best_val:
                    best_val  = v
                    best_ckpt = {k: p.cpu().clone()
                                 for k, p in model.state_dict().items()}
                    no_improv = 0
                else:
                    no_improv += 1

                pbar.set_postfix(
                    train=f"{t:.4f}", val=f"{v:.4f}", kl=f"{tk:.4f}"
                )
                if no_improv >= PATIENCE:
                    print(f"\n  Early stop @ epoch {ep}")
                    break

        # ── evaluate anomaly detection on full dataset ────────────────
        model.load_state_dict(best_ckpt)
        scores = reconstruction_errors(model, X_all, device)
        auc    = roc_auc_score(y_all, scores)

        thresh = np.percentile(scores[y_all == 0], THRESHOLD_PERCENTILE)
        print(f"  Fold {fold_i} → val_loss={best_val:.5f}  AUC={auc:.4f}  "
              f"threshold={thresh:.6f}")

        fold_results.append(
            dict(fold=fold_i, val_loss=best_val, auc=auc, threshold=thresh)
        )

        if auc > best_auc:
            best_auc     = auc
            best_state   = best_ckpt
            best_history = history
            best_fold    = fold_i

    print(f"\n  Best fold: {best_fold}  AUC={best_auc:.4f}")

    # ── build final model ─────────────────────────────────────────────
    best_model = VAE(WINDOW_SIZE, LATENT_DIM, BETA).to(device)
    best_model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MODEL_DIR, "best_vae.pt"))
    print(f"  Model saved → {MODEL_DIR}/best_vae.pt")

    return best_model, fold_results, best_history
