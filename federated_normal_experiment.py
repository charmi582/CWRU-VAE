"""Federated normal-only anomaly detection experiment.

This script simulates the journal extension scenario where each company owns
private normal bearing data. Clients train locally on normal windows only, the
server aggregates model weights with FedAvg, and no raw client windows are
shared. Fault windows, when available, are used only for optional evaluation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, BETA, LATENT_DIM, LEARNING_RATE, STRIDE, WINDOW_SIZE
from journal_experiment_utils import classification_metrics, ensure_dir, threshold_candidates
from paderborn_data_loader import DEFAULT_BEARINGS, PADERBORN_MAT_DIR, load_paderborn_with_metadata
from preprocessor import normalize_per_sample
from trainer import _eval_epoch, _train_epoch, get_device, reconstruction_errors
from vae_model import VAE
from ae_model import ConvAE


OUT_DIR = os.path.join("results", "federated_normal")


@dataclass
class FederatedRoundRow:
    round: int
    model: str
    client_id: str
    client_train_count: int
    client_val_count: int
    client_test_normal_count: int
    threshold_strategy: str
    threshold: float
    normal_false_alarm_rate: float
    normal_precision: float
    normal_recall: float
    optional_fault_count: int
    optional_fault_recall: float
    optional_fault_miss_rate: float
    optional_fault_f1: float
    mean_client_val_loss: float


def model_factory(model_name: str, input_size: int):
    if model_name == "vae":
        return lambda: VAE(input_size, LATENT_DIM, BETA)
    if model_name == "cnn-ae":
        return lambda: ConvAE(input_size, LATENT_DIM)
    raise ValueError(f"Unsupported federated model: {model_name}")


def split_indices(indices: np.ndarray, seed: int, train_fraction: float = 0.7, val_fraction: float = 0.15):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    n_train = max(1, int(round(len(shuffled) * train_fraction)))
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    if len(test) == 0:
        test = val
    return train, val, test


def build_clients(
    y: np.ndarray,
    metadata: dict[str, np.ndarray],
    clients_by: str,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    groups = metadata[clients_by].astype(str)
    clients = {}
    for group in sorted(np.unique(groups[y == 0])):
        idx = np.where((y == 0) & (groups == group))[0]
        if len(idx) < 20:
            continue
        tr, va, te = split_indices(idx, seed=seed + len(clients))
        clients[group] = {"train": tr, "val": va, "test": te}
    return clients


def local_train(
    global_state: dict[str, torch.Tensor],
    model_name: str,
    input_size: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    local_epochs: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    model = model_factory(model_name, input_size)().to(device)
    model.load_state_dict(global_state)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train).unsqueeze(1)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val).unsqueeze(1)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_loss = float("nan")
    for _ in range(local_epochs):
        _train_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = _eval_epoch(model, val_loader, device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, float(val_loss)


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    averaged = {}
    for key in states[0]:
        tensor = states[0][key]
        if torch.is_floating_point(tensor):
            acc = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
            averaged[key] = acc
        else:
            averaged[key] = tensor.clone()
    return averaged


def evaluate_round(
    round_i: int,
    model_name: str,
    input_size: int,
    global_state: dict[str, torch.Tensor],
    X: np.ndarray,
    y: np.ndarray,
    clients: dict[str, dict[str, np.ndarray]],
    fault_eval_idx: np.ndarray,
    mean_client_val_loss: float,
    device: torch.device,
) -> list[FederatedRoundRow]:
    model = model_factory(model_name, input_size)().to(device)
    model.load_state_dict(global_state)

    train_scores = []
    val_scores = []
    for client in clients.values():
        train_scores.append(reconstruction_errors(model, X[client["train"]], device))
        val_scores.append(reconstruction_errors(model, X[client["val"]], device))
    pooled_train = np.concatenate(train_scores)
    pooled_val = np.concatenate(val_scores)
    thresholds = threshold_candidates(pooled_train, pooled_val)

    fault_scores = np.empty((0,), dtype=np.float32)
    fault_labels = np.empty((0,), dtype=np.int32)
    if len(fault_eval_idx):
        fault_scores = reconstruction_errors(model, X[fault_eval_idx], device)
        fault_labels = y[fault_eval_idx]

    rows: list[FederatedRoundRow] = []
    for client_id, client in clients.items():
        normal_scores = reconstruction_errors(model, X[client["test"]], device)
        normal_labels = np.zeros(len(normal_scores), dtype=np.int32)
        eval_scores = normal_scores
        eval_labels = normal_labels
        if len(fault_scores):
            eval_scores = np.concatenate([normal_scores, fault_scores])
            eval_labels = np.concatenate([normal_labels, fault_labels])

        for strategy, threshold in thresholds.items():
            y_pred_normal = (normal_scores > threshold).astype(np.int32)
            far = float(y_pred_normal.mean()) if len(y_pred_normal) else 0.0
            if len(fault_scores):
                metrics = classification_metrics(eval_labels, eval_scores, threshold)
                fault_recall = float(metrics["recall"])
                fault_miss = float(metrics["miss_rate"])
                fault_f1 = float(metrics["f1"])
                precision = float(metrics["precision"])
                recall = float(metrics["recall"])
            else:
                fault_recall = 0.0
                fault_miss = 0.0
                fault_f1 = 0.0
                precision = 0.0
                recall = 0.0

            rows.append(
                FederatedRoundRow(
                    round=round_i,
                    model=model_name,
                    client_id=client_id,
                    client_train_count=len(client["train"]),
                    client_val_count=len(client["val"]),
                    client_test_normal_count=len(client["test"]),
                    threshold_strategy=strategy,
                    threshold=float(threshold),
                    normal_false_alarm_rate=far,
                    normal_precision=precision,
                    normal_recall=recall,
                    optional_fault_count=len(fault_eval_idx),
                    optional_fault_recall=fault_recall,
                    optional_fault_miss_rate=fault_miss,
                    optional_fault_f1=fault_f1,
                    mean_client_val_loss=mean_client_val_loss,
                )
            )
    return rows


def write_rows(rows: list[FederatedRoundRow], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FederatedRoundRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(rows: list[FederatedRoundRow], path: str) -> None:
    import pandas as pd

    df = pd.DataFrame([asdict(row) for row in rows])
    final_round = df["round"].max()
    final = df[df["round"].eq(final_round)]
    summary = (
        final.groupby(["model", "threshold_strategy"])[
            ["normal_false_alarm_rate", "optional_fault_recall", "optional_fault_miss_rate", "optional_fault_f1"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )

    def md_table(frame):
        cols = list(frame.columns)
        out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
            out.append("| " + " | ".join(values) + " |")
        return "\n".join(out)

    lines = [
        "# Federated Normal-Only Anomaly Detection Summary\n",
        "This experiment simulates multiple companies or sites as clients. Each client trains on its own normal bearing windows only. The server aggregates model weights with FedAvg and never receives raw client windows.\n",
        "## Final-Round Calibration Summary\n",
        md_table(summary),
        "\n## Journal Positioning\n",
        "- This supports a privacy-preserving deployment story: clients keep normal vibration data locally while contributing to a shared anomaly detector.",
        "- The main evaluation on normal-only clients is false alarm rate and threshold stability.",
        "- Fault windows are optional and used only as an external audit, not for federated training.",
        "- This should be presented as a deployment-oriented extension rather than a claim that federated learning automatically improves accuracy.\n",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vae", "cnn-ae"], default="cnn-ae")
    parser.add_argument("--clients-by", choices=["bearing", "condition"], default="bearing")
    parser.add_argument("--bearings", nargs="+", default=list(DEFAULT_BEARINGS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--max-files-per-condition", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mat-dir", type=Path, default=PADERBORN_MAT_DIR)
    parser.add_argument("--include-fault-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUT_DIR)
    X_raw, y, _, metadata = load_paderborn_with_metadata(
        mat_dir=args.mat_dir,
        bearings=tuple(args.bearings),
        max_files_per_condition=args.max_files_per_condition,
        window_size=args.window_size,
        stride=args.stride,
    )
    X = normalize_per_sample(X_raw)
    clients = build_clients(y, metadata, clients_by=args.clients_by, seed=args.seed)
    if len(clients) < 2:
        raise RuntimeError("Need at least two normal clients for federated learning.")

    fault_eval_idx = np.where(y == 1)[0] if args.include_fault_audit else np.array([], dtype=np.int64)
    device = get_device()
    global_model = model_factory(args.model, args.window_size)().to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    rows: list[FederatedRoundRow] = []
    for round_i in range(1, args.rounds + 1):
        print(f"\nFederated round {round_i}/{args.rounds}")
        local_states = []
        weights = []
        val_losses = []
        for client_id, split in clients.items():
            print(f"  client={client_id} train={len(split['train'])} val={len(split['val'])}")
            state, val_loss = local_train(
                global_state,
                args.model,
                args.window_size,
                X[split["train"]],
                X[split["val"]],
                local_epochs=args.local_epochs,
                device=device,
            )
            local_states.append(state)
            weights.append(len(split["train"]))
            if np.isfinite(val_loss):
                val_losses.append(val_loss)
        global_state = fedavg(local_states, weights)
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        rows.extend(
            evaluate_round(
                round_i,
                args.model,
                args.window_size,
                global_state,
                X,
                y,
                clients,
                fault_eval_idx,
                mean_val_loss,
                device,
            )
        )

    result_path = os.path.join(OUT_DIR, "federated_normal_metrics.csv")
    summary_path = os.path.join(OUT_DIR, "federated_normal_summary.md")
    write_rows(rows, result_path)
    write_summary(rows, summary_path)
    torch.save(global_state, os.path.join(OUT_DIR, "federated_global_model.pt"))
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
