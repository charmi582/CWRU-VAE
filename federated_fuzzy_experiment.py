"""Federated fuzzy health-index comparison experiment.

This script turns the earlier federated pilot into a formal deployment
comparison:

- local-only: each client trains its own normal-only model.
- centralized: a single model trains on pooled normal windows.
- fedavg: clients train locally and share only model weights for FedAvg.
- fedprox: clients use a proximal penalty during local training to reduce
  non-IID client drift.
- fedbn: clients aggregate non-BatchNorm parameters while keeping local
  BatchNorm statistics private.
- personalized variants: the federated global model is locally adapted on each
  client's normal windows before evaluation.

All modes are evaluated with the same fuzzy health-index decision layer. Fault
windows are used only for audit evaluation and never for training.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from ae_model import ConvAE
from config import BATCH_SIZE, BETA, LATENT_DIM, LEARNING_RATE
from fuzzy_health_index_experiment import fuzzy_membership, md_table
from journal_experiment_utils import ensure_dir
from paderborn_data_loader import PADERBORN_MAT_DIR, load_paderborn_with_metadata
from preprocessor import normalize_per_sample
from trainer import _eval_epoch, _train_epoch, get_device, reconstruction_errors
from vae_model import VAE


OUT_DIR = os.path.join("results", "federated_fuzzy")


@dataclass
class FederatedFuzzyRow:
    seed: int
    clients_by: str
    training_mode: str
    calibration_scope: str
    round: int
    model: str
    client_id: str
    client_train_count: int
    client_val_count: int
    client_test_normal_count: int
    fault_audit_count: int
    decision_policy: str
    p50: float
    p90: float
    p95: float
    p99: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    miss_rate: float
    uncertain_rate: float
    mean_health_normal: float
    mean_health_fault: float
    health_gap: float
    tn: int
    fp: int
    fn: int
    tp: int


def model_factory(model_name: str, input_size: int):
    if model_name == "vae":
        return lambda: VAE(input_size, LATENT_DIM, BETA)
    if model_name == "cnn-ae":
        return lambda: ConvAE(input_size, LATENT_DIM)
    raise ValueError(f"Unsupported model: {model_name}")


def split_indices(indices: np.ndarray, seed: int, train_fraction: float = 0.7, val_fraction: float = 0.15):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    n_train = max(1, int(round(len(shuffled) * train_fraction)))
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    if len(test) == 0:
        test = val
    return train, val, test


def build_clients(y: np.ndarray, metadata: dict[str, np.ndarray], clients_by: str, seed: int):
    groups = metadata[clients_by].astype(str)
    clients = {}
    for group in sorted(np.unique(groups[y == 0])):
        idx = np.where((y == 0) & (groups == group))[0]
        if len(idx) < 30:
            continue
        tr, va, te = split_indices(idx, seed=seed + len(clients))
        clients[group] = {"train": tr, "val": va, "test": te}
    if len(clients) < 2:
        raise RuntimeError("Need at least two normal clients.")
    return clients


def train_model_from_state(
    initial_state: dict[str, torch.Tensor],
    model_name: str,
    input_size: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    device: torch.device,
    prox_state: dict[str, torch.Tensor] | None = None,
    prox_mu: float = 0.0,
) -> tuple[dict[str, torch.Tensor], float]:
    model = model_factory(model_name, input_size)().to(device)
    model.load_state_dict(initial_state)
    prox_state_device = None
    if prox_state is not None and prox_mu > 0:
        prox_state_device = {k: v.detach().to(device) for k, v in prox_state.items()}
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
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
    for _ in range(epochs):
        if prox_state_device is None:
            _train_epoch(model, train_loader, optimiser, device)
        else:
            train_epoch_fedprox(model, train_loader, optimiser, device, prox_state_device, prox_mu)
        val_loss, _, _ = _eval_epoch(model, val_loader, device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, float(val_loss)


def train_epoch_fedprox(
    model,
    loader,
    optimiser,
    device: torch.device,
    prox_state: dict[str, torch.Tensor],
    prox_mu: float,
) -> float:
    model.train()
    total = 0.0
    named_params = dict(model.named_parameters())
    for (bx,) in loader:
        bx = bx.to(device)
        optimiser.zero_grad()
        x_hat, mu, logvar = model(bx)
        loss, _, _ = model.loss_function(bx, x_hat, mu, logvar)
        prox = torch.zeros((), device=device)
        for name, param in named_params.items():
            if name in prox_state:
                prox = prox + torch.sum((param - prox_state[name]) ** 2)
        loss = loss + 0.5 * prox_mu * prox
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
    return total / max(1, len(loader))


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    out = {}
    for key in states[0]:
        tensor = states[0][key]
        if torch.is_floating_point(tensor):
            out[key] = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
        else:
            out[key] = tensor.clone()
    return out


def batchnorm_state_keys(model) -> set[str]:
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            keys.update(prefix + key for key in module.state_dict().keys())
    return keys


def fedavg_excluding(
    states: list[dict[str, torch.Tensor]],
    weights: list[int],
    excluded_keys: set[str],
) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    out = {}
    for key in states[0]:
        tensor = states[0][key]
        if key in excluded_keys:
            out[key] = tensor.clone()
        elif torch.is_floating_point(tensor):
            out[key] = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
        else:
            out[key] = tensor.clone()
    return out


def merge_shared_state(
    client_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    local_keys: set[str],
) -> dict[str, torch.Tensor]:
    merged = copy.deepcopy(client_state)
    for key, value in global_state.items():
        if key not in local_keys:
            merged[key] = value.detach().cpu().clone()
    return merged


def binary_metrics(y_true: np.ndarray, score: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0
    miss = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": far,
        "miss_rate": miss,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return np.zeros_like(scores, dtype=np.float64)
    hi = float(np.max(finite))
    lo = float(np.min(finite))
    fill = hi + max(1e-6, hi - lo)
    return np.nan_to_num(scores, nan=fill, posinf=fill, neginf=lo)


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(mad, 1e-8)


def adaptive_calibration_anchors(
    client_scores: np.ndarray,
    pooled_scores: np.ndarray,
) -> tuple[float, float, float, float]:
    """Blend client and pooled anchors according to validation-score drift.

    The blend weight increases when a client's normal validation score
    distribution differs from the pooled validation distribution. This keeps
    clients close to the shared calibration when distributions match, while
    moving toward client-specific thresholds under stronger non-IID drift.
    """
    percentiles = (50, 90, 95, 99)
    client_anchors = np.array([np.percentile(client_scores, p) for p in percentiles], dtype=np.float64)
    pooled_anchors = np.array([np.percentile(pooled_scores, p) for p in percentiles], dtype=np.float64)

    client_median = float(np.median(client_scores))
    pooled_median = float(np.median(pooled_scores))
    client_mad = robust_mad(client_scores)
    pooled_mad = robust_mad(pooled_scores)
    location_shift = abs(client_median - pooled_median) / pooled_mad
    scale_shift = abs(np.log(client_mad / pooled_mad))
    drift = location_shift + scale_shift
    alpha = float(drift / (1.0 + drift))
    blended = alpha * client_anchors + (1.0 - alpha) * pooled_anchors
    return tuple(float(x) for x in blended)


def evaluate_state(
    seed: int,
    clients_by: str,
    training_mode: str,
    round_i: int,
    model_name: str,
    input_size: int,
    state: dict[str, torch.Tensor],
    X: np.ndarray,
    y: np.ndarray,
    clients: dict[str, dict[str, np.ndarray]],
    fault_idx: np.ndarray,
    device: torch.device,
    calibration_scopes: tuple[str, ...] = ("client_specific",),
) -> list[FederatedFuzzyRow]:
    model = model_factory(model_name, input_size)().to(device)
    model.load_state_dict(state)
    rows = []
    fault_scores = reconstruction_errors(model, X[fault_idx], device) if len(fault_idx) else np.array([])
    fault_scores = sanitize_scores(fault_scores) if len(fault_scores) else fault_scores
    fault_labels = y[fault_idx] if len(fault_idx) else np.array([], dtype=np.int32)
    val_scores_by_client = {
        client_id: sanitize_scores(reconstruction_errors(model, X[split["val"]], device))
        for client_id, split in clients.items()
    }
    pooled_val_scores = np.concatenate(list(val_scores_by_client.values()))
    pooled_anchors = tuple(float(np.percentile(pooled_val_scores, p)) for p in (50, 90, 95, 99))

    for client_id, split in clients.items():
        normal_scores = sanitize_scores(reconstruction_errors(model, X[split["test"]], device))
        eval_scores = normal_scores
        eval_labels = np.zeros(len(normal_scores), dtype=np.int32)
        if len(fault_scores):
            eval_scores = np.concatenate([normal_scores, fault_scores])
            eval_labels = np.concatenate([eval_labels, fault_labels])

        client_anchors = tuple(float(np.percentile(val_scores_by_client[client_id], p)) for p in (50, 90, 95, 99))
        for calibration_scope in calibration_scopes:
            if calibration_scope == "client_specific":
                p50, p90, p95, p99 = client_anchors
            elif calibration_scope == "pooled":
                p50, p90, p95, p99 = pooled_anchors
            elif calibration_scope == "adaptive":
                p50, p90, p95, p99 = adaptive_calibration_anchors(val_scores_by_client[client_id], pooled_val_scores)
            else:
                raise ValueError(f"Unsupported calibration scope: {calibration_scope}")

            _, _, _, health, _ = fuzzy_membership(eval_scores, p50, p90, p95, p99)
            uncertain = ((health >= 0.25) & (health < 0.75)).astype(np.int32)
            mean_normal = float(health[eval_labels == 0].mean()) if (eval_labels == 0).any() else 0.0
            mean_fault = float(health[eval_labels == 1].mean()) if (eval_labels == 1).any() else 0.0

            policies = {
                "hard_val_p95": (eval_scores > p95).astype(np.int32),
                "fuzzy_warning_as_alarm_hi50": (health >= 0.50).astype(np.int32),
                "fuzzy_fault_only_hi75": (health >= 0.75).astype(np.int32),
            }
            for policy, y_pred in policies.items():
                metrics = binary_metrics(eval_labels, health, y_pred)
                rows.append(
                    FederatedFuzzyRow(
                        seed=seed,
                        clients_by=clients_by,
                        training_mode=training_mode,
                        calibration_scope=calibration_scope,
                        round=round_i,
                        model=model_name,
                        client_id=client_id,
                        client_train_count=len(split["train"]),
                        client_val_count=len(split["val"]),
                        client_test_normal_count=len(split["test"]),
                        fault_audit_count=len(fault_idx),
                        decision_policy=policy,
                        p50=p50,
                        p90=p90,
                        p95=p95,
                        p99=p99,
                        uncertain_rate=float(uncertain.mean()),
                        mean_health_normal=mean_normal,
                        mean_health_fault=mean_fault,
                        health_gap=mean_fault - mean_normal,
                        **metrics,
                    )
                )
    return rows


def evaluate_client_states(
    seed: int,
    clients_by: str,
    training_mode: str,
    round_i: int,
    model_name: str,
    input_size: int,
    states_by_client: dict[str, dict[str, torch.Tensor]],
    X: np.ndarray,
    y: np.ndarray,
    clients: dict[str, dict[str, np.ndarray]],
    fault_idx: np.ndarray,
    device: torch.device,
    calibration_scopes: tuple[str, ...] = ("client_specific",),
) -> list[FederatedFuzzyRow]:
    rows = []
    val_scores_by_client = {}
    normal_scores_by_client = {}
    fault_scores_by_client = {}

    for client_id, split in clients.items():
        model = model_factory(model_name, input_size)().to(device)
        model.load_state_dict(states_by_client[client_id])
        val_scores_by_client[client_id] = sanitize_scores(reconstruction_errors(model, X[split["val"]], device))
        normal_scores_by_client[client_id] = sanitize_scores(reconstruction_errors(model, X[split["test"]], device))
        if len(fault_idx):
            fault_scores_by_client[client_id] = sanitize_scores(reconstruction_errors(model, X[fault_idx], device))
        else:
            fault_scores_by_client[client_id] = np.array([])

    pooled_val_scores = np.concatenate(list(val_scores_by_client.values()))
    pooled_anchors = tuple(float(np.percentile(pooled_val_scores, p)) for p in (50, 90, 95, 99))
    fault_labels = y[fault_idx] if len(fault_idx) else np.array([], dtype=np.int32)

    for client_id, split in clients.items():
        normal_scores = normal_scores_by_client[client_id]
        eval_scores = normal_scores
        eval_labels = np.zeros(len(normal_scores), dtype=np.int32)
        if len(fault_idx):
            eval_scores = np.concatenate([normal_scores, fault_scores_by_client[client_id]])
            eval_labels = np.concatenate([eval_labels, fault_labels])

        client_anchors = tuple(float(np.percentile(val_scores_by_client[client_id], p)) for p in (50, 90, 95, 99))
        for calibration_scope in calibration_scopes:
            if calibration_scope == "client_specific":
                p50, p90, p95, p99 = client_anchors
            elif calibration_scope == "pooled":
                p50, p90, p95, p99 = pooled_anchors
            elif calibration_scope == "adaptive":
                p50, p90, p95, p99 = adaptive_calibration_anchors(val_scores_by_client[client_id], pooled_val_scores)
            else:
                raise ValueError(f"Unsupported calibration scope: {calibration_scope}")

            _, _, _, health, _ = fuzzy_membership(eval_scores, p50, p90, p95, p99)
            uncertain = ((health >= 0.25) & (health < 0.75)).astype(np.int32)
            mean_normal = float(health[eval_labels == 0].mean()) if (eval_labels == 0).any() else 0.0
            mean_fault = float(health[eval_labels == 1].mean()) if (eval_labels == 1).any() else 0.0
            policies = {
                "hard_val_p95": (eval_scores > p95).astype(np.int32),
                "fuzzy_warning_as_alarm_hi50": (health >= 0.50).astype(np.int32),
                "fuzzy_fault_only_hi75": (health >= 0.75).astype(np.int32),
            }
            for policy, y_pred in policies.items():
                metrics = binary_metrics(eval_labels, health, y_pred)
                rows.append(
                    FederatedFuzzyRow(
                        seed=seed,
                        clients_by=clients_by,
                        training_mode=training_mode,
                        calibration_scope=calibration_scope,
                        round=round_i,
                        model=model_name,
                        client_id=client_id,
                        client_train_count=len(split["train"]),
                        client_val_count=len(split["val"]),
                        client_test_normal_count=len(split["test"]),
                        fault_audit_count=len(fault_idx),
                        decision_policy=policy,
                        p50=p50,
                        p90=p90,
                        p95=p95,
                        p99=p99,
                        uncertain_rate=float(uncertain.mean()),
                        mean_health_normal=mean_normal,
                        mean_health_fault=mean_fault,
                        health_gap=mean_fault - mean_normal,
                        **metrics,
                    )
                )
    return rows


def run_model(
    model_name: str,
    input_size: int,
    X: np.ndarray,
    y: np.ndarray,
    clients: dict[str, dict[str, np.ndarray]],
    clients_by: str,
    fault_idx: np.ndarray,
    rounds: int,
    local_epochs: int,
    centralized_epochs: int,
    seed: int,
    device: torch.device,
    evaluate_each_round: bool = False,
    calibration_scopes: tuple[str, ...] = ("client_specific", "pooled"),
    federated_methods: tuple[str, ...] = ("fedavg",),
    fedprox_mu: float = 0.01,
    personalize_epochs: int = 1,
) -> list[FederatedFuzzyRow]:
    torch.manual_seed(seed)
    base_model = model_factory(model_name, input_size)().to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    bn_keys = batchnorm_state_keys(base_model)

    rows = []

    print(f"\n[{model_name}] local-only")
    for client_id, split in clients.items():
        print(f"  client={client_id}")
        local_state, _ = train_model_from_state(
            copy.deepcopy(initial_state),
            model_name,
            input_size,
            X[split["train"]],
            X[split["val"]],
            epochs=local_epochs * rounds,
            device=device,
        )
        one_client = {client_id: split}
        rows.extend(
            evaluate_state(seed, clients_by, "local-only", rounds, model_name, input_size, local_state, X, y, one_client, fault_idx, device, calibration_scopes=("client_specific",))
        )

    print(f"[{model_name}] centralized")
    pooled_train = np.concatenate([split["train"] for split in clients.values()])
    pooled_val = np.concatenate([split["val"] for split in clients.values()])
    central_state, _ = train_model_from_state(
        copy.deepcopy(initial_state),
        model_name,
        input_size,
        X[pooled_train],
        X[pooled_val],
        epochs=centralized_epochs,
        device=device,
    )
    rows.extend(
        evaluate_state(seed, clients_by, "centralized", centralized_epochs, model_name, input_size, central_state, X, y, clients, fault_idx, device, calibration_scopes=calibration_scopes)
    )

    for method in federated_methods:
        print(f"[{model_name}] {method}")
        global_state = copy.deepcopy(initial_state)
        client_states = {client_id: copy.deepcopy(initial_state) for client_id in clients}
        for round_i in range(1, rounds + 1):
            local_states = []
            weights = []
            print(f"  round={round_i}/{rounds}")
            for client_id, split in clients.items():
                print(f"    client={client_id}")
                start_state = copy.deepcopy(global_state)
                if method == "fedbn":
                    start_state = merge_shared_state(client_states[client_id], global_state, bn_keys)
                state, _ = train_model_from_state(
                    start_state,
                    model_name,
                    input_size,
                    X[split["train"]],
                    X[split["val"]],
                    epochs=local_epochs,
                    device=device,
                    prox_state=global_state if method == "fedprox" else None,
                    prox_mu=fedprox_mu if method == "fedprox" else 0.0,
                )
                local_states.append(state)
                weights.append(len(split["train"]))
                client_states[client_id] = copy.deepcopy(state)
            if method == "fedbn":
                global_state = fedavg_excluding(local_states, weights, bn_keys)
            else:
                global_state = fedavg(local_states, weights)
            if evaluate_each_round:
                if method == "fedbn":
                    rows.extend(
                        evaluate_client_states(seed, clients_by, method, round_i, model_name, input_size, client_states, X, y, clients, fault_idx, device, calibration_scopes=calibration_scopes)
                    )
                else:
                    rows.extend(
                        evaluate_state(seed, clients_by, method, round_i, model_name, input_size, global_state, X, y, clients, fault_idx, device, calibration_scopes=calibration_scopes)
                    )
        if not evaluate_each_round:
            if method == "fedbn":
                rows.extend(
                    evaluate_client_states(seed, clients_by, method, rounds, model_name, input_size, client_states, X, y, clients, fault_idx, device, calibration_scopes=calibration_scopes)
                )
            else:
                rows.extend(
                    evaluate_state(seed, clients_by, method, rounds, model_name, input_size, global_state, X, y, clients, fault_idx, device, calibration_scopes=calibration_scopes)
                )
        if personalize_epochs > 0:
            for client_id, split in clients.items():
                start_state = global_state
                if method == "fedbn":
                    start_state = merge_shared_state(client_states[client_id], global_state, bn_keys)
                personalized_state, _ = train_model_from_state(
                    copy.deepcopy(start_state),
                    model_name,
                    input_size,
                    X[split["train"]],
                    X[split["val"]],
                    epochs=personalize_epochs,
                    device=device,
                )
                one_client = {client_id: split}
                rows.extend(
                    evaluate_state(
                        seed,
                        clients_by,
                        f"{method}-personalized",
                        rounds,
                        model_name,
                        input_size,
                        personalized_state,
                        X,
                        y,
                        one_client,
                        fault_idx,
                        device,
                        calibration_scopes=("client_specific",),
                    )
                )
    return rows


def write_rows(rows: list[FederatedFuzzyRow], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FederatedFuzzyRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(rows: list[FederatedFuzzyRow]) -> None:
    df = pd.DataFrame([asdict(row) for row in rows])
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "false_alarm_rate",
        "miss_rate",
        "uncertain_rate",
        "mean_health_normal",
        "mean_health_fault",
        "health_gap",
    ]
    summary = (
        df.groupby(["clients_by", "training_mode", "calibration_scope", "model", "decision_policy"])[metric_cols]
        .mean()
        .reset_index()
    )
    stability = (
        df.groupby(["clients_by", "training_mode", "calibration_scope", "model", "decision_policy"])[
            ["false_alarm_rate", "miss_rate", "uncertain_rate", "health_gap"]
        ]
        .std(ddof=0)
        .reset_index()
        .rename(
            columns={
                "false_alarm_rate": "false_alarm_rate_std",
                "miss_rate": "miss_rate_std",
                "uncertain_rate": "uncertain_rate_std",
                "health_gap": "health_gap_std",
            }
        )
    )
    client = df[
        [
            "training_mode",
            "seed",
            "clients_by",
            "calibration_scope",
            "model",
            "client_id",
            "decision_policy",
            "f1",
            "false_alarm_rate",
            "miss_rate",
            "uncertain_rate",
            "mean_health_normal",
            "mean_health_fault",
            "health_gap",
        ]
    ].copy()
    summary.to_csv(os.path.join(OUT_DIR, "federated_fuzzy_summary.csv"), index=False)
    stability.to_csv(os.path.join(OUT_DIR, "federated_fuzzy_client_stability.csv"), index=False)
    client.to_csv(os.path.join(OUT_DIR, "federated_fuzzy_client_detail.csv"), index=False)
    formal = summary[
        summary["model"].eq("cnn-ae")
        & summary["decision_policy"].eq("fuzzy_warning_as_alarm_hi50")
    ][
        [
            "clients_by",
            "training_mode",
            "calibration_scope",
            "roc_auc",
            "pr_auc",
            "f1",
            "false_alarm_rate",
            "miss_rate",
            "uncertain_rate",
            "health_gap",
        ]
    ].copy()
    for col in ["roc_auc", "pr_auc", "f1", "false_alarm_rate", "miss_rate", "uncertain_rate", "health_gap"]:
        formal[col] = formal[col].round(4)
    formal.to_csv(os.path.join(OUT_DIR, "federated_formal_cnn_ae_hard_p95_summary.csv"), index=False)
    convergence = (
        df[df["training_mode"].isin(["fedavg", "fedprox", "fedbn"])]
        .groupby(["clients_by", "training_mode", "calibration_scope", "model", "decision_policy", "round"])[metric_cols]
        .mean()
        .reset_index()
    )
    convergence.to_csv(os.path.join(OUT_DIR, "federated_fuzzy_convergence.csv"), index=False)

    lines = [
        "# Federated Fuzzy Health-Index Comparison\n\n",
        "This experiment compares local-only, centralized, FedAvg, FedProx, and personalized federated training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.\n\n",
        "## Mean Performance\n\n",
        md_table(summary),
        "\n\n## Client Stability\n\n",
        md_table(stability),
        "\n\n## Federated Convergence\n\n",
        md_table(convergence),
        "\n\n## Interpretation\n\n",
        "- `local-only` shows how each private site performs without collaboration.\n",
        "- `centralized` is the upper-reference setting that pools normal data and would require data sharing.\n",
        "- `fedavg` approximates collaborative normal-only training without sharing raw vibration windows.\n",
        "- `fedprox` adds a proximal penalty to reduce local client drift under non-IID data.\n",
        "- `fedbn` keeps BatchNorm parameters and running statistics local to each client.\n",
        "- `*-personalized` locally adapts the federated global model before client evaluation.\n",
        "- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.\n",
        "- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.\n",
    ]
    Path(os.path.join(OUT_DIR, "federated_fuzzy_summary.md")).write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["vae", "cnn-ae"], default=["cnn-ae", "vae"])
    parser.add_argument("--clients-by", choices=["bearing", "condition"], default="bearing")
    parser.add_argument("--client-partitions", nargs="+", choices=["bearing", "condition"], default=None)
    parser.add_argument(
        "--bearings",
        nargs="+",
        default=["K001", "K002", "K003", "K004", "K005", "K006", "KA01", "KA03", "KA05", "KA07", "KI01", "KI03", "KI05", "KI07"],
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--centralized-epochs", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--max-files-per-condition", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--evaluate-each-round", action="store_true")
    parser.add_argument("--calibration-scopes", nargs="+", choices=["client_specific", "pooled", "adaptive"], default=["client_specific", "pooled", "adaptive"])
    parser.add_argument("--federated-methods", nargs="+", choices=["fedavg", "fedprox", "fedbn"], default=["fedavg"])
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--personalize-epochs", type=int, default=1)
    parser.add_argument("--mat-dir", type=Path, default=PADERBORN_MAT_DIR)
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
    fault_idx = np.where(y == 1)[0]
    device = get_device()
    print(f"Using device: {device}")
    print(f"Fault audit windows: {len(fault_idx)}")
    X = normalize_per_sample(X_raw)

    rows = []
    partitions = args.client_partitions or [args.clients_by]
    seeds = args.seeds or [args.seed]
    for clients_by in partitions:
        for seed in seeds:
            clients = build_clients(y, metadata, clients_by=clients_by, seed=seed)
            print(f"\nclients_by={clients_by} seed={seed} clients={list(clients)}")
            for model_name in args.models:
                rows.extend(
                    run_model(
                        model_name,
                        input_size=args.window_size,
                        X=X,
                        y=y,
                        clients=clients,
                        clients_by=clients_by,
                        fault_idx=fault_idx,
                        rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        centralized_epochs=args.centralized_epochs,
                        seed=seed,
                        device=device,
                        evaluate_each_round=args.evaluate_each_round,
                        calibration_scopes=tuple(args.calibration_scopes),
                        federated_methods=tuple(args.federated_methods),
                        fedprox_mu=args.fedprox_mu,
                        personalize_epochs=args.personalize_epochs,
                    )
                )

    metrics_path = os.path.join(OUT_DIR, "federated_fuzzy_metrics.csv")
    write_rows(rows, metrics_path)
    summarize(rows)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {os.path.join(OUT_DIR, 'federated_fuzzy_summary.md')}")


if __name__ == "__main__":
    main()
