"""Generate figures for the federated fuzzy bearing-monitoring experiment."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from journal_experiment_utils import ensure_dir


OUT_DIR = os.path.join("results", "federated_fuzzy")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def _save(name: str) -> None:
    ensure_dir(FIG_DIR)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=220, bbox_inches="tight")
    plt.close()


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")

    clients = [("Client A", "bearing or condition\nsite"), ("Client B", "private normal\nwindows"), ("Client C", "local validation\ncalibration")]
    x_positions = [0.08, 0.08, 0.08]
    y_positions = [0.72, 0.48, 0.24]
    for (title, body), x, y in zip(clients, x_positions, y_positions):
        ax.add_patch(plt.Rectangle((x, y), 0.22, 0.15, fill=False, lw=1.8, ec="#34699a"))
        ax.text(x + 0.11, y + 0.095, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + 0.11, y + 0.045, body, ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((0.41, 0.43), 0.18, 0.22, fill=False, lw=1.8, ec="#c05a5a"))
    ax.text(0.50, 0.58, "FedAvg server", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.50, 0.51, "aggregates model\nweights only", ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((0.72, 0.54), 0.20, 0.15, fill=False, lw=1.8, ec="#51865c"))
    ax.text(0.82, 0.64, "Shared detector", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.82, 0.59, "CNN-AE / VAE", ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((0.72, 0.30), 0.20, 0.15, fill=False, lw=1.8, ec="#6a5b9a"))
    ax.text(0.82, 0.40, "Local decision", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.82, 0.35, "threshold + fuzzy HI", ha="center", va="center", fontsize=9)

    for y in y_positions:
        ax.annotate("", xy=(0.41, 0.54), xytext=(0.30, y + 0.075), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4d5b6a"))
    ax.annotate("", xy=(0.72, 0.62), xytext=(0.59, 0.57), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4d5b6a"))
    ax.annotate("", xy=(0.82, 0.45), xytext=(0.82, 0.54), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4d5b6a"))

    ax.text(0.50, 0.92, "Privacy-preserving federated bearing health monitoring", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.50, 0.86, "Raw vibration windows remain inside each client; only model parameters are transmitted.", ha="center", fontsize=10)
    _save("fig_federated_architecture.png")


def plot_mean_performance(summary: pd.DataFrame) -> None:
    data = summary[summary["decision_policy"].eq("fuzzy_warning_as_alarm_hi50")]
    data = data[data["calibration_scope"].eq("client_specific")]
    labels = data["clients_by"] + "\n" + data["training_mode"]
    x = range(len(data))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x, data["f1"], color="#6a8caf")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, max(0.25, data["f1"].max() * 1.25))
    ax.set_title("Local-only, centralized and federated performance")
    _save("fig_federated_mean_performance.png")


def plot_calibration_scope(summary: pd.DataFrame) -> None:
    data = summary[
        summary["decision_policy"].eq("fuzzy_warning_as_alarm_hi50")
        & summary["training_mode"].isin(["centralized", "fedavg", "fedprox"])
    ].copy()
    data["label"] = data["clients_by"] + " / " + data["training_mode"] + " / " + data["calibration_scope"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(range(len(data)), data["miss_rate"], color="#c77855", label="Miss rate")
    ax.plot(range(len(data)), data["false_alarm_rate"], marker="o", color="#2f5f8f", label="False alarm rate")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["label"], rotation=45, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Client-specific versus pooled threshold calibration")
    ax.legend()
    _save("fig_calibration_scope_comparison.png")


def _plot_client_metric(detail: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    data = detail[
        detail["decision_policy"].eq("fuzzy_warning_as_alarm_hi50")
        & detail["calibration_scope"].eq("client_specific")
    ].copy()
    grouped = data.groupby(["clients_by", "training_mode"], as_index=False)[metric].mean()
    labels = grouped["clients_by"] + "\n" + grouped["training_mode"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(range(len(grouped)), grouped[metric], color="#7aa874")
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by federated setting")
    _save(name)


def plot_convergence(convergence: pd.DataFrame) -> None:
    data = convergence[
        convergence["decision_policy"].eq("fuzzy_warning_as_alarm_hi50")
        & convergence["calibration_scope"].eq("client_specific")
    ]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for label, sub in data.groupby(["clients_by", "training_mode"]):
        sub = sub.sort_values("round")
        ax.plot(sub["round"], sub["f1"], marker="o", label=f"{label[0]} / {label[1]}")
    ax.set_xlabel("FedAvg round")
    ax.set_ylabel("F1")
    ax.set_title("FedAvg convergence under non-IID clients")
    ax.legend(title="Client split / method")
    _save("fig_fedavg_convergence.png")


def plot_communication(cost: pd.DataFrame) -> None:
    data = cost[cost["model"].eq("cnn-ae")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(data["client_partition"], data["bidirectional_mb_total"], color="#8a6f9e")
    ax.set_ylabel("Bidirectional traffic (MB)")
    ax.set_title("FedAvg communication cost over 10 rounds")
    for i, value in enumerate(data["bidirectional_mb_total"]):
        ax.text(i, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    _save("fig_communication_cost.png")


def main() -> None:
    summary = pd.read_csv(os.path.join(OUT_DIR, "federated_fuzzy_summary.csv"))
    detail = pd.read_csv(os.path.join(OUT_DIR, "federated_fuzzy_client_detail.csv"))
    convergence = pd.read_csv(os.path.join(OUT_DIR, "federated_fuzzy_convergence.csv"))
    cost = pd.read_csv(os.path.join(OUT_DIR, "federated_communication_cost_summary.csv"))

    plot_architecture()
    plot_mean_performance(summary)
    plot_calibration_scope(summary)
    _plot_client_metric(detail, "false_alarm_rate", "False alarm rate", "fig_client_false_alarm_rate.png")
    _plot_client_metric(detail, "miss_rate", "Miss rate", "fig_client_miss_rate.png")
    _plot_client_metric(detail, "health_gap", "Fuzzy health gap", "fig_client_health_gap.png")
    plot_convergence(convergence)
    plot_communication(cost)
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
