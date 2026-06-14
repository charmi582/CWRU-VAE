"""Deployment and communication-cost analysis for federated bearing monitoring."""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

from ae_model import ConvAE
from config import BETA, LATENT_DIM
from journal_experiment_utils import ensure_dir
from vae_model import VAE


OUT_DIR = os.path.join("results", "federated_fuzzy")


@dataclass
class CommunicationCostRow:
    model: str
    client_partition: str
    num_clients: int
    rounds: int
    parameters: int
    model_size_mb_fp32: float
    uplink_mb_total: float
    downlink_mb_total: float
    bidirectional_mb_total: float
    raw_data_shared: str


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_factory(model_name: str, input_size: int):
    if model_name == "cnn-ae":
        return ConvAE(input_size, LATENT_DIM)
    if model_name == "vae":
        return VAE(input_size, LATENT_DIM, BETA)
    raise ValueError(model_name)


def build_rows(input_size: int = 8192, rounds: int = 10) -> list[CommunicationCostRow]:
    client_counts = {"bearing": 6, "condition": 4}
    rows = []
    for model_name in ("cnn-ae", "vae"):
        model = model_factory(model_name, input_size)
        params = count_parameters(model)
        model_size_mb = params * 4 / (1024**2)
        for partition, num_clients in client_counts.items():
            uplink = model_size_mb * num_clients * rounds
            downlink = model_size_mb * num_clients * rounds
            rows.append(
                CommunicationCostRow(
                    model=model_name,
                    client_partition=partition,
                    num_clients=num_clients,
                    rounds=rounds,
                    parameters=params,
                    model_size_mb_fp32=round(model_size_mb, 4),
                    uplink_mb_total=round(uplink, 4),
                    downlink_mb_total=round(downlink, 4),
                    bidirectional_mb_total=round(uplink + downlink, 4),
                    raw_data_shared="No",
                )
            )
    return rows


def write_rows(rows: list[CommunicationCostRow]) -> str:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "federated_communication_cost_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CommunicationCostRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def main() -> None:
    path = write_rows(build_rows())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
