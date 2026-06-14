# Federated Learning Research Questions

This research stage studies privacy-preserving normal-only bearing anomaly detection across simulated companies or factory sites. Each client owns local normal bearing vibration windows and does not share raw data. The server receives only model weights and aggregates them with FedAvg.

## Core Research Questions

**RQ1. Centralized reference gap**

Can FedAvg approach the centralized reference setting without sharing raw bearing vibration windows?

Centralized training is treated as an upper-reference baseline because it pools normal windows from all clients. It is not privacy-preserving, but it shows what could be achieved if data sharing were allowed.

**RQ2. Local-only comparison**

Does federated training provide a more useful or more stable operating point than local-only training?

Local-only training represents each company or site training its own detector without collaboration. FedAvg is only meaningful if it offers a practical tradeoff relative to this baseline.

**RQ3. Client stability**

Which clients show false alarm or miss-rate drift under local-only, centralized, and federated training?

This question is important because industrial deployment is not judged only by mean F1. A method can look acceptable on average while one site suffers high false alarms or missed detections.

**RQ4. Fuzzy health drift**

Can the fuzzy health index reveal client-to-client health-score drift that is hidden by binary alarm metrics?

The fuzzy layer is used as a model-agnostic decision layer. It reports uncertain rate, mean health index for normal windows, mean health index for audit fault windows, and the health gap between both groups.

## Experimental Design

- Dataset: Paderborn bearing dataset.
- Clients: simulated by `bearing` and `condition`.
- Training data: normal windows only.
- Fault windows: audit-only evaluation; never used for training.
- Training modes:
  - `local-only`: each client trains independently.
  - `centralized`: pooled normal windows train one reference model.
  - `federated`: clients train locally and share model weights through FedAvg.
- Main model: CNN-AE.
- Supplementary model: VAE.
- Planned formal setting:
  - FedAvg rounds: 10 or 20.
  - Local epochs per round: 2 or 3.
  - Seeds: at least 3.
  - Client partitions: bearing-wise and condition-wise.

## Expected Contribution

The expected contribution is not that FedAvg automatically improves accuracy. The contribution is a privacy-preserving comparison framework that reports centralized gap, local-only tradeoffs, client stability, fuzzy health drift, and deployment-oriented alarm behavior.
