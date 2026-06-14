# Federated Journal Paper Plan

## Working Title

Privacy-Preserving Normal-Only Bearing Health Monitoring Using Federated Reconstruction Models and Fuzzy Health Decisions

## Central Thesis

This paper studies whether bearing health monitoring can be deployed across multiple companies or factory sites without collecting raw vibration data at a central server. The work does not claim that federated learning automatically improves accuracy. Instead, it treats FedAvg as a privacy-preserving deployment mechanism and evaluates its practical tradeoffs against local-only and centralized references.

## Research Questions

**RQ1. Centralized reference gap.** Can FedAvg approach a centralized reference model while keeping raw bearing vibration windows local to each client?

**RQ2. Local-only tradeoff.** Does FedAvg provide a more useful deployment point than training each client independently?

**RQ3. Client stability.** Which clients suffer false-alarm, miss-rate, or health-index drift under local-only, centralized, and federated settings?

**RQ4. Calibration under non-IID clients.** Does client-specific threshold calibration behave differently from pooled calibration when clients represent different bearings or operating conditions?

**RQ5. Deployability.** What communication cost is introduced by FedAvg, and how does it compare conceptually with sharing raw vibration streams?

## Main Contributions

1. A privacy-preserving normal-only bearing monitoring framework that trains reconstruction models through local client updates and FedAvg aggregation.
2. A deployment-oriented comparison among local-only, centralized, and federated settings under bearing-wise and condition-wise client partitions.
3. A calibration analysis showing that threshold anchors are not neutral under non-IID clients; client-specific and pooled calibration change FAR/miss-rate behavior.
4. A fuzzy health-index decision layer that converts anomaly scores into graded normal, warning, fault, and uncertain regions.
5. A reproducible experiment package with metrics, client-stability summaries, convergence curves, communication-cost estimates, and figure-generation scripts.

## Experimental Design

- Dataset: Paderborn bearing dataset.
- Clients:
  - bearing-wise clients
  - condition-wise clients
- Training data:
  - normal windows only
- Fault data:
  - audit-only evaluation, never used for training
- Training modes:
  - local-only
  - centralized
  - federated FedAvg
- Model:
  - main: CNN-AE
  - supplementary: VAE after stabilization
- Rounds and seeds:
  - FedAvg rounds: 10
  - local epochs: 2
  - centralized epochs: 20
  - seeds: 42, 202, 777
- Calibration:
  - client-specific validation anchors
  - pooled validation anchors

## Current Result Interpretation

FedAvg does not dominate local-only or centralized training in F1. This should be presented as an honest deployment result rather than a weakness. In realistic industrial settings, the value of federated learning is not automatic accuracy gain, but privacy-preserving collaboration. The important finding is that calibration and client stability can change substantially even when ranking metrics such as PR-AUC remain high.

## Figures Already Available

- `fig_federated_architecture.png`: privacy-preserving deployment architecture
- `fig_federated_mean_performance.png`: local-only vs centralized vs federated
- `fig_calibration_scope_comparison.png`: client-specific vs pooled calibration
- `fig_client_false_alarm_rate.png`: client-wise false-alarm analysis
- `fig_client_miss_rate.png`: client-wise miss-rate analysis
- `fig_client_health_gap.png`: fuzzy health-index drift
- `fig_fedavg_convergence.png`: FedAvg convergence across rounds
- `fig_communication_cost.png`: deployment communication traffic

## Recommended Journal Direction

The paper should target an AI-manufacturing or industrial informatics venue rather than a pure fault-diagnosis accuracy venue. Strong candidates are Journal of Intelligent Manufacturing, Computers in Industry, or an industrial AI special issue. The paper should avoid overclaiming predictive maintenance or remaining useful life unless time-to-failure labels are added later.
