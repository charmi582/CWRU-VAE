# Federated Journal Paper Plan

## Working Title

Privacy-Preserving Normal-Only Bearing Health Monitoring Using Federated Reconstruction Models and Fuzzy Health Decisions

## Central Thesis

This paper studies whether bearing health monitoring can be deployed across multiple companies or factory sites without collecting raw vibration data at a central server. The work does not claim that federated learning automatically improves accuracy. Instead, it treats federated learning as a privacy-preserving deployment mechanism and evaluates FedAvg, FedProx, FedBN, and personalized local adaptation against local-only and centralized references.

## Research Questions

**RQ1. Centralized reference gap.** Can FedAvg approach a centralized reference model while keeping raw bearing vibration windows local to each client?

**RQ2. Local-only tradeoff.** Do FedAvg, FedProx, FedBN, or personalized federated adaptation provide a more useful deployment point than training each client independently?

**RQ3. Client stability.** Which clients suffer false-alarm, miss-rate, or health-index drift under local-only, centralized, and federated settings?

**RQ4. Calibration under non-IID clients.** Does client-specific, pooled, or adaptive threshold calibration provide the best deployment tradeoff when clients represent different bearings or operating conditions?

**RQ5. Non-IID federated adaptation.** Does FedProx, FedBN, or one-step client personalization reduce the false-alarm, miss-rate, or health-index drift observed with plain FedAvg?

**RQ6. Deployability.** What communication cost is introduced by federated model updates, and how does it compare conceptually with sharing raw vibration streams?

## Main Contributions

1. A privacy-preserving normal-only bearing monitoring framework that trains reconstruction models through local client updates and federated aggregation.
2. A deployment-oriented comparison among local-only, centralized, FedAvg, FedProx, FedBN, and personalized federated settings under bearing-wise and condition-wise client partitions.
3. A calibration analysis showing that threshold anchors are not neutral under non-IID clients; client-specific, pooled, and adaptive calibration change FAR/miss-rate behavior and client stability.
4. A fuzzy health-index decision layer that converts anomaly scores into graded normal, warning, fault, and uncertain regions.
5. A reproducible experiment package with metrics, client-stability summaries, convergence curves, communication-cost estimates, and figure-generation scripts.

## Experimental Design

- Main dataset: Paderborn bearing dataset.
- External validation track: NASA IMS Bearing dataset.
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
  - FedAvg
  - FedProx
  - FedBN
  - FedAvg-personalized
  - FedProx-personalized
  - FedBN-personalized
- Model:
  - main: CNN-AE
  - supplementary: VAE after stabilization
- Rounds and seeds:
  - federated rounds: 10
  - local epochs: 2
  - personalized epochs: 1
  - centralized epochs: 20
  - seeds: 42, 202, 777
- Calibration:
  - client-specific validation anchors
  - pooled validation anchors
  - adaptive drift-aware blended anchors

## Current Result Interpretation

Plain FedAvg does not dominate local-only or centralized training in F1. This should be presented as an honest deployment result rather than a weakness. In realistic industrial settings, the value of federated learning is not automatic accuracy gain, but privacy-preserving collaboration. The stronger result is that non-IID adaptation matters: FedBN-personalized currently gives the strongest bearing-wise operating point, while personalized FedAvg, FedProx, and FedBN all improve over plain FedAvg under condition-wise clients. Calibration and client stability also change substantially even when ranking metrics such as PR-AUC remain high. Client-specific calibration tends to give stronger F1, pooled calibration often reduces false alarms at the cost of more missed detections, and adaptive calibration can reduce client-to-client instability by blending both.

## External Dataset Plan

IMS Bearing is selected as the next validation dataset because it contains run-to-failure bearing experiments from the NASA Prognostics Data Repository. The first protocol treats early-life files as normal training candidates and late-life files as audit fault windows. Clients can be simulated by bearing channel or run-to-failure test. This adds a stronger external validation axis than Paderborn alone while staying aligned with the normal-only anomaly-monitoring objective.

The first IMS pilot is complete. The dataset loader produced 37,184 windows using 8192-point windows and 4096-point stride, with 18,592 normal windows and 18,592 fault-audit windows. The pilot uses bearing-wise and condition-wise clients, CNN-AE reconstruction, FedAvg, FedProx, FedBN, personalized variants, and client-specific, pooled, and adaptive calibration. Initial results show that IMS is much more difficult than CWRU-style strict splits: local-only remains strong, personalized federated variants approach local-only under bearing-wise clients, and pooled calibration often increases miss rate. The formal next step is a multi-seed IMS run with longer federated training.

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
