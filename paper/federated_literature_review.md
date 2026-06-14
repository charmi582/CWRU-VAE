# Federated Bearing Health Monitoring Literature Review Notes

This note supports the journal paper section on privacy-preserving federated bearing health monitoring. The focus is 2021-2026 literature, plus a few foundational references needed to justify the algorithms and datasets.

## 1. Federated learning for heterogeneous industrial monitoring

Federated learning is relevant to predictive maintenance because factories, operators, or companies may not be able to centralize raw vibration data. The key technical difficulty is that client data are rarely IID: machines operate under different loads, speeds, sensor placements, degradation states, and maintenance histories. A federated monitoring paper should therefore avoid claiming that FL is automatically more accurate than centralized learning. The correct comparison is local-only vs centralized reference vs federated deployment.

The foundational FL baselines are FedAvg, FedProx, and FedBN. FedProx addresses system and statistical heterogeneity by adding a proximal term to the local objective, limiting excessive drift from the current global model. FedBN addresses feature-shift non-IID data by keeping BatchNorm statistics local while aggregating the remaining model parameters. These two baselines are directly relevant to bearing monitoring because operating-condition changes and bearing-to-bearing variation are feature-distribution shifts.

## 2. Federated fault diagnosis and predictive maintenance

Recent rotating-machinery fault diagnosis papers increasingly use federated learning to avoid raw data sharing across distributed clients. Most work remains supervised or classification-oriented, often assuming labeled fault classes at each client. This differs from the present study, where clients train only on normal windows and fault data are used only for audit evaluation. The present framing is closer to deployable industrial screening because many sites have normal operating histories but limited confirmed fault labels.

Important positioning:

- Prior FL fault diagnosis: focuses on privacy-preserving classification accuracy.
- Present work: focuses on normal-only anomaly monitoring, threshold calibration, client stability, and fuzzy health decisions.
- Prior non-IID FL: often reports average accuracy.
- Present work: reports false alarm rate, miss rate, uncertain rate, health-gap drift, calibration policy, and communication cost.

## 3. Non-IID client adaptation

Non-IID behavior is not a side issue in industrial FL; it is the central deployment problem. In bearing monitoring, non-IID can arise from:

- different bearings or bearing histories,
- different speed/load conditions,
- different measurement channels,
- different sensor mounting,
- different degradation trajectories,
- different client-specific normal-score distributions.

FedProx and FedBN are appropriate baselines for this reason. Personalization is also important because a single global model may not be optimal for every client. A short local adaptation step after federated aggregation can improve client-specific operating points while preserving the privacy-preserving training story.

## 4. Calibration in anomaly detection

Reconstruction models produce anomaly scores, not deployment decisions. A score threshold determines false alarms and missed detections. In non-IID federated settings, pooled calibration may suppress false alarms but increase missed detections, while client-specific calibration may improve local sensitivity but create larger client-to-client variability. Adaptive calibration can be positioned as a deployment policy that blends local and pooled anchors according to validation-score drift.

The paper should state clearly that calibration is not post-processing decoration. It is part of the monitoring system. For industrial maintenance, a high PR-AUC is insufficient if the operating threshold is unstable across clients.

## 5. Fuzzy health decisions and interpretability

Binary anomaly alarms are often too rigid for maintenance workflows. Fuzzy health states provide a graded interpretation such as normal, warning, fault, and uncertain. This is useful when anomaly scores are near thresholds or when a client has distribution drift. The fuzzy layer in this study should be positioned as a model-agnostic decision layer attached to reconstruction scores from local-only, centralized, and federated models.

## 6. External datasets

The paper currently uses Paderborn as the main external federated benchmark. The next external validation target is IMS Bearing from NASA PCoE because it contains run-to-failure experiments. For normal-only monitoring, early-life files can be used as normal training candidates and late-life files as audit fault windows. FEMTO/PRONOSTIA is also relevant, especially for RUL-oriented work, but IMS is the more direct next step for anomaly-monitoring external validation.

## Suggested related-work paragraph

Recent federated fault-diagnosis studies show that privacy-preserving training is attractive for distributed industrial assets, but many of them remain supervised and accuracy-centered. In contrast, normal-only health monitoring requires a different evaluation logic: the client may only provide healthy histories for training, while faults appear only during audit or deployment. Under this setting, non-IID client distributions affect not only model training but also threshold calibration and alarm stability. Therefore, our study compares FedAvg, FedProx, FedBN, and personalized variants, and evaluates calibration policies using false alarm rate, miss rate, uncertain rate, health-index drift, convergence, and communication cost rather than only mean classification accuracy.

## Sources Used

- FedProx: Li et al., MLSys 2020.
- FedBN: Li et al., ICLR 2021.
- Paderborn benchmark dataset: Lessmeier et al., PHM Society 2016.
- IMS Bearing dataset: Lee et al., NASA Prognostics Data Repository.
- FEMTO/PRONOSTIA: Nectoux et al., PRONOSTIA/IEEE PHM 2012.
- Recent FL fault diagnosis papers should be expanded during final manuscript polishing with exact journal metadata.
