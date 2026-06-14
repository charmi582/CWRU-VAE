# Federated Fuzzy Formal Experiment Notes

Setting: Paderborn normal-only clients, client partitions by bearing and by operating condition, seeds 42/202/777, federated rounds=10, local epochs=2, centralized epochs=20, max files per condition=1. Fault windows are audit-only.

## Added federated-specific analyses

- Client-specific vs pooled calibration: compares whether each site should calibrate thresholds from its own validation-normal scores or use shared pooled calibration anchors.
- FedProx: adds a proximal local-training penalty to reduce non-IID client drift relative to FedAvg.
- Personalized federated adaptation: fine-tunes each federated global model locally for one epoch before client evaluation.
- Communication cost: estimates FP32 model-weight upload/download traffic for FedAvg. Raw vibration windows are never transmitted.

## Main interpretation

- The contribution should be framed as privacy-preserving collaborative monitoring, not generic accuracy improvement.
- CNN-AE is the main model for the federated study; VAE should remain supplementary unless further stabilized.
- FedProx-personalized improves the current F1 operating point compared with plain FedAvg under both bearing-wise and condition-wise clients.
- Client-specific calibration is important because pooled calibration can reduce false alarms while increasing missed detections under non-IID clients.
- Communication cost is measurable and far smaller than centralizing continuous raw vibration streams in real deployments, but it is not zero.
