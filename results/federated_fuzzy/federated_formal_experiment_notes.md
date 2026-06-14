# Federated Fuzzy Formal Experiment Notes

Setting: Paderborn normal-only clients, client partitions by bearing and by operating condition, seeds 42/202/777, federated rounds=10, local epochs=2, centralized epochs=20, max files per condition=1. Fault windows are audit-only.

## Added federated-specific analyses

- Client-specific vs pooled calibration: compares whether each site should calibrate thresholds from its own validation-normal scores or use shared pooled calibration anchors.
- Adaptive calibration: blends client-specific and pooled anchors according to validation-score distribution drift. It moves toward client anchors when local normal-score distributions differ from the pooled reference.
- FedProx: adds a proximal local-training penalty to reduce non-IID client drift relative to FedAvg.
- FedBN: keeps BatchNorm parameters and running statistics local to each client while aggregating the remaining model parameters.
- Personalized federated adaptation: fine-tunes each federated global model locally for one epoch before client evaluation.
- Communication cost: estimates FP32 model-weight upload/download traffic for FedAvg. Raw vibration windows are never transmitted.

## Main interpretation

- The contribution should be framed as privacy-preserving collaborative monitoring, not generic accuracy improvement.
- CNN-AE is the main model for the federated study; VAE should remain supplementary unless further stabilized.
- FedBN-personalized provides the current strongest bearing-wise F1 operating point, while personalized FedAvg, FedProx, and FedBN all improve over plain FedAvg under condition-wise clients.
- Client-specific calibration provides stronger F1 operating points, while adaptive calibration often reduces client-to-client miss-rate instability compared with client-specific calibration.
- Pooled calibration can reduce false alarms while increasing missed detections under non-IID clients.
- Communication cost is measurable and far smaller than centralizing continuous raw vibration streams in real deployments, but it is not zero.
