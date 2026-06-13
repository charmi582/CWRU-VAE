# Federated Normal-Only Anomaly Detection Summary

This experiment simulates multiple companies or sites as clients. Each client trains on its own normal bearing windows only. The server aggregates model weights with FedAvg and never receives raw client windows.

## Final-Round Calibration Summary

| model | threshold_strategy | normal_false_alarm_rate | optional_fault_recall | optional_fault_miss_rate | optional_fault_f1 |
| --- | --- | --- | --- | --- | --- |
| cnn-ae | train_mad5 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| cnn-ae | train_p95 | 0.2639 | 0.1434 | 0.8566 | 0.2467 |
| cnn-ae | train_p99 | 0.1528 | 0.0676 | 0.9324 | 0.1254 |
| cnn-ae | val_p95 | 0.0833 | 0.0307 | 0.9693 | 0.0593 |
| cnn-ae | val_p99 | 0.0000 | 0.0041 | 0.9959 | 0.0082 |

## Journal Positioning

- This supports a privacy-preserving deployment story: clients keep normal vibration data locally while contributing to a shared anomaly detector.
- The main evaluation on normal-only clients is false alarm rate and threshold stability.
- Fault windows are optional and used only as an external audit, not for federated training.
- This should be presented as a deployment-oriented extension rather than a claim that federated learning automatically improves accuracy.
