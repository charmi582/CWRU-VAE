# IMS Federated External Validation Pilot

This note records the completed IMS external validation runs for the federated journal track.

## Dataset Preparation

- Dataset: NASA IMS Bearing.
- Window size: 8192.
- Stride: 4096.
- Normal-only training candidates: earliest 20% of files in each run-to-failure test.
- Fault audit windows: latest 20% of files in each run-to-failure test.
- Middle degradation region: excluded from this first anomaly-monitoring protocol.
- Loaded windows: 37,184.
- Normal windows: 18,592.
- Fault audit windows: 18,592.
- Client partitions:
  - Bearing-wise: 8 clients.
  - Condition-wise: 3 clients.

## Formal Command

```bash
python federated_fuzzy_experiment.py \
  --dataset ims \
  --models cnn-ae \
  --client-partitions bearing condition \
  --rounds 10 \
  --local-epochs 2 \
  --centralized-epochs 20 \
  --seeds 42 202 777 \
  --evaluate-each-round \
  --federated-methods fedavg fedprox fedbn \
  --fedprox-mu 0.01 \
  --personalize-epochs 1 \
  --calibration-scopes client_specific pooled adaptive
```

## Main Observations

The IMS formal run is substantially more challenging than CWRU-style strict splits. Under bearing-wise clients, local-only training reaches the strongest F1, while FedBN and personalized federated variants approach the local-only operating point. Plain FedAvg remains weaker, supporting the argument that federated learning should not be presented as an automatic accuracy booster.

Under condition-wise clients, local-only training again remains strong, but FedBN and personalized federated variants outperform plain FedAvg. This suggests that non-IID adaptation and local personalization are necessary when clients represent different run-to-failure tests or operating histories.

Calibration behavior is central. Client-specific calibration gives better F1 than pooled calibration in most IMS settings. Pooled calibration often increases miss rate, which is important for maintenance deployment because fewer false alarms can be obtained at the cost of missing more fault audit windows. Adaptive calibration provides an intermediate deployment policy, especially when the objective is to reduce extreme client-specific threshold behavior rather than maximize F1 alone.

## Best Formal Operating Points

Bearing-wise clients:

- Local-only, client-specific calibration: F1 = 0.8021, FAR = 0.0488, miss rate = 0.3010.
- FedBN, client-specific calibration: F1 = 0.7235, FAR = 0.0542, miss rate = 0.3871.
- FedProx-personalized, client-specific calibration: F1 = 0.7064, FAR = 0.0582, miss rate = 0.4091.
- FedAvg-personalized, client-specific calibration: F1 = 0.7032, FAR = 0.0522, miss rate = 0.4138.

Condition-wise clients:

- Local-only, client-specific calibration: F1 = 0.6639, FAR = 0.0546, miss rate = 0.4927.
- FedBN, client-specific calibration: F1 = 0.5491, FAR = 0.0547, miss rate = 0.6066.
- FedAvg-personalized, client-specific calibration: F1 = 0.5419, FAR = 0.0538, miss rate = 0.6136.
- FedProx-personalized, client-specific calibration: F1 = 0.5409, FAR = 0.0517, miss rate = 0.6141.

## Journal Interpretation

IMS strengthens the journal story because it shows that external run-to-failure validation is not a clean benchmark with perfect separability. The model can still rank many fault windows highly, as shown by high PR-AUC values, but thresholded deployment metrics remain difficult. This supports the paper's main claim: deployable federated bearing health monitoring requires model training, client-aware calibration, and fuzzy health-state reporting to be evaluated together.

The next paper-writing step is to compare Paderborn and IMS side by side. Paderborn can be used as the controlled multi-condition benchmark, while IMS can be used as the more realistic run-to-failure external validation dataset. The central interpretation should be that the federated model gives a privacy-preserving collaborative operating point, but calibration and non-IID adaptation determine whether the detector is deployable.
