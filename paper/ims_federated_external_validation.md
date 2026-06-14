# IMS Federated External Validation Pilot

This note records the first completed IMS external validation run for the federated journal track.

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

## Pilot Command

```bash
python federated_fuzzy_experiment.py \
  --dataset ims \
  --models cnn-ae \
  --client-partitions bearing condition \
  --rounds 3 \
  --local-epochs 1 \
  --centralized-epochs 3 \
  --seeds 42 \
  --evaluate-each-round \
  --federated-methods fedavg fedprox fedbn \
  --fedprox-mu 0.01 \
  --personalize-epochs 1 \
  --calibration-scopes client_specific pooled adaptive
```

## Main Observations

The IMS pilot is substantially more challenging than CWRU-style strict splits. Under bearing-wise clients, local-only training reaches the strongest pilot F1, while personalized FedProx, personalized FedBN, and personalized FedAvg approach the local-only operating point. Plain FedAvg remains weaker, supporting the argument that federated learning should not be presented as an automatic accuracy booster.

Under condition-wise clients, local-only training again remains strong, but FedBN and personalized federated variants outperform plain FedAvg. This suggests that non-IID adaptation and local personalization are necessary when clients represent different run-to-failure tests or operating histories.

Calibration behavior is central. Client-specific calibration gives better F1 than pooled calibration in most IMS settings. Pooled calibration often increases miss rate, which is important for maintenance deployment because fewer false alarms can be obtained at the cost of missing more fault audit windows. Adaptive calibration provides an intermediate deployment policy and should be evaluated in the formal multi-seed IMS run.

## Best Pilot Operating Points

Bearing-wise clients:

- Local-only, client-specific calibration: F1 = 0.7377, FAR = 0.0605, miss rate = 0.3658.
- FedProx-personalized, client-specific calibration: F1 = 0.7216, FAR = 0.0450, miss rate = 0.3861.
- FedBN-personalized, client-specific calibration: F1 = 0.7191, FAR = 0.0447, miss rate = 0.3890.

Condition-wise clients:

- Local-only, client-specific calibration: F1 = 0.6406, FAR = 0.0571, miss rate = 0.5188.
- FedBN-personalized, client-specific calibration: F1 = 0.5373, FAR = 0.0546, miss rate = 0.6232.
- FedBN, client-specific calibration: F1 = 0.5342, FAR = 0.0501, miss rate = 0.6168.

## Journal Interpretation

IMS strengthens the journal story because it shows that external run-to-failure validation is not a clean benchmark with perfect separability. The model can still rank many fault windows highly, as shown by high PR-AUC values, but thresholded deployment metrics remain difficult. This supports the paper's main claim: deployable federated bearing health monitoring requires model training, client-aware calibration, and fuzzy health-state reporting to be evaluated together.

The next formal step is a multi-seed IMS run with longer federated training, followed by a compact comparison against the Paderborn results.
