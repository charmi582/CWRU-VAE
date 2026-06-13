# Journal Strict CWRU Experiment Summary

This summary reports the CWRU-only journal extension experiments before adding external datasets. The experiments address stricter evaluation protocols, stronger baselines, threshold calibration, and deployment efficiency.

## Experiment Matrix

- Protocols: load-wise split, fault-size-wise split, file-wise split.
- Models: VAE, CNN-AE, LSTM-AE, Isolation Forest.
- Threshold calibration strategies: train_p95, train_p99, val_p95, val_p99, train_mad5.
- Total result rows: 220 = 11 held-out groups x 4 models x 5 threshold strategies.

## Threshold Calibration Summary

| threshold_strategy | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- |
| train_mad5 | 0.5525 | 0.5394 | 0.6839 | 0.2470 | 0.3161 |
| train_p95 | 0.7443 | 0.6570 | 0.9745 | 0.3437 | 0.0255 |
| train_p99 | 0.7571 | 0.7215 | 0.9361 | 0.3163 | 0.0639 |
| val_p95 | 0.8490 | 0.7986 | 0.9749 | 0.1785 | 0.0251 |
| val_p99 | 0.8666 | 0.8654 | 0.9401 | 0.1383 | 0.0599 |

Interpretation: validation-calibrated thresholds, especially `val_p99`, reduce average false alarm rate while preserving high recall. This supports repositioning threshold calibration as a core contribution rather than an auxiliary experiment.

## VAE Load-Wise Threshold Behavior

| threshold_strategy | f1 | false_alarm_rate | miss_rate | precision | recall |
| --- | --- | --- | --- | --- | --- |
| train_mad5 | 0.6924 | 0.5000 | 0.0000 | 0.6206 | 1.0000 |
| train_p95 | 0.6206 | 0.5400 | 0.0000 | 0.4967 | 1.0000 |
| train_p99 | 0.6517 | 0.5215 | 0.0000 | 0.5474 | 1.0000 |
| val_p95 | 0.7908 | 0.2834 | 0.0000 | 0.7321 | 1.0000 |
| val_p99 | 0.8274 | 0.2294 | 0.0000 | 0.7898 | 1.0000 |

Interpretation: load-wise ranking remains strong, but FAR is sensitive to calibration. Held-out load 3 still shows severe threshold shift under VAE, so the journal version should explicitly discuss cross-load threshold instability as the practical deployment challenge.

## Headline Results Using val_p99

| protocol | model | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fault-size-wise | cnn-ae | 1.0000 | 1.0000 | 0.9774 | 0.9558 | 1.0000 | 0.0196 | 0.0000 |
| fault-size-wise | iforest | 0.9841 | 0.9660 | 0.8581 | 0.9523 | 0.7837 | 0.0166 | 0.2163 |
| fault-size-wise | lstm-ae | 1.0000 | 1.0000 | 0.9924 | 0.9850 | 1.0000 | 0.0065 | 0.0000 |
| fault-size-wise | vae | 1.0000 | 1.0000 | 0.9702 | 0.9423 | 1.0000 | 0.0262 | 0.0000 |
| file-wise | cnn-ae | 1.0000 | 1.0000 | 0.8127 | 0.7656 | 1.0000 | 0.2544 | 0.0000 |
| file-wise | iforest | 0.9770 | 0.9405 | 0.8272 | 0.9352 | 0.7504 | 0.0140 | 0.2496 |
| file-wise | lstm-ae | 0.9974 | 0.9808 | 0.8682 | 0.8274 | 1.0000 | 0.2500 | 0.0000 |
| file-wise | vae | 1.0000 | 1.0000 | 0.8238 | 0.7803 | 1.0000 | 0.2283 | 0.0000 |
| load-wise | cnn-ae | 1.0000 | 1.0000 | 0.8253 | 0.7849 | 1.0000 | 0.2289 | 0.0000 |
| load-wise | iforest | 0.9781 | 0.9415 | 0.8331 | 0.9344 | 0.7526 | 0.0140 | 0.2474 |
| load-wise | lstm-ae | 0.9987 | 0.9887 | 0.8658 | 0.8254 | 1.0000 | 0.2500 | 0.0000 |
| load-wise | vae | 1.0000 | 1.0000 | 0.8274 | 0.7898 | 1.0000 | 0.2294 | 0.0000 |

## Best Threshold Strategy by Protocol and Model

| protocol | model | threshold_strategy | roc_auc | pr_auc | f1 | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fault-size-wise | cnn-ae | val_p99 | 1.0000 | 1.0000 | 0.9774 | 0.0196 | 0.0000 |
| fault-size-wise | iforest | train_p95 | 0.9841 | 0.9660 | 0.8809 | 0.0604 | 0.0964 |
| fault-size-wise | lstm-ae | val_p99 | 1.0000 | 1.0000 | 0.9924 | 0.0065 | 0.0000 |
| fault-size-wise | vae | val_p99 | 1.0000 | 1.0000 | 0.9702 | 0.0262 | 0.0000 |
| file-wise | cnn-ae | val_p99 | 1.0000 | 1.0000 | 0.8127 | 0.2544 | 0.0000 |
| file-wise | iforest | train_p95 | 0.9770 | 0.9405 | 0.8416 | 0.0662 | 0.1016 |
| file-wise | lstm-ae | train_p99 | 0.9974 | 0.9808 | 0.8685 | 0.2489 | 0.0000 |
| file-wise | vae | val_p99 | 1.0000 | 1.0000 | 0.8238 | 0.2283 | 0.0000 |
| load-wise | cnn-ae | val_p99 | 1.0000 | 1.0000 | 0.8253 | 0.2289 | 0.0000 |
| load-wise | iforest | train_p95 | 0.9781 | 0.9415 | 0.8404 | 0.0662 | 0.1062 |
| load-wise | lstm-ae | train_p99 | 0.9987 | 0.9887 | 0.8661 | 0.2489 | 0.0000 |
| load-wise | vae | val_p99 | 1.0000 | 1.0000 | 0.8274 | 0.2294 | 0.0000 |

## Efficiency Summary

| model | inference_ms_per_window_cpu | model_size_mb | param_count |
| --- | --- | --- | --- |
| cnn-ae | 1.6273 | 16.3574 | 4287265.0000 |
| iforest | 0.0000 | 0.0000 | 300.0000 |
| lstm-ae | 0.8932 | 0.2086 | 54689.0000 |
| vae | 1.3855 | 16.3887 | 4295489.0000 |

Notes: deep-model latency is CPU-only per-window inference latency measured by repeated forward passes. Isolation Forest is included as a classical baseline; its current size/latency fields are placeholders for model-family comparison and should be refined if exact serialized object size is needed.

## Journal Writing Implications

- Do not claim VAE is universally superior on CWRU. AE-family models remain very strong under strict splits.
- Claim that stricter splits expose deployment-relevant threshold instability even when AUC remains high.
- Present threshold calibration as the method-level contribution: held-out-normal calibration gives a practical way to tune FAR/Miss trade-offs under domain shift.
- External bearing datasets remain the next required step for a stronger Q3 submission.
