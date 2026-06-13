# Paderborn 8192-Window Deep Pilot Summary

This follow-up reruns the Paderborn pilot with an 8192-point window and 4096-point stride after the window-size sensitivity diagnostic showed clear gains for longer windows.

## Threshold Summary

| threshold_strategy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_mad5 | 0.6304 | 0.7261 | 0.0899 | 0.4482 | 0.0654 | 0.0245 | 0.9346 |
| train_p95 | 0.6304 | 0.7261 | 0.4167 | 0.6684 | 0.3868 | 0.2189 | 0.6132 |
| train_p99 | 0.6304 | 0.7261 | 0.2141 | 0.5187 | 0.2012 | 0.1643 | 0.7988 |
| val_p95 | 0.6304 | 0.7261 | 0.2127 | 0.5878 | 0.1759 | 0.0778 | 0.8241 |
| val_p99 | 0.6304 | 0.7261 | 0.0304 | 0.5134 | 0.0169 | 0.0241 | 0.9831 |

## train_p95 Results by Protocol and Model

| protocol | model | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paderborn-bearing-wise | cnn-ae | 0.4811 | 0.6631 | 0.7348 | 0.7379 | 0.7318 | 0.6447 | 0.2682 |
| paderborn-bearing-wise | iforest | 0.9454 | 0.9491 | 0.6188 | 0.9549 | 0.5654 | 0.0305 | 0.4346 |
| paderborn-bearing-wise | lstm-ae | 0.5137 | 0.7130 | 0.0939 | 0.6086 | 0.0512 | 0.0736 | 0.9488 |
| paderborn-bearing-wise | vae | 0.5352 | 0.7012 | 0.4520 | 0.6488 | 0.4774 | 0.3579 | 0.5226 |
| paderborn-condition-wise | cnn-ae | 0.5626 | 0.5188 | 0.6108 | 0.5386 | 0.7362 | 0.6155 | 0.2638 |
| paderborn-condition-wise | iforest | 0.8048 | 0.8166 | 0.6986 | 0.8423 | 0.6274 | 0.2118 | 0.3726 |
| paderborn-condition-wise | lstm-ae | 0.4898 | 0.4980 | 0.0939 | 0.5289 | 0.0522 | 0.0528 | 0.9478 |
| paderborn-condition-wise | vae | 0.4803 | 0.4446 | 0.0161 | 0.0625 | 0.0092 | 0.0557 | 0.9908 |
| paderborn-file-wise | cnn-ae | 0.7528 | 0.9150 | 0.7555 | 0.8787 | 0.6860 | 0.4385 | 0.3140 |
| paderborn-file-wise | iforest | 0.8624 | 0.9372 | 0.7161 | 0.9495 | 0.6019 | 0.1636 | 0.3981 |
| paderborn-file-wise | lstm-ae | 0.4886 | 0.7927 | 0.0933 | 0.7266 | 0.0501 | 0.0693 | 0.9499 |
| paderborn-file-wise | vae | 0.6247 | 0.8246 | 0.2328 | 0.6818 | 0.1919 | 0.0287 | 0.8081 |

## Best Threshold by Protocol and Model

| protocol | model | threshold_strategy | roc_auc | pr_auc | f1 | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paderborn-bearing-wise | cnn-ae | train_p95 | 0.4811 | 0.6631 | 0.7348 | 0.6447 | 0.2682 |
| paderborn-bearing-wise | iforest | train_p95 | 0.9454 | 0.9491 | 0.6188 | 0.0305 | 0.4346 |
| paderborn-bearing-wise | lstm-ae | train_p95 | 0.5137 | 0.7130 | 0.0939 | 0.0736 | 0.9488 |
| paderborn-bearing-wise | vae | train_p95 | 0.5352 | 0.7012 | 0.4520 | 0.3579 | 0.5226 |
| paderborn-condition-wise | cnn-ae | train_p95 | 0.5626 | 0.5188 | 0.6108 | 0.6155 | 0.2638 |
| paderborn-condition-wise | iforest | val_p95 | 0.8048 | 0.8166 | 0.7394 | 0.2237 | 0.3072 |
| paderborn-condition-wise | lstm-ae | train_p95 | 0.4898 | 0.4980 | 0.0939 | 0.0528 | 0.9478 |
| paderborn-condition-wise | vae | val_p95 | 0.4803 | 0.4446 | 0.0226 | 0.0659 | 0.9867 |
| paderborn-file-wise | cnn-ae | train_p95 | 0.7528 | 0.9150 | 0.7555 | 0.4385 | 0.3140 |
| paderborn-file-wise | iforest | train_p95 | 0.8624 | 0.9372 | 0.7161 | 0.1636 | 0.3981 |
| paderborn-file-wise | lstm-ae | train_p95 | 0.4886 | 0.7927 | 0.0933 | 0.0693 | 0.9499 |
| paderborn-file-wise | vae | train_p95 | 0.6247 | 0.8246 | 0.2328 | 0.0287 | 0.8081 |

## Efficiency

| model | inference_ms_per_window_cpu | model_size_mb | param_count | train_seconds |
| --- | --- | --- | --- | --- |
| cnn-ae | 8.2687 | 128.5761 | 33704737.0000 | 5.2630 |
| iforest | 0.0000 | 0.0000 | 300.0000 | 0.0000 |
| lstm-ae | 4.2028 | 0.2086 | 54689.0000 | 1.4056 |
| vae | 7.5873 | 128.6075 | 33712961.0000 | 3.8916 |

## Interpretation

- Longer Paderborn windows substantially improve external validation compared with the initial 1024-point pilot.
- Isolation Forest remains the strongest baseline on this subset, suggesting hand-crafted statistical features are more robust than reconstruction models under this small external pilot.
- VAE/CNN-AE/LSTM-AE still lag behind Isolation Forest, so the journal manuscript should not claim deep-model superiority on external data.
- The external-data story should be framed as sampling-rate-aware preprocessing plus threshold calibration, not as a universal VAE win.
