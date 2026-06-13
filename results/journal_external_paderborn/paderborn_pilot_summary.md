# Paderborn External Validation Pilot Summary

This pilot uses a subset of the official Paderborn Bearing Dataset: healthy bearings `K001`, `K002` and faulty bearings `KA01`, `KI01`, with two files per operating condition. Raw archives and extracted MATLAB files are intentionally excluded from git.

## Experiment Matrix

- Windows loaded: 16,070 total; 8,073 normal and 7,997 fault.
- Protocols: condition-wise, bearing-wise, file-wise.
- Models: VAE, CNN-AE, LSTM-AE, Isolation Forest.
- Threshold strategies: train_mad5, train_p95, train_p99, val_p95, val_p99.
- Total result rows: 200.

## Threshold Calibration Summary

| threshold_strategy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_mad5 | 0.5664 | 0.6888 | 0.0034 | 0.1108 | 0.0018 | 0.0117 | 0.9982 |
| train_p95 | 0.5664 | 0.6888 | 0.1501 | 0.5846 | 0.1048 | 0.1365 | 0.8952 |
| train_p99 | 0.5664 | 0.6888 | 0.0305 | 0.4301 | 0.0204 | 0.0547 | 0.9796 |
| val_p95 | 0.5664 | 0.6888 | 0.1264 | 0.5810 | 0.0853 | 0.0769 | 0.9147 |
| val_p99 | 0.5664 | 0.6888 | 0.0152 | 0.3999 | 0.0083 | 0.0230 | 0.9917 |

Interpretation: unlike CWRU, the Paderborn subset is not trivially separable. AUC and PR-AUC drop substantially, and conservative thresholds such as `val_p99` cause very high miss rates. This is important evidence that external validation exposes deployment risk that CWRU alone hides.

## Headline Results Using train_p95

| protocol | model | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paderborn-bearing-wise | cnn-ae | 0.4111 | 0.6269 | 0.0628 | 0.3099 | 0.0350 | 0.1799 | 0.9650 |
| paderborn-bearing-wise | iforest | 0.7667 | 0.8470 | 0.2791 | 0.8320 | 0.1780 | 0.0514 | 0.8220 |
| paderborn-bearing-wise | lstm-ae | 0.5250 | 0.6874 | 0.0463 | 0.5546 | 0.0241 | 0.0480 | 0.9759 |
| paderborn-bearing-wise | vae | 0.4473 | 0.6457 | 0.0768 | 0.3553 | 0.0432 | 0.1217 | 0.9568 |
| paderborn-condition-wise | cnn-ae | 0.4128 | 0.4159 | 0.0983 | 0.1467 | 0.0914 | 0.2479 | 0.9086 |
| paderborn-condition-wise | iforest | 0.7196 | 0.7373 | 0.3770 | 0.7853 | 0.2685 | 0.1434 | 0.7315 |
| paderborn-condition-wise | lstm-ae | 0.5468 | 0.4998 | 0.0556 | 0.3685 | 0.0305 | 0.0496 | 0.9695 |
| paderborn-condition-wise | vae | 0.4244 | 0.4214 | 0.1325 | 0.1845 | 0.1304 | 0.2996 | 0.8696 |
| paderborn-file-wise | cnn-ae | 0.5789 | 0.8385 | 0.0930 | 0.8132 | 0.0532 | 0.1147 | 0.9468 |
| paderborn-file-wise | iforest | 0.7040 | 0.8832 | 0.3572 | 0.8929 | 0.2453 | 0.1311 | 0.7547 |
| paderborn-file-wise | lstm-ae | 0.6558 | 0.8625 | 0.0544 | 0.8234 | 0.0281 | 0.0291 | 0.9719 |
| paderborn-file-wise | vae | 0.5467 | 0.8257 | 0.1007 | 0.8057 | 0.0605 | 0.1493 | 0.9395 |

## Best Threshold Strategy by Protocol and Model

| protocol | model | threshold_strategy | roc_auc | pr_auc | f1 | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paderborn-bearing-wise | cnn-ae | train_p95 | 0.4111 | 0.6269 | 0.0628 | 0.1799 | 0.9650 |
| paderborn-bearing-wise | iforest | train_p95 | 0.7667 | 0.8470 | 0.2791 | 0.0514 | 0.8220 |
| paderborn-bearing-wise | lstm-ae | train_p95 | 0.5250 | 0.6874 | 0.0463 | 0.0480 | 0.9759 |
| paderborn-bearing-wise | vae | train_p95 | 0.4473 | 0.6457 | 0.0768 | 0.1217 | 0.9568 |
| paderborn-condition-wise | cnn-ae | train_p95 | 0.4128 | 0.4159 | 0.0983 | 0.2479 | 0.9086 |
| paderborn-condition-wise | iforest | train_p95 | 0.7196 | 0.7373 | 0.3770 | 0.1434 | 0.7315 |
| paderborn-condition-wise | lstm-ae | val_p95 | 0.5468 | 0.4998 | 0.0599 | 0.0525 | 0.9670 |
| paderborn-condition-wise | vae | train_p95 | 0.4244 | 0.4214 | 0.1325 | 0.2996 | 0.8696 |
| paderborn-file-wise | cnn-ae | train_p95 | 0.5789 | 0.8385 | 0.0930 | 0.1147 | 0.9468 |
| paderborn-file-wise | iforest | val_p95 | 0.7040 | 0.8832 | 0.4034 | 0.1401 | 0.7178 |
| paderborn-file-wise | lstm-ae | val_p95 | 0.6558 | 0.8625 | 0.0622 | 0.0341 | 0.9676 |
| paderborn-file-wise | vae | train_p95 | 0.5467 | 0.8257 | 0.1007 | 0.1493 | 0.9395 |

## Efficiency Summary

| model | inference_ms_per_window_cpu | model_size_mb | param_count | train_seconds |
| --- | --- | --- | --- | --- |
| cnn-ae | 1.4932 | 16.3574 | 4287265.0000 | 7.4901 |
| iforest | 0.0000 | 0.0000 | 300.0000 | 0.0000 |
| lstm-ae | 0.8488 | 0.2086 | 54689.0000 | 6.2002 |
| vae | 1.3800 | 16.3887 | 4295489.0000 | 8.0422 |

## Journal Writing Implications

- The external dataset should be presented as a challenging generalisation check, not as a performance victory.
- The CWRU-only conclusion must be softened: high CWRU AUC does not guarantee cross-dataset transfer.
- Threshold calibration becomes more important because Paderborn shows high miss-rate sensitivity under conservative thresholds.
- The next formal run should expand the Paderborn subset to more healthy and damaged bearing states, then compare whether per-dataset calibration or shared calibration is more defensible.
