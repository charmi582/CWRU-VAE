# Expanded Paderborn 8192 Iforest Validation Summary

This formal expanded external validation uses Paderborn bearings `K001-K006`, `KA01`, `KA03`, `KA05`, `KA07`, `KI01`, `KI03`, `KI05`, and `KI07`. The main segmentation uses an 8192-point window and 4096-point stride. To keep the experiment tractable while expanding beyond the pilot, four measurement files per operating condition are used.

## Dataset Scale

- Windows: 13,704 total.
- Normal windows: 5,874.
- Fault windows: 7,830.
- Protocols: condition-wise, bearing-wise, file-wise.
- Model in this validation pass: Isolation Forest on statistical features.

## Threshold Strategy Summary

| threshold_strategy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_mad5 | 0.7854 | 0.8359 | 0.0303 | 0.6227 | 0.0189 | 0.0242 | 0.9811 |
| train_p95 | 0.7854 | 0.8359 | 0.5216 | 0.8594 | 0.4065 | 0.1218 | 0.5935 |
| train_p99 | 0.7854 | 0.8359 | 0.0527 | 0.6600 | 0.0315 | 0.0332 | 0.9685 |
| val_p95 | 0.7854 | 0.8359 | 0.5200 | 0.8589 | 0.4050 | 0.1194 | 0.5950 |
| val_p99 | 0.7854 | 0.8359 | 0.1184 | 0.7597 | 0.0744 | 0.0377 | 0.9256 |

## Protocol-Specific Calibration Summary

| protocol | threshold_strategy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paderborn-bearing-wise | train_mad5 | 0.8318 | 0.7421 | 0.0116 | 0.2605 | 0.0060 | 0.0051 | 0.9940 |
| paderborn-bearing-wise | train_p95 | 0.8318 | 0.7421 | 0.4712 | 0.7633 | 0.3983 | 0.0502 | 0.6017 |
| paderborn-bearing-wise | train_p99 | 0.8318 | 0.7421 | 0.0372 | 0.3653 | 0.0201 | 0.0145 | 0.9799 |
| paderborn-bearing-wise | val_p95 | 0.8318 | 0.7421 | 0.4711 | 0.7633 | 0.3982 | 0.0502 | 0.6018 |
| paderborn-bearing-wise | val_p99 | 0.8318 | 0.7421 | 0.1285 | 0.5699 | 0.0834 | 0.0153 | 0.9166 |
| paderborn-condition-wise | train_mad5 | 0.7896 | 0.8177 | 0.0877 | 0.9091 | 0.0600 | 0.0453 | 0.9400 |
| paderborn-condition-wise | train_p95 | 0.7896 | 0.8177 | 0.5744 | 0.8818 | 0.4478 | 0.1241 | 0.5522 |
| paderborn-condition-wise | train_p99 | 0.7896 | 0.8177 | 0.1059 | 0.8305 | 0.0724 | 0.0507 | 0.9276 |
| paderborn-condition-wise | val_p95 | 0.7896 | 0.8177 | 0.5817 | 0.8783 | 0.4543 | 0.1254 | 0.5457 |
| paderborn-condition-wise | val_p99 | 0.7896 | 0.8177 | 0.1877 | 0.8660 | 0.1224 | 0.0529 | 0.8776 |
| paderborn-file-wise | train_mad5 | 0.7207 | 0.9730 | 0.0169 | 0.9147 | 0.0086 | 0.0355 | 0.9914 |
| paderborn-file-wise | train_p95 | 0.7207 | 0.9730 | 0.5537 | 0.9726 | 0.3899 | 0.2158 | 0.6101 |
| paderborn-file-wise | train_p99 | 0.7207 | 0.9730 | 0.0378 | 0.9394 | 0.0194 | 0.0464 | 0.9806 |
| paderborn-file-wise | val_p95 | 0.7207 | 0.9730 | 0.5441 | 0.9735 | 0.3812 | 0.2077 | 0.6188 |
| paderborn-file-wise | val_p99 | 0.7207 | 0.9730 | 0.0586 | 0.9418 | 0.0305 | 0.0574 | 0.9695 |

## Best Strategy by Protocol

| protocol | model | threshold_strategy | roc_auc | pr_auc | f1 | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paderborn-bearing-wise | iforest | train_p95 | 0.8318 | 0.7421 | 0.4712 | 0.0502 | 0.6017 |
| paderborn-condition-wise | iforest | val_p95 | 0.7896 | 0.8177 | 0.5817 | 0.1254 | 0.5457 |
| paderborn-file-wise | iforest | train_p95 | 0.7207 | 0.9730 | 0.5537 | 0.2158 | 0.6101 |

## Interpretation

- The expanded external validation is no longer a four-bearing pilot and is suitable as a main external-dataset result block.
- `train_p95` and `val_p95` provide the practical operating region; `p99` and robust MAD thresholds are too conservative and lead to high miss rates.
- Condition-wise and file-wise protocols remain challenging, so the paper should emphasize threshold transferability and operating-condition shift rather than claiming CWRU-like near-perfect performance.
- This result supports dataset-specific calibration: thresholds that look acceptable on CWRU should not be reused blindly on Paderborn.
