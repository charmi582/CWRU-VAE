# Paderborn Window-Size Sensitivity Summary

This diagnostic keeps the Paderborn bearing subset fixed and changes only the segmentation window size.
It is intended to test whether the CWRU-style 1024-point window is too short for Paderborn.

## Threshold Summary by Window Size

| window_size | threshold_strategy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | train_mad5 | 0.7228 | 0.8176 | 0.0134 | 0.4431 | 0.0073 | 0.0468 | 0.9927 |
| 1024 | train_p95 | 0.7228 | 0.8176 | 0.3495 | 0.8377 | 0.2411 | 0.1201 | 0.7589 |
| 1024 | train_p99 | 0.7228 | 0.8176 | 0.0089 | 0.2394 | 0.0053 | 0.0453 | 0.9947 |
| 1024 | val_p95 | 0.7228 | 0.8176 | 0.3632 | 0.8446 | 0.2537 | 0.1248 | 0.7463 |
| 1024 | val_p99 | 0.7228 | 0.8176 | 0.0102 | 0.3604 | 0.0056 | 0.0435 | 0.9944 |
| 2048 | train_mad5 | 0.7815 | 0.8431 | 0.0088 | 0.3955 | 0.0054 | 0.0645 | 0.9946 |
| 2048 | train_p95 | 0.7815 | 0.8431 | 0.4380 | 0.8475 | 0.3253 | 0.1389 | 0.6747 |
| 2048 | train_p99 | 0.7815 | 0.8431 | 0.0116 | 0.4243 | 0.0073 | 0.0521 | 0.9927 |
| 2048 | val_p95 | 0.7815 | 0.8431 | 0.4656 | 0.8484 | 0.3503 | 0.1445 | 0.6497 |
| 2048 | val_p99 | 0.7815 | 0.8431 | 0.0189 | 0.5436 | 0.0133 | 0.0757 | 0.9867 |
| 4096 | train_mad5 | 0.8111 | 0.8654 | 0.0081 | 0.3394 | 0.0047 | 0.0702 | 0.9953 |
| 4096 | train_p95 | 0.8111 | 0.8654 | 0.5963 | 0.8945 | 0.5029 | 0.1585 | 0.4971 |
| 4096 | train_p99 | 0.8111 | 0.8654 | 0.0153 | 0.4812 | 0.0090 | 0.0674 | 0.9910 |
| 4096 | val_p95 | 0.8111 | 0.8654 | 0.6267 | 0.8923 | 0.5387 | 0.1781 | 0.4613 |
| 4096 | val_p99 | 0.8111 | 0.8654 | 0.0333 | 0.3539 | 0.0178 | 0.0442 | 0.9822 |
| 8192 | train_mad5 | 0.8560 | 0.8914 | 0.2672 | 0.7586 | 0.1982 | 0.0852 | 0.8018 |
| 8192 | train_p95 | 0.8560 | 0.8914 | 0.6896 | 0.9077 | 0.6048 | 0.1562 | 0.3952 |
| 8192 | train_p99 | 0.8560 | 0.8914 | 0.0108 | 0.2492 | 0.0055 | 0.0565 | 0.9945 |
| 8192 | val_p95 | 0.8560 | 0.8914 | 0.6699 | 0.9000 | 0.5942 | 0.1547 | 0.4058 |
| 8192 | val_p99 | 0.8560 | 0.8914 | 0.0258 | 0.4454 | 0.0141 | 0.0537 | 0.9859 |

## Best Threshold by Window Size, Protocol, and Model

| window_size | protocol | model | threshold_strategy | roc_auc | pr_auc | f1 | false_alarm_rate | miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | paderborn-bearing-wise | iforest | train_p95 | 0.7667 | 0.8470 | 0.2791 | 0.0514 | 0.8220 |
| 1024 | paderborn-condition-wise | iforest | train_p95 | 0.7196 | 0.7373 | 0.3770 | 0.1434 | 0.7315 |
| 1024 | paderborn-file-wise | iforest | val_p95 | 0.7040 | 0.8832 | 0.4034 | 0.1401 | 0.7178 |
| 2048 | paderborn-bearing-wise | iforest | train_p95 | 0.8380 | 0.8665 | 0.3415 | 0.0584 | 0.7546 |
| 2048 | paderborn-condition-wise | iforest | val_p95 | 0.7739 | 0.7771 | 0.5558 | 0.1837 | 0.5493 |
| 2048 | paderborn-file-wise | iforest | val_p95 | 0.7608 | 0.8973 | 0.4631 | 0.1515 | 0.6737 |
| 4096 | paderborn-bearing-wise | iforest | train_p95 | 0.9055 | 0.9215 | 0.5483 | 0.0299 | 0.5167 |
| 4096 | paderborn-condition-wise | iforest | val_p95 | 0.7874 | 0.7942 | 0.6827 | 0.2189 | 0.3899 |
| 4096 | paderborn-file-wise | iforest | val_p95 | 0.7876 | 0.9085 | 0.6140 | 0.2115 | 0.5012 |
| 8192 | paderborn-bearing-wise | iforest | train_p95 | 0.9454 | 0.9491 | 0.6188 | 0.0305 | 0.4346 |
| 8192 | paderborn-condition-wise | iforest | val_p95 | 0.8048 | 0.8166 | 0.7394 | 0.2237 | 0.3072 |
| 8192 | paderborn-file-wise | iforest | train_p95 | 0.8624 | 0.9372 | 0.7161 | 0.1636 | 0.3981 |

## Interpretation Template

- If longer windows improve ROC-AUC/PR-AUC, the paper should report Paderborn as a sampling-rate-sensitive external validation case.
- If longer windows do not improve results, the paper should frame Paderborn as evidence of cross-dataset domain shift rather than a simple segmentation issue.
- In both cases, threshold calibration should remain a core contribution because FAR/Miss trade-offs vary strongly by dataset and protocol.
