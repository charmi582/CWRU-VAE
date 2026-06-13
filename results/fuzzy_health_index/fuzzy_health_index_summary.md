# Fuzzy Health Index Summary

The fuzzy layer converts calibrated anomaly scores into a graded health index.

Membership anchors are estimated from validation-normal scores: P50, P90, P95, and P99. P95 is treated as the warning center and P99 as high-confidence fault membership.

## Policy-Level Mean

| dataset | decision_policy | f1 | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CWRU | fuzzy_fault_only_hi75 | 0.8557 | 0.0331 | 0.1722 | 0.0866 | 0.1122 | 0.8884 |
| CWRU | fuzzy_warning_as_alarm_hi50 | 0.8491 | 0.0696 | 0.1005 | 0.0866 | 0.1122 | 0.8884 |
| CWRU | hard_val_p95 | 0.8491 | 0.0696 | 0.1005 | 0.0866 | 0.1122 | 0.8884 |
| Paderborn | fuzzy_fault_only_hi75 | 0.3911 | 0.0683 | 0.7212 | 0.1558 | 0.1475 | 0.4115 |
| Paderborn | fuzzy_warning_as_alarm_hi50 | 0.5200 | 0.1194 | 0.5950 | 0.1558 | 0.1475 | 0.4115 |
| Paderborn | hard_val_p95 | 0.5200 | 0.1194 | 0.5950 | 0.1558 | 0.1475 | 0.4115 |

## Protocol-Level Mean

| dataset | protocol | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CWRU | fault-size-wise | fuzzy_fault_only_hi75 | 0.9785 | 0.9316 | 0.8836 | 0.9160 | 0.8584 | 0.0332 | 0.1416 | 0.0720 | 0.1037 | 0.9053 |
| CWRU | fault-size-wise | fuzzy_warning_as_alarm_hi50 | 0.9785 | 0.9316 | 0.8776 | 0.8501 | 0.9120 | 0.0680 | 0.0880 | 0.0720 | 0.1037 | 0.9053 |
| CWRU | fault-size-wise | hard_val_p95 | 0.9785 | 0.9316 | 0.8776 | 0.8501 | 0.9120 | 0.0680 | 0.0880 | 0.0720 | 0.1037 | 0.9053 |
| CWRU | file-wise | fuzzy_fault_only_hi75 | 0.9720 | 0.8930 | 0.8434 | 0.8810 | 0.8158 | 0.0330 | 0.1842 | 0.0935 | 0.1154 | 0.8832 |
| CWRU | file-wise | fuzzy_warning_as_alarm_hi50 | 0.9720 | 0.8930 | 0.8393 | 0.7919 | 0.8984 | 0.0702 | 0.1016 | 0.0935 | 0.1154 | 0.8832 |
| CWRU | file-wise | hard_val_p95 | 0.9720 | 0.8930 | 0.8393 | 0.7919 | 0.8984 | 0.0702 | 0.1016 | 0.0935 | 0.1154 | 0.8832 |
| CWRU | load-wise | fuzzy_fault_only_hi75 | 0.9731 | 0.8943 | 0.8472 | 0.8822 | 0.8170 | 0.0330 | 0.1830 | 0.0908 | 0.1154 | 0.8808 |
| CWRU | load-wise | fuzzy_warning_as_alarm_hi50 | 0.9731 | 0.8943 | 0.8376 | 0.7918 | 0.8913 | 0.0702 | 0.1087 | 0.0908 | 0.1154 | 0.8808 |
| CWRU | load-wise | hard_val_p95 | 0.9731 | 0.8943 | 0.8376 | 0.7918 | 0.8913 | 0.0702 | 0.1087 | 0.0908 | 0.1154 | 0.8808 |
| Paderborn | paderborn-bearing-wise | fuzzy_fault_only_hi75 | 0.8194 | 0.7396 | 0.3554 | 0.7086 | 0.2776 | 0.0281 | 0.7224 | 0.1193 | 0.0910 | 0.4064 |
| Paderborn | paderborn-bearing-wise | fuzzy_warning_as_alarm_hi50 | 0.8194 | 0.7396 | 0.4711 | 0.7633 | 0.3982 | 0.0502 | 0.6018 | 0.1193 | 0.0910 | 0.4064 |
| Paderborn | paderborn-bearing-wise | hard_val_p95 | 0.8194 | 0.7396 | 0.4711 | 0.7633 | 0.3982 | 0.0502 | 0.6018 | 0.1193 | 0.0910 | 0.4064 |
| Paderborn | paderborn-condition-wise | fuzzy_fault_only_hi75 | 0.7803 | 0.8121 | 0.4802 | 0.9105 | 0.3416 | 0.0748 | 0.6584 | 0.1392 | 0.1668 | 0.4636 |
| Paderborn | paderborn-condition-wise | fuzzy_warning_as_alarm_hi50 | 0.7803 | 0.8121 | 0.5817 | 0.8783 | 0.4543 | 0.1254 | 0.5457 | 0.1392 | 0.1668 | 0.4636 |
| Paderborn | paderborn-condition-wise | hard_val_p95 | 0.7803 | 0.8121 | 0.5817 | 0.8783 | 0.4543 | 0.1254 | 0.5457 | 0.1392 | 0.1668 | 0.4636 |
| Paderborn | paderborn-file-wise | fuzzy_fault_only_hi75 | 0.7095 | 0.9717 | 0.3793 | 0.9770 | 0.2384 | 0.1175 | 0.7616 | 0.2155 | 0.2100 | 0.3835 |
| Paderborn | paderborn-file-wise | fuzzy_warning_as_alarm_hi50 | 0.7095 | 0.9717 | 0.5441 | 0.9735 | 0.3812 | 0.2077 | 0.6188 | 0.2155 | 0.2100 | 0.3835 |
| Paderborn | paderborn-file-wise | hard_val_p95 | 0.7095 | 0.9717 | 0.5441 | 0.9735 | 0.3812 | 0.2077 | 0.6188 | 0.2155 | 0.2100 | 0.3835 |

## Interpretation

- `hard_val_p95` is the original calibrated hard-threshold baseline.
- `fuzzy_warning_as_alarm_hi50` treats warning and fault states as alarms.
- `fuzzy_fault_only_hi75` alarms only on high-confidence fuzzy fault membership.
- `uncertain_rate` quantifies the gray zone instead of forcing every window into a binary label.
- The fuzzy layer should be framed as a calibrated decision layer, not as a new
  representation learner.
- On CWRU, the high-confidence fuzzy fault policy reduces false alarms compared
  with P95, but increases miss rate. This is a deployment trade-off.
- On expanded Paderborn, fuzzy health values reveal a larger gray zone and lower
  fault-health separation, so the method should be used for graded warning and
  calibration analysis rather than claiming direct accuracy improvement.
