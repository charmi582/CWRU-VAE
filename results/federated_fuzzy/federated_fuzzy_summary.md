# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, and FedAvg training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| training_mode | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centralized | cnn-ae | fuzzy_fault_only_hi75 | 0.4726 | 0.9761 | 0.1970 | 0.6858 | 0.1313 | 0.0598 | 0.8687 | 0.0185 | 0.1294 | 0.1675 | 0.0381 |
| centralized | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4726 | 0.9761 | 0.2011 | 0.7063 | 0.1344 | 0.0689 | 0.8656 | 0.0185 | 0.1294 | 0.1675 | 0.0381 |
| centralized | cnn-ae | hard_val_p95 | 0.4726 | 0.9761 | 0.2011 | 0.7063 | 0.1344 | 0.0689 | 0.8656 | 0.0185 | 0.1294 | 0.1675 | 0.0381 |
| centralized | vae | fuzzy_fault_only_hi75 | 0.4462 | 0.9760 | 0.0307 | 0.9508 | 0.0158 | 0.0462 | 0.9842 | 0.0308 | 0.1235 | 0.0693 | -0.0542 |
| centralized | vae | fuzzy_warning_as_alarm_hi50 | 0.4462 | 0.9760 | 0.0603 | 0.9366 | 0.0320 | 0.0692 | 0.9680 | 0.0308 | 0.1235 | 0.0693 | -0.0542 |
| centralized | vae | hard_val_p95 | 0.4462 | 0.9760 | 0.0603 | 0.9366 | 0.0320 | 0.0692 | 0.9680 | 0.0308 | 0.1235 | 0.0693 | -0.0542 |
| federated | cnn-ae | fuzzy_fault_only_hi75 | 0.4810 | 0.9802 | 0.1063 | 0.9468 | 0.0607 | 0.0414 | 0.9393 | 0.0444 | 0.1143 | 0.1168 | 0.0025 |
| federated | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4810 | 0.9802 | 0.1426 | 0.9424 | 0.0850 | 0.0733 | 0.9150 | 0.0444 | 0.1143 | 0.1168 | 0.0025 |
| federated | cnn-ae | hard_val_p95 | 0.4810 | 0.9802 | 0.1426 | 0.9424 | 0.0850 | 0.0733 | 0.9150 | 0.0444 | 0.1143 | 0.1168 | 0.0025 |
| federated | vae | fuzzy_fault_only_hi75 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| federated | vae | fuzzy_warning_as_alarm_hi50 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| federated | vae | hard_val_p95 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local-only | cnn-ae | fuzzy_fault_only_hi75 | 0.5198 | 0.9834 | 0.1114 | 0.9933 | 0.0620 | 0.0185 | 0.9380 | 0.0448 | 0.0911 | 0.1358 | 0.0447 |
| local-only | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5198 | 0.9834 | 0.1517 | 0.9865 | 0.0867 | 0.0277 | 0.9133 | 0.0448 | 0.0911 | 0.1358 | 0.0447 |
| local-only | cnn-ae | hard_val_p95 | 0.5198 | 0.9834 | 0.1517 | 0.9865 | 0.0867 | 0.0277 | 0.9133 | 0.0448 | 0.0911 | 0.1358 | 0.0447 |
| local-only | vae | fuzzy_fault_only_hi75 | 0.4655 | 0.9793 | 0.0301 | 0.7578 | 0.0155 | 0.0231 | 0.9845 | 0.0269 | 0.0770 | 0.0577 | -0.0194 |
| local-only | vae | fuzzy_warning_as_alarm_hi50 | 0.4655 | 0.9793 | 0.0497 | 0.7854 | 0.0261 | 0.0370 | 0.9739 | 0.0269 | 0.0770 | 0.0577 | -0.0194 |
| local-only | vae | hard_val_p95 | 0.4655 | 0.9793 | 0.0497 | 0.7854 | 0.0261 | 0.0370 | 0.9739 | 0.0269 | 0.0770 | 0.0577 | -0.0194 |

## Client Stability

| training_mode | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- |
| centralized | cnn-ae | fuzzy_fault_only_hi75 | 0.0371 | 0.1650 | 0.0160 | 0.1841 |
| centralized | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0412 | 0.1671 | 0.0160 | 0.1841 |
| centralized | cnn-ae | hard_val_p95 | 0.0412 | 0.1671 | 0.0160 | 0.1841 |
| centralized | vae | fuzzy_fault_only_hi75 | 0.0674 | 0.0162 | 0.0311 | 0.0379 |
| centralized | vae | fuzzy_warning_as_alarm_hi50 | 0.0595 | 0.0286 | 0.0311 | 0.0379 |
| centralized | vae | hard_val_p95 | 0.0595 | 0.0286 | 0.0311 | 0.0379 |
| federated | cnn-ae | fuzzy_fault_only_hi75 | 0.0136 | 0.0711 | 0.0444 | 0.1133 |
| federated | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0457 | 0.0973 | 0.0444 | 0.1133 |
| federated | cnn-ae | hard_val_p95 | 0.0457 | 0.0973 | 0.0444 | 0.1133 |
| federated | vae | fuzzy_fault_only_hi75 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| federated | vae | fuzzy_warning_as_alarm_hi50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| federated | vae | hard_val_p95 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| local-only | cnn-ae | fuzzy_fault_only_hi75 | 0.0207 | 0.0575 | 0.0207 | 0.0743 |
| local-only | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0227 | 0.0721 | 0.0207 | 0.0743 |
| local-only | cnn-ae | hard_val_p95 | 0.0227 | 0.0721 | 0.0207 | 0.0743 |
| local-only | vae | fuzzy_fault_only_hi75 | 0.0191 | 0.0151 | 0.0208 | 0.0369 |
| local-only | vae | fuzzy_warning_as_alarm_hi50 | 0.0346 | 0.0243 | 0.0208 | 0.0369 |
| local-only | vae | hard_val_p95 | 0.0346 | 0.0243 | 0.0208 | 0.0369 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `federated` approximates collaborative normal-only training without sharing raw vibration windows.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
- In this initial formal pass, FedAvg should not be claimed as an automatic
  accuracy improvement. The CNN-AE local-only setting gives the strongest mean
  F1, while the federated setting gives a privacy-preserving collaborative
  reference with different stability trade-offs.
- The VAE federated setting is unstable in this low-epoch Paderborn pass and
  should be reported as evidence that model choice matters under federated
  non-IID client distributions.
- The deployment claim should focus on the comparison framework: privacy
  preservation, client-specific fuzzy calibration, and client-to-client
  stability analysis.
