# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, and FedAvg training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| clients_by | training_mode | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | cnn-ae | fuzzy_fault_only_hi75 | 0.5090 | 0.9816 | 0.1568 | 0.9136 | 0.0936 | 0.0383 | 0.9064 | 0.0439 | 0.0974 | 0.1470 | 0.0496 |
| bearing | centralized | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5090 | 0.9816 | 0.1745 | 0.9052 | 0.1051 | 0.0459 | 0.8949 | 0.0439 | 0.0974 | 0.1470 | 0.0496 |
| bearing | centralized | cnn-ae | hard_val_p95 | 0.5090 | 0.9816 | 0.1745 | 0.9052 | 0.1051 | 0.0459 | 0.8949 | 0.0439 | 0.0974 | 0.1470 | 0.0496 |
| bearing | centralized | vae | fuzzy_fault_only_hi75 | 0.5107 | 0.9824 | 0.1574 | 0.9663 | 0.0967 | 0.0352 | 0.9033 | 0.0525 | 0.1016 | 0.1541 | 0.0525 |
| bearing | centralized | vae | fuzzy_warning_as_alarm_hi50 | 0.5107 | 0.9824 | 0.1939 | 0.9691 | 0.1201 | 0.0522 | 0.8799 | 0.0525 | 0.1016 | 0.1541 | 0.0525 |
| bearing | centralized | vae | hard_val_p95 | 0.5107 | 0.9824 | 0.1939 | 0.9691 | 0.1201 | 0.0522 | 0.8799 | 0.0525 | 0.1016 | 0.1541 | 0.0525 |
| bearing | federated | cnn-ae | fuzzy_fault_only_hi75 | 0.4879 | 0.9810 | 0.0904 | 0.9144 | 0.0514 | 0.0432 | 0.9486 | 0.0483 | 0.1093 | 0.1140 | 0.0048 |
| bearing | federated | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4879 | 0.9810 | 0.1236 | 0.9299 | 0.0709 | 0.0658 | 0.9291 | 0.0483 | 0.1093 | 0.1140 | 0.0048 |
| bearing | federated | cnn-ae | hard_val_p95 | 0.4879 | 0.9810 | 0.1236 | 0.9299 | 0.0709 | 0.0658 | 0.9291 | 0.0483 | 0.1093 | 0.1140 | 0.0048 |
| bearing | federated | vae | fuzzy_fault_only_hi75 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | federated | vae | fuzzy_warning_as_alarm_hi50 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | federated | vae | hard_val_p95 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | local-only | cnn-ae | fuzzy_fault_only_hi75 | 0.5138 | 0.9802 | 0.0651 | 0.8987 | 0.0353 | 0.0384 | 0.9647 | 0.0542 | 0.1164 | 0.1181 | 0.0016 |
| bearing | local-only | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5138 | 0.9802 | 0.1039 | 0.9600 | 0.0580 | 0.0692 | 0.9420 | 0.0542 | 0.1164 | 0.1181 | 0.0016 |
| bearing | local-only | cnn-ae | hard_val_p95 | 0.5138 | 0.9802 | 0.1039 | 0.9600 | 0.0580 | 0.0692 | 0.9420 | 0.0542 | 0.1164 | 0.1181 | 0.0016 |
| bearing | local-only | vae | fuzzy_fault_only_hi75 | 0.4630 | 0.9776 | 0.0379 | 0.8524 | 0.0197 | 0.0444 | 0.9803 | 0.0326 | 0.1212 | 0.0792 | -0.0420 |
| bearing | local-only | vae | fuzzy_warning_as_alarm_hi50 | 0.4630 | 0.9776 | 0.0625 | 0.8472 | 0.0329 | 0.0750 | 0.9671 | 0.0326 | 0.1212 | 0.0792 | -0.0420 |
| bearing | local-only | vae | hard_val_p95 | 0.4630 | 0.9776 | 0.0625 | 0.8472 | 0.0329 | 0.0750 | 0.9671 | 0.0326 | 0.1212 | 0.0792 | -0.0420 |
| condition | centralized | cnn-ae | fuzzy_fault_only_hi75 | 0.4319 | 0.9700 | 0.1159 | 0.9167 | 0.0647 | 0.0378 | 0.9353 | 0.0471 | 0.1198 | 0.1147 | -0.0050 |
| condition | centralized | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4319 | 0.9700 | 0.1553 | 0.9011 | 0.0895 | 0.0664 | 0.9105 | 0.0471 | 0.1198 | 0.1147 | -0.0050 |
| condition | centralized | cnn-ae | hard_val_p95 | 0.4319 | 0.9700 | 0.1553 | 0.9011 | 0.0895 | 0.0664 | 0.9105 | 0.0471 | 0.1198 | 0.1147 | -0.0050 |
| condition | centralized | vae | fuzzy_fault_only_hi75 | 0.5068 | 0.9747 | 0.1436 | 0.9822 | 0.0852 | 0.0272 | 0.9148 | 0.0639 | 0.0981 | 0.1495 | 0.0515 |
| condition | centralized | vae | fuzzy_warning_as_alarm_hi50 | 0.5068 | 0.9747 | 0.1827 | 0.9793 | 0.1096 | 0.0469 | 0.8904 | 0.0639 | 0.0981 | 0.1495 | 0.0515 |
| condition | centralized | vae | hard_val_p95 | 0.5068 | 0.9747 | 0.1827 | 0.9793 | 0.1096 | 0.0469 | 0.8904 | 0.0639 | 0.0981 | 0.1495 | 0.0515 |
| condition | federated | cnn-ae | fuzzy_fault_only_hi75 | 0.4720 | 0.9706 | 0.0685 | 0.9185 | 0.0366 | 0.0375 | 0.9634 | 0.0498 | 0.1077 | 0.0969 | -0.0108 |
| condition | federated | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4720 | 0.9706 | 0.1067 | 0.9268 | 0.0581 | 0.0645 | 0.9419 | 0.0498 | 0.1077 | 0.0969 | -0.0108 |
| condition | federated | cnn-ae | hard_val_p95 | 0.4720 | 0.9706 | 0.1067 | 0.9268 | 0.0581 | 0.0645 | 0.9419 | 0.0498 | 0.1077 | 0.0969 | -0.0108 |
| condition | federated | vae | fuzzy_fault_only_hi75 | 0.4705 | 0.9683 | 0.0189 | 0.5913 | 0.0097 | 0.0261 | 0.9903 | 0.0197 | 0.0707 | 0.0446 | -0.0261 |
| condition | federated | vae | fuzzy_warning_as_alarm_hi50 | 0.4705 | 0.9683 | 0.0326 | 0.6103 | 0.0170 | 0.0392 | 0.9830 | 0.0197 | 0.0707 | 0.0446 | -0.0261 |
| condition | federated | vae | hard_val_p95 | 0.4705 | 0.9683 | 0.0326 | 0.6103 | 0.0170 | 0.0392 | 0.9830 | 0.0197 | 0.0707 | 0.0446 | -0.0261 |
| condition | local-only | cnn-ae | fuzzy_fault_only_hi75 | 0.4394 | 0.9684 | 0.0806 | 0.9612 | 0.0435 | 0.0376 | 0.9565 | 0.0337 | 0.1158 | 0.0947 | -0.0211 |
| condition | local-only | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4394 | 0.9684 | 0.1089 | 0.9558 | 0.0595 | 0.0602 | 0.9405 | 0.0337 | 0.1158 | 0.0947 | -0.0211 |
| condition | local-only | cnn-ae | hard_val_p95 | 0.4394 | 0.9684 | 0.1089 | 0.9558 | 0.0595 | 0.0602 | 0.9405 | 0.0337 | 0.1158 | 0.0947 | -0.0211 |
| condition | local-only | vae | fuzzy_fault_only_hi75 | 0.4728 | 0.9673 | 0.0222 | 0.7580 | 0.0113 | 0.0347 | 0.9887 | 0.0286 | 0.0955 | 0.0609 | -0.0347 |
| condition | local-only | vae | fuzzy_warning_as_alarm_hi50 | 0.4728 | 0.9673 | 0.0356 | 0.7612 | 0.0183 | 0.0498 | 0.9817 | 0.0286 | 0.0955 | 0.0609 | -0.0347 |
| condition | local-only | vae | hard_val_p95 | 0.4728 | 0.9673 | 0.0356 | 0.7612 | 0.0183 | 0.0498 | 0.9817 | 0.0286 | 0.0955 | 0.0609 | -0.0347 |

## Client Stability

| clients_by | training_mode | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | cnn-ae | fuzzy_fault_only_hi75 | 0.0418 | 0.1025 | 0.0410 | 0.1208 |
| bearing | centralized | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0473 | 0.1088 | 0.0410 | 0.1208 |
| bearing | centralized | cnn-ae | hard_val_p95 | 0.0473 | 0.1088 | 0.0410 | 0.1208 |
| bearing | centralized | vae | fuzzy_fault_only_hi75 | 0.0397 | 0.1188 | 0.0378 | 0.1272 |
| bearing | centralized | vae | fuzzy_warning_as_alarm_hi50 | 0.0410 | 0.1267 | 0.0378 | 0.1272 |
| bearing | centralized | vae | hard_val_p95 | 0.0410 | 0.1267 | 0.0378 | 0.1272 |
| bearing | federated | cnn-ae | fuzzy_fault_only_hi75 | 0.0481 | 0.0702 | 0.0336 | 0.0774 |
| bearing | federated | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0657 | 0.0782 | 0.0336 | 0.0774 |
| bearing | federated | cnn-ae | hard_val_p95 | 0.0657 | 0.0782 | 0.0336 | 0.0774 |
| bearing | federated | vae | fuzzy_fault_only_hi75 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | federated | vae | fuzzy_warning_as_alarm_hi50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | federated | vae | hard_val_p95 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | local-only | cnn-ae | fuzzy_fault_only_hi75 | 0.0370 | 0.0421 | 0.0398 | 0.0785 |
| bearing | local-only | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0435 | 0.0596 | 0.0398 | 0.0785 |
| bearing | local-only | cnn-ae | hard_val_p95 | 0.0435 | 0.0596 | 0.0398 | 0.0785 |
| bearing | local-only | vae | fuzzy_fault_only_hi75 | 0.0538 | 0.0197 | 0.0274 | 0.0503 |
| bearing | local-only | vae | fuzzy_warning_as_alarm_hi50 | 0.0710 | 0.0247 | 0.0274 | 0.0503 |
| bearing | local-only | vae | hard_val_p95 | 0.0710 | 0.0247 | 0.0274 | 0.0503 |
| condition | centralized | cnn-ae | fuzzy_fault_only_hi75 | 0.0336 | 0.0584 | 0.0238 | 0.0777 |
| condition | centralized | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0380 | 0.0775 | 0.0238 | 0.0777 |
| condition | centralized | cnn-ae | hard_val_p95 | 0.0380 | 0.0775 | 0.0238 | 0.0777 |
| condition | centralized | vae | fuzzy_fault_only_hi75 | 0.0229 | 0.0981 | 0.0251 | 0.1059 |
| condition | centralized | vae | fuzzy_warning_as_alarm_hi50 | 0.0263 | 0.1072 | 0.0251 | 0.1059 |
| condition | centralized | vae | hard_val_p95 | 0.0263 | 0.1072 | 0.0251 | 0.1059 |
| condition | federated | cnn-ae | fuzzy_fault_only_hi75 | 0.0331 | 0.0353 | 0.0249 | 0.0443 |
| condition | federated | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0460 | 0.0431 | 0.0249 | 0.0443 |
| condition | federated | cnn-ae | hard_val_p95 | 0.0460 | 0.0431 | 0.0249 | 0.0443 |
| condition | federated | vae | fuzzy_fault_only_hi75 | 0.0283 | 0.0136 | 0.0188 | 0.0285 |
| condition | federated | vae | fuzzy_warning_as_alarm_hi50 | 0.0393 | 0.0193 | 0.0188 | 0.0285 |
| condition | federated | vae | hard_val_p95 | 0.0393 | 0.0193 | 0.0188 | 0.0285 |
| condition | local-only | cnn-ae | fuzzy_fault_only_hi75 | 0.0238 | 0.0389 | 0.0174 | 0.0351 |
| condition | local-only | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0246 | 0.0451 | 0.0174 | 0.0351 |
| condition | local-only | cnn-ae | hard_val_p95 | 0.0246 | 0.0451 | 0.0174 | 0.0351 |
| condition | local-only | vae | fuzzy_fault_only_hi75 | 0.0291 | 0.0094 | 0.0186 | 0.0244 |
| condition | local-only | vae | fuzzy_warning_as_alarm_hi50 | 0.0324 | 0.0121 | 0.0186 | 0.0244 |
| condition | local-only | vae | hard_val_p95 | 0.0324 | 0.0121 | 0.0186 | 0.0244 |

## Federated Convergence

| clients_by | model | decision_policy | round | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4758 | 0.9802 | 0.0584 | 0.9719 | 0.0308 | 0.0368 | 0.9692 | 0.0401 | 0.1069 | 0.0926 | -0.0142 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5052 | 0.9826 | 0.1138 | 0.9881 | 0.0668 | 0.0354 | 0.9332 | 0.0614 | 0.1019 | 0.1311 | 0.0293 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4928 | 0.9810 | 0.1146 | 0.8944 | 0.0696 | 0.0385 | 0.9304 | 0.0442 | 0.1145 | 0.1312 | 0.0167 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4822 | 0.9809 | 0.1004 | 0.8719 | 0.0558 | 0.0461 | 0.9442 | 0.0494 | 0.1112 | 0.1160 | 0.0048 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4880 | 0.9812 | 0.1020 | 0.8703 | 0.0598 | 0.0461 | 0.9402 | 0.0551 | 0.1132 | 0.1198 | 0.0066 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4922 | 0.9815 | 0.0926 | 0.9031 | 0.0502 | 0.0400 | 0.9498 | 0.0548 | 0.1041 | 0.1121 | 0.0080 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.5037 | 0.9819 | 0.0924 | 0.9010 | 0.0526 | 0.0507 | 0.9474 | 0.0463 | 0.1075 | 0.1141 | 0.0066 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4855 | 0.9802 | 0.0728 | 0.8857 | 0.0412 | 0.0477 | 0.9588 | 0.0545 | 0.1158 | 0.1086 | -0.0071 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4812 | 0.9805 | 0.0791 | 0.8939 | 0.0450 | 0.0523 | 0.9550 | 0.0392 | 0.1102 | 0.1082 | -0.0019 |
| bearing | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4719 | 0.9804 | 0.0776 | 0.9634 | 0.0423 | 0.0385 | 0.9577 | 0.0375 | 0.1075 | 0.1064 | -0.0011 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4758 | 0.9802 | 0.0845 | 0.9760 | 0.0450 | 0.0521 | 0.9550 | 0.0401 | 0.1069 | 0.0926 | -0.0142 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5052 | 0.9826 | 0.1509 | 0.9868 | 0.0891 | 0.0491 | 0.9109 | 0.0614 | 0.1019 | 0.1311 | 0.0293 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4928 | 0.9810 | 0.1422 | 0.9273 | 0.0859 | 0.0677 | 0.9141 | 0.0442 | 0.1145 | 0.1312 | 0.0167 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4822 | 0.9809 | 0.1366 | 0.8939 | 0.0771 | 0.0691 | 0.9229 | 0.0494 | 0.1112 | 0.1160 | 0.0048 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4880 | 0.9812 | 0.1339 | 0.8869 | 0.0792 | 0.0721 | 0.9208 | 0.0551 | 0.1132 | 0.1198 | 0.0066 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4922 | 0.9815 | 0.1408 | 0.9146 | 0.0790 | 0.0600 | 0.9210 | 0.0548 | 0.1041 | 0.1121 | 0.0080 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.5037 | 0.9819 | 0.1299 | 0.9164 | 0.0747 | 0.0768 | 0.9253 | 0.0463 | 0.1075 | 0.1141 | 0.0066 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4855 | 0.9802 | 0.1048 | 0.9084 | 0.0599 | 0.0800 | 0.9401 | 0.0545 | 0.1158 | 0.1086 | -0.0071 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4812 | 0.9805 | 0.1067 | 0.9153 | 0.0610 | 0.0677 | 0.9390 | 0.0392 | 0.1102 | 0.1082 | -0.0019 |
| bearing | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4719 | 0.9804 | 0.1052 | 0.9731 | 0.0586 | 0.0631 | 0.9414 | 0.0375 | 0.1075 | 0.1064 | -0.0011 |
| bearing | cnn-ae | hard_val_p95 | 1 | 0.4758 | 0.9802 | 0.0845 | 0.9760 | 0.0450 | 0.0521 | 0.9550 | 0.0401 | 0.1069 | 0.0926 | -0.0142 |
| bearing | cnn-ae | hard_val_p95 | 2 | 0.5052 | 0.9826 | 0.1509 | 0.9868 | 0.0891 | 0.0491 | 0.9109 | 0.0614 | 0.1019 | 0.1311 | 0.0293 |
| bearing | cnn-ae | hard_val_p95 | 3 | 0.4928 | 0.9810 | 0.1422 | 0.9273 | 0.0859 | 0.0677 | 0.9141 | 0.0442 | 0.1145 | 0.1312 | 0.0167 |
| bearing | cnn-ae | hard_val_p95 | 4 | 0.4822 | 0.9809 | 0.1366 | 0.8939 | 0.0771 | 0.0691 | 0.9229 | 0.0494 | 0.1112 | 0.1160 | 0.0048 |
| bearing | cnn-ae | hard_val_p95 | 5 | 0.4880 | 0.9812 | 0.1339 | 0.8869 | 0.0792 | 0.0721 | 0.9208 | 0.0551 | 0.1132 | 0.1198 | 0.0066 |
| bearing | cnn-ae | hard_val_p95 | 6 | 0.4922 | 0.9815 | 0.1408 | 0.9146 | 0.0790 | 0.0600 | 0.9210 | 0.0548 | 0.1041 | 0.1121 | 0.0080 |
| bearing | cnn-ae | hard_val_p95 | 7 | 0.5037 | 0.9819 | 0.1299 | 0.9164 | 0.0747 | 0.0768 | 0.9253 | 0.0463 | 0.1075 | 0.1141 | 0.0066 |
| bearing | cnn-ae | hard_val_p95 | 8 | 0.4855 | 0.9802 | 0.1048 | 0.9084 | 0.0599 | 0.0800 | 0.9401 | 0.0545 | 0.1158 | 0.1086 | -0.0071 |
| bearing | cnn-ae | hard_val_p95 | 9 | 0.4812 | 0.9805 | 0.1067 | 0.9153 | 0.0610 | 0.0677 | 0.9390 | 0.0392 | 0.1102 | 0.1082 | -0.0019 |
| bearing | cnn-ae | hard_val_p95 | 10 | 0.4719 | 0.9804 | 0.1052 | 0.9731 | 0.0586 | 0.0631 | 0.9414 | 0.0375 | 0.1075 | 0.1064 | -0.0011 |
| bearing | vae | fuzzy_fault_only_hi75 | 1 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 2 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 3 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 4 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 5 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 6 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 7 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 8 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 9 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_fault_only_hi75 | 10 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 3 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 4 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 7 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 1 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 2 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 3 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 4 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 5 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 6 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 7 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 8 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 9 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bearing | vae | hard_val_p95 | 10 | 0.5000 | 0.9818 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4945 | 0.9712 | 0.0393 | 0.9192 | 0.0202 | 0.0271 | 0.9798 | 0.0515 | 0.0975 | 0.0912 | -0.0063 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4926 | 0.9745 | 0.1381 | 0.9260 | 0.0788 | 0.0302 | 0.9212 | 0.0652 | 0.1092 | 0.1405 | 0.0314 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.5107 | 0.9723 | 0.0734 | 0.8950 | 0.0387 | 0.0423 | 0.9613 | 0.0572 | 0.1023 | 0.1044 | 0.0021 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4823 | 0.9714 | 0.0746 | 0.8895 | 0.0393 | 0.0392 | 0.9607 | 0.0480 | 0.0947 | 0.0932 | -0.0015 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4945 | 0.9712 | 0.0652 | 0.9163 | 0.0343 | 0.0468 | 0.9657 | 0.0469 | 0.1086 | 0.0979 | -0.0106 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4799 | 0.9694 | 0.0409 | 0.9067 | 0.0213 | 0.0497 | 0.9787 | 0.0450 | 0.1100 | 0.0842 | -0.0258 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4457 | 0.9681 | 0.0358 | 0.9377 | 0.0185 | 0.0212 | 0.9815 | 0.0420 | 0.1017 | 0.0738 | -0.0279 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4430 | 0.9691 | 0.0671 | 0.9204 | 0.0352 | 0.0438 | 0.9648 | 0.0503 | 0.1181 | 0.0921 | -0.0260 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4504 | 0.9696 | 0.0647 | 0.9474 | 0.0339 | 0.0376 | 0.9661 | 0.0487 | 0.1126 | 0.0920 | -0.0207 |
| condition | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4265 | 0.9692 | 0.0861 | 0.9272 | 0.0461 | 0.0376 | 0.9539 | 0.0429 | 0.1223 | 0.0997 | -0.0226 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4945 | 0.9712 | 0.0783 | 0.9381 | 0.0414 | 0.0436 | 0.9586 | 0.0515 | 0.0975 | 0.0912 | -0.0063 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4926 | 0.9745 | 0.1797 | 0.9386 | 0.1041 | 0.0575 | 0.8959 | 0.0652 | 0.1092 | 0.1405 | 0.0314 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.5107 | 0.9723 | 0.1150 | 0.8944 | 0.0628 | 0.0663 | 0.9372 | 0.0572 | 0.1023 | 0.1044 | 0.0021 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4823 | 0.9714 | 0.1097 | 0.8928 | 0.0592 | 0.0588 | 0.9408 | 0.0480 | 0.0947 | 0.0932 | -0.0015 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4945 | 0.9712 | 0.1030 | 0.9340 | 0.0551 | 0.0692 | 0.9449 | 0.0469 | 0.1086 | 0.0979 | -0.0106 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4799 | 0.9694 | 0.0772 | 0.9348 | 0.0409 | 0.0678 | 0.9591 | 0.0450 | 0.1100 | 0.0842 | -0.0258 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4457 | 0.9681 | 0.0666 | 0.9232 | 0.0351 | 0.0513 | 0.9649 | 0.0420 | 0.1017 | 0.0738 | -0.0279 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4430 | 0.9691 | 0.1127 | 0.9338 | 0.0610 | 0.0859 | 0.9390 | 0.0503 | 0.1181 | 0.0921 | -0.0260 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4504 | 0.9696 | 0.1070 | 0.9413 | 0.0574 | 0.0707 | 0.9426 | 0.0487 | 0.1126 | 0.0920 | -0.0207 |
| condition | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4265 | 0.9692 | 0.1178 | 0.9365 | 0.0644 | 0.0739 | 0.9356 | 0.0429 | 0.1223 | 0.0997 | -0.0226 |
| condition | cnn-ae | hard_val_p95 | 1 | 0.4945 | 0.9712 | 0.0783 | 0.9381 | 0.0414 | 0.0436 | 0.9586 | 0.0515 | 0.0975 | 0.0912 | -0.0063 |
| condition | cnn-ae | hard_val_p95 | 2 | 0.4926 | 0.9745 | 0.1797 | 0.9386 | 0.1041 | 0.0575 | 0.8959 | 0.0652 | 0.1092 | 0.1405 | 0.0314 |
| condition | cnn-ae | hard_val_p95 | 3 | 0.5107 | 0.9723 | 0.1150 | 0.8944 | 0.0628 | 0.0663 | 0.9372 | 0.0572 | 0.1023 | 0.1044 | 0.0021 |
| condition | cnn-ae | hard_val_p95 | 4 | 0.4823 | 0.9714 | 0.1097 | 0.8928 | 0.0592 | 0.0588 | 0.9408 | 0.0480 | 0.0947 | 0.0932 | -0.0015 |
| condition | cnn-ae | hard_val_p95 | 5 | 0.4945 | 0.9712 | 0.1030 | 0.9340 | 0.0551 | 0.0692 | 0.9449 | 0.0469 | 0.1086 | 0.0979 | -0.0106 |
| condition | cnn-ae | hard_val_p95 | 6 | 0.4799 | 0.9694 | 0.0772 | 0.9348 | 0.0409 | 0.0678 | 0.9591 | 0.0450 | 0.1100 | 0.0842 | -0.0258 |
| condition | cnn-ae | hard_val_p95 | 7 | 0.4457 | 0.9681 | 0.0666 | 0.9232 | 0.0351 | 0.0513 | 0.9649 | 0.0420 | 0.1017 | 0.0738 | -0.0279 |
| condition | cnn-ae | hard_val_p95 | 8 | 0.4430 | 0.9691 | 0.1127 | 0.9338 | 0.0610 | 0.0859 | 0.9390 | 0.0503 | 0.1181 | 0.0921 | -0.0260 |
| condition | cnn-ae | hard_val_p95 | 9 | 0.4504 | 0.9696 | 0.1070 | 0.9413 | 0.0574 | 0.0707 | 0.9426 | 0.0487 | 0.1126 | 0.0920 | -0.0207 |
| condition | cnn-ae | hard_val_p95 | 10 | 0.4265 | 0.9692 | 0.1178 | 0.9365 | 0.0644 | 0.0739 | 0.9356 | 0.0429 | 0.1223 | 0.0997 | -0.0226 |
| condition | vae | fuzzy_fault_only_hi75 | 1 | 0.4624 | 0.9671 | 0.0154 | 0.5821 | 0.0079 | 0.0302 | 0.9921 | 0.0167 | 0.0743 | 0.0445 | -0.0297 |
| condition | vae | fuzzy_fault_only_hi75 | 2 | 0.4696 | 0.9670 | 0.0167 | 0.5901 | 0.0086 | 0.0256 | 0.9914 | 0.0205 | 0.0788 | 0.0464 | -0.0324 |
| condition | vae | fuzzy_fault_only_hi75 | 3 | 0.4652 | 0.9674 | 0.0149 | 0.5643 | 0.0076 | 0.0227 | 0.9924 | 0.0215 | 0.0720 | 0.0419 | -0.0302 |
| condition | vae | fuzzy_fault_only_hi75 | 4 | 0.4743 | 0.9682 | 0.0157 | 0.6099 | 0.0081 | 0.0211 | 0.9919 | 0.0166 | 0.0661 | 0.0421 | -0.0240 |
| condition | vae | fuzzy_fault_only_hi75 | 5 | 0.4701 | 0.9682 | 0.0159 | 0.6051 | 0.0082 | 0.0196 | 0.9918 | 0.0198 | 0.0668 | 0.0430 | -0.0237 |
| condition | vae | fuzzy_fault_only_hi75 | 6 | 0.4747 | 0.9695 | 0.0166 | 0.6010 | 0.0086 | 0.0226 | 0.9914 | 0.0217 | 0.0622 | 0.0420 | -0.0201 |
| condition | vae | fuzzy_fault_only_hi75 | 7 | 0.4741 | 0.9691 | 0.0237 | 0.5812 | 0.0122 | 0.0332 | 0.9878 | 0.0209 | 0.0729 | 0.0477 | -0.0252 |
| condition | vae | fuzzy_fault_only_hi75 | 8 | 0.4728 | 0.9688 | 0.0231 | 0.5931 | 0.0119 | 0.0302 | 0.9881 | 0.0218 | 0.0738 | 0.0478 | -0.0260 |
| condition | vae | fuzzy_fault_only_hi75 | 9 | 0.4684 | 0.9690 | 0.0211 | 0.5867 | 0.0109 | 0.0242 | 0.9891 | 0.0182 | 0.0667 | 0.0428 | -0.0238 |
| condition | vae | fuzzy_fault_only_hi75 | 10 | 0.4739 | 0.9688 | 0.0261 | 0.5995 | 0.0135 | 0.0318 | 0.9865 | 0.0193 | 0.0738 | 0.0476 | -0.0262 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4624 | 0.9671 | 0.0328 | 0.6105 | 0.0170 | 0.0392 | 0.9830 | 0.0167 | 0.0743 | 0.0445 | -0.0297 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4696 | 0.9670 | 0.0311 | 0.5927 | 0.0161 | 0.0513 | 0.9839 | 0.0205 | 0.0788 | 0.0464 | -0.0324 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4652 | 0.9674 | 0.0294 | 0.6088 | 0.0153 | 0.0392 | 0.9847 | 0.0215 | 0.0720 | 0.0419 | -0.0302 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4743 | 0.9682 | 0.0257 | 0.6105 | 0.0133 | 0.0317 | 0.9867 | 0.0166 | 0.0661 | 0.0421 | -0.0240 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4701 | 0.9682 | 0.0287 | 0.6115 | 0.0149 | 0.0301 | 0.9851 | 0.0198 | 0.0668 | 0.0430 | -0.0237 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4747 | 0.9695 | 0.0298 | 0.6176 | 0.0155 | 0.0317 | 0.9845 | 0.0217 | 0.0622 | 0.0420 | -0.0201 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4741 | 0.9691 | 0.0400 | 0.6126 | 0.0210 | 0.0437 | 0.9790 | 0.0209 | 0.0729 | 0.0477 | -0.0252 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4728 | 0.9688 | 0.0368 | 0.6150 | 0.0192 | 0.0422 | 0.9808 | 0.0218 | 0.0738 | 0.0478 | -0.0260 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4684 | 0.9690 | 0.0313 | 0.6067 | 0.0162 | 0.0362 | 0.9838 | 0.0182 | 0.0667 | 0.0428 | -0.0238 |
| condition | vae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4739 | 0.9688 | 0.0406 | 0.6167 | 0.0213 | 0.0468 | 0.9787 | 0.0193 | 0.0738 | 0.0476 | -0.0262 |
| condition | vae | hard_val_p95 | 1 | 0.4624 | 0.9671 | 0.0328 | 0.6105 | 0.0170 | 0.0392 | 0.9830 | 0.0167 | 0.0743 | 0.0445 | -0.0297 |
| condition | vae | hard_val_p95 | 2 | 0.4696 | 0.9670 | 0.0311 | 0.5927 | 0.0161 | 0.0513 | 0.9839 | 0.0205 | 0.0788 | 0.0464 | -0.0324 |
| condition | vae | hard_val_p95 | 3 | 0.4652 | 0.9674 | 0.0294 | 0.6088 | 0.0153 | 0.0392 | 0.9847 | 0.0215 | 0.0720 | 0.0419 | -0.0302 |
| condition | vae | hard_val_p95 | 4 | 0.4743 | 0.9682 | 0.0257 | 0.6105 | 0.0133 | 0.0317 | 0.9867 | 0.0166 | 0.0661 | 0.0421 | -0.0240 |
| condition | vae | hard_val_p95 | 5 | 0.4701 | 0.9682 | 0.0287 | 0.6115 | 0.0149 | 0.0301 | 0.9851 | 0.0198 | 0.0668 | 0.0430 | -0.0237 |
| condition | vae | hard_val_p95 | 6 | 0.4747 | 0.9695 | 0.0298 | 0.6176 | 0.0155 | 0.0317 | 0.9845 | 0.0217 | 0.0622 | 0.0420 | -0.0201 |
| condition | vae | hard_val_p95 | 7 | 0.4741 | 0.9691 | 0.0400 | 0.6126 | 0.0210 | 0.0437 | 0.9790 | 0.0209 | 0.0729 | 0.0477 | -0.0252 |
| condition | vae | hard_val_p95 | 8 | 0.4728 | 0.9688 | 0.0368 | 0.6150 | 0.0192 | 0.0422 | 0.9808 | 0.0218 | 0.0738 | 0.0478 | -0.0260 |
| condition | vae | hard_val_p95 | 9 | 0.4684 | 0.9690 | 0.0313 | 0.6067 | 0.0162 | 0.0362 | 0.9838 | 0.0182 | 0.0667 | 0.0428 | -0.0238 |
| condition | vae | hard_val_p95 | 10 | 0.4739 | 0.9688 | 0.0406 | 0.6167 | 0.0213 | 0.0468 | 0.9787 | 0.0193 | 0.0738 | 0.0476 | -0.0262 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `federated` approximates collaborative normal-only training without sharing raw vibration windows.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
