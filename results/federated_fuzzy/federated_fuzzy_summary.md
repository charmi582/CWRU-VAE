# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, and FedAvg training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| clients_by | training_mode | calibration_scope | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5138 | 0.9816 | 0.1728 | 0.8264 | 0.1065 | 0.0398 | 0.8935 | 0.0440 | 0.1030 | 0.1596 | 0.0565 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5138 | 0.9816 | 0.1998 | 0.8368 | 0.1253 | 0.0628 | 0.8747 | 0.0440 | 0.1030 | 0.1596 | 0.0565 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.5138 | 0.9816 | 0.1998 | 0.8368 | 0.1253 | 0.0628 | 0.8747 | 0.0440 | 0.1030 | 0.1596 | 0.0565 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4720 | 0.9803 | 0.0441 | 0.6583 | 0.0228 | 0.0230 | 0.9772 | 0.0521 | 0.0816 | 0.0756 | -0.0059 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4720 | 0.9803 | 0.0794 | 0.8593 | 0.0424 | 0.0460 | 0.9576 | 0.0521 | 0.0816 | 0.0756 | -0.0059 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.4720 | 0.9803 | 0.0794 | 0.8593 | 0.0424 | 0.0460 | 0.9576 | 0.0521 | 0.0816 | 0.0756 | -0.0059 |
| bearing | federated | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4880 | 0.9812 | 0.0941 | 0.9130 | 0.0536 | 0.0430 | 0.9464 | 0.0479 | 0.1099 | 0.1167 | 0.0068 |
| bearing | federated | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4880 | 0.9812 | 0.1268 | 0.9276 | 0.0729 | 0.0647 | 0.9271 | 0.0479 | 0.1099 | 0.1167 | 0.0068 |
| bearing | federated | client_specific | cnn-ae | hard_val_p95 | 0.4880 | 0.9812 | 0.1268 | 0.9276 | 0.0729 | 0.0647 | 0.9271 | 0.0479 | 0.1099 | 0.1167 | 0.0068 |
| bearing | federated | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4648 | 0.9802 | 0.0310 | 0.9513 | 0.0159 | 0.0208 | 0.9841 | 0.0430 | 0.0855 | 0.0722 | -0.0133 |
| bearing | federated | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4648 | 0.9802 | 0.0659 | 0.9726 | 0.0343 | 0.0399 | 0.9657 | 0.0430 | 0.0855 | 0.0722 | -0.0133 |
| bearing | federated | pooled | cnn-ae | hard_val_p95 | 0.4648 | 0.9802 | 0.0659 | 0.9726 | 0.0343 | 0.0399 | 0.9657 | 0.0430 | 0.0855 | 0.0722 | -0.0133 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5084 | 0.9804 | 0.0631 | 0.9548 | 0.0342 | 0.0476 | 0.9658 | 0.0495 | 0.1206 | 0.1135 | -0.0071 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5084 | 0.9804 | 0.1026 | 0.9527 | 0.0580 | 0.0814 | 0.9420 | 0.0495 | 0.1206 | 0.1135 | -0.0071 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.5084 | 0.9804 | 0.1026 | 0.9527 | 0.0580 | 0.0814 | 0.9420 | 0.0495 | 0.1206 | 0.1135 | -0.0071 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4445 | 0.9713 | 0.1201 | 0.8957 | 0.0669 | 0.0453 | 0.9331 | 0.0489 | 0.1181 | 0.1172 | -0.0009 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4445 | 0.9713 | 0.1520 | 0.9077 | 0.0865 | 0.0695 | 0.9135 | 0.0489 | 0.1181 | 0.1172 | -0.0009 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.4445 | 0.9713 | 0.1520 | 0.9077 | 0.0865 | 0.0695 | 0.9135 | 0.0489 | 0.1181 | 0.1172 | -0.0009 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4387 | 0.9695 | 0.0396 | 0.9112 | 0.0204 | 0.0287 | 0.9796 | 0.0385 | 0.0925 | 0.0653 | -0.0272 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4387 | 0.9695 | 0.0641 | 0.9244 | 0.0336 | 0.0544 | 0.9664 | 0.0385 | 0.0925 | 0.0653 | -0.0272 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.4387 | 0.9695 | 0.0641 | 0.9244 | 0.0336 | 0.0544 | 0.9664 | 0.0385 | 0.0925 | 0.0653 | -0.0272 |
| condition | federated | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4743 | 0.9710 | 0.0711 | 0.9240 | 0.0381 | 0.0371 | 0.9619 | 0.0509 | 0.1084 | 0.1001 | -0.0083 |
| condition | federated | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4743 | 0.9710 | 0.1089 | 0.9331 | 0.0594 | 0.0608 | 0.9406 | 0.0509 | 0.1084 | 0.1001 | -0.0083 |
| condition | federated | client_specific | cnn-ae | hard_val_p95 | 0.4743 | 0.9710 | 0.1089 | 0.9331 | 0.0594 | 0.0608 | 0.9406 | 0.0509 | 0.1084 | 0.1001 | -0.0083 |
| condition | federated | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4686 | 0.9703 | 0.0383 | 0.9392 | 0.0198 | 0.0246 | 0.9802 | 0.0444 | 0.0888 | 0.0748 | -0.0140 |
| condition | federated | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4686 | 0.9703 | 0.0681 | 0.9520 | 0.0357 | 0.0447 | 0.9643 | 0.0444 | 0.0888 | 0.0748 | -0.0140 |
| condition | federated | pooled | cnn-ae | hard_val_p95 | 0.4686 | 0.9703 | 0.0681 | 0.9520 | 0.0357 | 0.0447 | 0.9643 | 0.0444 | 0.0888 | 0.0748 | -0.0140 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4358 | 0.9685 | 0.0770 | 0.9511 | 0.0415 | 0.0421 | 0.9585 | 0.0358 | 0.1143 | 0.0928 | -0.0215 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4358 | 0.9685 | 0.1071 | 0.9500 | 0.0589 | 0.0587 | 0.9411 | 0.0358 | 0.1143 | 0.0928 | -0.0215 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.4358 | 0.9685 | 0.1071 | 0.9500 | 0.0589 | 0.0587 | 0.9411 | 0.0358 | 0.1143 | 0.0928 | -0.0215 |

## Client Stability

| clients_by | training_mode | calibration_scope | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0450 | 0.1198 | 0.0383 | 0.1379 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0597 | 0.1316 | 0.0383 | 0.1379 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0597 | 0.1316 | 0.0383 | 0.1379 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0381 | 0.0173 | 0.0456 | 0.0717 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0560 | 0.0334 | 0.0456 | 0.0717 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.0560 | 0.0334 | 0.0456 | 0.0717 |
| bearing | federated | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0441 | 0.0712 | 0.0335 | 0.0777 |
| bearing | federated | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0594 | 0.0795 | 0.0335 | 0.0777 |
| bearing | federated | client_specific | cnn-ae | hard_val_p95 | 0.0594 | 0.0795 | 0.0335 | 0.0777 |
| bearing | federated | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0264 | 0.0115 | 0.0151 | 0.0485 |
| bearing | federated | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0399 | 0.0154 | 0.0151 | 0.0485 |
| bearing | federated | pooled | cnn-ae | hard_val_p95 | 0.0399 | 0.0154 | 0.0151 | 0.0485 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0451 | 0.0413 | 0.0445 | 0.0727 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0543 | 0.0666 | 0.0445 | 0.0727 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0543 | 0.0666 | 0.0445 | 0.0727 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0217 | 0.0574 | 0.0224 | 0.0711 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0385 | 0.0682 | 0.0224 | 0.0711 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0385 | 0.0682 | 0.0224 | 0.0711 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0483 | 0.0150 | 0.0157 | 0.0933 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0856 | 0.0213 | 0.0157 | 0.0933 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.0856 | 0.0213 | 0.0157 | 0.0933 |
| condition | federated | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0340 | 0.0365 | 0.0276 | 0.0411 |
| condition | federated | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0469 | 0.0442 | 0.0276 | 0.0411 |
| condition | federated | client_specific | cnn-ae | hard_val_p95 | 0.0469 | 0.0442 | 0.0276 | 0.0411 |
| condition | federated | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0309 | 0.0146 | 0.0195 | 0.0474 |
| condition | federated | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0403 | 0.0214 | 0.0195 | 0.0474 |
| condition | federated | pooled | cnn-ae | hard_val_p95 | 0.0403 | 0.0214 | 0.0195 | 0.0474 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0278 | 0.0392 | 0.0193 | 0.0329 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0376 | 0.0495 | 0.0193 | 0.0329 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0376 | 0.0495 | 0.0193 | 0.0329 |

## Federated Convergence

| clients_by | calibration_scope | model | decision_policy | round | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4751 | 0.9802 | 0.0581 | 0.9715 | 0.0306 | 0.0384 | 0.9694 | 0.0403 | 0.1070 | 0.0927 | -0.0143 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5002 | 0.9826 | 0.1156 | 0.9869 | 0.0678 | 0.0369 | 0.9322 | 0.0581 | 0.1037 | 0.1307 | 0.0270 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4908 | 0.9810 | 0.1128 | 0.8755 | 0.0685 | 0.0369 | 0.9315 | 0.0362 | 0.1094 | 0.1266 | 0.0172 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4807 | 0.9808 | 0.1079 | 0.8861 | 0.0607 | 0.0537 | 0.9393 | 0.0481 | 0.1193 | 0.1211 | 0.0019 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4866 | 0.9810 | 0.1079 | 0.8697 | 0.0635 | 0.0415 | 0.9365 | 0.0547 | 0.1120 | 0.1242 | 0.0122 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4936 | 0.9820 | 0.0911 | 0.9029 | 0.0498 | 0.0354 | 0.9502 | 0.0503 | 0.0978 | 0.1104 | 0.0126 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.5013 | 0.9819 | 0.0913 | 0.8926 | 0.0516 | 0.0461 | 0.9484 | 0.0496 | 0.1073 | 0.1182 | 0.0109 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4852 | 0.9806 | 0.0799 | 0.8971 | 0.0441 | 0.0430 | 0.9559 | 0.0510 | 0.1114 | 0.1120 | 0.0005 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4899 | 0.9814 | 0.0909 | 0.8904 | 0.0525 | 0.0507 | 0.9475 | 0.0495 | 0.1117 | 0.1175 | 0.0058 |
| bearing | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4763 | 0.9803 | 0.0852 | 0.9571 | 0.0466 | 0.0475 | 0.9534 | 0.0411 | 0.1189 | 0.1132 | -0.0058 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4751 | 0.9802 | 0.0843 | 0.9760 | 0.0449 | 0.0506 | 0.9551 | 0.0403 | 0.1070 | 0.0927 | -0.0143 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5002 | 0.9826 | 0.1488 | 0.9858 | 0.0880 | 0.0523 | 0.9120 | 0.0581 | 0.1037 | 0.1307 | 0.0270 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4908 | 0.9810 | 0.1369 | 0.8947 | 0.0832 | 0.0661 | 0.9168 | 0.0362 | 0.1094 | 0.1266 | 0.0172 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4807 | 0.9808 | 0.1461 | 0.8980 | 0.0838 | 0.0768 | 0.9162 | 0.0481 | 0.1193 | 0.1211 | 0.0019 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4866 | 0.9810 | 0.1417 | 0.8867 | 0.0839 | 0.0645 | 0.9161 | 0.0547 | 0.1120 | 0.1242 | 0.0122 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4936 | 0.9820 | 0.1294 | 0.9173 | 0.0725 | 0.0569 | 0.9275 | 0.0503 | 0.0978 | 0.1104 | 0.0126 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.5013 | 0.9819 | 0.1326 | 0.9155 | 0.0755 | 0.0692 | 0.9245 | 0.0496 | 0.1073 | 0.1182 | 0.0109 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4852 | 0.9806 | 0.1144 | 0.9184 | 0.0647 | 0.0615 | 0.9353 | 0.0510 | 0.1114 | 0.1120 | 0.0005 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4899 | 0.9814 | 0.1258 | 0.9160 | 0.0727 | 0.0738 | 0.9273 | 0.0495 | 0.1117 | 0.1175 | 0.0058 |
| bearing | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4763 | 0.9803 | 0.1080 | 0.9671 | 0.0601 | 0.0752 | 0.9399 | 0.0411 | 0.1189 | 0.1132 | -0.0058 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4751 | 0.9802 | 0.0843 | 0.9760 | 0.0449 | 0.0506 | 0.9551 | 0.0403 | 0.1070 | 0.0927 | -0.0143 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 2 | 0.5002 | 0.9826 | 0.1488 | 0.9858 | 0.0880 | 0.0523 | 0.9120 | 0.0581 | 0.1037 | 0.1307 | 0.0270 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4908 | 0.9810 | 0.1369 | 0.8947 | 0.0832 | 0.0661 | 0.9168 | 0.0362 | 0.1094 | 0.1266 | 0.0172 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4807 | 0.9808 | 0.1461 | 0.8980 | 0.0838 | 0.0768 | 0.9162 | 0.0481 | 0.1193 | 0.1211 | 0.0019 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4866 | 0.9810 | 0.1417 | 0.8867 | 0.0839 | 0.0645 | 0.9161 | 0.0547 | 0.1120 | 0.1242 | 0.0122 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4936 | 0.9820 | 0.1294 | 0.9173 | 0.0725 | 0.0569 | 0.9275 | 0.0503 | 0.0978 | 0.1104 | 0.0126 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 7 | 0.5013 | 0.9819 | 0.1326 | 0.9155 | 0.0755 | 0.0692 | 0.9245 | 0.0496 | 0.1073 | 0.1182 | 0.0109 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4852 | 0.9806 | 0.1144 | 0.9184 | 0.0647 | 0.0615 | 0.9353 | 0.0510 | 0.1114 | 0.1120 | 0.0005 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4899 | 0.9814 | 0.1258 | 0.9160 | 0.0727 | 0.0738 | 0.9273 | 0.0495 | 0.1117 | 0.1175 | 0.0058 |
| bearing | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4763 | 0.9803 | 0.1080 | 0.9671 | 0.0601 | 0.0752 | 0.9399 | 0.0411 | 0.1189 | 0.1132 | -0.0058 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4778 | 0.9801 | 0.0236 | 0.9762 | 0.0119 | 0.0184 | 0.9881 | 0.0393 | 0.0916 | 0.0760 | -0.0155 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4883 | 0.9821 | 0.0351 | 0.9837 | 0.0179 | 0.0137 | 0.9821 | 0.0687 | 0.0860 | 0.0878 | 0.0018 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4668 | 0.9804 | 0.0274 | 0.9374 | 0.0140 | 0.0121 | 0.9860 | 0.0325 | 0.0840 | 0.0752 | -0.0088 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4798 | 0.9804 | 0.0354 | 0.9202 | 0.0182 | 0.0244 | 0.9818 | 0.0489 | 0.0883 | 0.0773 | -0.0110 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4738 | 0.9806 | 0.0296 | 0.9248 | 0.0152 | 0.0244 | 0.9848 | 0.0508 | 0.0922 | 0.0788 | -0.0134 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4798 | 0.9813 | 0.0401 | 0.9566 | 0.0206 | 0.0215 | 0.9794 | 0.0410 | 0.0744 | 0.0737 | -0.0007 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4764 | 0.9807 | 0.0301 | 0.9460 | 0.0155 | 0.0214 | 0.9845 | 0.0450 | 0.0771 | 0.0709 | -0.0062 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4423 | 0.9789 | 0.0267 | 0.9559 | 0.0136 | 0.0199 | 0.9864 | 0.0347 | 0.0825 | 0.0586 | -0.0239 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4328 | 0.9795 | 0.0185 | 0.9581 | 0.0094 | 0.0168 | 0.9906 | 0.0450 | 0.0800 | 0.0585 | -0.0215 |
| bearing | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4302 | 0.9783 | 0.0439 | 0.9540 | 0.0227 | 0.0352 | 0.9773 | 0.0240 | 0.0994 | 0.0652 | -0.0342 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4778 | 0.9801 | 0.0556 | 0.9788 | 0.0286 | 0.0368 | 0.9714 | 0.0393 | 0.0916 | 0.0760 | -0.0155 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4883 | 0.9821 | 0.0735 | 0.9835 | 0.0383 | 0.0321 | 0.9617 | 0.0687 | 0.0860 | 0.0878 | 0.0018 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4668 | 0.9804 | 0.0503 | 0.9494 | 0.0261 | 0.0275 | 0.9739 | 0.0325 | 0.0840 | 0.0752 | -0.0088 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4798 | 0.9804 | 0.0717 | 0.9678 | 0.0377 | 0.0444 | 0.9623 | 0.0489 | 0.0883 | 0.0773 | -0.0110 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4738 | 0.9806 | 0.0782 | 0.9654 | 0.0412 | 0.0552 | 0.9588 | 0.0508 | 0.0922 | 0.0788 | -0.0134 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4798 | 0.9813 | 0.0706 | 0.9776 | 0.0368 | 0.0368 | 0.9632 | 0.0410 | 0.0744 | 0.0737 | -0.0007 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4764 | 0.9807 | 0.0659 | 0.9756 | 0.0344 | 0.0352 | 0.9656 | 0.0450 | 0.0771 | 0.0709 | -0.0062 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4423 | 0.9789 | 0.0629 | 0.9758 | 0.0325 | 0.0428 | 0.9675 | 0.0347 | 0.0825 | 0.0586 | -0.0239 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4328 | 0.9795 | 0.0648 | 0.9793 | 0.0336 | 0.0382 | 0.9664 | 0.0450 | 0.0800 | 0.0585 | -0.0215 |
| bearing | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4302 | 0.9783 | 0.0654 | 0.9725 | 0.0339 | 0.0504 | 0.9661 | 0.0240 | 0.0994 | 0.0652 | -0.0342 |
| bearing | pooled | cnn-ae | hard_val_p95 | 1 | 0.4778 | 0.9801 | 0.0556 | 0.9788 | 0.0286 | 0.0368 | 0.9714 | 0.0393 | 0.0916 | 0.0760 | -0.0155 |
| bearing | pooled | cnn-ae | hard_val_p95 | 2 | 0.4883 | 0.9821 | 0.0735 | 0.9835 | 0.0383 | 0.0321 | 0.9617 | 0.0687 | 0.0860 | 0.0878 | 0.0018 |
| bearing | pooled | cnn-ae | hard_val_p95 | 3 | 0.4668 | 0.9804 | 0.0503 | 0.9494 | 0.0261 | 0.0275 | 0.9739 | 0.0325 | 0.0840 | 0.0752 | -0.0088 |
| bearing | pooled | cnn-ae | hard_val_p95 | 4 | 0.4798 | 0.9804 | 0.0717 | 0.9678 | 0.0377 | 0.0444 | 0.9623 | 0.0489 | 0.0883 | 0.0773 | -0.0110 |
| bearing | pooled | cnn-ae | hard_val_p95 | 5 | 0.4738 | 0.9806 | 0.0782 | 0.9654 | 0.0412 | 0.0552 | 0.9588 | 0.0508 | 0.0922 | 0.0788 | -0.0134 |
| bearing | pooled | cnn-ae | hard_val_p95 | 6 | 0.4798 | 0.9813 | 0.0706 | 0.9776 | 0.0368 | 0.0368 | 0.9632 | 0.0410 | 0.0744 | 0.0737 | -0.0007 |
| bearing | pooled | cnn-ae | hard_val_p95 | 7 | 0.4764 | 0.9807 | 0.0659 | 0.9756 | 0.0344 | 0.0352 | 0.9656 | 0.0450 | 0.0771 | 0.0709 | -0.0062 |
| bearing | pooled | cnn-ae | hard_val_p95 | 8 | 0.4423 | 0.9789 | 0.0629 | 0.9758 | 0.0325 | 0.0428 | 0.9675 | 0.0347 | 0.0825 | 0.0586 | -0.0239 |
| bearing | pooled | cnn-ae | hard_val_p95 | 9 | 0.4328 | 0.9795 | 0.0648 | 0.9793 | 0.0336 | 0.0382 | 0.9664 | 0.0450 | 0.0800 | 0.0585 | -0.0215 |
| bearing | pooled | cnn-ae | hard_val_p95 | 10 | 0.4302 | 0.9783 | 0.0654 | 0.9725 | 0.0339 | 0.0504 | 0.9661 | 0.0240 | 0.0994 | 0.0652 | -0.0342 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4939 | 0.9712 | 0.0393 | 0.9237 | 0.0202 | 0.0241 | 0.9798 | 0.0503 | 0.0957 | 0.0903 | -0.0054 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4983 | 0.9749 | 0.1368 | 0.9289 | 0.0783 | 0.0378 | 0.9217 | 0.0639 | 0.1113 | 0.1435 | 0.0322 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4997 | 0.9721 | 0.0789 | 0.9009 | 0.0421 | 0.0363 | 0.9579 | 0.0571 | 0.1023 | 0.1044 | 0.0021 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.5053 | 0.9719 | 0.0834 | 0.8914 | 0.0444 | 0.0468 | 0.9556 | 0.0582 | 0.1075 | 0.1060 | -0.0015 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4893 | 0.9716 | 0.0592 | 0.8895 | 0.0310 | 0.0392 | 0.9690 | 0.0595 | 0.1010 | 0.0952 | -0.0058 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4744 | 0.9698 | 0.0468 | 0.9148 | 0.0245 | 0.0422 | 0.9755 | 0.0462 | 0.1114 | 0.0879 | -0.0235 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4445 | 0.9697 | 0.0558 | 0.9384 | 0.0293 | 0.0317 | 0.9707 | 0.0497 | 0.1093 | 0.0890 | -0.0202 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4503 | 0.9695 | 0.0555 | 0.9304 | 0.0290 | 0.0363 | 0.9710 | 0.0372 | 0.1074 | 0.0844 | -0.0230 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4465 | 0.9700 | 0.0752 | 0.9679 | 0.0398 | 0.0317 | 0.9602 | 0.0521 | 0.1164 | 0.1036 | -0.0128 |
| condition | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4411 | 0.9693 | 0.0805 | 0.9539 | 0.0426 | 0.0453 | 0.9574 | 0.0351 | 0.1215 | 0.0968 | -0.0247 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4939 | 0.9712 | 0.0768 | 0.9409 | 0.0405 | 0.0391 | 0.9595 | 0.0503 | 0.0957 | 0.0903 | -0.0054 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4983 | 0.9749 | 0.1868 | 0.9384 | 0.1086 | 0.0575 | 0.8914 | 0.0639 | 0.1113 | 0.1435 | 0.0322 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4997 | 0.9721 | 0.1187 | 0.8936 | 0.0648 | 0.0709 | 0.9352 | 0.0571 | 0.1023 | 0.1044 | 0.0021 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.5053 | 0.9719 | 0.1218 | 0.8926 | 0.0662 | 0.0663 | 0.9338 | 0.0582 | 0.1075 | 0.1060 | -0.0015 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4893 | 0.9716 | 0.1056 | 0.9319 | 0.0579 | 0.0603 | 0.9421 | 0.0595 | 0.1010 | 0.0952 | -0.0058 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4744 | 0.9698 | 0.0890 | 0.9365 | 0.0476 | 0.0693 | 0.9524 | 0.0462 | 0.1114 | 0.0879 | -0.0235 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4445 | 0.9697 | 0.0847 | 0.9571 | 0.0449 | 0.0572 | 0.9551 | 0.0497 | 0.1093 | 0.0890 | -0.0202 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4503 | 0.9695 | 0.0861 | 0.9343 | 0.0458 | 0.0574 | 0.9542 | 0.0372 | 0.1074 | 0.0844 | -0.0230 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4465 | 0.9700 | 0.1087 | 0.9663 | 0.0583 | 0.0558 | 0.9417 | 0.0521 | 0.1164 | 0.1036 | -0.0128 |
| condition | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4411 | 0.9693 | 0.1107 | 0.9396 | 0.0597 | 0.0739 | 0.9403 | 0.0351 | 0.1215 | 0.0968 | -0.0247 |
| condition | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4939 | 0.9712 | 0.0768 | 0.9409 | 0.0405 | 0.0391 | 0.9595 | 0.0503 | 0.0957 | 0.0903 | -0.0054 |
| condition | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4983 | 0.9749 | 0.1868 | 0.9384 | 0.1086 | 0.0575 | 0.8914 | 0.0639 | 0.1113 | 0.1435 | 0.0322 |
| condition | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4997 | 0.9721 | 0.1187 | 0.8936 | 0.0648 | 0.0709 | 0.9352 | 0.0571 | 0.1023 | 0.1044 | 0.0021 |
| condition | client_specific | cnn-ae | hard_val_p95 | 4 | 0.5053 | 0.9719 | 0.1218 | 0.8926 | 0.0662 | 0.0663 | 0.9338 | 0.0582 | 0.1075 | 0.1060 | -0.0015 |
| condition | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4893 | 0.9716 | 0.1056 | 0.9319 | 0.0579 | 0.0603 | 0.9421 | 0.0595 | 0.1010 | 0.0952 | -0.0058 |
| condition | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4744 | 0.9698 | 0.0890 | 0.9365 | 0.0476 | 0.0693 | 0.9524 | 0.0462 | 0.1114 | 0.0879 | -0.0235 |
| condition | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4445 | 0.9697 | 0.0847 | 0.9571 | 0.0449 | 0.0572 | 0.9551 | 0.0497 | 0.1093 | 0.0890 | -0.0202 |
| condition | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4503 | 0.9695 | 0.0861 | 0.9343 | 0.0458 | 0.0574 | 0.9542 | 0.0372 | 0.1074 | 0.0844 | -0.0230 |
| condition | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4465 | 0.9700 | 0.1087 | 0.9663 | 0.0583 | 0.0558 | 0.9417 | 0.0521 | 0.1164 | 0.1036 | -0.0128 |
| condition | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4411 | 0.9693 | 0.1107 | 0.9396 | 0.0597 | 0.0739 | 0.9403 | 0.0351 | 0.1215 | 0.0968 | -0.0247 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4879 | 0.9709 | 0.0198 | 0.9335 | 0.0101 | 0.0181 | 0.9899 | 0.0434 | 0.0859 | 0.0758 | -0.0101 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4986 | 0.9725 | 0.0416 | 0.9334 | 0.0215 | 0.0272 | 0.9785 | 0.0522 | 0.0945 | 0.0916 | -0.0029 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4946 | 0.9719 | 0.0423 | 0.9219 | 0.0218 | 0.0257 | 0.9782 | 0.0472 | 0.0837 | 0.0823 | -0.0015 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4989 | 0.9716 | 0.0435 | 0.9297 | 0.0225 | 0.0272 | 0.9775 | 0.0607 | 0.0877 | 0.0854 | -0.0023 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4879 | 0.9712 | 0.0353 | 0.9444 | 0.0181 | 0.0212 | 0.9819 | 0.0457 | 0.0807 | 0.0709 | -0.0098 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4692 | 0.9688 | 0.0231 | 0.9374 | 0.0118 | 0.0242 | 0.9882 | 0.0400 | 0.0841 | 0.0600 | -0.0241 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4392 | 0.9693 | 0.0370 | 0.9603 | 0.0189 | 0.0196 | 0.9811 | 0.0440 | 0.0975 | 0.0738 | -0.0237 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4440 | 0.9684 | 0.0332 | 0.9330 | 0.0170 | 0.0242 | 0.9830 | 0.0324 | 0.0801 | 0.0585 | -0.0216 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4408 | 0.9694 | 0.0510 | 0.9502 | 0.0264 | 0.0287 | 0.9736 | 0.0489 | 0.1032 | 0.0841 | -0.0190 |
| condition | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4252 | 0.9685 | 0.0567 | 0.9479 | 0.0297 | 0.0302 | 0.9703 | 0.0297 | 0.0911 | 0.0658 | -0.0253 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4879 | 0.9709 | 0.0380 | 0.9552 | 0.0194 | 0.0317 | 0.9806 | 0.0434 | 0.0859 | 0.0758 | -0.0101 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4986 | 0.9725 | 0.0781 | 0.9378 | 0.0416 | 0.0484 | 0.9584 | 0.0522 | 0.0945 | 0.0916 | -0.0029 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4946 | 0.9719 | 0.0745 | 0.9589 | 0.0392 | 0.0424 | 0.9608 | 0.0472 | 0.0837 | 0.0823 | -0.0015 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4989 | 0.9716 | 0.0859 | 0.9615 | 0.0455 | 0.0423 | 0.9545 | 0.0607 | 0.0877 | 0.0854 | -0.0023 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4879 | 0.9712 | 0.0528 | 0.9473 | 0.0273 | 0.0392 | 0.9727 | 0.0457 | 0.0807 | 0.0709 | -0.0098 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4692 | 0.9688 | 0.0525 | 0.9391 | 0.0271 | 0.0498 | 0.9729 | 0.0400 | 0.0841 | 0.0600 | -0.0241 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4392 | 0.9693 | 0.0655 | 0.9573 | 0.0341 | 0.0437 | 0.9659 | 0.0440 | 0.0975 | 0.0738 | -0.0237 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4440 | 0.9684 | 0.0621 | 0.9500 | 0.0324 | 0.0453 | 0.9676 | 0.0324 | 0.0801 | 0.0585 | -0.0216 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4408 | 0.9694 | 0.0911 | 0.9676 | 0.0481 | 0.0453 | 0.9519 | 0.0489 | 0.1032 | 0.0841 | -0.0190 |
| condition | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4252 | 0.9685 | 0.0801 | 0.9449 | 0.0424 | 0.0589 | 0.9576 | 0.0297 | 0.0911 | 0.0658 | -0.0253 |
| condition | pooled | cnn-ae | hard_val_p95 | 1 | 0.4879 | 0.9709 | 0.0380 | 0.9552 | 0.0194 | 0.0317 | 0.9806 | 0.0434 | 0.0859 | 0.0758 | -0.0101 |
| condition | pooled | cnn-ae | hard_val_p95 | 2 | 0.4986 | 0.9725 | 0.0781 | 0.9378 | 0.0416 | 0.0484 | 0.9584 | 0.0522 | 0.0945 | 0.0916 | -0.0029 |
| condition | pooled | cnn-ae | hard_val_p95 | 3 | 0.4946 | 0.9719 | 0.0745 | 0.9589 | 0.0392 | 0.0424 | 0.9608 | 0.0472 | 0.0837 | 0.0823 | -0.0015 |
| condition | pooled | cnn-ae | hard_val_p95 | 4 | 0.4989 | 0.9716 | 0.0859 | 0.9615 | 0.0455 | 0.0423 | 0.9545 | 0.0607 | 0.0877 | 0.0854 | -0.0023 |
| condition | pooled | cnn-ae | hard_val_p95 | 5 | 0.4879 | 0.9712 | 0.0528 | 0.9473 | 0.0273 | 0.0392 | 0.9727 | 0.0457 | 0.0807 | 0.0709 | -0.0098 |
| condition | pooled | cnn-ae | hard_val_p95 | 6 | 0.4692 | 0.9688 | 0.0525 | 0.9391 | 0.0271 | 0.0498 | 0.9729 | 0.0400 | 0.0841 | 0.0600 | -0.0241 |
| condition | pooled | cnn-ae | hard_val_p95 | 7 | 0.4392 | 0.9693 | 0.0655 | 0.9573 | 0.0341 | 0.0437 | 0.9659 | 0.0440 | 0.0975 | 0.0738 | -0.0237 |
| condition | pooled | cnn-ae | hard_val_p95 | 8 | 0.4440 | 0.9684 | 0.0621 | 0.9500 | 0.0324 | 0.0453 | 0.9676 | 0.0324 | 0.0801 | 0.0585 | -0.0216 |
| condition | pooled | cnn-ae | hard_val_p95 | 9 | 0.4408 | 0.9694 | 0.0911 | 0.9676 | 0.0481 | 0.0453 | 0.9519 | 0.0489 | 0.1032 | 0.0841 | -0.0190 |
| condition | pooled | cnn-ae | hard_val_p95 | 10 | 0.4252 | 0.9685 | 0.0801 | 0.9449 | 0.0424 | 0.0589 | 0.9576 | 0.0297 | 0.0911 | 0.0658 | -0.0253 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `federated` approximates collaborative normal-only training without sharing raw vibration windows.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
