# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, FedAvg, FedProx, and personalized federated training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| clients_by | training_mode | calibration_scope | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5068 | 0.9808 | 0.1623 | 0.8509 | 0.0990 | 0.0475 | 0.9010 | 0.0438 | 0.1132 | 0.1528 | 0.0396 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5068 | 0.9808 | 0.1970 | 0.8966 | 0.1210 | 0.0676 | 0.8790 | 0.0438 | 0.1132 | 0.1528 | 0.0396 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.5068 | 0.9808 | 0.1970 | 0.8966 | 0.1210 | 0.0676 | 0.8790 | 0.0438 | 0.1132 | 0.1528 | 0.0396 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4653 | 0.9796 | 0.0327 | 0.9154 | 0.0169 | 0.0229 | 0.9831 | 0.0523 | 0.0837 | 0.0679 | -0.0158 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4653 | 0.9796 | 0.0713 | 0.8946 | 0.0377 | 0.0475 | 0.9623 | 0.0523 | 0.0837 | 0.0679 | -0.0158 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.4653 | 0.9796 | 0.0713 | 0.8946 | 0.0377 | 0.0475 | 0.9623 | 0.0523 | 0.0837 | 0.0679 | -0.0158 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4836 | 0.9811 | 0.0894 | 0.9174 | 0.0507 | 0.0425 | 0.9493 | 0.0501 | 0.1089 | 0.1133 | 0.0044 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4836 | 0.9811 | 0.1218 | 0.9254 | 0.0697 | 0.0629 | 0.9303 | 0.0501 | 0.1089 | 0.1133 | 0.0044 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.4836 | 0.9811 | 0.1218 | 0.9254 | 0.0697 | 0.0629 | 0.9303 | 0.0501 | 0.1089 | 0.1133 | 0.0044 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4624 | 0.9802 | 0.0271 | 0.9527 | 0.0138 | 0.0203 | 0.9862 | 0.0446 | 0.0830 | 0.0699 | -0.0131 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4624 | 0.9802 | 0.0607 | 0.9742 | 0.0314 | 0.0369 | 0.9686 | 0.0446 | 0.0830 | 0.0699 | -0.0131 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.4624 | 0.9802 | 0.0607 | 0.9742 | 0.0314 | 0.0369 | 0.9686 | 0.0446 | 0.0830 | 0.0699 | -0.0131 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5375 | 0.9842 | 0.1546 | 0.9828 | 0.0900 | 0.0246 | 0.9100 | 0.0645 | 0.1010 | 0.1708 | 0.0698 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5375 | 0.9842 | 0.2034 | 0.9838 | 0.1212 | 0.0554 | 0.8788 | 0.0645 | 0.1010 | 0.1708 | 0.0698 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5375 | 0.9842 | 0.2034 | 0.9838 | 0.1212 | 0.0554 | 0.8788 | 0.0645 | 0.1010 | 0.1708 | 0.0698 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4777 | 0.9805 | 0.0698 | 0.9444 | 0.0379 | 0.0400 | 0.9621 | 0.0466 | 0.1044 | 0.0985 | -0.0059 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4777 | 0.9805 | 0.0982 | 0.9521 | 0.0543 | 0.0588 | 0.9457 | 0.0466 | 0.1044 | 0.0985 | -0.0059 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.4777 | 0.9805 | 0.0982 | 0.9521 | 0.0543 | 0.0588 | 0.9457 | 0.0466 | 0.1044 | 0.0985 | -0.0059 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4584 | 0.9796 | 0.0322 | 0.9103 | 0.0165 | 0.0191 | 0.9835 | 0.0372 | 0.0834 | 0.0681 | -0.0153 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4584 | 0.9796 | 0.0575 | 0.9646 | 0.0300 | 0.0368 | 0.9700 | 0.0372 | 0.0834 | 0.0681 | -0.0153 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.4584 | 0.9796 | 0.0575 | 0.9646 | 0.0300 | 0.0368 | 0.9700 | 0.0372 | 0.0834 | 0.0681 | -0.0153 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5580 | 0.9849 | 0.1880 | 0.9833 | 0.1102 | 0.0323 | 0.8898 | 0.0648 | 0.0982 | 0.1877 | 0.0895 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5580 | 0.9849 | 0.2282 | 0.9807 | 0.1366 | 0.0569 | 0.8634 | 0.0648 | 0.0982 | 0.1877 | 0.0895 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5580 | 0.9849 | 0.2282 | 0.9807 | 0.1366 | 0.0569 | 0.8634 | 0.0648 | 0.0982 | 0.1877 | 0.0895 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5152 | 0.9808 | 0.0616 | 0.9640 | 0.0330 | 0.0369 | 0.9670 | 0.0494 | 0.1157 | 0.1136 | -0.0021 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5152 | 0.9808 | 0.0914 | 0.9568 | 0.0498 | 0.0708 | 0.9502 | 0.0494 | 0.1157 | 0.1136 | -0.0021 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.5152 | 0.9808 | 0.0914 | 0.9568 | 0.0498 | 0.0708 | 0.9502 | 0.0494 | 0.1157 | 0.1136 | -0.0021 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4572 | 0.9729 | 0.1362 | 0.9334 | 0.0769 | 0.0317 | 0.9231 | 0.0565 | 0.1108 | 0.1351 | 0.0243 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4572 | 0.9729 | 0.1835 | 0.9143 | 0.1055 | 0.0604 | 0.8945 | 0.0565 | 0.1108 | 0.1351 | 0.0243 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.4572 | 0.9729 | 0.1835 | 0.9143 | 0.1055 | 0.0604 | 0.8945 | 0.0565 | 0.1108 | 0.1351 | 0.0243 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4454 | 0.9707 | 0.0396 | 0.9170 | 0.0204 | 0.0227 | 0.9796 | 0.0557 | 0.0906 | 0.0762 | -0.0145 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4454 | 0.9707 | 0.0786 | 0.9270 | 0.0418 | 0.0408 | 0.9582 | 0.0557 | 0.0906 | 0.0762 | -0.0145 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.4454 | 0.9707 | 0.0786 | 0.9270 | 0.0418 | 0.0408 | 0.9582 | 0.0557 | 0.0906 | 0.0762 | -0.0145 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4721 | 0.9703 | 0.0670 | 0.9212 | 0.0359 | 0.0380 | 0.9641 | 0.0501 | 0.1093 | 0.0967 | -0.0126 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4721 | 0.9703 | 0.1033 | 0.9277 | 0.0562 | 0.0631 | 0.9438 | 0.0501 | 0.1093 | 0.0967 | -0.0126 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.4721 | 0.9703 | 0.1033 | 0.9277 | 0.0562 | 0.0631 | 0.9438 | 0.0501 | 0.1093 | 0.0967 | -0.0126 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4657 | 0.9696 | 0.0337 | 0.9373 | 0.0173 | 0.0249 | 0.9827 | 0.0443 | 0.0926 | 0.0729 | -0.0196 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4657 | 0.9696 | 0.0641 | 0.9466 | 0.0335 | 0.0498 | 0.9665 | 0.0443 | 0.0926 | 0.0729 | -0.0196 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.4657 | 0.9696 | 0.0641 | 0.9466 | 0.0335 | 0.0498 | 0.9665 | 0.0443 | 0.0926 | 0.0729 | -0.0196 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4464 | 0.9693 | 0.1005 | 0.9449 | 0.0541 | 0.0604 | 0.9459 | 0.0478 | 0.1267 | 0.1127 | -0.0140 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4464 | 0.9693 | 0.1384 | 0.9517 | 0.0766 | 0.0770 | 0.9234 | 0.0478 | 0.1267 | 0.1127 | -0.0140 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.4464 | 0.9693 | 0.1384 | 0.9517 | 0.0766 | 0.0770 | 0.9234 | 0.0478 | 0.1267 | 0.1127 | -0.0140 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4823 | 0.9721 | 0.0850 | 0.9305 | 0.0469 | 0.0287 | 0.9531 | 0.0499 | 0.1001 | 0.1082 | 0.0082 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4823 | 0.9721 | 0.1224 | 0.9335 | 0.0683 | 0.0510 | 0.9317 | 0.0499 | 0.1001 | 0.1082 | 0.0082 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.4823 | 0.9721 | 0.1224 | 0.9335 | 0.0683 | 0.0510 | 0.9317 | 0.0499 | 0.1001 | 0.1082 | 0.0082 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4744 | 0.9712 | 0.0355 | 0.9406 | 0.0183 | 0.0219 | 0.9817 | 0.0461 | 0.0899 | 0.0788 | -0.0110 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4744 | 0.9712 | 0.0707 | 0.9514 | 0.0372 | 0.0459 | 0.9628 | 0.0461 | 0.0899 | 0.0788 | -0.0110 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.4744 | 0.9712 | 0.0707 | 0.9514 | 0.0372 | 0.0459 | 0.9628 | 0.0461 | 0.0899 | 0.0788 | -0.0110 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4925 | 0.9732 | 0.1500 | 0.9569 | 0.0846 | 0.0377 | 0.9154 | 0.0491 | 0.1136 | 0.1494 | 0.0358 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4925 | 0.9732 | 0.1860 | 0.9507 | 0.1066 | 0.0634 | 0.8934 | 0.0491 | 0.1136 | 0.1494 | 0.0358 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.4925 | 0.9732 | 0.1860 | 0.9507 | 0.1066 | 0.0634 | 0.8934 | 0.0491 | 0.1136 | 0.1494 | 0.0358 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4402 | 0.9687 | 0.0807 | 0.9549 | 0.0437 | 0.0421 | 0.9563 | 0.0389 | 0.1215 | 0.1006 | -0.0209 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4402 | 0.9687 | 0.1154 | 0.9550 | 0.0637 | 0.0677 | 0.9363 | 0.0389 | 0.1215 | 0.1006 | -0.0209 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.4402 | 0.9687 | 0.1154 | 0.9550 | 0.0637 | 0.0677 | 0.9363 | 0.0389 | 0.1215 | 0.1006 | -0.0209 |

## Client Stability

| clients_by | training_mode | calibration_scope | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0445 | 0.1151 | 0.0360 | 0.1325 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0451 | 0.1207 | 0.0360 | 0.1325 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0451 | 0.1207 | 0.0360 | 0.1325 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0378 | 0.0156 | 0.0383 | 0.0618 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0573 | 0.0263 | 0.0383 | 0.0618 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.0573 | 0.0263 | 0.0383 | 0.0618 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0445 | 0.0691 | 0.0338 | 0.0764 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0584 | 0.0766 | 0.0338 | 0.0764 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0584 | 0.0766 | 0.0338 | 0.0764 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0274 | 0.0089 | 0.0164 | 0.0490 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0397 | 0.0118 | 0.0164 | 0.0490 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0397 | 0.0118 | 0.0164 | 0.0490 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0380 | 0.0850 | 0.0413 | 0.0906 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0499 | 0.0967 | 0.0413 | 0.0906 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0499 | 0.0967 | 0.0413 | 0.0906 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0428 | 0.0445 | 0.0446 | 0.0605 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0509 | 0.0556 | 0.0446 | 0.0605 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0509 | 0.0556 | 0.0446 | 0.0605 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0217 | 0.0128 | 0.0176 | 0.0421 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0342 | 0.0199 | 0.0176 | 0.0421 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0342 | 0.0199 | 0.0176 | 0.0421 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0335 | 0.0848 | 0.0545 | 0.0881 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0574 | 0.0928 | 0.0545 | 0.0881 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0574 | 0.0928 | 0.0545 | 0.0881 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0346 | 0.0369 | 0.0341 | 0.0760 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0501 | 0.0445 | 0.0341 | 0.0760 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0501 | 0.0445 | 0.0341 | 0.0760 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0276 | 0.0644 | 0.0262 | 0.0740 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0452 | 0.0697 | 0.0262 | 0.0740 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0452 | 0.0697 | 0.0262 | 0.0740 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0440 | 0.0152 | 0.0276 | 0.0884 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0639 | 0.0291 | 0.0276 | 0.0884 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.0639 | 0.0291 | 0.0276 | 0.0884 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0326 | 0.0361 | 0.0248 | 0.0425 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0444 | 0.0426 | 0.0248 | 0.0425 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0444 | 0.0426 | 0.0248 | 0.0425 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0301 | 0.0126 | 0.0179 | 0.0516 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0460 | 0.0189 | 0.0179 | 0.0516 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0460 | 0.0189 | 0.0179 | 0.0516 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0516 | 0.0347 | 0.0339 | 0.0593 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0458 | 0.0471 | 0.0339 | 0.0593 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0458 | 0.0471 | 0.0339 | 0.0593 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0313 | 0.0540 | 0.0262 | 0.0545 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0392 | 0.0590 | 0.0262 | 0.0545 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0392 | 0.0590 | 0.0262 | 0.0545 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0327 | 0.0144 | 0.0191 | 0.0590 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0529 | 0.0224 | 0.0191 | 0.0590 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0529 | 0.0224 | 0.0191 | 0.0590 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0203 | 0.0616 | 0.0256 | 0.0723 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0320 | 0.0656 | 0.0256 | 0.0723 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0320 | 0.0656 | 0.0256 | 0.0723 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0315 | 0.0415 | 0.0209 | 0.0334 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0382 | 0.0507 | 0.0209 | 0.0334 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0382 | 0.0507 | 0.0209 | 0.0334 |

## Federated Convergence

| clients_by | training_mode | calibration_scope | model | decision_policy | round | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4752 | 0.9802 | 0.0582 | 0.9722 | 0.0307 | 0.0368 | 0.9693 | 0.0397 | 0.1068 | 0.0926 | -0.0142 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5040 | 0.9827 | 0.1137 | 0.9852 | 0.0669 | 0.0369 | 0.9331 | 0.0573 | 0.0982 | 0.1289 | 0.0307 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4863 | 0.9809 | 0.1110 | 0.8922 | 0.0676 | 0.0415 | 0.9324 | 0.0436 | 0.1093 | 0.1271 | 0.0177 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4692 | 0.9805 | 0.0897 | 0.8929 | 0.0498 | 0.0430 | 0.9502 | 0.0504 | 0.1090 | 0.1091 | 0.0001 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4902 | 0.9820 | 0.1120 | 0.8757 | 0.0660 | 0.0352 | 0.9340 | 0.0623 | 0.1108 | 0.1314 | 0.0206 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4887 | 0.9817 | 0.0909 | 0.9001 | 0.0497 | 0.0430 | 0.9503 | 0.0448 | 0.1037 | 0.1073 | 0.0036 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4983 | 0.9819 | 0.0869 | 0.9025 | 0.0480 | 0.0414 | 0.9520 | 0.0568 | 0.1041 | 0.1113 | 0.0073 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4805 | 0.9804 | 0.0710 | 0.8995 | 0.0390 | 0.0492 | 0.9610 | 0.0498 | 0.1117 | 0.1040 | -0.0076 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4742 | 0.9804 | 0.0847 | 0.8954 | 0.0479 | 0.0552 | 0.9521 | 0.0518 | 0.1209 | 0.1135 | -0.0074 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4695 | 0.9805 | 0.0755 | 0.9583 | 0.0420 | 0.0430 | 0.9580 | 0.0447 | 0.1145 | 0.1081 | -0.0063 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4752 | 0.9802 | 0.0847 | 0.9763 | 0.0451 | 0.0506 | 0.9549 | 0.0397 | 0.1068 | 0.0926 | -0.0142 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5040 | 0.9827 | 0.1465 | 0.9867 | 0.0864 | 0.0430 | 0.9136 | 0.0573 | 0.0982 | 0.1289 | 0.0307 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4863 | 0.9809 | 0.1390 | 0.9017 | 0.0846 | 0.0585 | 0.9154 | 0.0436 | 0.1093 | 0.1271 | 0.0177 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4692 | 0.9805 | 0.1251 | 0.8938 | 0.0716 | 0.0599 | 0.9284 | 0.0504 | 0.1090 | 0.1091 | 0.0001 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4902 | 0.9820 | 0.1486 | 0.8875 | 0.0884 | 0.0582 | 0.9116 | 0.0623 | 0.1108 | 0.1314 | 0.0206 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4887 | 0.9817 | 0.1261 | 0.9090 | 0.0700 | 0.0707 | 0.9300 | 0.0448 | 0.1037 | 0.1073 | 0.0036 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4983 | 0.9819 | 0.1245 | 0.9156 | 0.0699 | 0.0659 | 0.9301 | 0.0568 | 0.1041 | 0.1113 | 0.0073 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4805 | 0.9804 | 0.1024 | 0.9075 | 0.0570 | 0.0737 | 0.9430 | 0.0498 | 0.1117 | 0.1040 | -0.0076 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4742 | 0.9804 | 0.1187 | 0.9084 | 0.0672 | 0.0782 | 0.9328 | 0.0518 | 0.1209 | 0.1135 | -0.0074 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4695 | 0.9805 | 0.1026 | 0.9674 | 0.0573 | 0.0707 | 0.9427 | 0.0447 | 0.1145 | 0.1081 | -0.0063 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4752 | 0.9802 | 0.0847 | 0.9763 | 0.0451 | 0.0506 | 0.9549 | 0.0397 | 0.1068 | 0.0926 | -0.0142 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.5040 | 0.9827 | 0.1465 | 0.9867 | 0.0864 | 0.0430 | 0.9136 | 0.0573 | 0.0982 | 0.1289 | 0.0307 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4863 | 0.9809 | 0.1390 | 0.9017 | 0.0846 | 0.0585 | 0.9154 | 0.0436 | 0.1093 | 0.1271 | 0.0177 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4692 | 0.9805 | 0.1251 | 0.8938 | 0.0716 | 0.0599 | 0.9284 | 0.0504 | 0.1090 | 0.1091 | 0.0001 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4902 | 0.9820 | 0.1486 | 0.8875 | 0.0884 | 0.0582 | 0.9116 | 0.0623 | 0.1108 | 0.1314 | 0.0206 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4887 | 0.9817 | 0.1261 | 0.9090 | 0.0700 | 0.0707 | 0.9300 | 0.0448 | 0.1037 | 0.1073 | 0.0036 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4983 | 0.9819 | 0.1245 | 0.9156 | 0.0699 | 0.0659 | 0.9301 | 0.0568 | 0.1041 | 0.1113 | 0.0073 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4805 | 0.9804 | 0.1024 | 0.9075 | 0.0570 | 0.0737 | 0.9430 | 0.0498 | 0.1117 | 0.1040 | -0.0076 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4742 | 0.9804 | 0.1187 | 0.9084 | 0.0672 | 0.0782 | 0.9328 | 0.0518 | 0.1209 | 0.1135 | -0.0074 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4695 | 0.9805 | 0.1026 | 0.9674 | 0.0573 | 0.0707 | 0.9427 | 0.0447 | 0.1145 | 0.1081 | -0.0063 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4766 | 0.9801 | 0.0229 | 0.9759 | 0.0116 | 0.0184 | 0.9884 | 0.0395 | 0.0917 | 0.0762 | -0.0155 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4895 | 0.9822 | 0.0258 | 0.9799 | 0.0131 | 0.0122 | 0.9869 | 0.0728 | 0.0821 | 0.0840 | 0.0020 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4691 | 0.9804 | 0.0238 | 0.9449 | 0.0121 | 0.0137 | 0.9879 | 0.0392 | 0.0840 | 0.0755 | -0.0085 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4701 | 0.9802 | 0.0306 | 0.9295 | 0.0157 | 0.0244 | 0.9843 | 0.0378 | 0.0824 | 0.0686 | -0.0138 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4759 | 0.9814 | 0.0336 | 0.9197 | 0.0172 | 0.0184 | 0.9828 | 0.0485 | 0.0883 | 0.0838 | -0.0045 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4825 | 0.9813 | 0.0369 | 0.9608 | 0.0189 | 0.0214 | 0.9811 | 0.0419 | 0.0732 | 0.0730 | -0.0002 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4781 | 0.9810 | 0.0306 | 0.9597 | 0.0157 | 0.0229 | 0.9843 | 0.0518 | 0.0756 | 0.0725 | -0.0031 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4445 | 0.9788 | 0.0218 | 0.9492 | 0.0111 | 0.0245 | 0.9889 | 0.0368 | 0.0785 | 0.0560 | -0.0226 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4242 | 0.9784 | 0.0228 | 0.9579 | 0.0116 | 0.0229 | 0.9884 | 0.0451 | 0.0884 | 0.0579 | -0.0305 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4131 | 0.9780 | 0.0222 | 0.9498 | 0.0112 | 0.0246 | 0.9888 | 0.0330 | 0.0858 | 0.0516 | -0.0342 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4766 | 0.9801 | 0.0565 | 0.9792 | 0.0291 | 0.0368 | 0.9709 | 0.0395 | 0.0917 | 0.0762 | -0.0155 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4895 | 0.9822 | 0.0693 | 0.9842 | 0.0360 | 0.0291 | 0.9640 | 0.0728 | 0.0821 | 0.0840 | 0.0020 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4691 | 0.9804 | 0.0491 | 0.9609 | 0.0254 | 0.0214 | 0.9746 | 0.0392 | 0.0840 | 0.0755 | -0.0085 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4701 | 0.9802 | 0.0604 | 0.9715 | 0.0314 | 0.0413 | 0.9686 | 0.0378 | 0.0824 | 0.0686 | -0.0138 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4759 | 0.9814 | 0.0635 | 0.9736 | 0.0331 | 0.0322 | 0.9669 | 0.0485 | 0.0883 | 0.0838 | -0.0045 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4825 | 0.9813 | 0.0696 | 0.9828 | 0.0361 | 0.0322 | 0.9639 | 0.0419 | 0.0732 | 0.0730 | -0.0002 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4781 | 0.9810 | 0.0695 | 0.9821 | 0.0361 | 0.0335 | 0.9639 | 0.0518 | 0.0756 | 0.0725 | -0.0031 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4445 | 0.9788 | 0.0547 | 0.9655 | 0.0283 | 0.0474 | 0.9717 | 0.0368 | 0.0785 | 0.0560 | -0.0226 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4242 | 0.9784 | 0.0608 | 0.9721 | 0.0315 | 0.0474 | 0.9685 | 0.0451 | 0.0884 | 0.0579 | -0.0305 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4131 | 0.9780 | 0.0533 | 0.9701 | 0.0274 | 0.0475 | 0.9726 | 0.0330 | 0.0858 | 0.0516 | -0.0342 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.4766 | 0.9801 | 0.0565 | 0.9792 | 0.0291 | 0.0368 | 0.9709 | 0.0395 | 0.0917 | 0.0762 | -0.0155 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.4895 | 0.9822 | 0.0693 | 0.9842 | 0.0360 | 0.0291 | 0.9640 | 0.0728 | 0.0821 | 0.0840 | 0.0020 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.4691 | 0.9804 | 0.0491 | 0.9609 | 0.0254 | 0.0214 | 0.9746 | 0.0392 | 0.0840 | 0.0755 | -0.0085 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.4701 | 0.9802 | 0.0604 | 0.9715 | 0.0314 | 0.0413 | 0.9686 | 0.0378 | 0.0824 | 0.0686 | -0.0138 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.4759 | 0.9814 | 0.0635 | 0.9736 | 0.0331 | 0.0322 | 0.9669 | 0.0485 | 0.0883 | 0.0838 | -0.0045 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.4825 | 0.9813 | 0.0696 | 0.9828 | 0.0361 | 0.0322 | 0.9639 | 0.0419 | 0.0732 | 0.0730 | -0.0002 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.4781 | 0.9810 | 0.0695 | 0.9821 | 0.0361 | 0.0335 | 0.9639 | 0.0518 | 0.0756 | 0.0725 | -0.0031 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.4445 | 0.9788 | 0.0547 | 0.9655 | 0.0283 | 0.0474 | 0.9717 | 0.0368 | 0.0785 | 0.0560 | -0.0226 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.4242 | 0.9784 | 0.0608 | 0.9721 | 0.0315 | 0.0474 | 0.9685 | 0.0451 | 0.0884 | 0.0579 | -0.0305 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.4131 | 0.9780 | 0.0533 | 0.9701 | 0.0274 | 0.0475 | 0.9726 | 0.0330 | 0.0858 | 0.0516 | -0.0342 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4864 | 0.9808 | 0.0600 | 0.9766 | 0.0318 | 0.0339 | 0.9682 | 0.0388 | 0.0998 | 0.0929 | -0.0069 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4938 | 0.9819 | 0.1270 | 0.9819 | 0.0730 | 0.0446 | 0.9270 | 0.0540 | 0.1156 | 0.1336 | 0.0181 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4569 | 0.9802 | 0.0705 | 0.8394 | 0.0380 | 0.0353 | 0.9620 | 0.0487 | 0.1095 | 0.0965 | -0.0131 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4708 | 0.9800 | 0.0684 | 0.8223 | 0.0362 | 0.0323 | 0.9638 | 0.0479 | 0.1032 | 0.0937 | -0.0094 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4689 | 0.9801 | 0.0626 | 0.9771 | 0.0331 | 0.0386 | 0.9669 | 0.0349 | 0.0975 | 0.0854 | -0.0122 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4798 | 0.9809 | 0.0582 | 0.9743 | 0.0303 | 0.0338 | 0.9697 | 0.0379 | 0.0884 | 0.0842 | -0.0042 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4877 | 0.9804 | 0.0563 | 0.9687 | 0.0297 | 0.0432 | 0.9703 | 0.0415 | 0.0986 | 0.0857 | -0.0129 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4786 | 0.9800 | 0.0618 | 0.9656 | 0.0330 | 0.0508 | 0.9670 | 0.0545 | 0.1140 | 0.1025 | -0.0115 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4809 | 0.9805 | 0.0623 | 0.9703 | 0.0348 | 0.0415 | 0.9652 | 0.0458 | 0.1059 | 0.0999 | -0.0059 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4737 | 0.9803 | 0.0713 | 0.9674 | 0.0395 | 0.0462 | 0.9605 | 0.0623 | 0.1121 | 0.1106 | -0.0014 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4864 | 0.9808 | 0.0850 | 0.9784 | 0.0459 | 0.0492 | 0.9541 | 0.0388 | 0.0998 | 0.0929 | -0.0069 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4938 | 0.9819 | 0.1634 | 0.9665 | 0.0957 | 0.0753 | 0.9043 | 0.0540 | 0.1156 | 0.1336 | 0.0181 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4569 | 0.9802 | 0.0975 | 0.8778 | 0.0542 | 0.0552 | 0.9458 | 0.0487 | 0.1095 | 0.0965 | -0.0131 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4708 | 0.9800 | 0.0919 | 0.8753 | 0.0493 | 0.0523 | 0.9507 | 0.0479 | 0.1032 | 0.0937 | -0.0094 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4689 | 0.9801 | 0.0877 | 0.9581 | 0.0471 | 0.0555 | 0.9529 | 0.0349 | 0.0975 | 0.0854 | -0.0122 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4798 | 0.9809 | 0.0852 | 0.9800 | 0.0453 | 0.0446 | 0.9547 | 0.0379 | 0.0884 | 0.0842 | -0.0042 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4877 | 0.9804 | 0.0801 | 0.9699 | 0.0426 | 0.0570 | 0.9574 | 0.0415 | 0.0986 | 0.0857 | -0.0129 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4786 | 0.9800 | 0.0968 | 0.9708 | 0.0537 | 0.0738 | 0.9463 | 0.0545 | 0.1140 | 0.1025 | -0.0115 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4809 | 0.9805 | 0.0904 | 0.9657 | 0.0509 | 0.0646 | 0.9491 | 0.0458 | 0.1059 | 0.0999 | -0.0059 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4737 | 0.9803 | 0.1043 | 0.9782 | 0.0587 | 0.0601 | 0.9413 | 0.0623 | 0.1121 | 0.1106 | -0.0014 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4864 | 0.9808 | 0.0850 | 0.9784 | 0.0459 | 0.0492 | 0.9541 | 0.0388 | 0.0998 | 0.0929 | -0.0069 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4938 | 0.9819 | 0.1634 | 0.9665 | 0.0957 | 0.0753 | 0.9043 | 0.0540 | 0.1156 | 0.1336 | 0.0181 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4569 | 0.9802 | 0.0975 | 0.8778 | 0.0542 | 0.0552 | 0.9458 | 0.0487 | 0.1095 | 0.0965 | -0.0131 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4708 | 0.9800 | 0.0919 | 0.8753 | 0.0493 | 0.0523 | 0.9507 | 0.0479 | 0.1032 | 0.0937 | -0.0094 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4689 | 0.9801 | 0.0877 | 0.9581 | 0.0471 | 0.0555 | 0.9529 | 0.0349 | 0.0975 | 0.0854 | -0.0122 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4798 | 0.9809 | 0.0852 | 0.9800 | 0.0453 | 0.0446 | 0.9547 | 0.0379 | 0.0884 | 0.0842 | -0.0042 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4877 | 0.9804 | 0.0801 | 0.9699 | 0.0426 | 0.0570 | 0.9574 | 0.0415 | 0.0986 | 0.0857 | -0.0129 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4786 | 0.9800 | 0.0968 | 0.9708 | 0.0537 | 0.0738 | 0.9463 | 0.0545 | 0.1140 | 0.1025 | -0.0115 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4809 | 0.9805 | 0.0904 | 0.9657 | 0.0509 | 0.0646 | 0.9491 | 0.0458 | 0.1059 | 0.0999 | -0.0059 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4737 | 0.9803 | 0.1043 | 0.9782 | 0.0587 | 0.0601 | 0.9413 | 0.0623 | 0.1121 | 0.1106 | -0.0014 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4818 | 0.9807 | 0.0282 | 0.9752 | 0.0143 | 0.0200 | 0.9857 | 0.0417 | 0.0897 | 0.0791 | -0.0106 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4762 | 0.9810 | 0.0573 | 0.9783 | 0.0298 | 0.0231 | 0.9702 | 0.0457 | 0.0944 | 0.0903 | -0.0040 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4548 | 0.9783 | 0.0312 | 0.6583 | 0.0160 | 0.0200 | 0.9840 | 0.0435 | 0.0977 | 0.0749 | -0.0228 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4684 | 0.9800 | 0.0344 | 0.6590 | 0.0177 | 0.0154 | 0.9823 | 0.0450 | 0.0842 | 0.0764 | -0.0078 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4622 | 0.9802 | 0.0379 | 0.9720 | 0.0196 | 0.0169 | 0.9804 | 0.0395 | 0.0797 | 0.0718 | -0.0079 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4795 | 0.9806 | 0.0362 | 0.9834 | 0.0186 | 0.0153 | 0.9814 | 0.0347 | 0.0782 | 0.0720 | -0.0062 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4662 | 0.9795 | 0.0277 | 0.9576 | 0.0141 | 0.0262 | 0.9859 | 0.0327 | 0.0791 | 0.0615 | -0.0176 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4342 | 0.9782 | 0.0267 | 0.9686 | 0.0136 | 0.0246 | 0.9864 | 0.0277 | 0.0827 | 0.0555 | -0.0272 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4322 | 0.9784 | 0.0185 | 0.9653 | 0.0094 | 0.0185 | 0.9906 | 0.0293 | 0.0746 | 0.0470 | -0.0276 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4283 | 0.9786 | 0.0238 | 0.9849 | 0.0121 | 0.0107 | 0.9879 | 0.0325 | 0.0741 | 0.0527 | -0.0214 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4818 | 0.9807 | 0.0635 | 0.9777 | 0.0331 | 0.0400 | 0.9669 | 0.0417 | 0.0897 | 0.0791 | -0.0106 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4762 | 0.9810 | 0.0950 | 0.9802 | 0.0504 | 0.0460 | 0.9496 | 0.0457 | 0.0944 | 0.0903 | -0.0040 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4548 | 0.9783 | 0.0527 | 0.9436 | 0.0274 | 0.0338 | 0.9726 | 0.0435 | 0.0977 | 0.0749 | -0.0228 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4684 | 0.9800 | 0.0640 | 0.9445 | 0.0336 | 0.0306 | 0.9664 | 0.0450 | 0.0842 | 0.0764 | -0.0078 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4622 | 0.9802 | 0.0641 | 0.9662 | 0.0337 | 0.0339 | 0.9663 | 0.0395 | 0.0797 | 0.0718 | -0.0079 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4795 | 0.9806 | 0.0595 | 0.9665 | 0.0310 | 0.0352 | 0.9690 | 0.0347 | 0.0782 | 0.0720 | -0.0062 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4662 | 0.9795 | 0.0489 | 0.9673 | 0.0252 | 0.0383 | 0.9748 | 0.0327 | 0.0791 | 0.0615 | -0.0176 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4342 | 0.9782 | 0.0426 | 0.9619 | 0.0220 | 0.0429 | 0.9780 | 0.0277 | 0.0827 | 0.0555 | -0.0272 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4322 | 0.9784 | 0.0387 | 0.9651 | 0.0198 | 0.0352 | 0.9802 | 0.0293 | 0.0746 | 0.0470 | -0.0276 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4283 | 0.9786 | 0.0458 | 0.9736 | 0.0237 | 0.0322 | 0.9763 | 0.0325 | 0.0741 | 0.0527 | -0.0214 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.4818 | 0.9807 | 0.0635 | 0.9777 | 0.0331 | 0.0400 | 0.9669 | 0.0417 | 0.0897 | 0.0791 | -0.0106 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.4762 | 0.9810 | 0.0950 | 0.9802 | 0.0504 | 0.0460 | 0.9496 | 0.0457 | 0.0944 | 0.0903 | -0.0040 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.4548 | 0.9783 | 0.0527 | 0.9436 | 0.0274 | 0.0338 | 0.9726 | 0.0435 | 0.0977 | 0.0749 | -0.0228 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.4684 | 0.9800 | 0.0640 | 0.9445 | 0.0336 | 0.0306 | 0.9664 | 0.0450 | 0.0842 | 0.0764 | -0.0078 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.4622 | 0.9802 | 0.0641 | 0.9662 | 0.0337 | 0.0339 | 0.9663 | 0.0395 | 0.0797 | 0.0718 | -0.0079 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.4795 | 0.9806 | 0.0595 | 0.9665 | 0.0310 | 0.0352 | 0.9690 | 0.0347 | 0.0782 | 0.0720 | -0.0062 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.4662 | 0.9795 | 0.0489 | 0.9673 | 0.0252 | 0.0383 | 0.9748 | 0.0327 | 0.0791 | 0.0615 | -0.0176 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.4342 | 0.9782 | 0.0426 | 0.9619 | 0.0220 | 0.0429 | 0.9780 | 0.0277 | 0.0827 | 0.0555 | -0.0272 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.4322 | 0.9784 | 0.0387 | 0.9651 | 0.0198 | 0.0352 | 0.9802 | 0.0293 | 0.0746 | 0.0470 | -0.0276 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.4283 | 0.9786 | 0.0458 | 0.9736 | 0.0237 | 0.0322 | 0.9763 | 0.0325 | 0.0741 | 0.0527 | -0.0214 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4938 | 0.9712 | 0.0384 | 0.9206 | 0.0197 | 0.0241 | 0.9803 | 0.0514 | 0.0963 | 0.0902 | -0.0061 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4932 | 0.9747 | 0.1401 | 0.9377 | 0.0797 | 0.0303 | 0.9203 | 0.0649 | 0.1087 | 0.1436 | 0.0349 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.5028 | 0.9719 | 0.0758 | 0.8900 | 0.0401 | 0.0498 | 0.9599 | 0.0476 | 0.1062 | 0.1010 | -0.0053 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.5097 | 0.9718 | 0.0852 | 0.8875 | 0.0455 | 0.0468 | 0.9545 | 0.0582 | 0.1084 | 0.1054 | -0.0030 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4943 | 0.9713 | 0.0622 | 0.8891 | 0.0326 | 0.0407 | 0.9674 | 0.0604 | 0.1027 | 0.0949 | -0.0078 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4816 | 0.9698 | 0.0485 | 0.9247 | 0.0252 | 0.0467 | 0.9748 | 0.0448 | 0.1109 | 0.0877 | -0.0232 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4522 | 0.9689 | 0.0366 | 0.9486 | 0.0189 | 0.0151 | 0.9811 | 0.0495 | 0.0998 | 0.0796 | -0.0201 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4445 | 0.9689 | 0.0613 | 0.9315 | 0.0323 | 0.0468 | 0.9677 | 0.0447 | 0.1161 | 0.0880 | -0.0281 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4304 | 0.9682 | 0.0540 | 0.9482 | 0.0281 | 0.0347 | 0.9719 | 0.0434 | 0.1157 | 0.0871 | -0.0286 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4189 | 0.9666 | 0.0682 | 0.9344 | 0.0367 | 0.0452 | 0.9633 | 0.0363 | 0.1283 | 0.0892 | -0.0391 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4938 | 0.9712 | 0.0761 | 0.9392 | 0.0402 | 0.0421 | 0.9598 | 0.0514 | 0.0963 | 0.0902 | -0.0061 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4932 | 0.9747 | 0.1916 | 0.9303 | 0.1109 | 0.0605 | 0.8891 | 0.0649 | 0.1087 | 0.1436 | 0.0349 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.5028 | 0.9719 | 0.1078 | 0.8891 | 0.0582 | 0.0723 | 0.9418 | 0.0476 | 0.1062 | 0.1010 | -0.0053 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.5097 | 0.9718 | 0.1215 | 0.8857 | 0.0662 | 0.0784 | 0.9338 | 0.0582 | 0.1084 | 0.1054 | -0.0030 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4943 | 0.9713 | 0.1044 | 0.9235 | 0.0565 | 0.0618 | 0.9435 | 0.0604 | 0.1027 | 0.0949 | -0.0078 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4816 | 0.9698 | 0.0831 | 0.9410 | 0.0439 | 0.0678 | 0.9561 | 0.0448 | 0.1109 | 0.0877 | -0.0232 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4522 | 0.9689 | 0.0706 | 0.9435 | 0.0371 | 0.0468 | 0.9629 | 0.0495 | 0.0998 | 0.0796 | -0.0201 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4445 | 0.9689 | 0.0965 | 0.9416 | 0.0516 | 0.0633 | 0.9484 | 0.0447 | 0.1161 | 0.0880 | -0.0281 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4304 | 0.9682 | 0.0947 | 0.9525 | 0.0506 | 0.0680 | 0.9494 | 0.0434 | 0.1157 | 0.0871 | -0.0286 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4189 | 0.9666 | 0.0870 | 0.9300 | 0.0469 | 0.0694 | 0.9531 | 0.0363 | 0.1283 | 0.0892 | -0.0391 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4938 | 0.9712 | 0.0761 | 0.9392 | 0.0402 | 0.0421 | 0.9598 | 0.0514 | 0.0963 | 0.0902 | -0.0061 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4932 | 0.9747 | 0.1916 | 0.9303 | 0.1109 | 0.0605 | 0.8891 | 0.0649 | 0.1087 | 0.1436 | 0.0349 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.5028 | 0.9719 | 0.1078 | 0.8891 | 0.0582 | 0.0723 | 0.9418 | 0.0476 | 0.1062 | 0.1010 | -0.0053 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.5097 | 0.9718 | 0.1215 | 0.8857 | 0.0662 | 0.0784 | 0.9338 | 0.0582 | 0.1084 | 0.1054 | -0.0030 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4943 | 0.9713 | 0.1044 | 0.9235 | 0.0565 | 0.0618 | 0.9435 | 0.0604 | 0.1027 | 0.0949 | -0.0078 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4816 | 0.9698 | 0.0831 | 0.9410 | 0.0439 | 0.0678 | 0.9561 | 0.0448 | 0.1109 | 0.0877 | -0.0232 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4522 | 0.9689 | 0.0706 | 0.9435 | 0.0371 | 0.0468 | 0.9629 | 0.0495 | 0.0998 | 0.0796 | -0.0201 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4445 | 0.9689 | 0.0965 | 0.9416 | 0.0516 | 0.0633 | 0.9484 | 0.0447 | 0.1161 | 0.0880 | -0.0281 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4304 | 0.9682 | 0.0947 | 0.9525 | 0.0506 | 0.0680 | 0.9494 | 0.0434 | 0.1157 | 0.0871 | -0.0286 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4189 | 0.9666 | 0.0870 | 0.9300 | 0.0469 | 0.0694 | 0.9531 | 0.0363 | 0.1283 | 0.0892 | -0.0391 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4897 | 0.9708 | 0.0201 | 0.9474 | 0.0102 | 0.0166 | 0.9898 | 0.0431 | 0.0862 | 0.0756 | -0.0106 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4908 | 0.9725 | 0.0496 | 0.9343 | 0.0257 | 0.0272 | 0.9743 | 0.0478 | 0.0940 | 0.0936 | -0.0004 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4986 | 0.9717 | 0.0380 | 0.9118 | 0.0196 | 0.0257 | 0.9804 | 0.0449 | 0.0851 | 0.0806 | -0.0045 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.5020 | 0.9717 | 0.0518 | 0.9480 | 0.0269 | 0.0242 | 0.9731 | 0.0493 | 0.0874 | 0.0857 | -0.0018 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4877 | 0.9709 | 0.0346 | 0.9333 | 0.0177 | 0.0212 | 0.9823 | 0.0550 | 0.0863 | 0.0765 | -0.0099 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4750 | 0.9691 | 0.0307 | 0.9367 | 0.0157 | 0.0317 | 0.9843 | 0.0385 | 0.0890 | 0.0642 | -0.0248 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4469 | 0.9679 | 0.0199 | 0.9347 | 0.0101 | 0.0257 | 0.9899 | 0.0469 | 0.1001 | 0.0675 | -0.0326 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4363 | 0.9684 | 0.0330 | 0.9462 | 0.0169 | 0.0226 | 0.9831 | 0.0388 | 0.0890 | 0.0601 | -0.0289 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4246 | 0.9674 | 0.0347 | 0.9451 | 0.0177 | 0.0242 | 0.9823 | 0.0385 | 0.0954 | 0.0638 | -0.0316 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4054 | 0.9652 | 0.0242 | 0.9359 | 0.0123 | 0.0303 | 0.9877 | 0.0397 | 0.1132 | 0.0618 | -0.0514 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4897 | 0.9708 | 0.0370 | 0.9513 | 0.0189 | 0.0332 | 0.9811 | 0.0431 | 0.0862 | 0.0756 | -0.0106 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4908 | 0.9725 | 0.0877 | 0.9386 | 0.0469 | 0.0499 | 0.9531 | 0.0478 | 0.0940 | 0.0936 | -0.0004 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4986 | 0.9717 | 0.0689 | 0.9542 | 0.0361 | 0.0423 | 0.9639 | 0.0449 | 0.0851 | 0.0806 | -0.0045 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.5020 | 0.9717 | 0.0842 | 0.9526 | 0.0446 | 0.0453 | 0.9554 | 0.0493 | 0.0874 | 0.0857 | -0.0018 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4877 | 0.9709 | 0.0679 | 0.9454 | 0.0354 | 0.0497 | 0.9646 | 0.0550 | 0.0863 | 0.0765 | -0.0099 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4750 | 0.9691 | 0.0547 | 0.9341 | 0.0283 | 0.0558 | 0.9717 | 0.0385 | 0.0890 | 0.0642 | -0.0248 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4469 | 0.9679 | 0.0568 | 0.9512 | 0.0293 | 0.0544 | 0.9707 | 0.0469 | 0.1001 | 0.0675 | -0.0326 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4363 | 0.9684 | 0.0592 | 0.9422 | 0.0307 | 0.0513 | 0.9693 | 0.0388 | 0.0890 | 0.0601 | -0.0289 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4246 | 0.9674 | 0.0640 | 0.9512 | 0.0332 | 0.0513 | 0.9668 | 0.0385 | 0.0954 | 0.0638 | -0.0316 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4054 | 0.9652 | 0.0611 | 0.9456 | 0.0317 | 0.0650 | 0.9683 | 0.0397 | 0.1132 | 0.0618 | -0.0514 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.4897 | 0.9708 | 0.0370 | 0.9513 | 0.0189 | 0.0332 | 0.9811 | 0.0431 | 0.0862 | 0.0756 | -0.0106 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.4908 | 0.9725 | 0.0877 | 0.9386 | 0.0469 | 0.0499 | 0.9531 | 0.0478 | 0.0940 | 0.0936 | -0.0004 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.4986 | 0.9717 | 0.0689 | 0.9542 | 0.0361 | 0.0423 | 0.9639 | 0.0449 | 0.0851 | 0.0806 | -0.0045 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.5020 | 0.9717 | 0.0842 | 0.9526 | 0.0446 | 0.0453 | 0.9554 | 0.0493 | 0.0874 | 0.0857 | -0.0018 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.4877 | 0.9709 | 0.0679 | 0.9454 | 0.0354 | 0.0497 | 0.9646 | 0.0550 | 0.0863 | 0.0765 | -0.0099 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.4750 | 0.9691 | 0.0547 | 0.9341 | 0.0283 | 0.0558 | 0.9717 | 0.0385 | 0.0890 | 0.0642 | -0.0248 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.4469 | 0.9679 | 0.0568 | 0.9512 | 0.0293 | 0.0544 | 0.9707 | 0.0469 | 0.1001 | 0.0675 | -0.0326 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.4363 | 0.9684 | 0.0592 | 0.9422 | 0.0307 | 0.0513 | 0.9693 | 0.0388 | 0.0890 | 0.0601 | -0.0289 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.4246 | 0.9674 | 0.0640 | 0.9512 | 0.0332 | 0.0513 | 0.9668 | 0.0385 | 0.0954 | 0.0638 | -0.0316 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.4054 | 0.9652 | 0.0611 | 0.9456 | 0.0317 | 0.0650 | 0.9683 | 0.0397 | 0.1132 | 0.0618 | -0.0514 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4926 | 0.9711 | 0.0468 | 0.9367 | 0.0241 | 0.0301 | 0.9759 | 0.0447 | 0.0977 | 0.0895 | -0.0082 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4929 | 0.9751 | 0.1268 | 0.9456 | 0.0732 | 0.0182 | 0.9268 | 0.0531 | 0.0970 | 0.1351 | 0.0381 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4920 | 0.9747 | 0.1459 | 0.9064 | 0.0857 | 0.0257 | 0.9143 | 0.0467 | 0.1076 | 0.1462 | 0.0386 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.5002 | 0.9735 | 0.1456 | 0.9351 | 0.0834 | 0.0257 | 0.9166 | 0.0559 | 0.1051 | 0.1468 | 0.0417 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5054 | 0.9732 | 0.0888 | 0.9024 | 0.0474 | 0.0287 | 0.9526 | 0.0508 | 0.0905 | 0.1077 | 0.0172 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5039 | 0.9719 | 0.0549 | 0.9289 | 0.0286 | 0.0392 | 0.9714 | 0.0512 | 0.0970 | 0.0918 | -0.0052 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4775 | 0.9716 | 0.0514 | 0.9168 | 0.0266 | 0.0241 | 0.9734 | 0.0475 | 0.0822 | 0.0785 | -0.0037 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4383 | 0.9688 | 0.0369 | 0.9300 | 0.0189 | 0.0302 | 0.9811 | 0.0538 | 0.1073 | 0.0824 | -0.0249 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4712 | 0.9710 | 0.0760 | 0.9467 | 0.0404 | 0.0333 | 0.9596 | 0.0426 | 0.1014 | 0.0981 | -0.0032 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4493 | 0.9702 | 0.0771 | 0.9559 | 0.0410 | 0.0317 | 0.9590 | 0.0525 | 0.1152 | 0.1065 | -0.0087 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4926 | 0.9711 | 0.0718 | 0.9434 | 0.0376 | 0.0422 | 0.9624 | 0.0447 | 0.0977 | 0.0895 | -0.0082 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4929 | 0.9751 | 0.1745 | 0.9224 | 0.1019 | 0.0424 | 0.8981 | 0.0531 | 0.0970 | 0.1351 | 0.0381 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4920 | 0.9747 | 0.1837 | 0.9215 | 0.1080 | 0.0559 | 0.8920 | 0.0467 | 0.1076 | 0.1462 | 0.0386 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.5002 | 0.9735 | 0.1819 | 0.9268 | 0.1059 | 0.0559 | 0.8941 | 0.0559 | 0.1051 | 0.1468 | 0.0417 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5054 | 0.9732 | 0.1212 | 0.9177 | 0.0660 | 0.0468 | 0.9340 | 0.0508 | 0.0905 | 0.1077 | 0.0172 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5039 | 0.9719 | 0.1004 | 0.9348 | 0.0540 | 0.0588 | 0.9460 | 0.0512 | 0.0970 | 0.0918 | -0.0052 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4775 | 0.9716 | 0.0888 | 0.9285 | 0.0475 | 0.0437 | 0.9525 | 0.0475 | 0.0822 | 0.0785 | -0.0037 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4383 | 0.9688 | 0.0781 | 0.9391 | 0.0412 | 0.0542 | 0.9588 | 0.0538 | 0.1073 | 0.0824 | -0.0249 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4712 | 0.9710 | 0.1098 | 0.9563 | 0.0595 | 0.0545 | 0.9405 | 0.0426 | 0.1014 | 0.0981 | -0.0032 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4493 | 0.9702 | 0.1138 | 0.9440 | 0.0613 | 0.0559 | 0.9387 | 0.0525 | 0.1152 | 0.1065 | -0.0087 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4926 | 0.9711 | 0.0718 | 0.9434 | 0.0376 | 0.0422 | 0.9624 | 0.0447 | 0.0977 | 0.0895 | -0.0082 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4929 | 0.9751 | 0.1745 | 0.9224 | 0.1019 | 0.0424 | 0.8981 | 0.0531 | 0.0970 | 0.1351 | 0.0381 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4920 | 0.9747 | 0.1837 | 0.9215 | 0.1080 | 0.0559 | 0.8920 | 0.0467 | 0.1076 | 0.1462 | 0.0386 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.5002 | 0.9735 | 0.1819 | 0.9268 | 0.1059 | 0.0559 | 0.8941 | 0.0559 | 0.1051 | 0.1468 | 0.0417 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.5054 | 0.9732 | 0.1212 | 0.9177 | 0.0660 | 0.0468 | 0.9340 | 0.0508 | 0.0905 | 0.1077 | 0.0172 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.5039 | 0.9719 | 0.1004 | 0.9348 | 0.0540 | 0.0588 | 0.9460 | 0.0512 | 0.0970 | 0.0918 | -0.0052 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4775 | 0.9716 | 0.0888 | 0.9285 | 0.0475 | 0.0437 | 0.9525 | 0.0475 | 0.0822 | 0.0785 | -0.0037 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4383 | 0.9688 | 0.0781 | 0.9391 | 0.0412 | 0.0542 | 0.9588 | 0.0538 | 0.1073 | 0.0824 | -0.0249 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4712 | 0.9710 | 0.1098 | 0.9563 | 0.0595 | 0.0545 | 0.9405 | 0.0426 | 0.1014 | 0.0981 | -0.0032 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4493 | 0.9702 | 0.1138 | 0.9440 | 0.0613 | 0.0559 | 0.9387 | 0.0525 | 0.1152 | 0.1065 | -0.0087 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4926 | 0.9708 | 0.0294 | 0.9562 | 0.0150 | 0.0212 | 0.9850 | 0.0402 | 0.0932 | 0.0809 | -0.0123 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4952 | 0.9728 | 0.0258 | 0.9299 | 0.0131 | 0.0182 | 0.9869 | 0.0472 | 0.0811 | 0.0798 | -0.0013 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4940 | 0.9724 | 0.0376 | 0.9260 | 0.0196 | 0.0167 | 0.9804 | 0.0461 | 0.0899 | 0.0868 | -0.0031 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4948 | 0.9727 | 0.0490 | 0.9423 | 0.0254 | 0.0226 | 0.9746 | 0.0517 | 0.0926 | 0.0950 | 0.0024 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5000 | 0.9729 | 0.0502 | 0.9376 | 0.0261 | 0.0302 | 0.9739 | 0.0468 | 0.0857 | 0.0870 | 0.0013 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4982 | 0.9713 | 0.0264 | 0.9171 | 0.0135 | 0.0348 | 0.9865 | 0.0475 | 0.0883 | 0.0734 | -0.0150 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4549 | 0.9706 | 0.0268 | 0.9536 | 0.0136 | 0.0182 | 0.9864 | 0.0453 | 0.0785 | 0.0614 | -0.0171 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4268 | 0.9683 | 0.0238 | 0.9410 | 0.0121 | 0.0242 | 0.9879 | 0.0411 | 0.0995 | 0.0674 | -0.0321 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4530 | 0.9706 | 0.0478 | 0.9596 | 0.0247 | 0.0151 | 0.9753 | 0.0437 | 0.0868 | 0.0769 | -0.0098 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4349 | 0.9697 | 0.0387 | 0.9431 | 0.0199 | 0.0181 | 0.9801 | 0.0512 | 0.1029 | 0.0795 | -0.0234 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4926 | 0.9708 | 0.0555 | 0.9594 | 0.0286 | 0.0453 | 0.9714 | 0.0402 | 0.0932 | 0.0809 | -0.0123 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4952 | 0.9728 | 0.0606 | 0.9338 | 0.0317 | 0.0333 | 0.9683 | 0.0472 | 0.0811 | 0.0798 | -0.0013 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4940 | 0.9724 | 0.0708 | 0.9224 | 0.0382 | 0.0469 | 0.9618 | 0.0461 | 0.0899 | 0.0868 | -0.0031 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4948 | 0.9727 | 0.0817 | 0.9468 | 0.0435 | 0.0407 | 0.9565 | 0.0517 | 0.0926 | 0.0950 | 0.0024 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5000 | 0.9729 | 0.0838 | 0.9596 | 0.0445 | 0.0484 | 0.9555 | 0.0468 | 0.0857 | 0.0870 | 0.0013 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4982 | 0.9713 | 0.0663 | 0.9538 | 0.0344 | 0.0589 | 0.9656 | 0.0475 | 0.0883 | 0.0734 | -0.0150 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4549 | 0.9706 | 0.0600 | 0.9646 | 0.0310 | 0.0408 | 0.9690 | 0.0453 | 0.0785 | 0.0614 | -0.0171 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4268 | 0.9683 | 0.0519 | 0.9484 | 0.0268 | 0.0483 | 0.9732 | 0.0411 | 0.0995 | 0.0674 | -0.0321 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4530 | 0.9706 | 0.0885 | 0.9590 | 0.0469 | 0.0439 | 0.9531 | 0.0437 | 0.0868 | 0.0769 | -0.0098 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4349 | 0.9697 | 0.0873 | 0.9661 | 0.0460 | 0.0529 | 0.9540 | 0.0512 | 0.1029 | 0.0795 | -0.0234 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.4926 | 0.9708 | 0.0555 | 0.9594 | 0.0286 | 0.0453 | 0.9714 | 0.0402 | 0.0932 | 0.0809 | -0.0123 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.4952 | 0.9728 | 0.0606 | 0.9338 | 0.0317 | 0.0333 | 0.9683 | 0.0472 | 0.0811 | 0.0798 | -0.0013 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.4940 | 0.9724 | 0.0708 | 0.9224 | 0.0382 | 0.0469 | 0.9618 | 0.0461 | 0.0899 | 0.0868 | -0.0031 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.4948 | 0.9727 | 0.0817 | 0.9468 | 0.0435 | 0.0407 | 0.9565 | 0.0517 | 0.0926 | 0.0950 | 0.0024 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.5000 | 0.9729 | 0.0838 | 0.9596 | 0.0445 | 0.0484 | 0.9555 | 0.0468 | 0.0857 | 0.0870 | 0.0013 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.4982 | 0.9713 | 0.0663 | 0.9538 | 0.0344 | 0.0589 | 0.9656 | 0.0475 | 0.0883 | 0.0734 | -0.0150 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.4549 | 0.9706 | 0.0600 | 0.9646 | 0.0310 | 0.0408 | 0.9690 | 0.0453 | 0.0785 | 0.0614 | -0.0171 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.4268 | 0.9683 | 0.0519 | 0.9484 | 0.0268 | 0.0483 | 0.9732 | 0.0411 | 0.0995 | 0.0674 | -0.0321 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.4530 | 0.9706 | 0.0885 | 0.9590 | 0.0469 | 0.0439 | 0.9531 | 0.0437 | 0.0868 | 0.0769 | -0.0098 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.4349 | 0.9697 | 0.0873 | 0.9661 | 0.0460 | 0.0529 | 0.9540 | 0.0512 | 0.1029 | 0.0795 | -0.0234 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `fedavg` approximates collaborative normal-only training without sharing raw vibration windows.
- `fedprox` adds a proximal penalty to reduce local client drift under non-IID data.
- `*-personalized` locally adapts the federated global model before client evaluation.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
