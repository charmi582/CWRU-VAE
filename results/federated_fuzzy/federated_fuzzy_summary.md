# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, FedAvg, FedProx, and personalized federated training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| clients_by | training_mode | calibration_scope | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5162 | 0.9822 | 0.1642 | 0.8550 | 0.0988 | 0.0230 | 0.9012 | 0.0628 | 0.0906 | 0.1576 | 0.0670 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5162 | 0.9822 | 0.2029 | 0.9422 | 0.1250 | 0.0476 | 0.8750 | 0.0628 | 0.0906 | 0.1576 | 0.0670 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.5162 | 0.9822 | 0.2029 | 0.9422 | 0.1250 | 0.0476 | 0.8750 | 0.0628 | 0.0906 | 0.1576 | 0.0670 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4777 | 0.9809 | 0.0508 | 0.9661 | 0.0264 | 0.0153 | 0.9736 | 0.0548 | 0.0782 | 0.0801 | 0.0019 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4777 | 0.9809 | 0.0868 | 0.9178 | 0.0464 | 0.0445 | 0.9536 | 0.0548 | 0.0782 | 0.0801 | 0.0019 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.4777 | 0.9809 | 0.0868 | 0.9178 | 0.0464 | 0.0445 | 0.9536 | 0.0548 | 0.0782 | 0.0801 | 0.0019 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4849 | 0.9811 | 0.0878 | 0.9172 | 0.0497 | 0.0410 | 0.9503 | 0.0496 | 0.1082 | 0.1125 | 0.0043 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4849 | 0.9811 | 0.1222 | 0.9404 | 0.0698 | 0.0627 | 0.9302 | 0.0496 | 0.1082 | 0.1125 | 0.0043 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.4849 | 0.9811 | 0.1222 | 0.9404 | 0.0698 | 0.0627 | 0.9302 | 0.0496 | 0.1082 | 0.1125 | 0.0043 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4667 | 0.9802 | 0.0282 | 0.9533 | 0.0144 | 0.0202 | 0.9856 | 0.0449 | 0.0834 | 0.0699 | -0.0135 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4667 | 0.9802 | 0.0630 | 0.9738 | 0.0327 | 0.0392 | 0.9673 | 0.0449 | 0.0834 | 0.0699 | -0.0135 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.4667 | 0.9802 | 0.0630 | 0.9738 | 0.0327 | 0.0392 | 0.9673 | 0.0449 | 0.0834 | 0.0699 | -0.0135 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5491 | 0.9847 | 0.1728 | 0.9807 | 0.1024 | 0.0323 | 0.8976 | 0.0733 | 0.0993 | 0.1827 | 0.0834 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5491 | 0.9847 | 0.2163 | 0.9846 | 0.1309 | 0.0537 | 0.8691 | 0.0733 | 0.0993 | 0.1827 | 0.0834 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5491 | 0.9847 | 0.2163 | 0.9846 | 0.1309 | 0.0537 | 0.8691 | 0.0733 | 0.0993 | 0.1827 | 0.0834 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5296 | 0.9830 | 0.1347 | 0.9513 | 0.0789 | 0.0420 | 0.9211 | 0.0594 | 0.1075 | 0.1510 | 0.0434 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5296 | 0.9830 | 0.1745 | 0.9648 | 0.1043 | 0.0642 | 0.8957 | 0.0594 | 0.1075 | 0.1510 | 0.0434 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.5296 | 0.9830 | 0.1745 | 0.9648 | 0.1043 | 0.0642 | 0.8957 | 0.0594 | 0.1075 | 0.1510 | 0.0434 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.5066 | 0.9821 | 0.0748 | 0.7714 | 0.0434 | 0.0247 | 0.9566 | 0.0552 | 0.0890 | 0.1086 | 0.0196 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5066 | 0.9821 | 0.1125 | 0.8434 | 0.0672 | 0.0461 | 0.9328 | 0.0552 | 0.0890 | 0.1086 | 0.0196 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.5066 | 0.9821 | 0.1125 | 0.8434 | 0.0672 | 0.0461 | 0.9328 | 0.0552 | 0.0890 | 0.1086 | 0.0196 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5651 | 0.9852 | 0.1667 | 0.9614 | 0.0955 | 0.0367 | 0.9045 | 0.0807 | 0.1044 | 0.1846 | 0.0802 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5651 | 0.9852 | 0.2353 | 0.9836 | 0.1419 | 0.0598 | 0.8581 | 0.0807 | 0.1044 | 0.1846 | 0.0802 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5651 | 0.9852 | 0.2353 | 0.9836 | 0.1419 | 0.0598 | 0.8581 | 0.0807 | 0.1044 | 0.1846 | 0.0802 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4763 | 0.9806 | 0.0699 | 0.9409 | 0.0380 | 0.0374 | 0.9620 | 0.0452 | 0.1036 | 0.0982 | -0.0054 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4763 | 0.9806 | 0.0988 | 0.9529 | 0.0545 | 0.0592 | 0.9455 | 0.0452 | 0.1036 | 0.0982 | -0.0054 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.4763 | 0.9806 | 0.0988 | 0.9529 | 0.0545 | 0.0592 | 0.9455 | 0.0452 | 0.1036 | 0.0982 | -0.0054 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4584 | 0.9796 | 0.0296 | 0.9309 | 0.0152 | 0.0191 | 0.9848 | 0.0394 | 0.0835 | 0.0672 | -0.0163 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4584 | 0.9796 | 0.0575 | 0.9625 | 0.0300 | 0.0379 | 0.9700 | 0.0394 | 0.0835 | 0.0672 | -0.0163 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.4584 | 0.9796 | 0.0575 | 0.9625 | 0.0300 | 0.0379 | 0.9700 | 0.0394 | 0.0835 | 0.0672 | -0.0163 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5265 | 0.9829 | 0.1450 | 0.9746 | 0.0859 | 0.0262 | 0.9141 | 0.0666 | 0.1057 | 0.1641 | 0.0584 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5265 | 0.9829 | 0.1882 | 0.9671 | 0.1138 | 0.0630 | 0.8862 | 0.0666 | 0.1057 | 0.1641 | 0.0584 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5265 | 0.9829 | 0.1882 | 0.9671 | 0.1138 | 0.0630 | 0.8862 | 0.0666 | 0.1057 | 0.1641 | 0.0584 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5054 | 0.9802 | 0.0540 | 0.9156 | 0.0289 | 0.0338 | 0.9711 | 0.0540 | 0.1149 | 0.1120 | -0.0028 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5054 | 0.9802 | 0.0958 | 0.9142 | 0.0528 | 0.0676 | 0.9472 | 0.0540 | 0.1149 | 0.1120 | -0.0028 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.5054 | 0.9802 | 0.0958 | 0.9142 | 0.0528 | 0.0676 | 0.9472 | 0.0540 | 0.1149 | 0.1120 | -0.0028 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4464 | 0.9714 | 0.1324 | 0.9150 | 0.0743 | 0.0393 | 0.9257 | 0.0506 | 0.1166 | 0.1240 | 0.0074 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4464 | 0.9714 | 0.1699 | 0.9125 | 0.0974 | 0.0710 | 0.9026 | 0.0506 | 0.1166 | 0.1240 | 0.0074 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.4464 | 0.9714 | 0.1699 | 0.9125 | 0.0974 | 0.0710 | 0.9026 | 0.0506 | 0.1166 | 0.1240 | 0.0074 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4338 | 0.9698 | 0.0487 | 0.9175 | 0.0254 | 0.0333 | 0.9746 | 0.0371 | 0.0937 | 0.0707 | -0.0231 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4338 | 0.9698 | 0.0734 | 0.9169 | 0.0392 | 0.0514 | 0.9608 | 0.0371 | 0.0937 | 0.0707 | -0.0231 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.4338 | 0.9698 | 0.0734 | 0.9169 | 0.0392 | 0.0514 | 0.9608 | 0.0371 | 0.0937 | 0.0707 | -0.0231 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4783 | 0.9713 | 0.0782 | 0.9218 | 0.0419 | 0.0383 | 0.9581 | 0.0541 | 0.1094 | 0.1046 | -0.0048 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4783 | 0.9713 | 0.1197 | 0.9313 | 0.0656 | 0.0629 | 0.9344 | 0.0541 | 0.1094 | 0.1046 | -0.0048 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.4783 | 0.9713 | 0.1197 | 0.9313 | 0.0656 | 0.0629 | 0.9344 | 0.0541 | 0.1094 | 0.1046 | -0.0048 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4728 | 0.9705 | 0.0422 | 0.9153 | 0.0219 | 0.0215 | 0.9781 | 0.0466 | 0.0896 | 0.0789 | -0.0107 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4728 | 0.9705 | 0.0777 | 0.9537 | 0.0411 | 0.0465 | 0.9589 | 0.0466 | 0.0896 | 0.0789 | -0.0107 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.4728 | 0.9705 | 0.0777 | 0.9537 | 0.0411 | 0.0465 | 0.9589 | 0.0466 | 0.0896 | 0.0789 | -0.0107 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4826 | 0.9735 | 0.1356 | 0.9700 | 0.0746 | 0.0409 | 0.9254 | 0.0615 | 0.1161 | 0.1433 | 0.0272 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4826 | 0.9735 | 0.1933 | 0.9702 | 0.1113 | 0.0620 | 0.8887 | 0.0615 | 0.1161 | 0.1433 | 0.0272 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.4826 | 0.9735 | 0.1933 | 0.9702 | 0.1113 | 0.0620 | 0.8887 | 0.0615 | 0.1161 | 0.1433 | 0.0272 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5085 | 0.9737 | 0.1082 | 0.9336 | 0.0608 | 0.0319 | 0.9392 | 0.0601 | 0.0993 | 0.1262 | 0.0268 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5085 | 0.9737 | 0.1511 | 0.9447 | 0.0869 | 0.0535 | 0.9131 | 0.0601 | 0.0993 | 0.1262 | 0.0268 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.5085 | 0.9737 | 0.1511 | 0.9447 | 0.0869 | 0.0535 | 0.9131 | 0.0601 | 0.0993 | 0.1262 | 0.0268 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.5072 | 0.9736 | 0.0726 | 0.7147 | 0.0420 | 0.0226 | 0.9580 | 0.0613 | 0.0861 | 0.1038 | 0.0177 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5072 | 0.9736 | 0.1150 | 0.7673 | 0.0693 | 0.0448 | 0.9307 | 0.0613 | 0.0861 | 0.1038 | 0.0177 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.5072 | 0.9736 | 0.1150 | 0.7673 | 0.0693 | 0.0448 | 0.9307 | 0.0613 | 0.0861 | 0.1038 | 0.0177 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5147 | 0.9755 | 0.1247 | 0.9467 | 0.0691 | 0.0181 | 0.9309 | 0.0611 | 0.0898 | 0.1375 | 0.0477 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5147 | 0.9755 | 0.1754 | 0.9574 | 0.0999 | 0.0438 | 0.9001 | 0.0611 | 0.0898 | 0.1375 | 0.0477 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5147 | 0.9755 | 0.1754 | 0.9574 | 0.0999 | 0.0438 | 0.9001 | 0.0611 | 0.0898 | 0.1375 | 0.0477 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4827 | 0.9719 | 0.0896 | 0.9346 | 0.0496 | 0.0317 | 0.9504 | 0.0530 | 0.1044 | 0.1117 | 0.0073 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4827 | 0.9719 | 0.1319 | 0.9317 | 0.0739 | 0.0576 | 0.9261 | 0.0530 | 0.1044 | 0.1117 | 0.0073 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.4827 | 0.9719 | 0.1319 | 0.9317 | 0.0739 | 0.0576 | 0.9261 | 0.0530 | 0.1044 | 0.1117 | 0.0073 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4760 | 0.9711 | 0.0409 | 0.9443 | 0.0212 | 0.0246 | 0.9788 | 0.0498 | 0.0932 | 0.0811 | -0.0120 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4760 | 0.9711 | 0.0780 | 0.9507 | 0.0413 | 0.0521 | 0.9587 | 0.0498 | 0.0932 | 0.0811 | -0.0120 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.4760 | 0.9711 | 0.0780 | 0.9507 | 0.0413 | 0.0521 | 0.9587 | 0.0498 | 0.0932 | 0.0811 | -0.0120 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5171 | 0.9746 | 0.1504 | 0.9476 | 0.0856 | 0.0363 | 0.9144 | 0.0520 | 0.0994 | 0.1469 | 0.0475 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5171 | 0.9746 | 0.1867 | 0.9536 | 0.1074 | 0.0528 | 0.8926 | 0.0520 | 0.0994 | 0.1469 | 0.0475 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5171 | 0.9746 | 0.1867 | 0.9536 | 0.1074 | 0.0528 | 0.8926 | 0.0520 | 0.0994 | 0.1469 | 0.0475 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4322 | 0.9691 | 0.0877 | 0.9682 | 0.0473 | 0.0376 | 0.9527 | 0.0390 | 0.1175 | 0.0997 | -0.0178 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4322 | 0.9691 | 0.1227 | 0.9513 | 0.0676 | 0.0677 | 0.9324 | 0.0390 | 0.1175 | 0.0997 | -0.0178 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.4322 | 0.9691 | 0.1227 | 0.9513 | 0.0676 | 0.0677 | 0.9324 | 0.0390 | 0.1175 | 0.0997 | -0.0178 |

## Client Stability

| clients_by | training_mode | calibration_scope | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0209 | 0.1074 | 0.0679 | 0.1252 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0315 | 0.1219 | 0.0679 | 0.1252 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0315 | 0.1219 | 0.0679 | 0.1252 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0265 | 0.0189 | 0.0328 | 0.0662 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0567 | 0.0314 | 0.0328 | 0.0662 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.0567 | 0.0314 | 0.0328 | 0.0662 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0455 | 0.0671 | 0.0319 | 0.0789 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0572 | 0.0755 | 0.0319 | 0.0789 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0572 | 0.0755 | 0.0319 | 0.0789 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0264 | 0.0101 | 0.0142 | 0.0505 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0441 | 0.0127 | 0.0142 | 0.0505 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0441 | 0.0127 | 0.0142 | 0.0505 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0361 | 0.0961 | 0.0452 | 0.0947 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0418 | 0.1072 | 0.0452 | 0.0947 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0418 | 0.1072 | 0.0452 | 0.0947 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0455 | 0.0900 | 0.0455 | 0.1015 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0563 | 0.1034 | 0.0455 | 0.1015 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0563 | 0.1034 | 0.0455 | 0.1015 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0408 | 0.0748 | 0.0571 | 0.0787 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0629 | 0.0994 | 0.0571 | 0.0787 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.0629 | 0.0994 | 0.0571 | 0.0787 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0396 | 0.0709 | 0.0808 | 0.0946 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0556 | 0.0994 | 0.0808 | 0.0946 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0556 | 0.0994 | 0.0808 | 0.0946 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0385 | 0.0442 | 0.0352 | 0.0606 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0484 | 0.0540 | 0.0352 | 0.0606 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0484 | 0.0540 | 0.0352 | 0.0606 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0230 | 0.0109 | 0.0190 | 0.0436 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0360 | 0.0193 | 0.0190 | 0.0436 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0360 | 0.0193 | 0.0190 | 0.0436 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0196 | 0.0958 | 0.0556 | 0.1116 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0421 | 0.1079 | 0.0556 | 0.1116 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0421 | 0.1079 | 0.0556 | 0.1116 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0418 | 0.0345 | 0.0430 | 0.0790 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0571 | 0.0516 | 0.0430 | 0.0790 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0571 | 0.0516 | 0.0430 | 0.0790 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0323 | 0.0614 | 0.0300 | 0.0858 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0496 | 0.0702 | 0.0300 | 0.0858 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0496 | 0.0702 | 0.0300 | 0.0858 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0579 | 0.0212 | 0.0171 | 0.0990 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0859 | 0.0330 | 0.0171 | 0.0990 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.0859 | 0.0330 | 0.0171 | 0.0990 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0343 | 0.0361 | 0.0295 | 0.0422 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0473 | 0.0452 | 0.0295 | 0.0422 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0473 | 0.0452 | 0.0295 | 0.0422 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0269 | 0.0180 | 0.0189 | 0.0493 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0429 | 0.0252 | 0.0189 | 0.0493 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0429 | 0.0252 | 0.0189 | 0.0493 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0365 | 0.0430 | 0.0510 | 0.0581 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0430 | 0.0681 | 0.0510 | 0.0581 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0430 | 0.0681 | 0.0510 | 0.0581 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0325 | 0.0638 | 0.0394 | 0.0692 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0480 | 0.0772 | 0.0394 | 0.0692 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0480 | 0.0772 | 0.0394 | 0.0692 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0385 | 0.0733 | 0.0747 | 0.0588 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0631 | 0.1027 | 0.0747 | 0.0588 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.0631 | 0.1027 | 0.0747 | 0.0588 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0129 | 0.0526 | 0.0414 | 0.0609 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0360 | 0.0639 | 0.0414 | 0.0609 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0360 | 0.0639 | 0.0414 | 0.0609 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0304 | 0.0556 | 0.0300 | 0.0551 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0392 | 0.0602 | 0.0300 | 0.0551 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0392 | 0.0602 | 0.0300 | 0.0551 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0327 | 0.0168 | 0.0222 | 0.0618 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0594 | 0.0257 | 0.0222 | 0.0618 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0594 | 0.0257 | 0.0222 | 0.0618 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0235 | 0.0692 | 0.0238 | 0.0517 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0334 | 0.0690 | 0.0238 | 0.0517 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0334 | 0.0690 | 0.0238 | 0.0517 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0271 | 0.0383 | 0.0192 | 0.0367 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0394 | 0.0485 | 0.0192 | 0.0367 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0394 | 0.0485 | 0.0192 | 0.0367 |

## Federated Convergence

| clients_by | training_mode | calibration_scope | model | decision_policy | round | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4748 | 0.9802 | 0.0581 | 0.9708 | 0.0306 | 0.0384 | 0.9694 | 0.0406 | 0.1068 | 0.0927 | -0.0141 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5021 | 0.9825 | 0.1178 | 0.9869 | 0.0694 | 0.0385 | 0.9306 | 0.0578 | 0.1037 | 0.1323 | 0.0286 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4952 | 0.9808 | 0.1080 | 0.8897 | 0.0661 | 0.0415 | 0.9339 | 0.0414 | 0.1100 | 0.1262 | 0.0162 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4870 | 0.9811 | 0.1046 | 0.8729 | 0.0591 | 0.0492 | 0.9409 | 0.0550 | 0.1156 | 0.1206 | 0.0050 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4823 | 0.9813 | 0.1068 | 0.8857 | 0.0623 | 0.0385 | 0.9377 | 0.0491 | 0.1086 | 0.1229 | 0.0144 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4824 | 0.9817 | 0.0851 | 0.9046 | 0.0463 | 0.0370 | 0.9537 | 0.0490 | 0.0997 | 0.1053 | 0.0055 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4888 | 0.9816 | 0.0814 | 0.9067 | 0.0444 | 0.0399 | 0.9556 | 0.0482 | 0.1003 | 0.1043 | 0.0040 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4833 | 0.9805 | 0.0752 | 0.8942 | 0.0413 | 0.0415 | 0.9587 | 0.0514 | 0.1094 | 0.1058 | -0.0036 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4767 | 0.9807 | 0.0781 | 0.9007 | 0.0431 | 0.0523 | 0.9569 | 0.0523 | 0.1179 | 0.1107 | -0.0072 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4761 | 0.9804 | 0.0633 | 0.9602 | 0.0342 | 0.0337 | 0.9658 | 0.0517 | 0.1101 | 0.1040 | -0.0061 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4748 | 0.9802 | 0.0851 | 0.9757 | 0.0453 | 0.0521 | 0.9547 | 0.0406 | 0.1068 | 0.0927 | -0.0141 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5021 | 0.9825 | 0.1510 | 0.9857 | 0.0895 | 0.0507 | 0.9105 | 0.0578 | 0.1037 | 0.1323 | 0.0286 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4952 | 0.9808 | 0.1345 | 0.9189 | 0.0821 | 0.0646 | 0.9179 | 0.0414 | 0.1100 | 0.1262 | 0.0162 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4870 | 0.9811 | 0.1492 | 0.8950 | 0.0865 | 0.0723 | 0.9135 | 0.0550 | 0.1156 | 0.1206 | 0.0050 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4823 | 0.9813 | 0.1421 | 0.8867 | 0.0832 | 0.0553 | 0.9168 | 0.0491 | 0.1086 | 0.1229 | 0.0144 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4824 | 0.9817 | 0.1249 | 0.9215 | 0.0691 | 0.0615 | 0.9309 | 0.0490 | 0.0997 | 0.1053 | 0.0055 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4888 | 0.9816 | 0.1167 | 0.9153 | 0.0648 | 0.0598 | 0.9352 | 0.0482 | 0.1003 | 0.1043 | 0.0040 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4833 | 0.9805 | 0.1082 | 0.9668 | 0.0608 | 0.0738 | 0.9392 | 0.0514 | 0.1094 | 0.1058 | -0.0036 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4767 | 0.9807 | 0.1165 | 0.9687 | 0.0655 | 0.0737 | 0.9345 | 0.0523 | 0.1179 | 0.1107 | -0.0072 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4761 | 0.9804 | 0.0938 | 0.9700 | 0.0515 | 0.0629 | 0.9485 | 0.0517 | 0.1101 | 0.1040 | -0.0061 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4748 | 0.9802 | 0.0851 | 0.9757 | 0.0453 | 0.0521 | 0.9547 | 0.0406 | 0.1068 | 0.0927 | -0.0141 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.5021 | 0.9825 | 0.1510 | 0.9857 | 0.0895 | 0.0507 | 0.9105 | 0.0578 | 0.1037 | 0.1323 | 0.0286 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4952 | 0.9808 | 0.1345 | 0.9189 | 0.0821 | 0.0646 | 0.9179 | 0.0414 | 0.1100 | 0.1262 | 0.0162 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4870 | 0.9811 | 0.1492 | 0.8950 | 0.0865 | 0.0723 | 0.9135 | 0.0550 | 0.1156 | 0.1206 | 0.0050 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4823 | 0.9813 | 0.1421 | 0.8867 | 0.0832 | 0.0553 | 0.9168 | 0.0491 | 0.1086 | 0.1229 | 0.0144 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4824 | 0.9817 | 0.1249 | 0.9215 | 0.0691 | 0.0615 | 0.9309 | 0.0490 | 0.0997 | 0.1053 | 0.0055 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4888 | 0.9816 | 0.1167 | 0.9153 | 0.0648 | 0.0598 | 0.9352 | 0.0482 | 0.1003 | 0.1043 | 0.0040 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4833 | 0.9805 | 0.1082 | 0.9668 | 0.0608 | 0.0738 | 0.9392 | 0.0514 | 0.1094 | 0.1058 | -0.0036 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4767 | 0.9807 | 0.1165 | 0.9687 | 0.0655 | 0.0737 | 0.9345 | 0.0523 | 0.1179 | 0.1107 | -0.0072 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4761 | 0.9804 | 0.0938 | 0.9700 | 0.0515 | 0.0629 | 0.9485 | 0.0517 | 0.1101 | 0.1040 | -0.0061 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4759 | 0.9801 | 0.0232 | 0.9761 | 0.0118 | 0.0184 | 0.9882 | 0.0392 | 0.0921 | 0.0765 | -0.0156 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4889 | 0.9822 | 0.0321 | 0.9832 | 0.0164 | 0.0122 | 0.9836 | 0.0644 | 0.0825 | 0.0843 | 0.0018 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4769 | 0.9804 | 0.0247 | 0.9241 | 0.0126 | 0.0122 | 0.9874 | 0.0368 | 0.0840 | 0.0745 | -0.0095 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4858 | 0.9806 | 0.0448 | 0.9310 | 0.0232 | 0.0275 | 0.9768 | 0.0420 | 0.0873 | 0.0791 | -0.0082 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4766 | 0.9808 | 0.0205 | 0.9370 | 0.0104 | 0.0214 | 0.9896 | 0.0507 | 0.0847 | 0.0743 | -0.0104 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4817 | 0.9813 | 0.0337 | 0.9625 | 0.0172 | 0.0199 | 0.9828 | 0.0551 | 0.0750 | 0.0752 | 0.0002 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4692 | 0.9808 | 0.0342 | 0.9629 | 0.0176 | 0.0184 | 0.9824 | 0.0426 | 0.0753 | 0.0683 | -0.0071 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4485 | 0.9788 | 0.0237 | 0.9539 | 0.0121 | 0.0230 | 0.9879 | 0.0404 | 0.0764 | 0.0551 | -0.0214 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4284 | 0.9784 | 0.0264 | 0.9579 | 0.0135 | 0.0261 | 0.9865 | 0.0406 | 0.0911 | 0.0579 | -0.0332 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4352 | 0.9784 | 0.0185 | 0.9443 | 0.0094 | 0.0230 | 0.9906 | 0.0368 | 0.0853 | 0.0536 | -0.0316 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4759 | 0.9801 | 0.0572 | 0.9781 | 0.0295 | 0.0383 | 0.9705 | 0.0392 | 0.0921 | 0.0765 | -0.0156 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4889 | 0.9822 | 0.0670 | 0.9850 | 0.0348 | 0.0275 | 0.9652 | 0.0644 | 0.0825 | 0.0843 | 0.0018 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4769 | 0.9804 | 0.0453 | 0.9555 | 0.0233 | 0.0275 | 0.9767 | 0.0368 | 0.0840 | 0.0745 | -0.0095 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4858 | 0.9806 | 0.0722 | 0.9718 | 0.0378 | 0.0414 | 0.9622 | 0.0420 | 0.0873 | 0.0791 | -0.0082 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4766 | 0.9808 | 0.0630 | 0.9702 | 0.0327 | 0.0414 | 0.9673 | 0.0507 | 0.0847 | 0.0743 | -0.0104 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4817 | 0.9813 | 0.0838 | 0.9823 | 0.0440 | 0.0383 | 0.9560 | 0.0551 | 0.0750 | 0.0752 | 0.0002 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4692 | 0.9808 | 0.0613 | 0.9782 | 0.0317 | 0.0352 | 0.9683 | 0.0426 | 0.0753 | 0.0683 | -0.0071 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4485 | 0.9788 | 0.0571 | 0.9727 | 0.0295 | 0.0428 | 0.9705 | 0.0404 | 0.0764 | 0.0551 | -0.0214 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4284 | 0.9784 | 0.0657 | 0.9705 | 0.0341 | 0.0551 | 0.9659 | 0.0406 | 0.0911 | 0.0579 | -0.0332 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4352 | 0.9784 | 0.0572 | 0.9733 | 0.0295 | 0.0445 | 0.9705 | 0.0368 | 0.0853 | 0.0536 | -0.0316 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.4759 | 0.9801 | 0.0572 | 0.9781 | 0.0295 | 0.0383 | 0.9705 | 0.0392 | 0.0921 | 0.0765 | -0.0156 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.4889 | 0.9822 | 0.0670 | 0.9850 | 0.0348 | 0.0275 | 0.9652 | 0.0644 | 0.0825 | 0.0843 | 0.0018 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.4769 | 0.9804 | 0.0453 | 0.9555 | 0.0233 | 0.0275 | 0.9767 | 0.0368 | 0.0840 | 0.0745 | -0.0095 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.4858 | 0.9806 | 0.0722 | 0.9718 | 0.0378 | 0.0414 | 0.9622 | 0.0420 | 0.0873 | 0.0791 | -0.0082 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.4766 | 0.9808 | 0.0630 | 0.9702 | 0.0327 | 0.0414 | 0.9673 | 0.0507 | 0.0847 | 0.0743 | -0.0104 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.4817 | 0.9813 | 0.0838 | 0.9823 | 0.0440 | 0.0383 | 0.9560 | 0.0551 | 0.0750 | 0.0752 | 0.0002 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.4692 | 0.9808 | 0.0613 | 0.9782 | 0.0317 | 0.0352 | 0.9683 | 0.0426 | 0.0753 | 0.0683 | -0.0071 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.4485 | 0.9788 | 0.0571 | 0.9727 | 0.0295 | 0.0428 | 0.9705 | 0.0404 | 0.0764 | 0.0551 | -0.0214 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.4284 | 0.9784 | 0.0657 | 0.9705 | 0.0341 | 0.0551 | 0.9659 | 0.0406 | 0.0911 | 0.0579 | -0.0332 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.4352 | 0.9784 | 0.0572 | 0.9733 | 0.0295 | 0.0445 | 0.9705 | 0.0368 | 0.0853 | 0.0536 | -0.0316 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5014 | 0.9817 | 0.0866 | 0.9766 | 0.0474 | 0.0308 | 0.9526 | 0.0566 | 0.1076 | 0.1149 | 0.0074 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5007 | 0.9822 | 0.1372 | 0.9810 | 0.0790 | 0.0631 | 0.9210 | 0.0582 | 0.1281 | 0.1463 | 0.0183 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4941 | 0.9809 | 0.1246 | 0.9183 | 0.0744 | 0.0523 | 0.9256 | 0.0464 | 0.1195 | 0.1360 | 0.0165 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4937 | 0.9813 | 0.1465 | 0.9175 | 0.0867 | 0.0601 | 0.9133 | 0.0521 | 0.1298 | 0.1498 | 0.0200 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5133 | 0.9823 | 0.1180 | 0.9475 | 0.0700 | 0.0355 | 0.9300 | 0.0508 | 0.1056 | 0.1381 | 0.0325 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5153 | 0.9823 | 0.1255 | 0.9176 | 0.0734 | 0.0523 | 0.9266 | 0.0513 | 0.1102 | 0.1399 | 0.0296 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.5273 | 0.9823 | 0.1025 | 0.9475 | 0.0596 | 0.0354 | 0.9404 | 0.0503 | 0.0985 | 0.1337 | 0.0352 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5566 | 0.9846 | 0.1188 | 0.9778 | 0.0665 | 0.0308 | 0.9335 | 0.0716 | 0.0892 | 0.1477 | 0.0585 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5689 | 0.9851 | 0.1543 | 0.9674 | 0.0902 | 0.0277 | 0.9098 | 0.0707 | 0.0920 | 0.1753 | 0.0833 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6247 | 0.9871 | 0.2332 | 0.9612 | 0.1419 | 0.0324 | 0.8581 | 0.0854 | 0.0950 | 0.2280 | 0.1330 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5014 | 0.9817 | 0.1202 | 0.9712 | 0.0681 | 0.0554 | 0.9319 | 0.0566 | 0.1076 | 0.1149 | 0.0074 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5007 | 0.9822 | 0.1740 | 0.9827 | 0.1018 | 0.0769 | 0.8982 | 0.0582 | 0.1281 | 0.1463 | 0.0183 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4941 | 0.9809 | 0.1540 | 0.9453 | 0.0913 | 0.0739 | 0.9087 | 0.0464 | 0.1195 | 0.1360 | 0.0165 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4937 | 0.9813 | 0.1811 | 0.9212 | 0.1074 | 0.0863 | 0.8926 | 0.0521 | 0.1298 | 0.1498 | 0.0200 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5133 | 0.9823 | 0.1533 | 0.9441 | 0.0906 | 0.0616 | 0.9094 | 0.0508 | 0.1056 | 0.1381 | 0.0325 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5153 | 0.9823 | 0.1663 | 0.9661 | 0.0974 | 0.0784 | 0.9026 | 0.0513 | 0.1102 | 0.1399 | 0.0296 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.5273 | 0.9823 | 0.1266 | 0.9666 | 0.0744 | 0.0508 | 0.9256 | 0.0503 | 0.0985 | 0.1337 | 0.0352 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5566 | 0.9846 | 0.1684 | 0.9844 | 0.0983 | 0.0508 | 0.9017 | 0.0716 | 0.0892 | 0.1477 | 0.0585 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5689 | 0.9851 | 0.2025 | 0.9822 | 0.1237 | 0.0461 | 0.8763 | 0.0707 | 0.0920 | 0.1753 | 0.0833 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6247 | 0.9871 | 0.2988 | 0.9840 | 0.1901 | 0.0616 | 0.8099 | 0.0854 | 0.0950 | 0.2280 | 0.1330 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.5014 | 0.9817 | 0.1202 | 0.9712 | 0.0681 | 0.0554 | 0.9319 | 0.0566 | 0.1076 | 0.1149 | 0.0074 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.5007 | 0.9822 | 0.1740 | 0.9827 | 0.1018 | 0.0769 | 0.8982 | 0.0582 | 0.1281 | 0.1463 | 0.0183 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4941 | 0.9809 | 0.1540 | 0.9453 | 0.0913 | 0.0739 | 0.9087 | 0.0464 | 0.1195 | 0.1360 | 0.0165 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4937 | 0.9813 | 0.1811 | 0.9212 | 0.1074 | 0.0863 | 0.8926 | 0.0521 | 0.1298 | 0.1498 | 0.0200 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 5 | 0.5133 | 0.9823 | 0.1533 | 0.9441 | 0.0906 | 0.0616 | 0.9094 | 0.0508 | 0.1056 | 0.1381 | 0.0325 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 6 | 0.5153 | 0.9823 | 0.1663 | 0.9661 | 0.0974 | 0.0784 | 0.9026 | 0.0513 | 0.1102 | 0.1399 | 0.0296 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 7 | 0.5273 | 0.9823 | 0.1266 | 0.9666 | 0.0744 | 0.0508 | 0.9256 | 0.0503 | 0.0985 | 0.1337 | 0.0352 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 8 | 0.5566 | 0.9846 | 0.1684 | 0.9844 | 0.0983 | 0.0508 | 0.9017 | 0.0716 | 0.0892 | 0.1477 | 0.0585 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 9 | 0.5689 | 0.9851 | 0.2025 | 0.9822 | 0.1237 | 0.0461 | 0.8763 | 0.0707 | 0.0920 | 0.1753 | 0.0833 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 10 | 0.6247 | 0.9871 | 0.2988 | 0.9840 | 0.1901 | 0.0616 | 0.8099 | 0.0854 | 0.0950 | 0.2280 | 0.1330 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5000 | 0.9815 | 0.0348 | 0.9798 | 0.0178 | 0.0199 | 0.9822 | 0.0459 | 0.0935 | 0.0876 | -0.0059 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4987 | 0.9822 | 0.0758 | 0.8226 | 0.0407 | 0.0322 | 0.9593 | 0.0709 | 0.0984 | 0.1127 | 0.0143 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4875 | 0.9806 | 0.0767 | 0.9154 | 0.0429 | 0.0322 | 0.9571 | 0.0559 | 0.1048 | 0.1083 | 0.0035 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4865 | 0.9810 | 0.0846 | 0.9434 | 0.0464 | 0.0337 | 0.9536 | 0.0665 | 0.1072 | 0.1186 | 0.0113 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5097 | 0.9823 | 0.0887 | 0.9481 | 0.0527 | 0.0215 | 0.9473 | 0.0472 | 0.0853 | 0.1180 | 0.0327 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5074 | 0.9817 | 0.0808 | 0.5936 | 0.0496 | 0.0154 | 0.9504 | 0.0679 | 0.0752 | 0.1158 | 0.0406 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4998 | 0.9809 | 0.0877 | 0.6026 | 0.0517 | 0.0231 | 0.9483 | 0.0574 | 0.0852 | 0.1157 | 0.0305 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5116 | 0.9826 | 0.0743 | 0.7812 | 0.0440 | 0.0277 | 0.9560 | 0.0569 | 0.0825 | 0.1049 | 0.0225 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5068 | 0.9829 | 0.0660 | 0.4930 | 0.0387 | 0.0154 | 0.9613 | 0.0382 | 0.0735 | 0.0920 | 0.0185 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5579 | 0.9848 | 0.0785 | 0.6347 | 0.0492 | 0.0262 | 0.9508 | 0.0450 | 0.0840 | 0.1122 | 0.0282 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5000 | 0.9815 | 0.0659 | 0.9785 | 0.0344 | 0.0461 | 0.9656 | 0.0459 | 0.0935 | 0.0876 | -0.0059 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4987 | 0.9822 | 0.1291 | 0.9859 | 0.0733 | 0.0460 | 0.9267 | 0.0709 | 0.0984 | 0.1127 | 0.0143 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4875 | 0.9806 | 0.1173 | 0.9400 | 0.0662 | 0.0644 | 0.9338 | 0.0559 | 0.1048 | 0.1083 | 0.0035 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4865 | 0.9810 | 0.1289 | 0.9323 | 0.0722 | 0.0522 | 0.9278 | 0.0665 | 0.1072 | 0.1186 | 0.0113 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5097 | 0.9823 | 0.1213 | 0.9704 | 0.0723 | 0.0354 | 0.9277 | 0.0472 | 0.0853 | 0.1180 | 0.0327 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5074 | 0.9817 | 0.1321 | 0.8588 | 0.0879 | 0.0339 | 0.9121 | 0.0679 | 0.0752 | 0.1158 | 0.0406 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4998 | 0.9809 | 0.1179 | 0.7680 | 0.0708 | 0.0477 | 0.9292 | 0.0574 | 0.0852 | 0.1157 | 0.0305 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5116 | 0.9826 | 0.1152 | 0.8121 | 0.0704 | 0.0478 | 0.9296 | 0.0569 | 0.0825 | 0.1049 | 0.0225 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5068 | 0.9829 | 0.0907 | 0.4799 | 0.0563 | 0.0401 | 0.9437 | 0.0382 | 0.0735 | 0.0920 | 0.0185 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5579 | 0.9848 | 0.1066 | 0.7083 | 0.0679 | 0.0478 | 0.9321 | 0.0450 | 0.0840 | 0.1122 | 0.0282 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.5000 | 0.9815 | 0.0659 | 0.9785 | 0.0344 | 0.0461 | 0.9656 | 0.0459 | 0.0935 | 0.0876 | -0.0059 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.4987 | 0.9822 | 0.1291 | 0.9859 | 0.0733 | 0.0460 | 0.9267 | 0.0709 | 0.0984 | 0.1127 | 0.0143 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.4875 | 0.9806 | 0.1173 | 0.9400 | 0.0662 | 0.0644 | 0.9338 | 0.0559 | 0.1048 | 0.1083 | 0.0035 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 4 | 0.4865 | 0.9810 | 0.1289 | 0.9323 | 0.0722 | 0.0522 | 0.9278 | 0.0665 | 0.1072 | 0.1186 | 0.0113 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 5 | 0.5097 | 0.9823 | 0.1213 | 0.9704 | 0.0723 | 0.0354 | 0.9277 | 0.0472 | 0.0853 | 0.1180 | 0.0327 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 6 | 0.5074 | 0.9817 | 0.1321 | 0.8588 | 0.0879 | 0.0339 | 0.9121 | 0.0679 | 0.0752 | 0.1158 | 0.0406 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 7 | 0.4998 | 0.9809 | 0.1179 | 0.7680 | 0.0708 | 0.0477 | 0.9292 | 0.0574 | 0.0852 | 0.1157 | 0.0305 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 8 | 0.5116 | 0.9826 | 0.1152 | 0.8121 | 0.0704 | 0.0478 | 0.9296 | 0.0569 | 0.0825 | 0.1049 | 0.0225 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 9 | 0.5068 | 0.9829 | 0.0907 | 0.4799 | 0.0563 | 0.0401 | 0.9437 | 0.0382 | 0.0735 | 0.0920 | 0.0185 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 10 | 0.5579 | 0.9848 | 0.1066 | 0.7083 | 0.0679 | 0.0478 | 0.9321 | 0.0450 | 0.0840 | 0.1122 | 0.0282 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4860 | 0.9808 | 0.0598 | 0.9764 | 0.0317 | 0.0339 | 0.9683 | 0.0391 | 0.0999 | 0.0930 | -0.0069 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4904 | 0.9817 | 0.1271 | 0.9837 | 0.0733 | 0.0431 | 0.9267 | 0.0542 | 0.1172 | 0.1337 | 0.0165 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4637 | 0.9803 | 0.0758 | 0.8727 | 0.0408 | 0.0338 | 0.9592 | 0.0447 | 0.1128 | 0.1005 | -0.0123 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4610 | 0.9798 | 0.0631 | 0.8234 | 0.0331 | 0.0276 | 0.9669 | 0.0514 | 0.1024 | 0.0903 | -0.0121 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4654 | 0.9802 | 0.0675 | 0.9243 | 0.0357 | 0.0324 | 0.9643 | 0.0398 | 0.0943 | 0.0890 | -0.0053 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4822 | 0.9810 | 0.0568 | 0.9691 | 0.0298 | 0.0308 | 0.9702 | 0.0382 | 0.0868 | 0.0852 | -0.0016 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4789 | 0.9808 | 0.0536 | 0.9704 | 0.0281 | 0.0323 | 0.9719 | 0.0414 | 0.0905 | 0.0831 | -0.0073 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4951 | 0.9808 | 0.0589 | 0.9569 | 0.0318 | 0.0476 | 0.9682 | 0.0491 | 0.1083 | 0.0993 | -0.0090 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4692 | 0.9795 | 0.0651 | 0.9537 | 0.0356 | 0.0539 | 0.9644 | 0.0492 | 0.1172 | 0.1012 | -0.0160 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4712 | 0.9810 | 0.0717 | 0.9786 | 0.0396 | 0.0384 | 0.9604 | 0.0448 | 0.1068 | 0.1069 | 0.0000 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4860 | 0.9808 | 0.0847 | 0.9785 | 0.0457 | 0.0492 | 0.9543 | 0.0391 | 0.0999 | 0.0930 | -0.0069 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4904 | 0.9817 | 0.1644 | 0.9680 | 0.0967 | 0.0769 | 0.9033 | 0.0542 | 0.1172 | 0.1337 | 0.0165 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4637 | 0.9803 | 0.1003 | 0.8873 | 0.0549 | 0.0645 | 0.9451 | 0.0447 | 0.1128 | 0.1005 | -0.0123 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4610 | 0.9798 | 0.0986 | 0.8891 | 0.0530 | 0.0614 | 0.9470 | 0.0514 | 0.1024 | 0.0903 | -0.0121 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4654 | 0.9802 | 0.0975 | 0.9698 | 0.0524 | 0.0508 | 0.9476 | 0.0398 | 0.0943 | 0.0890 | -0.0053 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4822 | 0.9810 | 0.0857 | 0.9690 | 0.0458 | 0.0415 | 0.9542 | 0.0382 | 0.0868 | 0.0852 | -0.0016 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4789 | 0.9808 | 0.0782 | 0.9797 | 0.0416 | 0.0430 | 0.9584 | 0.0414 | 0.0905 | 0.0831 | -0.0073 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4951 | 0.9808 | 0.0848 | 0.9610 | 0.0463 | 0.0644 | 0.9537 | 0.0491 | 0.1083 | 0.0993 | -0.0090 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4692 | 0.9795 | 0.0906 | 0.9590 | 0.0507 | 0.0724 | 0.9493 | 0.0492 | 0.1172 | 0.1012 | -0.0160 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4712 | 0.9810 | 0.1036 | 0.9677 | 0.0583 | 0.0676 | 0.9417 | 0.0448 | 0.1068 | 0.1069 | 0.0000 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4860 | 0.9808 | 0.0847 | 0.9785 | 0.0457 | 0.0492 | 0.9543 | 0.0391 | 0.0999 | 0.0930 | -0.0069 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4904 | 0.9817 | 0.1644 | 0.9680 | 0.0967 | 0.0769 | 0.9033 | 0.0542 | 0.1172 | 0.1337 | 0.0165 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4637 | 0.9803 | 0.1003 | 0.8873 | 0.0549 | 0.0645 | 0.9451 | 0.0447 | 0.1128 | 0.1005 | -0.0123 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4610 | 0.9798 | 0.0986 | 0.8891 | 0.0530 | 0.0614 | 0.9470 | 0.0514 | 0.1024 | 0.0903 | -0.0121 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4654 | 0.9802 | 0.0975 | 0.9698 | 0.0524 | 0.0508 | 0.9476 | 0.0398 | 0.0943 | 0.0890 | -0.0053 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4822 | 0.9810 | 0.0857 | 0.9690 | 0.0458 | 0.0415 | 0.9542 | 0.0382 | 0.0868 | 0.0852 | -0.0016 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4789 | 0.9808 | 0.0782 | 0.9797 | 0.0416 | 0.0430 | 0.9584 | 0.0414 | 0.0905 | 0.0831 | -0.0073 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4951 | 0.9808 | 0.0848 | 0.9610 | 0.0463 | 0.0644 | 0.9537 | 0.0491 | 0.1083 | 0.0993 | -0.0090 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4692 | 0.9795 | 0.0906 | 0.9590 | 0.0507 | 0.0724 | 0.9493 | 0.0492 | 0.1172 | 0.1012 | -0.0160 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4712 | 0.9810 | 0.1036 | 0.9677 | 0.0583 | 0.0676 | 0.9417 | 0.0448 | 0.1068 | 0.1069 | 0.0000 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4828 | 0.9807 | 0.0278 | 0.9769 | 0.0141 | 0.0184 | 0.9859 | 0.0410 | 0.0896 | 0.0792 | -0.0105 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4765 | 0.9810 | 0.0536 | 0.9723 | 0.0278 | 0.0292 | 0.9722 | 0.0487 | 0.0959 | 0.0904 | -0.0055 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4584 | 0.9786 | 0.0313 | 0.8917 | 0.0160 | 0.0153 | 0.9840 | 0.0438 | 0.0946 | 0.0759 | -0.0187 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4580 | 0.9797 | 0.0293 | 0.6605 | 0.0150 | 0.0123 | 0.9850 | 0.0458 | 0.0874 | 0.0727 | -0.0147 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4671 | 0.9801 | 0.0339 | 0.9708 | 0.0174 | 0.0169 | 0.9826 | 0.0465 | 0.0807 | 0.0728 | -0.0079 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4800 | 0.9807 | 0.0313 | 0.9672 | 0.0160 | 0.0200 | 0.9840 | 0.0383 | 0.0713 | 0.0646 | -0.0067 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4669 | 0.9798 | 0.0294 | 0.9674 | 0.0150 | 0.0200 | 0.9850 | 0.0352 | 0.0794 | 0.0637 | -0.0157 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4540 | 0.9791 | 0.0235 | 0.9675 | 0.0119 | 0.0231 | 0.9881 | 0.0358 | 0.0795 | 0.0554 | -0.0241 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4194 | 0.9774 | 0.0212 | 0.9619 | 0.0107 | 0.0246 | 0.9893 | 0.0271 | 0.0874 | 0.0518 | -0.0356 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4211 | 0.9787 | 0.0152 | 0.9731 | 0.0077 | 0.0108 | 0.9923 | 0.0313 | 0.0694 | 0.0456 | -0.0238 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4828 | 0.9807 | 0.0616 | 0.9769 | 0.0320 | 0.0400 | 0.9680 | 0.0410 | 0.0896 | 0.0792 | -0.0105 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4765 | 0.9810 | 0.0956 | 0.9794 | 0.0506 | 0.0491 | 0.9494 | 0.0487 | 0.0959 | 0.0904 | -0.0055 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4584 | 0.9786 | 0.0575 | 0.9306 | 0.0300 | 0.0322 | 0.9700 | 0.0438 | 0.0946 | 0.0759 | -0.0187 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4580 | 0.9797 | 0.0668 | 0.9376 | 0.0351 | 0.0444 | 0.9649 | 0.0458 | 0.0874 | 0.0727 | -0.0147 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4671 | 0.9801 | 0.0695 | 0.9668 | 0.0366 | 0.0368 | 0.9634 | 0.0465 | 0.0807 | 0.0728 | -0.0079 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4800 | 0.9807 | 0.0546 | 0.9649 | 0.0285 | 0.0337 | 0.9715 | 0.0383 | 0.0713 | 0.0646 | -0.0067 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4669 | 0.9798 | 0.0470 | 0.9660 | 0.0242 | 0.0352 | 0.9758 | 0.0352 | 0.0794 | 0.0637 | -0.0157 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4540 | 0.9791 | 0.0481 | 0.9657 | 0.0249 | 0.0444 | 0.9751 | 0.0358 | 0.0795 | 0.0554 | -0.0241 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4194 | 0.9774 | 0.0393 | 0.9620 | 0.0201 | 0.0399 | 0.9799 | 0.0271 | 0.0874 | 0.0518 | -0.0356 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4211 | 0.9787 | 0.0354 | 0.9751 | 0.0181 | 0.0230 | 0.9819 | 0.0313 | 0.0694 | 0.0456 | -0.0238 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.4828 | 0.9807 | 0.0616 | 0.9769 | 0.0320 | 0.0400 | 0.9680 | 0.0410 | 0.0896 | 0.0792 | -0.0105 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.4765 | 0.9810 | 0.0956 | 0.9794 | 0.0506 | 0.0491 | 0.9494 | 0.0487 | 0.0959 | 0.0904 | -0.0055 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.4584 | 0.9786 | 0.0575 | 0.9306 | 0.0300 | 0.0322 | 0.9700 | 0.0438 | 0.0946 | 0.0759 | -0.0187 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.4580 | 0.9797 | 0.0668 | 0.9376 | 0.0351 | 0.0444 | 0.9649 | 0.0458 | 0.0874 | 0.0727 | -0.0147 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.4671 | 0.9801 | 0.0695 | 0.9668 | 0.0366 | 0.0368 | 0.9634 | 0.0465 | 0.0807 | 0.0728 | -0.0079 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.4800 | 0.9807 | 0.0546 | 0.9649 | 0.0285 | 0.0337 | 0.9715 | 0.0383 | 0.0713 | 0.0646 | -0.0067 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.4669 | 0.9798 | 0.0470 | 0.9660 | 0.0242 | 0.0352 | 0.9758 | 0.0352 | 0.0794 | 0.0637 | -0.0157 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.4540 | 0.9791 | 0.0481 | 0.9657 | 0.0249 | 0.0444 | 0.9751 | 0.0358 | 0.0795 | 0.0554 | -0.0241 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.4194 | 0.9774 | 0.0393 | 0.9620 | 0.0201 | 0.0399 | 0.9799 | 0.0271 | 0.0874 | 0.0518 | -0.0356 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.4211 | 0.9787 | 0.0354 | 0.9751 | 0.0181 | 0.0230 | 0.9819 | 0.0313 | 0.0694 | 0.0456 | -0.0238 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4932 | 0.9712 | 0.0393 | 0.9198 | 0.0202 | 0.0271 | 0.9798 | 0.0516 | 0.0969 | 0.0911 | -0.0057 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4968 | 0.9749 | 0.1344 | 0.9287 | 0.0766 | 0.0318 | 0.9234 | 0.0624 | 0.1059 | 0.1392 | 0.0333 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.5104 | 0.9725 | 0.0807 | 0.9038 | 0.0427 | 0.0318 | 0.9573 | 0.0563 | 0.1001 | 0.1067 | 0.0066 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4892 | 0.9714 | 0.0780 | 0.8936 | 0.0415 | 0.0423 | 0.9585 | 0.0619 | 0.1046 | 0.1000 | -0.0046 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5068 | 0.9720 | 0.0737 | 0.8819 | 0.0388 | 0.0512 | 0.9612 | 0.0590 | 0.1075 | 0.1059 | -0.0016 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4831 | 0.9696 | 0.0613 | 0.8850 | 0.0324 | 0.0588 | 0.9676 | 0.0525 | 0.1248 | 0.0995 | -0.0253 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4500 | 0.9689 | 0.0497 | 0.9308 | 0.0258 | 0.0347 | 0.9742 | 0.0457 | 0.1123 | 0.0859 | -0.0264 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4575 | 0.9707 | 0.0877 | 0.9484 | 0.0465 | 0.0423 | 0.9535 | 0.0539 | 0.1166 | 0.1062 | -0.0104 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4523 | 0.9710 | 0.0809 | 0.9623 | 0.0428 | 0.0287 | 0.9572 | 0.0505 | 0.1071 | 0.1016 | -0.0054 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4436 | 0.9704 | 0.0960 | 0.9637 | 0.0516 | 0.0347 | 0.9484 | 0.0473 | 0.1183 | 0.1097 | -0.0086 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4932 | 0.9712 | 0.0772 | 0.9412 | 0.0408 | 0.0376 | 0.9592 | 0.0516 | 0.0969 | 0.0911 | -0.0057 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4968 | 0.9749 | 0.1785 | 0.9381 | 0.1036 | 0.0514 | 0.8964 | 0.0624 | 0.1059 | 0.1392 | 0.0333 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.5104 | 0.9725 | 0.1227 | 0.8974 | 0.0670 | 0.0649 | 0.9330 | 0.0563 | 0.1001 | 0.1067 | 0.0066 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4892 | 0.9714 | 0.1184 | 0.8890 | 0.0646 | 0.0709 | 0.9354 | 0.0619 | 0.1046 | 0.1000 | -0.0046 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5068 | 0.9720 | 0.1234 | 0.8880 | 0.0681 | 0.0738 | 0.9319 | 0.0590 | 0.1075 | 0.1059 | -0.0016 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4831 | 0.9696 | 0.1060 | 0.9311 | 0.0573 | 0.0798 | 0.9427 | 0.0525 | 0.1248 | 0.0995 | -0.0253 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4500 | 0.9689 | 0.0824 | 0.9427 | 0.0438 | 0.0633 | 0.9562 | 0.0457 | 0.1123 | 0.0859 | -0.0264 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4575 | 0.9707 | 0.1302 | 0.9510 | 0.0704 | 0.0680 | 0.9296 | 0.0539 | 0.1166 | 0.1062 | -0.0104 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4523 | 0.9710 | 0.1172 | 0.9668 | 0.0630 | 0.0528 | 0.9370 | 0.0505 | 0.1071 | 0.1016 | -0.0054 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4436 | 0.9704 | 0.1410 | 0.9673 | 0.0775 | 0.0664 | 0.9225 | 0.0473 | 0.1183 | 0.1097 | -0.0086 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4932 | 0.9712 | 0.0772 | 0.9412 | 0.0408 | 0.0376 | 0.9592 | 0.0516 | 0.0969 | 0.0911 | -0.0057 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4968 | 0.9749 | 0.1785 | 0.9381 | 0.1036 | 0.0514 | 0.8964 | 0.0624 | 0.1059 | 0.1392 | 0.0333 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.5104 | 0.9725 | 0.1227 | 0.8974 | 0.0670 | 0.0649 | 0.9330 | 0.0563 | 0.1001 | 0.1067 | 0.0066 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4892 | 0.9714 | 0.1184 | 0.8890 | 0.0646 | 0.0709 | 0.9354 | 0.0619 | 0.1046 | 0.1000 | -0.0046 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.5068 | 0.9720 | 0.1234 | 0.8880 | 0.0681 | 0.0738 | 0.9319 | 0.0590 | 0.1075 | 0.1059 | -0.0016 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4831 | 0.9696 | 0.1060 | 0.9311 | 0.0573 | 0.0798 | 0.9427 | 0.0525 | 0.1248 | 0.0995 | -0.0253 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4500 | 0.9689 | 0.0824 | 0.9427 | 0.0438 | 0.0633 | 0.9562 | 0.0457 | 0.1123 | 0.0859 | -0.0264 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4575 | 0.9707 | 0.1302 | 0.9510 | 0.0704 | 0.0680 | 0.9296 | 0.0539 | 0.1166 | 0.1062 | -0.0104 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4523 | 0.9710 | 0.1172 | 0.9668 | 0.0630 | 0.0528 | 0.9370 | 0.0505 | 0.1071 | 0.1016 | -0.0054 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4436 | 0.9704 | 0.1410 | 0.9673 | 0.0775 | 0.0664 | 0.9225 | 0.0473 | 0.1183 | 0.1097 | -0.0086 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4880 | 0.9709 | 0.0201 | 0.9316 | 0.0102 | 0.0196 | 0.9898 | 0.0430 | 0.0860 | 0.0753 | -0.0108 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4975 | 0.9726 | 0.0468 | 0.9339 | 0.0242 | 0.0272 | 0.9758 | 0.0464 | 0.0943 | 0.0919 | -0.0024 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.5040 | 0.9722 | 0.0383 | 0.6608 | 0.0198 | 0.0167 | 0.9802 | 0.0455 | 0.0808 | 0.0822 | 0.0014 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4893 | 0.9712 | 0.0352 | 0.9397 | 0.0181 | 0.0212 | 0.9819 | 0.0628 | 0.0834 | 0.0796 | -0.0038 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5047 | 0.9715 | 0.0453 | 0.9432 | 0.0235 | 0.0271 | 0.9765 | 0.0411 | 0.0869 | 0.0797 | -0.0073 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4880 | 0.9690 | 0.0300 | 0.9278 | 0.0153 | 0.0317 | 0.9847 | 0.0464 | 0.0929 | 0.0704 | -0.0225 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4427 | 0.9684 | 0.0250 | 0.9260 | 0.0128 | 0.0196 | 0.9872 | 0.0442 | 0.0947 | 0.0654 | -0.0292 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4486 | 0.9702 | 0.0576 | 0.9578 | 0.0300 | 0.0166 | 0.9700 | 0.0479 | 0.0865 | 0.0801 | -0.0064 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4402 | 0.9702 | 0.0618 | 0.9641 | 0.0322 | 0.0181 | 0.9678 | 0.0460 | 0.0950 | 0.0825 | -0.0125 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4254 | 0.9695 | 0.0619 | 0.9683 | 0.0327 | 0.0166 | 0.9673 | 0.0426 | 0.0959 | 0.0820 | -0.0139 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4880 | 0.9709 | 0.0357 | 0.9524 | 0.0182 | 0.0317 | 0.9818 | 0.0430 | 0.0860 | 0.0753 | -0.0108 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4975 | 0.9726 | 0.0809 | 0.9453 | 0.0429 | 0.0424 | 0.9571 | 0.0464 | 0.0943 | 0.0919 | -0.0024 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.5040 | 0.9722 | 0.0740 | 0.9610 | 0.0389 | 0.0379 | 0.9611 | 0.0455 | 0.0808 | 0.0822 | 0.0014 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4893 | 0.9712 | 0.0869 | 0.9529 | 0.0462 | 0.0483 | 0.9538 | 0.0628 | 0.0834 | 0.0796 | -0.0038 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5047 | 0.9715 | 0.0750 | 0.9485 | 0.0394 | 0.0498 | 0.9606 | 0.0411 | 0.0869 | 0.0797 | -0.0073 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4880 | 0.9690 | 0.0670 | 0.9400 | 0.0349 | 0.0589 | 0.9651 | 0.0464 | 0.0929 | 0.0704 | -0.0225 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4427 | 0.9684 | 0.0513 | 0.9441 | 0.0264 | 0.0498 | 0.9736 | 0.0442 | 0.0947 | 0.0654 | -0.0292 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4486 | 0.9702 | 0.1052 | 0.9570 | 0.0562 | 0.0544 | 0.9438 | 0.0479 | 0.0865 | 0.0801 | -0.0064 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4402 | 0.9702 | 0.0963 | 0.9686 | 0.0510 | 0.0453 | 0.9490 | 0.0460 | 0.0950 | 0.0825 | -0.0125 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4254 | 0.9695 | 0.1048 | 0.9675 | 0.0566 | 0.0469 | 0.9434 | 0.0426 | 0.0959 | 0.0820 | -0.0139 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.4880 | 0.9709 | 0.0357 | 0.9524 | 0.0182 | 0.0317 | 0.9818 | 0.0430 | 0.0860 | 0.0753 | -0.0108 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.4975 | 0.9726 | 0.0809 | 0.9453 | 0.0429 | 0.0424 | 0.9571 | 0.0464 | 0.0943 | 0.0919 | -0.0024 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.5040 | 0.9722 | 0.0740 | 0.9610 | 0.0389 | 0.0379 | 0.9611 | 0.0455 | 0.0808 | 0.0822 | 0.0014 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.4893 | 0.9712 | 0.0869 | 0.9529 | 0.0462 | 0.0483 | 0.9538 | 0.0628 | 0.0834 | 0.0796 | -0.0038 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.5047 | 0.9715 | 0.0750 | 0.9485 | 0.0394 | 0.0498 | 0.9606 | 0.0411 | 0.0869 | 0.0797 | -0.0073 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.4880 | 0.9690 | 0.0670 | 0.9400 | 0.0349 | 0.0589 | 0.9651 | 0.0464 | 0.0929 | 0.0704 | -0.0225 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.4427 | 0.9684 | 0.0513 | 0.9441 | 0.0264 | 0.0498 | 0.9736 | 0.0442 | 0.0947 | 0.0654 | -0.0292 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.4486 | 0.9702 | 0.1052 | 0.9570 | 0.0562 | 0.0544 | 0.9438 | 0.0479 | 0.0865 | 0.0801 | -0.0064 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.4402 | 0.9702 | 0.0963 | 0.9686 | 0.0510 | 0.0453 | 0.9490 | 0.0460 | 0.0950 | 0.0825 | -0.0125 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.4254 | 0.9695 | 0.1048 | 0.9675 | 0.0566 | 0.0469 | 0.9434 | 0.0426 | 0.0959 | 0.0820 | -0.0139 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5157 | 0.9745 | 0.1245 | 0.9779 | 0.0677 | 0.0393 | 0.9323 | 0.0698 | 0.1094 | 0.1378 | 0.0284 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4794 | 0.9731 | 0.1285 | 0.9624 | 0.0743 | 0.0303 | 0.9257 | 0.0527 | 0.1195 | 0.1371 | 0.0176 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4853 | 0.9734 | 0.1607 | 0.9003 | 0.0940 | 0.0438 | 0.9060 | 0.0528 | 0.1183 | 0.1555 | 0.0372 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4825 | 0.9717 | 0.0781 | 0.9114 | 0.0419 | 0.0363 | 0.9581 | 0.0539 | 0.0982 | 0.1030 | 0.0048 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4936 | 0.9725 | 0.0747 | 0.9018 | 0.0396 | 0.0302 | 0.9604 | 0.0514 | 0.0906 | 0.0996 | 0.0090 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4969 | 0.9712 | 0.0659 | 0.9391 | 0.0345 | 0.0453 | 0.9655 | 0.0423 | 0.1069 | 0.0977 | -0.0092 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4884 | 0.9709 | 0.0677 | 0.9353 | 0.0364 | 0.0332 | 0.9636 | 0.0571 | 0.1043 | 0.0989 | -0.0054 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5451 | 0.9762 | 0.1291 | 0.9087 | 0.0765 | 0.0211 | 0.9235 | 0.0827 | 0.0873 | 0.1546 | 0.0673 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5440 | 0.9762 | 0.1322 | 0.9469 | 0.0760 | 0.0287 | 0.9240 | 0.0666 | 0.0831 | 0.1406 | 0.0575 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5542 | 0.9775 | 0.1209 | 0.9524 | 0.0668 | 0.0106 | 0.9332 | 0.0721 | 0.0756 | 0.1369 | 0.0613 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5157 | 0.9745 | 0.1658 | 0.9793 | 0.0927 | 0.0603 | 0.9073 | 0.0698 | 0.1094 | 0.1378 | 0.0284 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4794 | 0.9731 | 0.1652 | 0.9568 | 0.0970 | 0.0575 | 0.9030 | 0.0527 | 0.1195 | 0.1371 | 0.0176 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4853 | 0.9734 | 0.2034 | 0.9169 | 0.1206 | 0.0696 | 0.8794 | 0.0528 | 0.1183 | 0.1555 | 0.0372 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4825 | 0.9717 | 0.1140 | 0.9218 | 0.0622 | 0.0514 | 0.9378 | 0.0539 | 0.0982 | 0.1030 | 0.0048 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4936 | 0.9725 | 0.1106 | 0.9416 | 0.0596 | 0.0499 | 0.9404 | 0.0514 | 0.0906 | 0.0996 | 0.0090 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4969 | 0.9712 | 0.0939 | 0.9338 | 0.0498 | 0.0589 | 0.9502 | 0.0423 | 0.1069 | 0.0977 | -0.0092 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4884 | 0.9709 | 0.1055 | 0.9352 | 0.0588 | 0.0694 | 0.9412 | 0.0571 | 0.1043 | 0.0989 | -0.0054 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5451 | 0.9762 | 0.1924 | 0.9497 | 0.1169 | 0.0377 | 0.8831 | 0.0827 | 0.0873 | 0.1546 | 0.0673 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5440 | 0.9762 | 0.1846 | 0.9487 | 0.1091 | 0.0454 | 0.8909 | 0.0666 | 0.0831 | 0.1406 | 0.0575 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5542 | 0.9775 | 0.1757 | 0.9628 | 0.1022 | 0.0347 | 0.8978 | 0.0721 | 0.0756 | 0.1369 | 0.0613 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.5157 | 0.9745 | 0.1658 | 0.9793 | 0.0927 | 0.0603 | 0.9073 | 0.0698 | 0.1094 | 0.1378 | 0.0284 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4794 | 0.9731 | 0.1652 | 0.9568 | 0.0970 | 0.0575 | 0.9030 | 0.0527 | 0.1195 | 0.1371 | 0.0176 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4853 | 0.9734 | 0.2034 | 0.9169 | 0.1206 | 0.0696 | 0.8794 | 0.0528 | 0.1183 | 0.1555 | 0.0372 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4825 | 0.9717 | 0.1140 | 0.9218 | 0.0622 | 0.0514 | 0.9378 | 0.0539 | 0.0982 | 0.1030 | 0.0048 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4936 | 0.9725 | 0.1106 | 0.9416 | 0.0596 | 0.0499 | 0.9404 | 0.0514 | 0.0906 | 0.0996 | 0.0090 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4969 | 0.9712 | 0.0939 | 0.9338 | 0.0498 | 0.0589 | 0.9502 | 0.0423 | 0.1069 | 0.0977 | -0.0092 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4884 | 0.9709 | 0.1055 | 0.9352 | 0.0588 | 0.0694 | 0.9412 | 0.0571 | 0.1043 | 0.0989 | -0.0054 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 8 | 0.5451 | 0.9762 | 0.1924 | 0.9497 | 0.1169 | 0.0377 | 0.8831 | 0.0827 | 0.0873 | 0.1546 | 0.0673 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 9 | 0.5440 | 0.9762 | 0.1846 | 0.9487 | 0.1091 | 0.0454 | 0.8909 | 0.0666 | 0.0831 | 0.1406 | 0.0575 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 10 | 0.5542 | 0.9775 | 0.1757 | 0.9628 | 0.1022 | 0.0347 | 0.8978 | 0.0721 | 0.0756 | 0.1369 | 0.0613 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5095 | 0.9740 | 0.0791 | 0.7407 | 0.0444 | 0.0300 | 0.9556 | 0.0687 | 0.1020 | 0.1134 | 0.0114 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4734 | 0.9716 | 0.0766 | 0.8577 | 0.0426 | 0.0332 | 0.9574 | 0.0559 | 0.1129 | 0.1124 | -0.0005 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4846 | 0.9735 | 0.1157 | 0.8248 | 0.0662 | 0.0361 | 0.9338 | 0.0735 | 0.1175 | 0.1342 | 0.0167 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4835 | 0.9722 | 0.0596 | 0.6731 | 0.0328 | 0.0257 | 0.9672 | 0.0532 | 0.0881 | 0.0916 | 0.0036 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4919 | 0.9722 | 0.0594 | 0.9065 | 0.0316 | 0.0227 | 0.9684 | 0.0589 | 0.0799 | 0.0911 | 0.0112 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4955 | 0.9712 | 0.0394 | 0.5770 | 0.0210 | 0.0136 | 0.9790 | 0.0496 | 0.0729 | 0.0818 | 0.0089 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4904 | 0.9707 | 0.0560 | 0.5158 | 0.0337 | 0.0273 | 0.9663 | 0.0521 | 0.0792 | 0.0821 | 0.0029 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5524 | 0.9770 | 0.0940 | 0.4978 | 0.0643 | 0.0136 | 0.9357 | 0.0773 | 0.0753 | 0.1226 | 0.0473 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5394 | 0.9757 | 0.0829 | 0.8170 | 0.0478 | 0.0151 | 0.9522 | 0.0663 | 0.0727 | 0.1142 | 0.0415 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5519 | 0.9774 | 0.0634 | 0.7365 | 0.0359 | 0.0091 | 0.9641 | 0.0576 | 0.0608 | 0.0947 | 0.0338 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5095 | 0.9740 | 0.1201 | 0.8159 | 0.0699 | 0.0510 | 0.9301 | 0.0687 | 0.1020 | 0.1134 | 0.0114 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4734 | 0.9716 | 0.1181 | 0.8653 | 0.0659 | 0.0589 | 0.9341 | 0.0559 | 0.1129 | 0.1124 | -0.0005 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4846 | 0.9735 | 0.1682 | 0.9189 | 0.0994 | 0.0691 | 0.9006 | 0.0735 | 0.1175 | 0.1342 | 0.0167 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4835 | 0.9722 | 0.0917 | 0.8459 | 0.0518 | 0.0514 | 0.9482 | 0.0532 | 0.0881 | 0.0916 | 0.0036 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4919 | 0.9722 | 0.0996 | 0.9062 | 0.0548 | 0.0394 | 0.9452 | 0.0589 | 0.0799 | 0.0911 | 0.0112 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4955 | 0.9712 | 0.0709 | 0.5916 | 0.0395 | 0.0317 | 0.9605 | 0.0496 | 0.0729 | 0.0818 | 0.0089 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4904 | 0.9707 | 0.0849 | 0.6199 | 0.0551 | 0.0439 | 0.9449 | 0.0521 | 0.0792 | 0.0821 | 0.0029 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5524 | 0.9770 | 0.1497 | 0.4959 | 0.1059 | 0.0394 | 0.8941 | 0.0773 | 0.0753 | 0.1226 | 0.0473 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5394 | 0.9757 | 0.1353 | 0.8717 | 0.0823 | 0.0302 | 0.9177 | 0.0663 | 0.0727 | 0.1142 | 0.0415 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5519 | 0.9774 | 0.1118 | 0.7414 | 0.0683 | 0.0333 | 0.9317 | 0.0576 | 0.0608 | 0.0947 | 0.0338 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.5095 | 0.9740 | 0.1201 | 0.8159 | 0.0699 | 0.0510 | 0.9301 | 0.0687 | 0.1020 | 0.1134 | 0.0114 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.4734 | 0.9716 | 0.1181 | 0.8653 | 0.0659 | 0.0589 | 0.9341 | 0.0559 | 0.1129 | 0.1124 | -0.0005 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.4846 | 0.9735 | 0.1682 | 0.9189 | 0.0994 | 0.0691 | 0.9006 | 0.0735 | 0.1175 | 0.1342 | 0.0167 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 4 | 0.4835 | 0.9722 | 0.0917 | 0.8459 | 0.0518 | 0.0514 | 0.9482 | 0.0532 | 0.0881 | 0.0916 | 0.0036 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 5 | 0.4919 | 0.9722 | 0.0996 | 0.9062 | 0.0548 | 0.0394 | 0.9452 | 0.0589 | 0.0799 | 0.0911 | 0.0112 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 6 | 0.4955 | 0.9712 | 0.0709 | 0.5916 | 0.0395 | 0.0317 | 0.9605 | 0.0496 | 0.0729 | 0.0818 | 0.0089 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 7 | 0.4904 | 0.9707 | 0.0849 | 0.6199 | 0.0551 | 0.0439 | 0.9449 | 0.0521 | 0.0792 | 0.0821 | 0.0029 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 8 | 0.5524 | 0.9770 | 0.1497 | 0.4959 | 0.1059 | 0.0394 | 0.8941 | 0.0773 | 0.0753 | 0.1226 | 0.0473 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 9 | 0.5394 | 0.9757 | 0.1353 | 0.8717 | 0.0823 | 0.0302 | 0.9177 | 0.0663 | 0.0727 | 0.1142 | 0.0415 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 10 | 0.5519 | 0.9774 | 0.1118 | 0.7414 | 0.0683 | 0.0333 | 0.9317 | 0.0576 | 0.0608 | 0.0947 | 0.0338 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4930 | 0.9712 | 0.0469 | 0.9369 | 0.0242 | 0.0301 | 0.9758 | 0.0434 | 0.0970 | 0.0891 | -0.0079 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4962 | 0.9752 | 0.1324 | 0.9511 | 0.0765 | 0.0167 | 0.9235 | 0.0544 | 0.0957 | 0.1367 | 0.0410 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4929 | 0.9747 | 0.1398 | 0.9373 | 0.0826 | 0.0212 | 0.9174 | 0.0580 | 0.1037 | 0.1451 | 0.0414 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4969 | 0.9734 | 0.1505 | 0.9311 | 0.0866 | 0.0332 | 0.9134 | 0.0653 | 0.1162 | 0.1527 | 0.0364 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5066 | 0.9733 | 0.0818 | 0.9473 | 0.0432 | 0.0196 | 0.9568 | 0.0530 | 0.0867 | 0.1063 | 0.0196 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4980 | 0.9711 | 0.0549 | 0.9051 | 0.0286 | 0.0392 | 0.9714 | 0.0498 | 0.0995 | 0.0891 | -0.0105 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4606 | 0.9698 | 0.0555 | 0.9253 | 0.0290 | 0.0332 | 0.9710 | 0.0408 | 0.0974 | 0.0811 | -0.0163 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4536 | 0.9696 | 0.0590 | 0.9324 | 0.0309 | 0.0393 | 0.9691 | 0.0534 | 0.1114 | 0.0945 | -0.0169 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4640 | 0.9710 | 0.0962 | 0.9380 | 0.0526 | 0.0453 | 0.9474 | 0.0542 | 0.1175 | 0.1156 | -0.0019 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4650 | 0.9703 | 0.0795 | 0.9420 | 0.0422 | 0.0392 | 0.9578 | 0.0580 | 0.1191 | 0.1071 | -0.0120 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4930 | 0.9712 | 0.0716 | 0.9375 | 0.0375 | 0.0452 | 0.9625 | 0.0434 | 0.0970 | 0.0891 | -0.0079 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4962 | 0.9752 | 0.1739 | 0.9258 | 0.1015 | 0.0453 | 0.8985 | 0.0544 | 0.0957 | 0.1367 | 0.0410 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4929 | 0.9747 | 0.1783 | 0.9259 | 0.1048 | 0.0378 | 0.8952 | 0.0580 | 0.1037 | 0.1451 | 0.0414 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4969 | 0.9734 | 0.2068 | 0.9188 | 0.1211 | 0.0846 | 0.8789 | 0.0653 | 0.1162 | 0.1527 | 0.0364 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5066 | 0.9733 | 0.1273 | 0.9276 | 0.0693 | 0.0452 | 0.9307 | 0.0530 | 0.0867 | 0.1063 | 0.0196 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4980 | 0.9711 | 0.0967 | 0.9099 | 0.0516 | 0.0649 | 0.9484 | 0.0498 | 0.0995 | 0.0891 | -0.0105 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4606 | 0.9698 | 0.0855 | 0.9348 | 0.0454 | 0.0512 | 0.9546 | 0.0408 | 0.0974 | 0.0811 | -0.0163 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4536 | 0.9696 | 0.1061 | 0.9481 | 0.0572 | 0.0544 | 0.9428 | 0.0534 | 0.1114 | 0.0945 | -0.0169 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4640 | 0.9710 | 0.1426 | 0.9488 | 0.0790 | 0.0723 | 0.9210 | 0.0542 | 0.1175 | 0.1156 | -0.0019 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4650 | 0.9703 | 0.1307 | 0.9399 | 0.0714 | 0.0754 | 0.9286 | 0.0580 | 0.1191 | 0.1071 | -0.0120 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4930 | 0.9712 | 0.0716 | 0.9375 | 0.0375 | 0.0452 | 0.9625 | 0.0434 | 0.0970 | 0.0891 | -0.0079 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4962 | 0.9752 | 0.1739 | 0.9258 | 0.1015 | 0.0453 | 0.8985 | 0.0544 | 0.0957 | 0.1367 | 0.0410 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4929 | 0.9747 | 0.1783 | 0.9259 | 0.1048 | 0.0378 | 0.8952 | 0.0580 | 0.1037 | 0.1451 | 0.0414 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4969 | 0.9734 | 0.2068 | 0.9188 | 0.1211 | 0.0846 | 0.8789 | 0.0653 | 0.1162 | 0.1527 | 0.0364 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.5066 | 0.9733 | 0.1273 | 0.9276 | 0.0693 | 0.0452 | 0.9307 | 0.0530 | 0.0867 | 0.1063 | 0.0196 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4980 | 0.9711 | 0.0967 | 0.9099 | 0.0516 | 0.0649 | 0.9484 | 0.0498 | 0.0995 | 0.0891 | -0.0105 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4606 | 0.9698 | 0.0855 | 0.9348 | 0.0454 | 0.0512 | 0.9546 | 0.0408 | 0.0974 | 0.0811 | -0.0163 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4536 | 0.9696 | 0.1061 | 0.9481 | 0.0572 | 0.0544 | 0.9428 | 0.0534 | 0.1114 | 0.0945 | -0.0169 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4640 | 0.9710 | 0.1426 | 0.9488 | 0.0790 | 0.0723 | 0.9210 | 0.0542 | 0.1175 | 0.1156 | -0.0019 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4650 | 0.9703 | 0.1307 | 0.9399 | 0.0714 | 0.0754 | 0.9286 | 0.0580 | 0.1191 | 0.1071 | -0.0120 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4911 | 0.9708 | 0.0297 | 0.9568 | 0.0152 | 0.0212 | 0.9848 | 0.0382 | 0.0914 | 0.0795 | -0.0119 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4994 | 0.9730 | 0.0271 | 0.9416 | 0.0138 | 0.0121 | 0.9862 | 0.0518 | 0.0856 | 0.0849 | -0.0006 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4945 | 0.9727 | 0.0328 | 0.9310 | 0.0170 | 0.0182 | 0.9830 | 0.0550 | 0.0897 | 0.0878 | -0.0019 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4889 | 0.9726 | 0.0490 | 0.9395 | 0.0254 | 0.0256 | 0.9746 | 0.0679 | 0.0987 | 0.0959 | -0.0028 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5022 | 0.9729 | 0.0501 | 0.9508 | 0.0261 | 0.0227 | 0.9739 | 0.0479 | 0.0806 | 0.0822 | 0.0016 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4881 | 0.9702 | 0.0257 | 0.9095 | 0.0131 | 0.0377 | 0.9869 | 0.0483 | 0.0922 | 0.0698 | -0.0223 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4491 | 0.9689 | 0.0368 | 0.9486 | 0.0189 | 0.0257 | 0.9811 | 0.0383 | 0.0873 | 0.0616 | -0.0256 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4461 | 0.9694 | 0.0369 | 0.9532 | 0.0189 | 0.0257 | 0.9811 | 0.0512 | 0.0997 | 0.0764 | -0.0233 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4558 | 0.9710 | 0.0676 | 0.9553 | 0.0356 | 0.0302 | 0.9644 | 0.0518 | 0.0993 | 0.0901 | -0.0092 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4450 | 0.9699 | 0.0532 | 0.9565 | 0.0276 | 0.0272 | 0.9724 | 0.0483 | 0.1073 | 0.0830 | -0.0243 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4911 | 0.9708 | 0.0539 | 0.9599 | 0.0278 | 0.0423 | 0.9722 | 0.0382 | 0.0914 | 0.0795 | -0.0119 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4994 | 0.9730 | 0.0757 | 0.9336 | 0.0400 | 0.0453 | 0.9600 | 0.0518 | 0.0856 | 0.0849 | -0.0006 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4945 | 0.9727 | 0.0682 | 0.9287 | 0.0365 | 0.0393 | 0.9635 | 0.0550 | 0.0897 | 0.0878 | -0.0019 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4889 | 0.9726 | 0.0922 | 0.9468 | 0.0493 | 0.0588 | 0.9507 | 0.0679 | 0.0987 | 0.0959 | -0.0028 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5022 | 0.9729 | 0.0792 | 0.9671 | 0.0418 | 0.0393 | 0.9582 | 0.0479 | 0.0806 | 0.0822 | 0.0016 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4881 | 0.9702 | 0.0631 | 0.9514 | 0.0327 | 0.0589 | 0.9673 | 0.0483 | 0.0922 | 0.0698 | -0.0223 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4491 | 0.9689 | 0.0660 | 0.9555 | 0.0343 | 0.0529 | 0.9657 | 0.0383 | 0.0873 | 0.0616 | -0.0256 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4461 | 0.9694 | 0.0772 | 0.9515 | 0.0407 | 0.0529 | 0.9593 | 0.0512 | 0.0997 | 0.0764 | -0.0233 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4558 | 0.9710 | 0.1135 | 0.9573 | 0.0615 | 0.0650 | 0.9385 | 0.0518 | 0.0993 | 0.0901 | -0.0092 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4450 | 0.9699 | 0.0910 | 0.9547 | 0.0482 | 0.0665 | 0.9518 | 0.0483 | 0.1073 | 0.0830 | -0.0243 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.4911 | 0.9708 | 0.0539 | 0.9599 | 0.0278 | 0.0423 | 0.9722 | 0.0382 | 0.0914 | 0.0795 | -0.0119 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.4994 | 0.9730 | 0.0757 | 0.9336 | 0.0400 | 0.0453 | 0.9600 | 0.0518 | 0.0856 | 0.0849 | -0.0006 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.4945 | 0.9727 | 0.0682 | 0.9287 | 0.0365 | 0.0393 | 0.9635 | 0.0550 | 0.0897 | 0.0878 | -0.0019 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.4889 | 0.9726 | 0.0922 | 0.9468 | 0.0493 | 0.0588 | 0.9507 | 0.0679 | 0.0987 | 0.0959 | -0.0028 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.5022 | 0.9729 | 0.0792 | 0.9671 | 0.0418 | 0.0393 | 0.9582 | 0.0479 | 0.0806 | 0.0822 | 0.0016 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.4881 | 0.9702 | 0.0631 | 0.9514 | 0.0327 | 0.0589 | 0.9673 | 0.0483 | 0.0922 | 0.0698 | -0.0223 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.4491 | 0.9689 | 0.0660 | 0.9555 | 0.0343 | 0.0529 | 0.9657 | 0.0383 | 0.0873 | 0.0616 | -0.0256 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.4461 | 0.9694 | 0.0772 | 0.9515 | 0.0407 | 0.0529 | 0.9593 | 0.0512 | 0.0997 | 0.0764 | -0.0233 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.4558 | 0.9710 | 0.1135 | 0.9573 | 0.0615 | 0.0650 | 0.9385 | 0.0518 | 0.0993 | 0.0901 | -0.0092 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.4450 | 0.9699 | 0.0910 | 0.9547 | 0.0482 | 0.0665 | 0.9518 | 0.0483 | 0.1073 | 0.0830 | -0.0243 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `fedavg` approximates collaborative normal-only training without sharing raw vibration windows.
- `fedprox` adds a proximal penalty to reduce local client drift under non-IID data.
- `fedbn` keeps BatchNorm parameters and running statistics local to each client.
- `*-personalized` locally adapts the federated global model before client evaluation.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
