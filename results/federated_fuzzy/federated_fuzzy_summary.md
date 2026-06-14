# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, FedAvg, FedProx, and personalized federated training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| clients_by | training_mode | calibration_scope | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.5122 | 0.9816 | 0.0826 | 0.9010 | 0.0456 | 0.0200 | 0.9544 | 0.0600 | 0.0797 | 0.1073 | 0.0276 |
| bearing | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5122 | 0.9816 | 0.1266 | 0.8932 | 0.0727 | 0.0430 | 0.9273 | 0.0600 | 0.0797 | 0.1073 | 0.0276 |
| bearing | centralized | adaptive | cnn-ae | hard_val_p95 | 0.5122 | 0.9816 | 0.1266 | 0.8932 | 0.0727 | 0.0430 | 0.9273 | 0.0600 | 0.0797 | 0.1073 | 0.0276 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5246 | 0.9819 | 0.1734 | 0.8985 | 0.1066 | 0.0353 | 0.8934 | 0.0545 | 0.1002 | 0.1622 | 0.0620 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5246 | 0.9819 | 0.2055 | 0.8891 | 0.1296 | 0.0644 | 0.8704 | 0.0545 | 0.1002 | 0.1622 | 0.0620 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.5246 | 0.9819 | 0.2055 | 0.8891 | 0.1296 | 0.0644 | 0.8704 | 0.0545 | 0.1002 | 0.1622 | 0.0620 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4833 | 0.9807 | 0.0410 | 0.9134 | 0.0211 | 0.0215 | 0.9789 | 0.0534 | 0.0795 | 0.0755 | -0.0040 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4833 | 0.9807 | 0.0799 | 0.9227 | 0.0424 | 0.0476 | 0.9576 | 0.0534 | 0.0795 | 0.0755 | -0.0040 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.4833 | 0.9807 | 0.0799 | 0.9227 | 0.0424 | 0.0476 | 0.9576 | 0.0534 | 0.0795 | 0.0755 | -0.0040 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.4826 | 0.9808 | 0.0458 | 0.9359 | 0.0244 | 0.0221 | 0.9756 | 0.0503 | 0.0889 | 0.0870 | -0.0020 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4826 | 0.9808 | 0.0807 | 0.9518 | 0.0435 | 0.0446 | 0.9565 | 0.0503 | 0.0889 | 0.0870 | -0.0020 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.4826 | 0.9808 | 0.0807 | 0.9518 | 0.0435 | 0.0446 | 0.9565 | 0.0503 | 0.0889 | 0.0870 | -0.0020 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4866 | 0.9810 | 0.0887 | 0.9206 | 0.0502 | 0.0420 | 0.9498 | 0.0508 | 0.1114 | 0.1142 | 0.0029 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4866 | 0.9810 | 0.1238 | 0.9307 | 0.0708 | 0.0695 | 0.9292 | 0.0508 | 0.1114 | 0.1142 | 0.0029 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.4866 | 0.9810 | 0.1238 | 0.9307 | 0.0708 | 0.0695 | 0.9292 | 0.0508 | 0.1114 | 0.1142 | 0.0029 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4648 | 0.9800 | 0.0266 | 0.9495 | 0.0136 | 0.0217 | 0.9864 | 0.0449 | 0.0850 | 0.0692 | -0.0158 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4648 | 0.9800 | 0.0599 | 0.9725 | 0.0310 | 0.0414 | 0.9690 | 0.0449 | 0.0850 | 0.0692 | -0.0158 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.4648 | 0.9800 | 0.0599 | 0.9725 | 0.0310 | 0.0414 | 0.9690 | 0.0449 | 0.0850 | 0.0692 | -0.0158 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5573 | 0.9849 | 0.2072 | 0.9791 | 0.1248 | 0.0385 | 0.8752 | 0.0677 | 0.1142 | 0.2060 | 0.0919 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5573 | 0.9849 | 0.2517 | 0.9861 | 0.1560 | 0.0708 | 0.8440 | 0.0677 | 0.1142 | 0.2060 | 0.0919 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5573 | 0.9849 | 0.2517 | 0.9861 | 0.1560 | 0.0708 | 0.8440 | 0.0677 | 0.1142 | 0.2060 | 0.0919 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.5221 | 0.9828 | 0.0747 | 0.8848 | 0.0422 | 0.0252 | 0.9578 | 0.0550 | 0.0860 | 0.1135 | 0.0275 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5221 | 0.9828 | 0.1129 | 0.9256 | 0.0650 | 0.0422 | 0.9350 | 0.0550 | 0.0860 | 0.1135 | 0.0275 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.5221 | 0.9828 | 0.1129 | 0.9256 | 0.0650 | 0.0422 | 0.9350 | 0.0550 | 0.0860 | 0.1135 | 0.0275 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5261 | 0.9829 | 0.1290 | 0.9478 | 0.0747 | 0.0430 | 0.9253 | 0.0593 | 0.1068 | 0.1451 | 0.0383 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5261 | 0.9829 | 0.1680 | 0.9585 | 0.0989 | 0.0642 | 0.9011 | 0.0593 | 0.1068 | 0.1451 | 0.0383 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.5261 | 0.9829 | 0.1680 | 0.9585 | 0.0989 | 0.0642 | 0.9011 | 0.0593 | 0.1068 | 0.1451 | 0.0383 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.5037 | 0.9819 | 0.0623 | 0.7690 | 0.0353 | 0.0234 | 0.9647 | 0.0598 | 0.0878 | 0.1019 | 0.0141 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5037 | 0.9819 | 0.1044 | 0.8293 | 0.0615 | 0.0457 | 0.9385 | 0.0598 | 0.0878 | 0.1019 | 0.0141 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.5037 | 0.9819 | 0.1044 | 0.8293 | 0.0615 | 0.0457 | 0.9385 | 0.0598 | 0.0878 | 0.1019 | 0.0141 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6025 | 0.9864 | 0.2152 | 0.9885 | 0.1319 | 0.0293 | 0.8681 | 0.0862 | 0.0917 | 0.2201 | 0.1285 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6025 | 0.9864 | 0.2785 | 0.9908 | 0.1759 | 0.0432 | 0.8241 | 0.0862 | 0.0917 | 0.2201 | 0.1285 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.6025 | 0.9864 | 0.2785 | 0.9908 | 0.1759 | 0.0432 | 0.8241 | 0.0862 | 0.0917 | 0.2201 | 0.1285 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.4718 | 0.9803 | 0.0470 | 0.9409 | 0.0245 | 0.0240 | 0.9755 | 0.0440 | 0.0903 | 0.0832 | -0.0071 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4718 | 0.9803 | 0.0763 | 0.9552 | 0.0406 | 0.0444 | 0.9594 | 0.0440 | 0.0903 | 0.0832 | -0.0071 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.4718 | 0.9803 | 0.0763 | 0.9552 | 0.0406 | 0.0444 | 0.9594 | 0.0440 | 0.0903 | 0.0832 | -0.0071 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4781 | 0.9806 | 0.0727 | 0.9476 | 0.0394 | 0.0377 | 0.9606 | 0.0452 | 0.1052 | 0.0993 | -0.0059 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4781 | 0.9806 | 0.1014 | 0.9504 | 0.0559 | 0.0607 | 0.9441 | 0.0452 | 0.1052 | 0.0993 | -0.0059 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.4781 | 0.9806 | 0.1014 | 0.9504 | 0.0559 | 0.0607 | 0.9441 | 0.0452 | 0.1052 | 0.0993 | -0.0059 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4588 | 0.9796 | 0.0348 | 0.9323 | 0.0179 | 0.0214 | 0.9821 | 0.0385 | 0.0865 | 0.0701 | -0.0164 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4588 | 0.9796 | 0.0638 | 0.9653 | 0.0334 | 0.0413 | 0.9666 | 0.0385 | 0.0865 | 0.0701 | -0.0164 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.4588 | 0.9796 | 0.0638 | 0.9653 | 0.0334 | 0.0413 | 0.9666 | 0.0385 | 0.0865 | 0.0701 | -0.0164 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5478 | 0.9842 | 0.1600 | 0.9754 | 0.0938 | 0.0322 | 0.9062 | 0.0711 | 0.1037 | 0.1732 | 0.0695 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5478 | 0.9842 | 0.2022 | 0.9734 | 0.1200 | 0.0629 | 0.8800 | 0.0711 | 0.1037 | 0.1732 | 0.0695 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5478 | 0.9842 | 0.2022 | 0.9734 | 0.1200 | 0.0629 | 0.8800 | 0.0711 | 0.1037 | 0.1732 | 0.0695 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5072 | 0.9807 | 0.0612 | 0.8989 | 0.0330 | 0.0338 | 0.9670 | 0.0508 | 0.1142 | 0.1123 | -0.0019 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5072 | 0.9807 | 0.0945 | 0.9412 | 0.0518 | 0.0707 | 0.9482 | 0.0508 | 0.1142 | 0.1123 | -0.0019 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.5072 | 0.9807 | 0.0945 | 0.9412 | 0.0518 | 0.0707 | 0.9482 | 0.0508 | 0.1142 | 0.1123 | -0.0019 |
| condition | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.4484 | 0.9722 | 0.0750 | 0.9167 | 0.0393 | 0.0317 | 0.9607 | 0.0390 | 0.0883 | 0.0868 | -0.0014 |
| condition | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4484 | 0.9722 | 0.1072 | 0.9160 | 0.0573 | 0.0437 | 0.9427 | 0.0390 | 0.0883 | 0.0868 | -0.0014 |
| condition | centralized | adaptive | cnn-ae | hard_val_p95 | 0.4484 | 0.9722 | 0.1072 | 0.9160 | 0.0573 | 0.0437 | 0.9427 | 0.0390 | 0.0883 | 0.0868 | -0.0014 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4530 | 0.9722 | 0.1477 | 0.9136 | 0.0837 | 0.0498 | 0.9163 | 0.0520 | 0.1193 | 0.1347 | 0.0155 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4530 | 0.9722 | 0.1870 | 0.9109 | 0.1087 | 0.0784 | 0.8913 | 0.0520 | 0.1193 | 0.1347 | 0.0155 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.4530 | 0.9722 | 0.1870 | 0.9109 | 0.1087 | 0.0784 | 0.8913 | 0.0520 | 0.1193 | 0.1347 | 0.0155 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4396 | 0.9700 | 0.0624 | 0.9237 | 0.0327 | 0.0392 | 0.9673 | 0.0368 | 0.1001 | 0.0781 | -0.0220 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4396 | 0.9700 | 0.0891 | 0.9356 | 0.0475 | 0.0634 | 0.9525 | 0.0368 | 0.1001 | 0.0781 | -0.0220 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.4396 | 0.9700 | 0.0891 | 0.9356 | 0.0475 | 0.0634 | 0.9525 | 0.0368 | 0.1001 | 0.0781 | -0.0220 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.4691 | 0.9702 | 0.0393 | 0.9308 | 0.0203 | 0.0225 | 0.9797 | 0.0463 | 0.0919 | 0.0784 | -0.0135 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4691 | 0.9702 | 0.0765 | 0.9324 | 0.0405 | 0.0489 | 0.9595 | 0.0463 | 0.0919 | 0.0784 | -0.0135 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.4691 | 0.9702 | 0.0765 | 0.9324 | 0.0405 | 0.0489 | 0.9595 | 0.0463 | 0.0919 | 0.0784 | -0.0135 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4729 | 0.9705 | 0.0696 | 0.9237 | 0.0373 | 0.0406 | 0.9627 | 0.0494 | 0.1113 | 0.0991 | -0.0123 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4729 | 0.9705 | 0.1085 | 0.9308 | 0.0591 | 0.0661 | 0.9409 | 0.0494 | 0.1113 | 0.0991 | -0.0123 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.4729 | 0.9705 | 0.1085 | 0.9308 | 0.0591 | 0.0661 | 0.9409 | 0.0494 | 0.1113 | 0.0991 | -0.0123 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4674 | 0.9698 | 0.0333 | 0.9360 | 0.0171 | 0.0248 | 0.9829 | 0.0456 | 0.0929 | 0.0744 | -0.0185 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4674 | 0.9698 | 0.0712 | 0.9491 | 0.0374 | 0.0525 | 0.9626 | 0.0456 | 0.0929 | 0.0744 | -0.0185 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.4674 | 0.9698 | 0.0712 | 0.9491 | 0.0374 | 0.0525 | 0.9626 | 0.0456 | 0.0929 | 0.0744 | -0.0185 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4709 | 0.9721 | 0.1154 | 0.9539 | 0.0634 | 0.0407 | 0.9366 | 0.0459 | 0.1114 | 0.1216 | 0.0101 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4709 | 0.9721 | 0.1559 | 0.9536 | 0.0873 | 0.0649 | 0.9127 | 0.0459 | 0.1114 | 0.1216 | 0.0101 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.4709 | 0.9721 | 0.1559 | 0.9536 | 0.0873 | 0.0649 | 0.9127 | 0.0459 | 0.1114 | 0.1216 | 0.0101 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.5062 | 0.9737 | 0.0777 | 0.8350 | 0.0444 | 0.0199 | 0.9556 | 0.0520 | 0.0805 | 0.1033 | 0.0227 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5062 | 0.9737 | 0.1151 | 0.8784 | 0.0668 | 0.0392 | 0.9332 | 0.0520 | 0.0805 | 0.1033 | 0.0227 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.5062 | 0.9737 | 0.1151 | 0.8784 | 0.0668 | 0.0392 | 0.9332 | 0.0520 | 0.0805 | 0.1033 | 0.0227 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5071 | 0.9737 | 0.1173 | 0.9316 | 0.0668 | 0.0359 | 0.9332 | 0.0611 | 0.1011 | 0.1305 | 0.0294 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5071 | 0.9737 | 0.1591 | 0.9411 | 0.0926 | 0.0572 | 0.9074 | 0.0611 | 0.1011 | 0.1305 | 0.0294 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.5071 | 0.9737 | 0.1591 | 0.9411 | 0.0926 | 0.0572 | 0.9074 | 0.0611 | 0.1011 | 0.1305 | 0.0294 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.5066 | 0.9734 | 0.0766 | 0.7540 | 0.0451 | 0.0220 | 0.9549 | 0.0618 | 0.0894 | 0.1077 | 0.0183 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5066 | 0.9734 | 0.1186 | 0.7923 | 0.0714 | 0.0472 | 0.9286 | 0.0618 | 0.0894 | 0.1077 | 0.0183 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.5066 | 0.9734 | 0.1186 | 0.7923 | 0.0714 | 0.0472 | 0.9286 | 0.0618 | 0.0894 | 0.1077 | 0.0183 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.5237 | 0.9755 | 0.1233 | 0.9327 | 0.0684 | 0.0212 | 0.9316 | 0.0672 | 0.0900 | 0.1368 | 0.0467 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5237 | 0.9755 | 0.1686 | 0.9760 | 0.0954 | 0.0498 | 0.9046 | 0.0672 | 0.0900 | 0.1368 | 0.0467 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.5237 | 0.9755 | 0.1686 | 0.9760 | 0.0954 | 0.0498 | 0.9046 | 0.0672 | 0.0900 | 0.1368 | 0.0467 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.4823 | 0.9719 | 0.0485 | 0.9303 | 0.0254 | 0.0208 | 0.9746 | 0.0489 | 0.0857 | 0.0858 | 0.0000 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4823 | 0.9719 | 0.0849 | 0.9377 | 0.0453 | 0.0396 | 0.9547 | 0.0489 | 0.0857 | 0.0858 | 0.0000 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.4823 | 0.9719 | 0.0849 | 0.9377 | 0.0453 | 0.0396 | 0.9547 | 0.0489 | 0.0857 | 0.0858 | 0.0000 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4855 | 0.9722 | 0.0897 | 0.9288 | 0.0499 | 0.0273 | 0.9501 | 0.0492 | 0.0984 | 0.1100 | 0.0116 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4855 | 0.9722 | 0.1251 | 0.9314 | 0.0702 | 0.0494 | 0.9298 | 0.0492 | 0.0984 | 0.1100 | 0.0116 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.4855 | 0.9722 | 0.1251 | 0.9314 | 0.0702 | 0.0494 | 0.9298 | 0.0492 | 0.0984 | 0.1100 | 0.0116 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.4790 | 0.9712 | 0.0386 | 0.9364 | 0.0199 | 0.0248 | 0.9801 | 0.0464 | 0.0904 | 0.0794 | -0.0110 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4790 | 0.9712 | 0.0735 | 0.9495 | 0.0388 | 0.0488 | 0.9612 | 0.0464 | 0.0904 | 0.0794 | -0.0110 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.4790 | 0.9712 | 0.0735 | 0.9495 | 0.0388 | 0.0488 | 0.9612 | 0.0464 | 0.0904 | 0.0794 | -0.0110 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4979 | 0.9739 | 0.1442 | 0.9305 | 0.0810 | 0.0423 | 0.9190 | 0.0442 | 0.1025 | 0.1401 | 0.0376 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4979 | 0.9739 | 0.1811 | 0.9453 | 0.1037 | 0.0529 | 0.8963 | 0.0442 | 0.1025 | 0.1401 | 0.0376 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.4979 | 0.9739 | 0.1811 | 0.9453 | 0.1037 | 0.0529 | 0.8963 | 0.0442 | 0.1025 | 0.1401 | 0.0376 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.4326 | 0.9679 | 0.0854 | 0.9603 | 0.0461 | 0.0452 | 0.9539 | 0.0309 | 0.1199 | 0.0960 | -0.0239 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.4326 | 0.9679 | 0.1075 | 0.9583 | 0.0586 | 0.0558 | 0.9414 | 0.0309 | 0.1199 | 0.0960 | -0.0239 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.4326 | 0.9679 | 0.1075 | 0.9583 | 0.0586 | 0.0558 | 0.9414 | 0.0309 | 0.1199 | 0.0960 | -0.0239 |

## Client Stability

| clients_by | training_mode | calibration_scope | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0258 | 0.0544 | 0.0428 | 0.0922 |
| bearing | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0452 | 0.0804 | 0.0428 | 0.0922 |
| bearing | centralized | adaptive | cnn-ae | hard_val_p95 | 0.0452 | 0.0804 | 0.0428 | 0.0922 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0302 | 0.1195 | 0.0426 | 0.1395 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0376 | 0.1380 | 0.0426 | 0.1395 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0376 | 0.1380 | 0.0426 | 0.1395 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0252 | 0.0146 | 0.0327 | 0.0624 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0493 | 0.0282 | 0.0327 | 0.0624 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.0493 | 0.0282 | 0.0327 | 0.0624 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0284 | 0.0331 | 0.0270 | 0.0683 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0441 | 0.0403 | 0.0270 | 0.0683 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.0441 | 0.0403 | 0.0270 | 0.0683 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0440 | 0.0669 | 0.0348 | 0.0783 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0620 | 0.0756 | 0.0348 | 0.0783 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0620 | 0.0756 | 0.0348 | 0.0783 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0291 | 0.0089 | 0.0155 | 0.0493 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0436 | 0.0127 | 0.0155 | 0.0493 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0436 | 0.0127 | 0.0155 | 0.0493 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0426 | 0.1047 | 0.0338 | 0.1058 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0595 | 0.1214 | 0.0338 | 0.1058 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0595 | 0.1214 | 0.0338 | 0.1058 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0331 | 0.0637 | 0.0464 | 0.0790 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0489 | 0.0786 | 0.0464 | 0.0790 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.0489 | 0.0786 | 0.0464 | 0.0790 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0474 | 0.0833 | 0.0422 | 0.0966 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0593 | 0.0928 | 0.0422 | 0.0966 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0593 | 0.0928 | 0.0422 | 0.0966 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0401 | 0.0609 | 0.0661 | 0.0709 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0651 | 0.0899 | 0.0661 | 0.0709 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.0651 | 0.0899 | 0.0661 | 0.0709 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0300 | 0.1179 | 0.0807 | 0.1260 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0395 | 0.1302 | 0.0807 | 0.1260 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0395 | 0.1302 | 0.0807 | 0.1260 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0239 | 0.0226 | 0.0309 | 0.0567 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0352 | 0.0322 | 0.0309 | 0.0567 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.0352 | 0.0322 | 0.0309 | 0.0567 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0376 | 0.0434 | 0.0387 | 0.0609 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0490 | 0.0537 | 0.0387 | 0.0609 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0490 | 0.0537 | 0.0387 | 0.0609 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0237 | 0.0127 | 0.0194 | 0.0456 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0360 | 0.0209 | 0.0194 | 0.0456 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0360 | 0.0209 | 0.0194 | 0.0456 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0308 | 0.0891 | 0.0473 | 0.0995 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0456 | 0.0933 | 0.0473 | 0.0995 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0456 | 0.0933 | 0.0473 | 0.0995 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0386 | 0.0400 | 0.0340 | 0.0741 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0427 | 0.0492 | 0.0340 | 0.0741 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0427 | 0.0492 | 0.0340 | 0.0741 |
| condition | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0297 | 0.0182 | 0.0161 | 0.0594 |
| condition | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0342 | 0.0253 | 0.0161 | 0.0594 |
| condition | centralized | adaptive | cnn-ae | hard_val_p95 | 0.0342 | 0.0253 | 0.0161 | 0.0594 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0287 | 0.0665 | 0.0217 | 0.0750 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0407 | 0.0787 | 0.0217 | 0.0750 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0407 | 0.0787 | 0.0217 | 0.0750 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0418 | 0.0225 | 0.0139 | 0.0968 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0958 | 0.0300 | 0.0139 | 0.0968 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.0958 | 0.0300 | 0.0139 | 0.0968 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0214 | 0.0167 | 0.0205 | 0.0398 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0325 | 0.0259 | 0.0205 | 0.0398 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.0325 | 0.0259 | 0.0205 | 0.0398 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0389 | 0.0369 | 0.0244 | 0.0425 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0446 | 0.0424 | 0.0244 | 0.0425 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0446 | 0.0424 | 0.0244 | 0.0425 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0299 | 0.0121 | 0.0183 | 0.0483 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0450 | 0.0210 | 0.0183 | 0.0483 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0450 | 0.0210 | 0.0183 | 0.0483 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0315 | 0.0480 | 0.0332 | 0.0565 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0403 | 0.0531 | 0.0332 | 0.0565 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0403 | 0.0531 | 0.0332 | 0.0565 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0294 | 0.0692 | 0.0442 | 0.0694 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0439 | 0.0839 | 0.0442 | 0.0694 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.0439 | 0.0839 | 0.0442 | 0.0694 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0377 | 0.0726 | 0.0408 | 0.0781 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0471 | 0.0854 | 0.0408 | 0.0781 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0471 | 0.0854 | 0.0408 | 0.0781 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0325 | 0.0828 | 0.0721 | 0.0643 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0572 | 0.1039 | 0.0721 | 0.0643 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.0572 | 0.1039 | 0.0721 | 0.0643 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0208 | 0.0540 | 0.0325 | 0.0656 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0443 | 0.0601 | 0.0325 | 0.0656 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0443 | 0.0601 | 0.0325 | 0.0656 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0248 | 0.0234 | 0.0241 | 0.0498 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0353 | 0.0317 | 0.0241 | 0.0498 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.0353 | 0.0317 | 0.0241 | 0.0498 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0252 | 0.0580 | 0.0283 | 0.0570 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0367 | 0.0633 | 0.0283 | 0.0570 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0367 | 0.0633 | 0.0283 | 0.0570 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0342 | 0.0164 | 0.0213 | 0.0642 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0607 | 0.0259 | 0.0213 | 0.0642 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0607 | 0.0259 | 0.0213 | 0.0642 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0393 | 0.0591 | 0.0201 | 0.0564 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0505 | 0.0669 | 0.0201 | 0.0564 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0505 | 0.0669 | 0.0201 | 0.0564 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0335 | 0.0399 | 0.0116 | 0.0383 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0281 | 0.0425 | 0.0116 | 0.0383 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0281 | 0.0425 | 0.0116 | 0.0383 |

## Federated Convergence

| clients_by | training_mode | calibration_scope | model | decision_policy | round | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4767 | 0.9802 | 0.0295 | 0.9782 | 0.0150 | 0.0184 | 0.9850 | 0.0407 | 0.0944 | 0.0793 | -0.0151 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4944 | 0.9825 | 0.0707 | 0.9817 | 0.0389 | 0.0199 | 0.9611 | 0.0726 | 0.0893 | 0.1078 | 0.0186 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4875 | 0.9809 | 0.0680 | 0.9085 | 0.0392 | 0.0107 | 0.9608 | 0.0421 | 0.0806 | 0.0973 | 0.0167 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4660 | 0.9797 | 0.0325 | 0.8973 | 0.0166 | 0.0184 | 0.9834 | 0.0503 | 0.0891 | 0.0736 | -0.0155 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5028 | 0.9815 | 0.0418 | 0.8724 | 0.0217 | 0.0215 | 0.9783 | 0.0657 | 0.0828 | 0.0875 | 0.0048 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4885 | 0.9814 | 0.0457 | 0.8961 | 0.0235 | 0.0230 | 0.9765 | 0.0481 | 0.0832 | 0.0814 | -0.0018 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4940 | 0.9813 | 0.0459 | 0.9675 | 0.0238 | 0.0245 | 0.9762 | 0.0557 | 0.0900 | 0.0866 | -0.0033 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4819 | 0.9803 | 0.0379 | 0.9419 | 0.0197 | 0.0245 | 0.9803 | 0.0417 | 0.0881 | 0.0823 | -0.0058 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4656 | 0.9800 | 0.0423 | 0.9566 | 0.0222 | 0.0307 | 0.9778 | 0.0445 | 0.0938 | 0.0834 | -0.0104 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4682 | 0.9801 | 0.0438 | 0.9591 | 0.0231 | 0.0292 | 0.9769 | 0.0417 | 0.0980 | 0.0903 | -0.0077 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4767 | 0.9802 | 0.0625 | 0.9759 | 0.0324 | 0.0429 | 0.9676 | 0.0407 | 0.0944 | 0.0793 | -0.0151 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4944 | 0.9825 | 0.1130 | 0.9831 | 0.0629 | 0.0353 | 0.9371 | 0.0726 | 0.0893 | 0.1078 | 0.0186 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4875 | 0.9809 | 0.0887 | 0.9507 | 0.0508 | 0.0260 | 0.9492 | 0.0421 | 0.0806 | 0.0973 | 0.0167 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4660 | 0.9797 | 0.0659 | 0.9411 | 0.0344 | 0.0476 | 0.9656 | 0.0503 | 0.0891 | 0.0736 | -0.0155 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5028 | 0.9815 | 0.0863 | 0.8858 | 0.0471 | 0.0384 | 0.9529 | 0.0657 | 0.0828 | 0.0875 | 0.0048 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4885 | 0.9814 | 0.0838 | 0.9549 | 0.0441 | 0.0476 | 0.9559 | 0.0481 | 0.0832 | 0.0814 | -0.0018 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4940 | 0.9813 | 0.0866 | 0.9547 | 0.0458 | 0.0490 | 0.9542 | 0.0557 | 0.0900 | 0.0866 | -0.0033 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4819 | 0.9803 | 0.0740 | 0.9648 | 0.0391 | 0.0568 | 0.9609 | 0.0417 | 0.0881 | 0.0823 | -0.0058 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4656 | 0.9800 | 0.0697 | 0.9519 | 0.0370 | 0.0521 | 0.9630 | 0.0445 | 0.0938 | 0.0834 | -0.0104 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4682 | 0.9801 | 0.0765 | 0.9555 | 0.0410 | 0.0506 | 0.9590 | 0.0417 | 0.0980 | 0.0903 | -0.0077 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 1 | 0.4767 | 0.9802 | 0.0625 | 0.9759 | 0.0324 | 0.0429 | 0.9676 | 0.0407 | 0.0944 | 0.0793 | -0.0151 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 2 | 0.4944 | 0.9825 | 0.1130 | 0.9831 | 0.0629 | 0.0353 | 0.9371 | 0.0726 | 0.0893 | 0.1078 | 0.0186 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 3 | 0.4875 | 0.9809 | 0.0887 | 0.9507 | 0.0508 | 0.0260 | 0.9492 | 0.0421 | 0.0806 | 0.0973 | 0.0167 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 4 | 0.4660 | 0.9797 | 0.0659 | 0.9411 | 0.0344 | 0.0476 | 0.9656 | 0.0503 | 0.0891 | 0.0736 | -0.0155 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 5 | 0.5028 | 0.9815 | 0.0863 | 0.8858 | 0.0471 | 0.0384 | 0.9529 | 0.0657 | 0.0828 | 0.0875 | 0.0048 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 6 | 0.4885 | 0.9814 | 0.0838 | 0.9549 | 0.0441 | 0.0476 | 0.9559 | 0.0481 | 0.0832 | 0.0814 | -0.0018 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 7 | 0.4940 | 0.9813 | 0.0866 | 0.9547 | 0.0458 | 0.0490 | 0.9542 | 0.0557 | 0.0900 | 0.0866 | -0.0033 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 8 | 0.4819 | 0.9803 | 0.0740 | 0.9648 | 0.0391 | 0.0568 | 0.9609 | 0.0417 | 0.0881 | 0.0823 | -0.0058 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 9 | 0.4656 | 0.9800 | 0.0697 | 0.9519 | 0.0370 | 0.0521 | 0.9630 | 0.0445 | 0.0938 | 0.0834 | -0.0104 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 10 | 0.4682 | 0.9801 | 0.0765 | 0.9555 | 0.0410 | 0.0506 | 0.9590 | 0.0417 | 0.0980 | 0.0903 | -0.0077 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4750 | 0.9802 | 0.0577 | 0.9720 | 0.0304 | 0.0368 | 0.9696 | 0.0404 | 0.1064 | 0.0924 | -0.0140 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5027 | 0.9825 | 0.1164 | 0.9853 | 0.0683 | 0.0385 | 0.9317 | 0.0607 | 0.1054 | 0.1319 | 0.0265 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4903 | 0.9809 | 0.1075 | 0.9084 | 0.0655 | 0.0370 | 0.9345 | 0.0436 | 0.1109 | 0.1258 | 0.0149 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4655 | 0.9799 | 0.0909 | 0.8954 | 0.0504 | 0.0493 | 0.9496 | 0.0570 | 0.1219 | 0.1133 | -0.0086 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4963 | 0.9814 | 0.1000 | 0.8721 | 0.0592 | 0.0384 | 0.9408 | 0.0571 | 0.1075 | 0.1216 | 0.0141 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4909 | 0.9816 | 0.0902 | 0.9016 | 0.0494 | 0.0431 | 0.9506 | 0.0489 | 0.1075 | 0.1100 | 0.0025 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4999 | 0.9817 | 0.0947 | 0.9009 | 0.0525 | 0.0445 | 0.9475 | 0.0518 | 0.1119 | 0.1160 | 0.0041 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4927 | 0.9808 | 0.0760 | 0.9085 | 0.0417 | 0.0384 | 0.9583 | 0.0491 | 0.1113 | 0.1107 | -0.0006 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4737 | 0.9804 | 0.0736 | 0.9017 | 0.0402 | 0.0507 | 0.9598 | 0.0445 | 0.1129 | 0.1028 | -0.0101 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4788 | 0.9807 | 0.0801 | 0.9604 | 0.0440 | 0.0430 | 0.9560 | 0.0546 | 0.1178 | 0.1179 | 0.0001 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4750 | 0.9802 | 0.0843 | 0.9748 | 0.0449 | 0.0521 | 0.9551 | 0.0404 | 0.1064 | 0.0924 | -0.0140 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5027 | 0.9825 | 0.1529 | 0.9803 | 0.0905 | 0.0615 | 0.9095 | 0.0607 | 0.1054 | 0.1319 | 0.0265 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4903 | 0.9809 | 0.1395 | 0.9028 | 0.0848 | 0.0677 | 0.9152 | 0.0436 | 0.1109 | 0.1258 | 0.0149 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4655 | 0.9799 | 0.1355 | 0.8958 | 0.0775 | 0.0816 | 0.9225 | 0.0570 | 0.1219 | 0.1133 | -0.0086 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4963 | 0.9814 | 0.1410 | 0.8838 | 0.0846 | 0.0676 | 0.9154 | 0.0571 | 0.1075 | 0.1216 | 0.0141 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4909 | 0.9816 | 0.1301 | 0.9083 | 0.0725 | 0.0768 | 0.9275 | 0.0489 | 0.1075 | 0.1100 | 0.0025 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4999 | 0.9817 | 0.1314 | 0.9125 | 0.0733 | 0.0752 | 0.9267 | 0.0518 | 0.1119 | 0.1160 | 0.0041 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4927 | 0.9808 | 0.1099 | 0.9668 | 0.0611 | 0.0722 | 0.9389 | 0.0491 | 0.1113 | 0.1107 | -0.0006 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4737 | 0.9804 | 0.0925 | 0.9103 | 0.0508 | 0.0691 | 0.9492 | 0.0445 | 0.1129 | 0.1028 | -0.0101 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4788 | 0.9807 | 0.1209 | 0.9713 | 0.0679 | 0.0707 | 0.9321 | 0.0546 | 0.1178 | 0.1179 | 0.0001 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4750 | 0.9802 | 0.0843 | 0.9748 | 0.0449 | 0.0521 | 0.9551 | 0.0404 | 0.1064 | 0.0924 | -0.0140 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.5027 | 0.9825 | 0.1529 | 0.9803 | 0.0905 | 0.0615 | 0.9095 | 0.0607 | 0.1054 | 0.1319 | 0.0265 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4903 | 0.9809 | 0.1395 | 0.9028 | 0.0848 | 0.0677 | 0.9152 | 0.0436 | 0.1109 | 0.1258 | 0.0149 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4655 | 0.9799 | 0.1355 | 0.8958 | 0.0775 | 0.0816 | 0.9225 | 0.0570 | 0.1219 | 0.1133 | -0.0086 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4963 | 0.9814 | 0.1410 | 0.8838 | 0.0846 | 0.0676 | 0.9154 | 0.0571 | 0.1075 | 0.1216 | 0.0141 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4909 | 0.9816 | 0.1301 | 0.9083 | 0.0725 | 0.0768 | 0.9275 | 0.0489 | 0.1075 | 0.1100 | 0.0025 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4999 | 0.9817 | 0.1314 | 0.9125 | 0.0733 | 0.0752 | 0.9267 | 0.0518 | 0.1119 | 0.1160 | 0.0041 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4927 | 0.9808 | 0.1099 | 0.9668 | 0.0611 | 0.0722 | 0.9389 | 0.0491 | 0.1113 | 0.1107 | -0.0006 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4737 | 0.9804 | 0.0925 | 0.9103 | 0.0508 | 0.0691 | 0.9492 | 0.0445 | 0.1129 | 0.1028 | -0.0101 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4788 | 0.9807 | 0.1209 | 0.9713 | 0.0679 | 0.0707 | 0.9321 | 0.0546 | 0.1178 | 0.1179 | 0.0001 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4772 | 0.9801 | 0.0236 | 0.9762 | 0.0119 | 0.0184 | 0.9881 | 0.0392 | 0.0915 | 0.0760 | -0.0155 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4887 | 0.9822 | 0.0315 | 0.9821 | 0.0160 | 0.0137 | 0.9840 | 0.0711 | 0.0850 | 0.0860 | 0.0010 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4678 | 0.9805 | 0.0212 | 0.9568 | 0.0107 | 0.0091 | 0.9893 | 0.0335 | 0.0754 | 0.0676 | -0.0078 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4671 | 0.9796 | 0.0235 | 0.9174 | 0.0119 | 0.0184 | 0.9881 | 0.0475 | 0.0856 | 0.0682 | -0.0173 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4843 | 0.9810 | 0.0201 | 0.9124 | 0.0102 | 0.0199 | 0.9898 | 0.0469 | 0.0845 | 0.0732 | -0.0113 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4833 | 0.9811 | 0.0404 | 0.9514 | 0.0208 | 0.0276 | 0.9792 | 0.0479 | 0.0810 | 0.0779 | -0.0031 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4803 | 0.9805 | 0.0345 | 0.9573 | 0.0177 | 0.0275 | 0.9823 | 0.0524 | 0.0855 | 0.0749 | -0.0106 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4448 | 0.9788 | 0.0241 | 0.9532 | 0.0123 | 0.0261 | 0.9877 | 0.0385 | 0.0815 | 0.0553 | -0.0262 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4251 | 0.9784 | 0.0277 | 0.9511 | 0.0141 | 0.0276 | 0.9859 | 0.0348 | 0.0878 | 0.0561 | -0.0317 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4293 | 0.9783 | 0.0198 | 0.9366 | 0.0101 | 0.0292 | 0.9899 | 0.0375 | 0.0925 | 0.0569 | -0.0356 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4772 | 0.9801 | 0.0568 | 0.9795 | 0.0293 | 0.0368 | 0.9707 | 0.0392 | 0.0915 | 0.0760 | -0.0155 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4887 | 0.9822 | 0.0729 | 0.9826 | 0.0380 | 0.0337 | 0.9620 | 0.0711 | 0.0850 | 0.0860 | 0.0010 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4678 | 0.9805 | 0.0350 | 0.9594 | 0.0179 | 0.0183 | 0.9821 | 0.0335 | 0.0754 | 0.0676 | -0.0078 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4671 | 0.9796 | 0.0559 | 0.9722 | 0.0290 | 0.0353 | 0.9710 | 0.0475 | 0.0856 | 0.0682 | -0.0173 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4843 | 0.9810 | 0.0480 | 0.9638 | 0.0247 | 0.0444 | 0.9753 | 0.0469 | 0.0845 | 0.0732 | -0.0113 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4833 | 0.9811 | 0.0773 | 0.9798 | 0.0404 | 0.0430 | 0.9596 | 0.0479 | 0.0810 | 0.0779 | -0.0031 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4803 | 0.9805 | 0.0738 | 0.9792 | 0.0385 | 0.0428 | 0.9615 | 0.0524 | 0.0855 | 0.0749 | -0.0106 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4448 | 0.9788 | 0.0600 | 0.9700 | 0.0310 | 0.0521 | 0.9690 | 0.0385 | 0.0815 | 0.0553 | -0.0262 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4251 | 0.9784 | 0.0629 | 0.9696 | 0.0327 | 0.0567 | 0.9673 | 0.0348 | 0.0878 | 0.0561 | -0.0317 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4293 | 0.9783 | 0.0561 | 0.9686 | 0.0290 | 0.0506 | 0.9710 | 0.0375 | 0.0925 | 0.0569 | -0.0356 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.4772 | 0.9801 | 0.0568 | 0.9795 | 0.0293 | 0.0368 | 0.9707 | 0.0392 | 0.0915 | 0.0760 | -0.0155 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.4887 | 0.9822 | 0.0729 | 0.9826 | 0.0380 | 0.0337 | 0.9620 | 0.0711 | 0.0850 | 0.0860 | 0.0010 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.4678 | 0.9805 | 0.0350 | 0.9594 | 0.0179 | 0.0183 | 0.9821 | 0.0335 | 0.0754 | 0.0676 | -0.0078 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.4671 | 0.9796 | 0.0559 | 0.9722 | 0.0290 | 0.0353 | 0.9710 | 0.0475 | 0.0856 | 0.0682 | -0.0173 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.4843 | 0.9810 | 0.0480 | 0.9638 | 0.0247 | 0.0444 | 0.9753 | 0.0469 | 0.0845 | 0.0732 | -0.0113 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.4833 | 0.9811 | 0.0773 | 0.9798 | 0.0404 | 0.0430 | 0.9596 | 0.0479 | 0.0810 | 0.0779 | -0.0031 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.4803 | 0.9805 | 0.0738 | 0.9792 | 0.0385 | 0.0428 | 0.9615 | 0.0524 | 0.0855 | 0.0749 | -0.0106 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.4448 | 0.9788 | 0.0600 | 0.9700 | 0.0310 | 0.0521 | 0.9690 | 0.0385 | 0.0815 | 0.0553 | -0.0262 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.4251 | 0.9784 | 0.0629 | 0.9696 | 0.0327 | 0.0567 | 0.9673 | 0.0348 | 0.0878 | 0.0561 | -0.0317 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.4293 | 0.9783 | 0.0561 | 0.9686 | 0.0290 | 0.0506 | 0.9710 | 0.0375 | 0.0925 | 0.0569 | -0.0356 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4994 | 0.9816 | 0.0460 | 0.9786 | 0.0237 | 0.0215 | 0.9763 | 0.0532 | 0.0940 | 0.0943 | 0.0002 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4984 | 0.9825 | 0.0869 | 0.9320 | 0.0469 | 0.0322 | 0.9531 | 0.0730 | 0.1024 | 0.1201 | 0.0176 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4902 | 0.9805 | 0.0936 | 0.9063 | 0.0544 | 0.0492 | 0.9456 | 0.0605 | 0.1145 | 0.1219 | 0.0075 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4940 | 0.9813 | 0.0998 | 0.9171 | 0.0568 | 0.0293 | 0.9432 | 0.0646 | 0.1063 | 0.1269 | 0.0206 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5235 | 0.9823 | 0.0955 | 0.9578 | 0.0553 | 0.0354 | 0.9447 | 0.0650 | 0.0945 | 0.1251 | 0.0306 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5269 | 0.9830 | 0.0618 | 0.9027 | 0.0356 | 0.0215 | 0.9644 | 0.0692 | 0.0778 | 0.1085 | 0.0307 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.5388 | 0.9831 | 0.0729 | 0.9837 | 0.0412 | 0.0216 | 0.9588 | 0.0491 | 0.0794 | 0.1158 | 0.0363 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5381 | 0.9838 | 0.0736 | 0.5904 | 0.0432 | 0.0169 | 0.9568 | 0.0276 | 0.0648 | 0.1056 | 0.0407 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5729 | 0.9857 | 0.0697 | 0.8547 | 0.0399 | 0.0092 | 0.9601 | 0.0482 | 0.0622 | 0.1217 | 0.0595 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5390 | 0.9839 | 0.0473 | 0.8248 | 0.0251 | 0.0154 | 0.9749 | 0.0392 | 0.0644 | 0.0953 | 0.0309 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4994 | 0.9816 | 0.0797 | 0.9796 | 0.0419 | 0.0400 | 0.9581 | 0.0532 | 0.0940 | 0.0943 | 0.0002 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4984 | 0.9825 | 0.1408 | 0.9827 | 0.0803 | 0.0584 | 0.9197 | 0.0730 | 0.1024 | 0.1201 | 0.0176 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4902 | 0.9805 | 0.1386 | 0.9278 | 0.0802 | 0.0738 | 0.9198 | 0.0605 | 0.1145 | 0.1219 | 0.0075 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4940 | 0.9813 | 0.1471 | 0.9170 | 0.0849 | 0.0585 | 0.9151 | 0.0646 | 0.1063 | 0.1269 | 0.0206 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5235 | 0.9823 | 0.1389 | 0.9743 | 0.0811 | 0.0462 | 0.9189 | 0.0650 | 0.0945 | 0.1251 | 0.0306 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5269 | 0.9830 | 0.1026 | 0.9779 | 0.0594 | 0.0400 | 0.9406 | 0.0692 | 0.0778 | 0.1085 | 0.0307 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.5388 | 0.9831 | 0.1104 | 0.9799 | 0.0647 | 0.0385 | 0.9353 | 0.0491 | 0.0794 | 0.1158 | 0.0363 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5381 | 0.9838 | 0.0911 | 0.7574 | 0.0541 | 0.0277 | 0.9459 | 0.0276 | 0.0648 | 0.1056 | 0.0407 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5729 | 0.9857 | 0.1084 | 0.8766 | 0.0646 | 0.0169 | 0.9354 | 0.0482 | 0.0622 | 0.1217 | 0.0595 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5390 | 0.9839 | 0.0716 | 0.8826 | 0.0385 | 0.0216 | 0.9615 | 0.0392 | 0.0644 | 0.0953 | 0.0309 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 1 | 0.4994 | 0.9816 | 0.0797 | 0.9796 | 0.0419 | 0.0400 | 0.9581 | 0.0532 | 0.0940 | 0.0943 | 0.0002 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 2 | 0.4984 | 0.9825 | 0.1408 | 0.9827 | 0.0803 | 0.0584 | 0.9197 | 0.0730 | 0.1024 | 0.1201 | 0.0176 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 3 | 0.4902 | 0.9805 | 0.1386 | 0.9278 | 0.0802 | 0.0738 | 0.9198 | 0.0605 | 0.1145 | 0.1219 | 0.0075 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 4 | 0.4940 | 0.9813 | 0.1471 | 0.9170 | 0.0849 | 0.0585 | 0.9151 | 0.0646 | 0.1063 | 0.1269 | 0.0206 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 5 | 0.5235 | 0.9823 | 0.1389 | 0.9743 | 0.0811 | 0.0462 | 0.9189 | 0.0650 | 0.0945 | 0.1251 | 0.0306 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 6 | 0.5269 | 0.9830 | 0.1026 | 0.9779 | 0.0594 | 0.0400 | 0.9406 | 0.0692 | 0.0778 | 0.1085 | 0.0307 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 7 | 0.5388 | 0.9831 | 0.1104 | 0.9799 | 0.0647 | 0.0385 | 0.9353 | 0.0491 | 0.0794 | 0.1158 | 0.0363 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 8 | 0.5381 | 0.9838 | 0.0911 | 0.7574 | 0.0541 | 0.0277 | 0.9459 | 0.0276 | 0.0648 | 0.1056 | 0.0407 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 9 | 0.5729 | 0.9857 | 0.1084 | 0.8766 | 0.0646 | 0.0169 | 0.9354 | 0.0482 | 0.0622 | 0.1217 | 0.0595 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 10 | 0.5390 | 0.9839 | 0.0716 | 0.8826 | 0.0385 | 0.0216 | 0.9615 | 0.0392 | 0.0644 | 0.0953 | 0.0309 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5010 | 0.9816 | 0.0843 | 0.9733 | 0.0461 | 0.0323 | 0.9539 | 0.0576 | 0.1062 | 0.1137 | 0.0076 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5016 | 0.9825 | 0.1360 | 0.9832 | 0.0783 | 0.0569 | 0.9217 | 0.0596 | 0.1252 | 0.1442 | 0.0190 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4926 | 0.9807 | 0.1268 | 0.9152 | 0.0753 | 0.0554 | 0.9247 | 0.0548 | 0.1267 | 0.1403 | 0.0137 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4970 | 0.9814 | 0.1234 | 0.9189 | 0.0730 | 0.0463 | 0.9270 | 0.0631 | 0.1213 | 0.1414 | 0.0201 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5211 | 0.9825 | 0.1337 | 0.9502 | 0.0783 | 0.0571 | 0.9217 | 0.0615 | 0.1134 | 0.1448 | 0.0314 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5256 | 0.9831 | 0.1171 | 0.9216 | 0.0678 | 0.0492 | 0.9322 | 0.0621 | 0.1061 | 0.1384 | 0.0323 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.5459 | 0.9834 | 0.1166 | 0.9142 | 0.0673 | 0.0385 | 0.9327 | 0.0441 | 0.0952 | 0.1377 | 0.0425 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5470 | 0.9840 | 0.1376 | 0.9309 | 0.0783 | 0.0353 | 0.9217 | 0.0559 | 0.0933 | 0.1517 | 0.0584 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5825 | 0.9860 | 0.1980 | 0.9914 | 0.1189 | 0.0262 | 0.8811 | 0.0637 | 0.0911 | 0.1975 | 0.1064 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5466 | 0.9840 | 0.1165 | 0.9791 | 0.0640 | 0.0323 | 0.9360 | 0.0705 | 0.0896 | 0.1414 | 0.0519 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5010 | 0.9816 | 0.1174 | 0.9722 | 0.0664 | 0.0493 | 0.9336 | 0.0576 | 0.1062 | 0.1137 | 0.0076 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5016 | 0.9825 | 0.1739 | 0.9834 | 0.1015 | 0.0815 | 0.8985 | 0.0596 | 0.1252 | 0.1442 | 0.0190 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4926 | 0.9807 | 0.1654 | 0.9111 | 0.0979 | 0.0847 | 0.9021 | 0.0548 | 0.1267 | 0.1403 | 0.0137 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4970 | 0.9814 | 0.1737 | 0.9223 | 0.1029 | 0.0771 | 0.8971 | 0.0631 | 0.1213 | 0.1414 | 0.0201 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5211 | 0.9825 | 0.1718 | 0.9442 | 0.1015 | 0.0709 | 0.8985 | 0.0615 | 0.1134 | 0.1448 | 0.0314 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5256 | 0.9831 | 0.1561 | 0.9546 | 0.0905 | 0.0722 | 0.9095 | 0.0621 | 0.1061 | 0.1384 | 0.0323 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.5459 | 0.9834 | 0.1443 | 0.9716 | 0.0847 | 0.0539 | 0.9153 | 0.0441 | 0.0952 | 0.1377 | 0.0425 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5470 | 0.9840 | 0.1718 | 0.9752 | 0.0999 | 0.0538 | 0.9001 | 0.0559 | 0.0933 | 0.1517 | 0.0584 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5825 | 0.9860 | 0.2372 | 0.9688 | 0.1457 | 0.0478 | 0.8543 | 0.0637 | 0.0911 | 0.1975 | 0.1064 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5466 | 0.9840 | 0.1689 | 0.9820 | 0.0979 | 0.0508 | 0.9021 | 0.0705 | 0.0896 | 0.1414 | 0.0519 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.5010 | 0.9816 | 0.1174 | 0.9722 | 0.0664 | 0.0493 | 0.9336 | 0.0576 | 0.1062 | 0.1137 | 0.0076 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.5016 | 0.9825 | 0.1739 | 0.9834 | 0.1015 | 0.0815 | 0.8985 | 0.0596 | 0.1252 | 0.1442 | 0.0190 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4926 | 0.9807 | 0.1654 | 0.9111 | 0.0979 | 0.0847 | 0.9021 | 0.0548 | 0.1267 | 0.1403 | 0.0137 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4970 | 0.9814 | 0.1737 | 0.9223 | 0.1029 | 0.0771 | 0.8971 | 0.0631 | 0.1213 | 0.1414 | 0.0201 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 5 | 0.5211 | 0.9825 | 0.1718 | 0.9442 | 0.1015 | 0.0709 | 0.8985 | 0.0615 | 0.1134 | 0.1448 | 0.0314 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 6 | 0.5256 | 0.9831 | 0.1561 | 0.9546 | 0.0905 | 0.0722 | 0.9095 | 0.0621 | 0.1061 | 0.1384 | 0.0323 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 7 | 0.5459 | 0.9834 | 0.1443 | 0.9716 | 0.0847 | 0.0539 | 0.9153 | 0.0441 | 0.0952 | 0.1377 | 0.0425 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 8 | 0.5470 | 0.9840 | 0.1718 | 0.9752 | 0.0999 | 0.0538 | 0.9001 | 0.0559 | 0.0933 | 0.1517 | 0.0584 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 9 | 0.5825 | 0.9860 | 0.2372 | 0.9688 | 0.1457 | 0.0478 | 0.8543 | 0.0637 | 0.0911 | 0.1975 | 0.1064 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 10 | 0.5466 | 0.9840 | 0.1689 | 0.9820 | 0.0979 | 0.0508 | 0.9021 | 0.0705 | 0.0896 | 0.1414 | 0.0519 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4998 | 0.9814 | 0.0335 | 0.9810 | 0.0171 | 0.0184 | 0.9829 | 0.0457 | 0.0912 | 0.0854 | -0.0057 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4924 | 0.9824 | 0.0725 | 0.8758 | 0.0391 | 0.0353 | 0.9609 | 0.0697 | 0.0987 | 0.1117 | 0.0130 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4812 | 0.9802 | 0.0684 | 0.9301 | 0.0381 | 0.0322 | 0.9619 | 0.0628 | 0.1120 | 0.1077 | -0.0043 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4932 | 0.9813 | 0.0766 | 0.9070 | 0.0420 | 0.0246 | 0.9580 | 0.0696 | 0.1009 | 0.1150 | 0.0141 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5220 | 0.9823 | 0.0816 | 0.9635 | 0.0469 | 0.0261 | 0.9531 | 0.0663 | 0.0872 | 0.1168 | 0.0296 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5165 | 0.9825 | 0.0707 | 0.6838 | 0.0422 | 0.0184 | 0.9578 | 0.0791 | 0.0750 | 0.1139 | 0.0389 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.5167 | 0.9819 | 0.0622 | 0.7120 | 0.0352 | 0.0216 | 0.9648 | 0.0569 | 0.0807 | 0.1043 | 0.0236 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5044 | 0.9824 | 0.0739 | 0.4281 | 0.0458 | 0.0324 | 0.9542 | 0.0394 | 0.0827 | 0.0979 | 0.0152 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5192 | 0.9834 | 0.0453 | 0.6032 | 0.0251 | 0.0123 | 0.9749 | 0.0675 | 0.0778 | 0.0930 | 0.0153 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4915 | 0.9817 | 0.0384 | 0.6057 | 0.0213 | 0.0123 | 0.9787 | 0.0414 | 0.0720 | 0.0732 | 0.0013 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4998 | 0.9814 | 0.0630 | 0.9798 | 0.0328 | 0.0400 | 0.9672 | 0.0457 | 0.0912 | 0.0854 | -0.0057 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4924 | 0.9824 | 0.1268 | 0.9836 | 0.0722 | 0.0507 | 0.9278 | 0.0697 | 0.0987 | 0.1117 | 0.0130 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4812 | 0.9802 | 0.1194 | 0.9437 | 0.0669 | 0.0737 | 0.9331 | 0.0628 | 0.1120 | 0.1077 | -0.0043 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4932 | 0.9813 | 0.1250 | 0.9427 | 0.0699 | 0.0475 | 0.9301 | 0.0696 | 0.1009 | 0.1150 | 0.0141 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5220 | 0.9823 | 0.1293 | 0.9727 | 0.0761 | 0.0445 | 0.9239 | 0.0663 | 0.0872 | 0.1168 | 0.0296 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5165 | 0.9825 | 0.1205 | 0.8672 | 0.0783 | 0.0338 | 0.9217 | 0.0791 | 0.0750 | 0.1139 | 0.0389 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.5167 | 0.9819 | 0.1028 | 0.8066 | 0.0595 | 0.0369 | 0.9405 | 0.0569 | 0.0807 | 0.1043 | 0.0236 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5044 | 0.9824 | 0.0999 | 0.4827 | 0.0654 | 0.0508 | 0.9346 | 0.0394 | 0.0827 | 0.0979 | 0.0152 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5192 | 0.9834 | 0.0950 | 0.6535 | 0.0580 | 0.0431 | 0.9420 | 0.0675 | 0.0778 | 0.0930 | 0.0153 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4915 | 0.9817 | 0.0625 | 0.6608 | 0.0360 | 0.0355 | 0.9640 | 0.0414 | 0.0720 | 0.0732 | 0.0013 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.4998 | 0.9814 | 0.0630 | 0.9798 | 0.0328 | 0.0400 | 0.9672 | 0.0457 | 0.0912 | 0.0854 | -0.0057 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.4924 | 0.9824 | 0.1268 | 0.9836 | 0.0722 | 0.0507 | 0.9278 | 0.0697 | 0.0987 | 0.1117 | 0.0130 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.4812 | 0.9802 | 0.1194 | 0.9437 | 0.0669 | 0.0737 | 0.9331 | 0.0628 | 0.1120 | 0.1077 | -0.0043 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 4 | 0.4932 | 0.9813 | 0.1250 | 0.9427 | 0.0699 | 0.0475 | 0.9301 | 0.0696 | 0.1009 | 0.1150 | 0.0141 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 5 | 0.5220 | 0.9823 | 0.1293 | 0.9727 | 0.0761 | 0.0445 | 0.9239 | 0.0663 | 0.0872 | 0.1168 | 0.0296 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 6 | 0.5165 | 0.9825 | 0.1205 | 0.8672 | 0.0783 | 0.0338 | 0.9217 | 0.0791 | 0.0750 | 0.1139 | 0.0389 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 7 | 0.5167 | 0.9819 | 0.1028 | 0.8066 | 0.0595 | 0.0369 | 0.9405 | 0.0569 | 0.0807 | 0.1043 | 0.0236 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 8 | 0.5044 | 0.9824 | 0.0999 | 0.4827 | 0.0654 | 0.0508 | 0.9346 | 0.0394 | 0.0827 | 0.0979 | 0.0152 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 9 | 0.5192 | 0.9834 | 0.0950 | 0.6535 | 0.0580 | 0.0431 | 0.9420 | 0.0675 | 0.0778 | 0.0930 | 0.0153 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 10 | 0.4915 | 0.9817 | 0.0625 | 0.6608 | 0.0360 | 0.0355 | 0.9640 | 0.0414 | 0.0720 | 0.0732 | 0.0013 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4840 | 0.9808 | 0.0323 | 0.9743 | 0.0165 | 0.0184 | 0.9835 | 0.0415 | 0.0902 | 0.0814 | -0.0088 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4852 | 0.9816 | 0.0914 | 0.9568 | 0.0493 | 0.0384 | 0.9507 | 0.0563 | 0.1070 | 0.1146 | 0.0075 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4628 | 0.9799 | 0.0418 | 0.8789 | 0.0217 | 0.0246 | 0.9783 | 0.0527 | 0.0966 | 0.0809 | -0.0157 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4650 | 0.9798 | 0.0363 | 0.7697 | 0.0187 | 0.0138 | 0.9813 | 0.0470 | 0.0869 | 0.0770 | -0.0099 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4623 | 0.9802 | 0.0478 | 0.9816 | 0.0247 | 0.0185 | 0.9753 | 0.0453 | 0.0869 | 0.0791 | -0.0078 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4678 | 0.9805 | 0.0521 | 0.9769 | 0.0270 | 0.0169 | 0.9730 | 0.0374 | 0.0772 | 0.0772 | 0.0000 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4756 | 0.9805 | 0.0424 | 0.9800 | 0.0219 | 0.0200 | 0.9781 | 0.0347 | 0.0835 | 0.0757 | -0.0079 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4846 | 0.9801 | 0.0354 | 0.9576 | 0.0182 | 0.0323 | 0.9818 | 0.0436 | 0.0898 | 0.0774 | -0.0124 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4610 | 0.9796 | 0.0437 | 0.9638 | 0.0228 | 0.0307 | 0.9772 | 0.0450 | 0.0963 | 0.0851 | -0.0112 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4698 | 0.9802 | 0.0467 | 0.9696 | 0.0247 | 0.0262 | 0.9753 | 0.0362 | 0.0884 | 0.0836 | -0.0049 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4840 | 0.9808 | 0.0666 | 0.9784 | 0.0347 | 0.0369 | 0.9653 | 0.0415 | 0.0902 | 0.0814 | -0.0088 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4852 | 0.9816 | 0.1412 | 0.9741 | 0.0784 | 0.0691 | 0.9216 | 0.0563 | 0.1070 | 0.1146 | 0.0075 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4628 | 0.9799 | 0.0666 | 0.9181 | 0.0350 | 0.0430 | 0.9650 | 0.0527 | 0.0966 | 0.0809 | -0.0157 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4650 | 0.9798 | 0.0713 | 0.8898 | 0.0377 | 0.0399 | 0.9623 | 0.0470 | 0.0869 | 0.0770 | -0.0099 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4623 | 0.9802 | 0.0773 | 0.9583 | 0.0408 | 0.0431 | 0.9592 | 0.0453 | 0.0869 | 0.0791 | -0.0078 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4678 | 0.9805 | 0.0775 | 0.9783 | 0.0409 | 0.0292 | 0.9591 | 0.0374 | 0.0772 | 0.0772 | 0.0000 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4756 | 0.9805 | 0.0657 | 0.9642 | 0.0344 | 0.0384 | 0.9656 | 0.0347 | 0.0835 | 0.0757 | -0.0079 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4846 | 0.9801 | 0.0577 | 0.9613 | 0.0302 | 0.0522 | 0.9698 | 0.0436 | 0.0898 | 0.0774 | -0.0124 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4610 | 0.9796 | 0.0689 | 0.9600 | 0.0368 | 0.0491 | 0.9632 | 0.0450 | 0.0963 | 0.0851 | -0.0112 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4698 | 0.9802 | 0.0698 | 0.9699 | 0.0372 | 0.0430 | 0.9628 | 0.0362 | 0.0884 | 0.0836 | -0.0049 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 1 | 0.4840 | 0.9808 | 0.0666 | 0.9784 | 0.0347 | 0.0369 | 0.9653 | 0.0415 | 0.0902 | 0.0814 | -0.0088 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 2 | 0.4852 | 0.9816 | 0.1412 | 0.9741 | 0.0784 | 0.0691 | 0.9216 | 0.0563 | 0.1070 | 0.1146 | 0.0075 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 3 | 0.4628 | 0.9799 | 0.0666 | 0.9181 | 0.0350 | 0.0430 | 0.9650 | 0.0527 | 0.0966 | 0.0809 | -0.0157 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 4 | 0.4650 | 0.9798 | 0.0713 | 0.8898 | 0.0377 | 0.0399 | 0.9623 | 0.0470 | 0.0869 | 0.0770 | -0.0099 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 5 | 0.4623 | 0.9802 | 0.0773 | 0.9583 | 0.0408 | 0.0431 | 0.9592 | 0.0453 | 0.0869 | 0.0791 | -0.0078 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 6 | 0.4678 | 0.9805 | 0.0775 | 0.9783 | 0.0409 | 0.0292 | 0.9591 | 0.0374 | 0.0772 | 0.0772 | 0.0000 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 7 | 0.4756 | 0.9805 | 0.0657 | 0.9642 | 0.0344 | 0.0384 | 0.9656 | 0.0347 | 0.0835 | 0.0757 | -0.0079 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 8 | 0.4846 | 0.9801 | 0.0577 | 0.9613 | 0.0302 | 0.0522 | 0.9698 | 0.0436 | 0.0898 | 0.0774 | -0.0124 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 9 | 0.4610 | 0.9796 | 0.0689 | 0.9600 | 0.0368 | 0.0491 | 0.9632 | 0.0450 | 0.0963 | 0.0851 | -0.0112 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 10 | 0.4698 | 0.9802 | 0.0698 | 0.9699 | 0.0372 | 0.0430 | 0.9628 | 0.0362 | 0.0884 | 0.0836 | -0.0049 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4864 | 0.9808 | 0.0599 | 0.9762 | 0.0318 | 0.0339 | 0.9682 | 0.0385 | 0.1003 | 0.0932 | -0.0071 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4925 | 0.9818 | 0.1301 | 0.9820 | 0.0751 | 0.0477 | 0.9249 | 0.0529 | 0.1180 | 0.1359 | 0.0179 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4656 | 0.9805 | 0.0750 | 0.8849 | 0.0405 | 0.0353 | 0.9595 | 0.0506 | 0.1103 | 0.0998 | -0.0105 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4653 | 0.9798 | 0.0610 | 0.8218 | 0.0320 | 0.0292 | 0.9680 | 0.0459 | 0.1021 | 0.0888 | -0.0133 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4609 | 0.9801 | 0.0689 | 0.9773 | 0.0364 | 0.0355 | 0.9636 | 0.0407 | 0.0984 | 0.0881 | -0.0104 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4723 | 0.9807 | 0.0698 | 0.9799 | 0.0367 | 0.0262 | 0.9633 | 0.0403 | 0.0950 | 0.0916 | -0.0034 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4868 | 0.9809 | 0.0674 | 0.9729 | 0.0360 | 0.0416 | 0.9640 | 0.0391 | 0.0973 | 0.0912 | -0.0061 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4956 | 0.9804 | 0.0609 | 0.9524 | 0.0327 | 0.0477 | 0.9673 | 0.0497 | 0.1094 | 0.0987 | -0.0108 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4718 | 0.9802 | 0.0611 | 0.9585 | 0.0326 | 0.0430 | 0.9674 | 0.0530 | 0.1120 | 0.1008 | -0.0111 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4842 | 0.9805 | 0.0731 | 0.9704 | 0.0403 | 0.0370 | 0.9597 | 0.0417 | 0.1090 | 0.1050 | -0.0040 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4864 | 0.9808 | 0.0857 | 0.9785 | 0.0463 | 0.0492 | 0.9537 | 0.0385 | 0.1003 | 0.0932 | -0.0071 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4925 | 0.9818 | 0.1688 | 0.9699 | 0.0993 | 0.0799 | 0.9007 | 0.0529 | 0.1180 | 0.1359 | 0.0179 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4656 | 0.9805 | 0.1015 | 0.8866 | 0.0558 | 0.0614 | 0.9442 | 0.0506 | 0.1103 | 0.0998 | -0.0105 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4653 | 0.9798 | 0.0897 | 0.8597 | 0.0482 | 0.0553 | 0.9518 | 0.0459 | 0.1021 | 0.0888 | -0.0133 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4609 | 0.9801 | 0.0943 | 0.9715 | 0.0506 | 0.0539 | 0.9494 | 0.0407 | 0.0984 | 0.0881 | -0.0104 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4723 | 0.9807 | 0.0952 | 0.9746 | 0.0511 | 0.0478 | 0.9489 | 0.0403 | 0.0950 | 0.0916 | -0.0034 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4868 | 0.9809 | 0.0974 | 0.9797 | 0.0533 | 0.0586 | 0.9467 | 0.0391 | 0.0973 | 0.0912 | -0.0061 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4956 | 0.9804 | 0.0854 | 0.9589 | 0.0465 | 0.0692 | 0.9535 | 0.0497 | 0.1094 | 0.0987 | -0.0108 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4718 | 0.9802 | 0.0978 | 0.9607 | 0.0540 | 0.0661 | 0.9460 | 0.0530 | 0.1120 | 0.1008 | -0.0111 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4842 | 0.9805 | 0.0978 | 0.9640 | 0.0540 | 0.0661 | 0.9460 | 0.0417 | 0.1090 | 0.1050 | -0.0040 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4864 | 0.9808 | 0.0857 | 0.9785 | 0.0463 | 0.0492 | 0.9537 | 0.0385 | 0.1003 | 0.0932 | -0.0071 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4925 | 0.9818 | 0.1688 | 0.9699 | 0.0993 | 0.0799 | 0.9007 | 0.0529 | 0.1180 | 0.1359 | 0.0179 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4656 | 0.9805 | 0.1015 | 0.8866 | 0.0558 | 0.0614 | 0.9442 | 0.0506 | 0.1103 | 0.0998 | -0.0105 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4653 | 0.9798 | 0.0897 | 0.8597 | 0.0482 | 0.0553 | 0.9518 | 0.0459 | 0.1021 | 0.0888 | -0.0133 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4609 | 0.9801 | 0.0943 | 0.9715 | 0.0506 | 0.0539 | 0.9494 | 0.0407 | 0.0984 | 0.0881 | -0.0104 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4723 | 0.9807 | 0.0952 | 0.9746 | 0.0511 | 0.0478 | 0.9489 | 0.0403 | 0.0950 | 0.0916 | -0.0034 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4868 | 0.9809 | 0.0974 | 0.9797 | 0.0533 | 0.0586 | 0.9467 | 0.0391 | 0.0973 | 0.0912 | -0.0061 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4956 | 0.9804 | 0.0854 | 0.9589 | 0.0465 | 0.0692 | 0.9535 | 0.0497 | 0.1094 | 0.0987 | -0.0108 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4718 | 0.9802 | 0.0978 | 0.9607 | 0.0540 | 0.0661 | 0.9460 | 0.0530 | 0.1120 | 0.1008 | -0.0111 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4842 | 0.9805 | 0.0978 | 0.9640 | 0.0540 | 0.0661 | 0.9460 | 0.0417 | 0.1090 | 0.1050 | -0.0040 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4812 | 0.9807 | 0.0288 | 0.9777 | 0.0147 | 0.0184 | 0.9853 | 0.0409 | 0.0901 | 0.0794 | -0.0107 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4741 | 0.9810 | 0.0590 | 0.9768 | 0.0307 | 0.0292 | 0.9693 | 0.0481 | 0.1004 | 0.0945 | -0.0059 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4623 | 0.9786 | 0.0309 | 0.9092 | 0.0158 | 0.0184 | 0.9842 | 0.0507 | 0.0950 | 0.0753 | -0.0197 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4631 | 0.9798 | 0.0299 | 0.6606 | 0.0153 | 0.0123 | 0.9847 | 0.0454 | 0.0849 | 0.0732 | -0.0117 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4653 | 0.9802 | 0.0410 | 0.9629 | 0.0211 | 0.0200 | 0.9789 | 0.0454 | 0.0850 | 0.0768 | -0.0082 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4684 | 0.9804 | 0.0468 | 0.9786 | 0.0242 | 0.0169 | 0.9758 | 0.0374 | 0.0763 | 0.0737 | -0.0027 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4604 | 0.9798 | 0.0323 | 0.9708 | 0.0165 | 0.0184 | 0.9835 | 0.0291 | 0.0811 | 0.0659 | -0.0152 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4570 | 0.9789 | 0.0218 | 0.9581 | 0.0111 | 0.0277 | 0.9889 | 0.0325 | 0.0808 | 0.0528 | -0.0280 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4273 | 0.9782 | 0.0291 | 0.9566 | 0.0148 | 0.0307 | 0.9852 | 0.0253 | 0.0879 | 0.0552 | -0.0327 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4289 | 0.9784 | 0.0281 | 0.9720 | 0.0143 | 0.0215 | 0.9857 | 0.0306 | 0.0835 | 0.0539 | -0.0296 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4812 | 0.9807 | 0.0639 | 0.9768 | 0.0332 | 0.0415 | 0.9668 | 0.0409 | 0.0901 | 0.0794 | -0.0107 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4741 | 0.9810 | 0.1083 | 0.9809 | 0.0578 | 0.0537 | 0.9422 | 0.0481 | 0.1004 | 0.0945 | -0.0059 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4623 | 0.9786 | 0.0536 | 0.9444 | 0.0279 | 0.0400 | 0.9721 | 0.0507 | 0.0950 | 0.0753 | -0.0197 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4631 | 0.9798 | 0.0666 | 0.9428 | 0.0351 | 0.0399 | 0.9649 | 0.0454 | 0.0849 | 0.0732 | -0.0117 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4653 | 0.9802 | 0.0745 | 0.9702 | 0.0394 | 0.0385 | 0.9606 | 0.0454 | 0.0850 | 0.0768 | -0.0082 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4684 | 0.9804 | 0.0732 | 0.9714 | 0.0385 | 0.0337 | 0.9615 | 0.0374 | 0.0763 | 0.0737 | -0.0027 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4604 | 0.9798 | 0.0543 | 0.9674 | 0.0281 | 0.0337 | 0.9719 | 0.0291 | 0.0811 | 0.0659 | -0.0152 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4570 | 0.9789 | 0.0441 | 0.9651 | 0.0227 | 0.0445 | 0.9773 | 0.0325 | 0.0808 | 0.0528 | -0.0280 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4273 | 0.9782 | 0.0435 | 0.9617 | 0.0223 | 0.0460 | 0.9777 | 0.0253 | 0.0879 | 0.0552 | -0.0327 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4289 | 0.9784 | 0.0563 | 0.9727 | 0.0291 | 0.0414 | 0.9709 | 0.0306 | 0.0835 | 0.0539 | -0.0296 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.4812 | 0.9807 | 0.0639 | 0.9768 | 0.0332 | 0.0415 | 0.9668 | 0.0409 | 0.0901 | 0.0794 | -0.0107 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.4741 | 0.9810 | 0.1083 | 0.9809 | 0.0578 | 0.0537 | 0.9422 | 0.0481 | 0.1004 | 0.0945 | -0.0059 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.4623 | 0.9786 | 0.0536 | 0.9444 | 0.0279 | 0.0400 | 0.9721 | 0.0507 | 0.0950 | 0.0753 | -0.0197 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.4631 | 0.9798 | 0.0666 | 0.9428 | 0.0351 | 0.0399 | 0.9649 | 0.0454 | 0.0849 | 0.0732 | -0.0117 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.4653 | 0.9802 | 0.0745 | 0.9702 | 0.0394 | 0.0385 | 0.9606 | 0.0454 | 0.0850 | 0.0768 | -0.0082 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.4684 | 0.9804 | 0.0732 | 0.9714 | 0.0385 | 0.0337 | 0.9615 | 0.0374 | 0.0763 | 0.0737 | -0.0027 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.4604 | 0.9798 | 0.0543 | 0.9674 | 0.0281 | 0.0337 | 0.9719 | 0.0291 | 0.0811 | 0.0659 | -0.0152 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.4570 | 0.9789 | 0.0441 | 0.9651 | 0.0227 | 0.0445 | 0.9773 | 0.0325 | 0.0808 | 0.0528 | -0.0280 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.4273 | 0.9782 | 0.0435 | 0.9617 | 0.0223 | 0.0460 | 0.9777 | 0.0253 | 0.0879 | 0.0552 | -0.0327 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.4289 | 0.9784 | 0.0563 | 0.9727 | 0.0291 | 0.0414 | 0.9709 | 0.0306 | 0.0835 | 0.0539 | -0.0296 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4883 | 0.9710 | 0.0207 | 0.9340 | 0.0105 | 0.0166 | 0.9895 | 0.0414 | 0.0834 | 0.0749 | -0.0084 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4960 | 0.9739 | 0.0790 | 0.9355 | 0.0420 | 0.0257 | 0.9580 | 0.0654 | 0.0990 | 0.1145 | 0.0155 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.5042 | 0.9722 | 0.0471 | 0.9037 | 0.0244 | 0.0196 | 0.9756 | 0.0443 | 0.0791 | 0.0829 | 0.0037 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4916 | 0.9717 | 0.0461 | 0.9029 | 0.0239 | 0.0197 | 0.9761 | 0.0570 | 0.0853 | 0.0853 | 0.0001 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4834 | 0.9701 | 0.0368 | 0.9234 | 0.0189 | 0.0271 | 0.9811 | 0.0452 | 0.0894 | 0.0709 | -0.0185 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4624 | 0.9687 | 0.0225 | 0.9189 | 0.0114 | 0.0242 | 0.9886 | 0.0430 | 0.0900 | 0.0633 | -0.0267 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4566 | 0.9690 | 0.0311 | 0.9355 | 0.0160 | 0.0257 | 0.9840 | 0.0476 | 0.0970 | 0.0757 | -0.0212 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4488 | 0.9692 | 0.0341 | 0.9391 | 0.0174 | 0.0226 | 0.9826 | 0.0361 | 0.0867 | 0.0667 | -0.0200 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4355 | 0.9684 | 0.0369 | 0.9560 | 0.0189 | 0.0257 | 0.9811 | 0.0467 | 0.1082 | 0.0788 | -0.0294 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4237 | 0.9677 | 0.0383 | 0.9593 | 0.0196 | 0.0182 | 0.9804 | 0.0361 | 0.1009 | 0.0709 | -0.0299 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4883 | 0.9710 | 0.0382 | 0.9484 | 0.0195 | 0.0271 | 0.9805 | 0.0414 | 0.0834 | 0.0749 | -0.0084 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4960 | 0.9739 | 0.1401 | 0.9376 | 0.0771 | 0.0484 | 0.9229 | 0.0654 | 0.0990 | 0.1145 | 0.0155 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.5042 | 0.9722 | 0.0735 | 0.8952 | 0.0387 | 0.0347 | 0.9613 | 0.0443 | 0.0791 | 0.0829 | 0.0037 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4916 | 0.9717 | 0.0898 | 0.8959 | 0.0478 | 0.0513 | 0.9522 | 0.0570 | 0.0853 | 0.0853 | 0.0001 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4834 | 0.9701 | 0.0707 | 0.9286 | 0.0369 | 0.0603 | 0.9631 | 0.0452 | 0.0894 | 0.0709 | -0.0185 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4624 | 0.9687 | 0.0557 | 0.9216 | 0.0288 | 0.0543 | 0.9712 | 0.0430 | 0.0900 | 0.0633 | -0.0267 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4566 | 0.9690 | 0.0742 | 0.9496 | 0.0389 | 0.0557 | 0.9611 | 0.0476 | 0.0970 | 0.0757 | -0.0212 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4488 | 0.9692 | 0.0703 | 0.9469 | 0.0368 | 0.0452 | 0.9632 | 0.0361 | 0.0867 | 0.0667 | -0.0200 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4355 | 0.9684 | 0.0787 | 0.9527 | 0.0413 | 0.0573 | 0.9587 | 0.0467 | 0.1082 | 0.0788 | -0.0294 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4237 | 0.9677 | 0.0742 | 0.9477 | 0.0389 | 0.0544 | 0.9611 | 0.0361 | 0.1009 | 0.0709 | -0.0299 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 1 | 0.4883 | 0.9710 | 0.0382 | 0.9484 | 0.0195 | 0.0271 | 0.9805 | 0.0414 | 0.0834 | 0.0749 | -0.0084 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 2 | 0.4960 | 0.9739 | 0.1401 | 0.9376 | 0.0771 | 0.0484 | 0.9229 | 0.0654 | 0.0990 | 0.1145 | 0.0155 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 3 | 0.5042 | 0.9722 | 0.0735 | 0.8952 | 0.0387 | 0.0347 | 0.9613 | 0.0443 | 0.0791 | 0.0829 | 0.0037 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 4 | 0.4916 | 0.9717 | 0.0898 | 0.8959 | 0.0478 | 0.0513 | 0.9522 | 0.0570 | 0.0853 | 0.0853 | 0.0001 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 5 | 0.4834 | 0.9701 | 0.0707 | 0.9286 | 0.0369 | 0.0603 | 0.9631 | 0.0452 | 0.0894 | 0.0709 | -0.0185 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 6 | 0.4624 | 0.9687 | 0.0557 | 0.9216 | 0.0288 | 0.0543 | 0.9712 | 0.0430 | 0.0900 | 0.0633 | -0.0267 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 7 | 0.4566 | 0.9690 | 0.0742 | 0.9496 | 0.0389 | 0.0557 | 0.9611 | 0.0476 | 0.0970 | 0.0757 | -0.0212 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 8 | 0.4488 | 0.9692 | 0.0703 | 0.9469 | 0.0368 | 0.0452 | 0.9632 | 0.0361 | 0.0867 | 0.0667 | -0.0200 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 9 | 0.4355 | 0.9684 | 0.0787 | 0.9527 | 0.0413 | 0.0573 | 0.9587 | 0.0467 | 0.1082 | 0.0788 | -0.0294 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 10 | 0.4237 | 0.9677 | 0.0742 | 0.9477 | 0.0389 | 0.0544 | 0.9611 | 0.0361 | 0.1009 | 0.0709 | -0.0299 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4954 | 0.9712 | 0.0389 | 0.9262 | 0.0200 | 0.0211 | 0.9800 | 0.0510 | 0.0962 | 0.0902 | -0.0060 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.5000 | 0.9748 | 0.1406 | 0.9308 | 0.0803 | 0.0379 | 0.9197 | 0.0632 | 0.1123 | 0.1444 | 0.0321 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.5063 | 0.9720 | 0.0794 | 0.8977 | 0.0424 | 0.0393 | 0.9576 | 0.0510 | 0.1016 | 0.1031 | 0.0015 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4927 | 0.9716 | 0.0813 | 0.8902 | 0.0433 | 0.0423 | 0.9567 | 0.0532 | 0.1054 | 0.1025 | -0.0029 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4815 | 0.9702 | 0.0589 | 0.8758 | 0.0307 | 0.0528 | 0.9693 | 0.0474 | 0.1106 | 0.0910 | -0.0196 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4691 | 0.9693 | 0.0514 | 0.9161 | 0.0272 | 0.0483 | 0.9728 | 0.0528 | 0.1198 | 0.0935 | -0.0263 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4590 | 0.9692 | 0.0503 | 0.9462 | 0.0263 | 0.0377 | 0.9737 | 0.0484 | 0.1105 | 0.0881 | -0.0224 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4539 | 0.9696 | 0.0592 | 0.9465 | 0.0311 | 0.0393 | 0.9689 | 0.0431 | 0.1114 | 0.0898 | -0.0216 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4415 | 0.9688 | 0.0683 | 0.9570 | 0.0362 | 0.0423 | 0.9638 | 0.0455 | 0.1253 | 0.0983 | -0.0270 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4291 | 0.9683 | 0.0679 | 0.9500 | 0.0358 | 0.0453 | 0.9642 | 0.0381 | 0.1203 | 0.0900 | -0.0303 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4954 | 0.9712 | 0.0766 | 0.9404 | 0.0405 | 0.0391 | 0.9595 | 0.0510 | 0.0962 | 0.0902 | -0.0060 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.5000 | 0.9748 | 0.1891 | 0.9341 | 0.1097 | 0.0605 | 0.8903 | 0.0632 | 0.1123 | 0.1444 | 0.0321 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.5063 | 0.9720 | 0.1143 | 0.8931 | 0.0622 | 0.0663 | 0.9378 | 0.0510 | 0.1016 | 0.1031 | 0.0015 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4927 | 0.9716 | 0.1236 | 0.8879 | 0.0674 | 0.0709 | 0.9326 | 0.0532 | 0.1054 | 0.1025 | -0.0029 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4815 | 0.9702 | 0.0913 | 0.9328 | 0.0483 | 0.0663 | 0.9517 | 0.0474 | 0.1106 | 0.0910 | -0.0196 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4691 | 0.9693 | 0.0949 | 0.9237 | 0.0509 | 0.0739 | 0.9491 | 0.0528 | 0.1198 | 0.0935 | -0.0263 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4590 | 0.9692 | 0.0897 | 0.9486 | 0.0477 | 0.0693 | 0.9523 | 0.0484 | 0.1105 | 0.0881 | -0.0224 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4539 | 0.9696 | 0.0988 | 0.9497 | 0.0530 | 0.0694 | 0.9470 | 0.0431 | 0.1114 | 0.0898 | -0.0216 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4415 | 0.9688 | 0.1105 | 0.9530 | 0.0595 | 0.0770 | 0.9405 | 0.0455 | 0.1253 | 0.0983 | -0.0270 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4291 | 0.9683 | 0.0963 | 0.9446 | 0.0516 | 0.0679 | 0.9484 | 0.0381 | 0.1203 | 0.0900 | -0.0303 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4954 | 0.9712 | 0.0766 | 0.9404 | 0.0405 | 0.0391 | 0.9595 | 0.0510 | 0.0962 | 0.0902 | -0.0060 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.5000 | 0.9748 | 0.1891 | 0.9341 | 0.1097 | 0.0605 | 0.8903 | 0.0632 | 0.1123 | 0.1444 | 0.0321 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.5063 | 0.9720 | 0.1143 | 0.8931 | 0.0622 | 0.0663 | 0.9378 | 0.0510 | 0.1016 | 0.1031 | 0.0015 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4927 | 0.9716 | 0.1236 | 0.8879 | 0.0674 | 0.0709 | 0.9326 | 0.0532 | 0.1054 | 0.1025 | -0.0029 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.4815 | 0.9702 | 0.0913 | 0.9328 | 0.0483 | 0.0663 | 0.9517 | 0.0474 | 0.1106 | 0.0910 | -0.0196 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4691 | 0.9693 | 0.0949 | 0.9237 | 0.0509 | 0.0739 | 0.9491 | 0.0528 | 0.1198 | 0.0935 | -0.0263 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4590 | 0.9692 | 0.0897 | 0.9486 | 0.0477 | 0.0693 | 0.9523 | 0.0484 | 0.1105 | 0.0881 | -0.0224 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4539 | 0.9696 | 0.0988 | 0.9497 | 0.0530 | 0.0694 | 0.9470 | 0.0431 | 0.1114 | 0.0898 | -0.0216 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4415 | 0.9688 | 0.1105 | 0.9530 | 0.0595 | 0.0770 | 0.9405 | 0.0455 | 0.1253 | 0.0983 | -0.0270 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4291 | 0.9683 | 0.0963 | 0.9446 | 0.0516 | 0.0679 | 0.9484 | 0.0381 | 0.1203 | 0.0900 | -0.0303 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4879 | 0.9708 | 0.0188 | 0.9363 | 0.0095 | 0.0181 | 0.9905 | 0.0433 | 0.0851 | 0.0748 | -0.0103 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4988 | 0.9724 | 0.0486 | 0.9441 | 0.0252 | 0.0288 | 0.9748 | 0.0518 | 0.1012 | 0.0971 | -0.0040 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4989 | 0.9719 | 0.0434 | 0.9303 | 0.0225 | 0.0242 | 0.9775 | 0.0426 | 0.0835 | 0.0810 | -0.0025 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4917 | 0.9714 | 0.0400 | 0.9318 | 0.0206 | 0.0272 | 0.9794 | 0.0578 | 0.0849 | 0.0831 | -0.0018 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.4837 | 0.9702 | 0.0340 | 0.9357 | 0.0174 | 0.0271 | 0.9826 | 0.0460 | 0.0882 | 0.0700 | -0.0181 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4612 | 0.9682 | 0.0192 | 0.9143 | 0.0097 | 0.0257 | 0.9903 | 0.0463 | 0.0890 | 0.0619 | -0.0271 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4517 | 0.9689 | 0.0309 | 0.9330 | 0.0158 | 0.0256 | 0.9842 | 0.0466 | 0.1015 | 0.0754 | -0.0261 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4477 | 0.9690 | 0.0311 | 0.9386 | 0.0158 | 0.0212 | 0.9842 | 0.0398 | 0.0850 | 0.0632 | -0.0219 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4316 | 0.9680 | 0.0327 | 0.9535 | 0.0167 | 0.0257 | 0.9833 | 0.0469 | 0.1049 | 0.0701 | -0.0348 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4206 | 0.9673 | 0.0343 | 0.9426 | 0.0176 | 0.0242 | 0.9824 | 0.0353 | 0.1052 | 0.0672 | -0.0379 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4879 | 0.9708 | 0.0341 | 0.9502 | 0.0174 | 0.0317 | 0.9826 | 0.0433 | 0.0851 | 0.0748 | -0.0103 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4988 | 0.9724 | 0.0967 | 0.9374 | 0.0521 | 0.0560 | 0.9479 | 0.0518 | 0.1012 | 0.0971 | -0.0040 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4989 | 0.9719 | 0.0698 | 0.9592 | 0.0366 | 0.0439 | 0.9634 | 0.0426 | 0.0835 | 0.0810 | -0.0025 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4917 | 0.9714 | 0.0833 | 0.9511 | 0.0441 | 0.0453 | 0.9559 | 0.0578 | 0.0849 | 0.0831 | -0.0018 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.4837 | 0.9702 | 0.0708 | 0.9429 | 0.0370 | 0.0588 | 0.9630 | 0.0460 | 0.0882 | 0.0700 | -0.0181 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4612 | 0.9682 | 0.0561 | 0.9388 | 0.0290 | 0.0543 | 0.9710 | 0.0463 | 0.0890 | 0.0619 | -0.0271 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4517 | 0.9689 | 0.0759 | 0.9545 | 0.0397 | 0.0603 | 0.9603 | 0.0466 | 0.1015 | 0.0754 | -0.0261 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4477 | 0.9690 | 0.0739 | 0.9548 | 0.0387 | 0.0482 | 0.9613 | 0.0398 | 0.0850 | 0.0632 | -0.0219 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4316 | 0.9680 | 0.0725 | 0.9558 | 0.0378 | 0.0543 | 0.9622 | 0.0469 | 0.1049 | 0.0701 | -0.0348 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4206 | 0.9673 | 0.0791 | 0.9459 | 0.0414 | 0.0725 | 0.9586 | 0.0353 | 0.1052 | 0.0672 | -0.0379 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.4879 | 0.9708 | 0.0341 | 0.9502 | 0.0174 | 0.0317 | 0.9826 | 0.0433 | 0.0851 | 0.0748 | -0.0103 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.4988 | 0.9724 | 0.0967 | 0.9374 | 0.0521 | 0.0560 | 0.9479 | 0.0518 | 0.1012 | 0.0971 | -0.0040 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.4989 | 0.9719 | 0.0698 | 0.9592 | 0.0366 | 0.0439 | 0.9634 | 0.0426 | 0.0835 | 0.0810 | -0.0025 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.4917 | 0.9714 | 0.0833 | 0.9511 | 0.0441 | 0.0453 | 0.9559 | 0.0578 | 0.0849 | 0.0831 | -0.0018 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.4837 | 0.9702 | 0.0708 | 0.9429 | 0.0370 | 0.0588 | 0.9630 | 0.0460 | 0.0882 | 0.0700 | -0.0181 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.4612 | 0.9682 | 0.0561 | 0.9388 | 0.0290 | 0.0543 | 0.9710 | 0.0463 | 0.0890 | 0.0619 | -0.0271 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.4517 | 0.9689 | 0.0759 | 0.9545 | 0.0397 | 0.0603 | 0.9603 | 0.0466 | 0.1015 | 0.0754 | -0.0261 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.4477 | 0.9690 | 0.0739 | 0.9548 | 0.0387 | 0.0482 | 0.9613 | 0.0398 | 0.0850 | 0.0632 | -0.0219 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.4316 | 0.9680 | 0.0725 | 0.9558 | 0.0378 | 0.0543 | 0.9622 | 0.0469 | 0.1049 | 0.0701 | -0.0348 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.4206 | 0.9673 | 0.0791 | 0.9459 | 0.0414 | 0.0725 | 0.9586 | 0.0353 | 0.1052 | 0.0672 | -0.0379 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5124 | 0.9745 | 0.0748 | 0.9099 | 0.0406 | 0.0165 | 0.9594 | 0.0647 | 0.0910 | 0.1088 | 0.0178 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4688 | 0.9716 | 0.1139 | 0.9484 | 0.0637 | 0.0332 | 0.9363 | 0.0611 | 0.1209 | 0.1323 | 0.0115 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4877 | 0.9733 | 0.1210 | 0.9075 | 0.0689 | 0.0256 | 0.9311 | 0.0669 | 0.1042 | 0.1352 | 0.0310 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4932 | 0.9734 | 0.0853 | 0.9252 | 0.0472 | 0.0212 | 0.9528 | 0.0525 | 0.0821 | 0.1062 | 0.0242 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5082 | 0.9728 | 0.0485 | 0.9177 | 0.0252 | 0.0212 | 0.9748 | 0.0539 | 0.0758 | 0.0834 | 0.0076 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4842 | 0.9707 | 0.0243 | 0.6113 | 0.0126 | 0.0121 | 0.9874 | 0.0371 | 0.0690 | 0.0643 | -0.0048 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4794 | 0.9715 | 0.0243 | 0.6799 | 0.0126 | 0.0196 | 0.9874 | 0.0268 | 0.0700 | 0.0619 | -0.0081 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5446 | 0.9766 | 0.1025 | 0.6620 | 0.0630 | 0.0121 | 0.9370 | 0.0432 | 0.0611 | 0.1141 | 0.0530 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5541 | 0.9773 | 0.0955 | 0.9353 | 0.0609 | 0.0196 | 0.9391 | 0.0683 | 0.0694 | 0.1246 | 0.0553 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5293 | 0.9752 | 0.0873 | 0.8533 | 0.0487 | 0.0182 | 0.9513 | 0.0455 | 0.0617 | 0.1018 | 0.0401 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5124 | 0.9745 | 0.1144 | 0.8927 | 0.0642 | 0.0436 | 0.9358 | 0.0647 | 0.0910 | 0.1088 | 0.0178 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4688 | 0.9716 | 0.1675 | 0.9458 | 0.0956 | 0.0589 | 0.9044 | 0.0611 | 0.1209 | 0.1323 | 0.0115 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4877 | 0.9733 | 0.1758 | 0.9194 | 0.1015 | 0.0573 | 0.8985 | 0.0669 | 0.1042 | 0.1352 | 0.0310 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4932 | 0.9734 | 0.1161 | 0.9318 | 0.0652 | 0.0348 | 0.9348 | 0.0525 | 0.0821 | 0.1062 | 0.0242 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5082 | 0.9728 | 0.0896 | 0.9415 | 0.0478 | 0.0438 | 0.9522 | 0.0539 | 0.0758 | 0.0834 | 0.0076 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4842 | 0.9707 | 0.0490 | 0.8478 | 0.0262 | 0.0378 | 0.9738 | 0.0371 | 0.0690 | 0.0643 | -0.0048 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4794 | 0.9715 | 0.0414 | 0.7728 | 0.0220 | 0.0287 | 0.9780 | 0.0268 | 0.0700 | 0.0619 | -0.0081 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5446 | 0.9766 | 0.1366 | 0.6884 | 0.0874 | 0.0287 | 0.9126 | 0.0432 | 0.0611 | 0.1141 | 0.0530 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5541 | 0.9773 | 0.1408 | 0.9604 | 0.0896 | 0.0332 | 0.9104 | 0.0683 | 0.0694 | 0.1246 | 0.0553 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5293 | 0.9752 | 0.1203 | 0.8833 | 0.0687 | 0.0257 | 0.9313 | 0.0455 | 0.0617 | 0.1018 | 0.0401 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 1 | 0.5124 | 0.9745 | 0.1144 | 0.8927 | 0.0642 | 0.0436 | 0.9358 | 0.0647 | 0.0910 | 0.1088 | 0.0178 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 2 | 0.4688 | 0.9716 | 0.1675 | 0.9458 | 0.0956 | 0.0589 | 0.9044 | 0.0611 | 0.1209 | 0.1323 | 0.0115 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 3 | 0.4877 | 0.9733 | 0.1758 | 0.9194 | 0.1015 | 0.0573 | 0.8985 | 0.0669 | 0.1042 | 0.1352 | 0.0310 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 4 | 0.4932 | 0.9734 | 0.1161 | 0.9318 | 0.0652 | 0.0348 | 0.9348 | 0.0525 | 0.0821 | 0.1062 | 0.0242 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 5 | 0.5082 | 0.9728 | 0.0896 | 0.9415 | 0.0478 | 0.0438 | 0.9522 | 0.0539 | 0.0758 | 0.0834 | 0.0076 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 6 | 0.4842 | 0.9707 | 0.0490 | 0.8478 | 0.0262 | 0.0378 | 0.9738 | 0.0371 | 0.0690 | 0.0643 | -0.0048 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 7 | 0.4794 | 0.9715 | 0.0414 | 0.7728 | 0.0220 | 0.0287 | 0.9780 | 0.0268 | 0.0700 | 0.0619 | -0.0081 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 8 | 0.5446 | 0.9766 | 0.1366 | 0.6884 | 0.0874 | 0.0287 | 0.9126 | 0.0432 | 0.0611 | 0.1141 | 0.0530 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 9 | 0.5541 | 0.9773 | 0.1408 | 0.9604 | 0.0896 | 0.0332 | 0.9104 | 0.0683 | 0.0694 | 0.1246 | 0.0553 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 10 | 0.5293 | 0.9752 | 0.1203 | 0.8833 | 0.0687 | 0.0257 | 0.9313 | 0.0455 | 0.0617 | 0.1018 | 0.0401 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5144 | 0.9746 | 0.1239 | 0.9797 | 0.0674 | 0.0377 | 0.9326 | 0.0692 | 0.1089 | 0.1380 | 0.0291 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4712 | 0.9725 | 0.1425 | 0.9424 | 0.0819 | 0.0498 | 0.9181 | 0.0543 | 0.1336 | 0.1465 | 0.0129 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4868 | 0.9731 | 0.1563 | 0.9032 | 0.0919 | 0.0393 | 0.9081 | 0.0539 | 0.1163 | 0.1492 | 0.0329 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4907 | 0.9731 | 0.1032 | 0.9211 | 0.0576 | 0.0348 | 0.9424 | 0.0519 | 0.0957 | 0.1177 | 0.0220 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5074 | 0.9727 | 0.0728 | 0.9178 | 0.0390 | 0.0333 | 0.9610 | 0.0527 | 0.0852 | 0.0929 | 0.0077 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4922 | 0.9707 | 0.0442 | 0.9234 | 0.0228 | 0.0302 | 0.9772 | 0.0436 | 0.0875 | 0.0808 | -0.0067 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4829 | 0.9712 | 0.0553 | 0.9263 | 0.0289 | 0.0317 | 0.9711 | 0.0521 | 0.1035 | 0.0938 | -0.0098 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5433 | 0.9764 | 0.1452 | 0.9291 | 0.0861 | 0.0347 | 0.9139 | 0.0657 | 0.0913 | 0.1525 | 0.0612 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5569 | 0.9772 | 0.1808 | 0.9244 | 0.1087 | 0.0317 | 0.8913 | 0.0839 | 0.0899 | 0.1792 | 0.0892 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5251 | 0.9752 | 0.1488 | 0.9489 | 0.0835 | 0.0362 | 0.9165 | 0.0833 | 0.0988 | 0.1541 | 0.0553 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5144 | 0.9746 | 0.1669 | 0.9796 | 0.0933 | 0.0602 | 0.9067 | 0.0692 | 0.1089 | 0.1380 | 0.0291 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4712 | 0.9725 | 0.1884 | 0.9406 | 0.1101 | 0.0755 | 0.8899 | 0.0543 | 0.1336 | 0.1465 | 0.0129 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4868 | 0.9731 | 0.1861 | 0.8991 | 0.1100 | 0.0665 | 0.8900 | 0.0539 | 0.1163 | 0.1492 | 0.0329 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4907 | 0.9731 | 0.1382 | 0.9265 | 0.0780 | 0.0515 | 0.9220 | 0.0519 | 0.0957 | 0.1177 | 0.0220 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5074 | 0.9727 | 0.0979 | 0.9401 | 0.0529 | 0.0514 | 0.9471 | 0.0527 | 0.0852 | 0.0929 | 0.0077 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4922 | 0.9707 | 0.0746 | 0.9350 | 0.0394 | 0.0438 | 0.9606 | 0.0436 | 0.0875 | 0.0808 | -0.0067 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4829 | 0.9712 | 0.0988 | 0.9302 | 0.0532 | 0.0634 | 0.9468 | 0.0521 | 0.1035 | 0.0938 | -0.0098 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5433 | 0.9764 | 0.1918 | 0.9519 | 0.1169 | 0.0498 | 0.8831 | 0.0657 | 0.0913 | 0.1525 | 0.0612 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5569 | 0.9772 | 0.2365 | 0.9540 | 0.1461 | 0.0513 | 0.8539 | 0.0839 | 0.0899 | 0.1792 | 0.0892 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5251 | 0.9752 | 0.2123 | 0.9538 | 0.1262 | 0.0588 | 0.8738 | 0.0833 | 0.0988 | 0.1541 | 0.0553 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.5144 | 0.9746 | 0.1669 | 0.9796 | 0.0933 | 0.0602 | 0.9067 | 0.0692 | 0.1089 | 0.1380 | 0.0291 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4712 | 0.9725 | 0.1884 | 0.9406 | 0.1101 | 0.0755 | 0.8899 | 0.0543 | 0.1336 | 0.1465 | 0.0129 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4868 | 0.9731 | 0.1861 | 0.8991 | 0.1100 | 0.0665 | 0.8900 | 0.0539 | 0.1163 | 0.1492 | 0.0329 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4907 | 0.9731 | 0.1382 | 0.9265 | 0.0780 | 0.0515 | 0.9220 | 0.0519 | 0.0957 | 0.1177 | 0.0220 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 5 | 0.5074 | 0.9727 | 0.0979 | 0.9401 | 0.0529 | 0.0514 | 0.9471 | 0.0527 | 0.0852 | 0.0929 | 0.0077 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 6 | 0.4922 | 0.9707 | 0.0746 | 0.9350 | 0.0394 | 0.0438 | 0.9606 | 0.0436 | 0.0875 | 0.0808 | -0.0067 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4829 | 0.9712 | 0.0988 | 0.9302 | 0.0532 | 0.0634 | 0.9468 | 0.0521 | 0.1035 | 0.0938 | -0.0098 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 8 | 0.5433 | 0.9764 | 0.1918 | 0.9519 | 0.1169 | 0.0498 | 0.8831 | 0.0657 | 0.0913 | 0.1525 | 0.0612 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 9 | 0.5569 | 0.9772 | 0.2365 | 0.9540 | 0.1461 | 0.0513 | 0.8539 | 0.0839 | 0.0899 | 0.1792 | 0.0892 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 10 | 0.5251 | 0.9752 | 0.2123 | 0.9538 | 0.1262 | 0.0588 | 0.8738 | 0.0833 | 0.0988 | 0.1541 | 0.0553 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.5118 | 0.9741 | 0.0793 | 0.8254 | 0.0446 | 0.0255 | 0.9554 | 0.0703 | 0.1011 | 0.1131 | 0.0120 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4693 | 0.9713 | 0.1023 | 0.8647 | 0.0573 | 0.0362 | 0.9427 | 0.0587 | 0.1199 | 0.1259 | 0.0061 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4873 | 0.9731 | 0.1143 | 0.8239 | 0.0653 | 0.0256 | 0.9347 | 0.0733 | 0.1102 | 0.1340 | 0.0238 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4912 | 0.9733 | 0.0786 | 0.8416 | 0.0440 | 0.0197 | 0.9560 | 0.0542 | 0.0825 | 0.1023 | 0.0197 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5107 | 0.9729 | 0.0502 | 0.9200 | 0.0263 | 0.0197 | 0.9737 | 0.0559 | 0.0786 | 0.0879 | 0.0092 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4933 | 0.9710 | 0.0388 | 0.6084 | 0.0208 | 0.0227 | 0.9792 | 0.0615 | 0.0829 | 0.0813 | -0.0016 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4860 | 0.9715 | 0.0335 | 0.5898 | 0.0181 | 0.0166 | 0.9819 | 0.0468 | 0.0768 | 0.0719 | -0.0049 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.5408 | 0.9762 | 0.0969 | 0.5787 | 0.0645 | 0.0166 | 0.9355 | 0.0644 | 0.0788 | 0.1239 | 0.0451 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.5497 | 0.9763 | 0.0919 | 0.7007 | 0.0647 | 0.0181 | 0.9353 | 0.0709 | 0.0864 | 0.1313 | 0.0449 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.5258 | 0.9744 | 0.0802 | 0.7871 | 0.0452 | 0.0197 | 0.9548 | 0.0621 | 0.0764 | 0.1055 | 0.0291 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.5118 | 0.9741 | 0.1218 | 0.8117 | 0.0708 | 0.0556 | 0.9292 | 0.0703 | 0.1011 | 0.1131 | 0.0120 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4693 | 0.9713 | 0.1511 | 0.9458 | 0.0865 | 0.0619 | 0.9135 | 0.0587 | 0.1199 | 0.1259 | 0.0061 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4873 | 0.9731 | 0.1762 | 0.9129 | 0.1023 | 0.0708 | 0.8977 | 0.0733 | 0.1102 | 0.1340 | 0.0238 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4912 | 0.9733 | 0.1081 | 0.8477 | 0.0611 | 0.0393 | 0.9389 | 0.0542 | 0.0825 | 0.1023 | 0.0197 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5107 | 0.9729 | 0.0994 | 0.9396 | 0.0542 | 0.0468 | 0.9458 | 0.0559 | 0.0786 | 0.0879 | 0.0092 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4933 | 0.9710 | 0.0805 | 0.6171 | 0.0470 | 0.0485 | 0.9530 | 0.0615 | 0.0829 | 0.0813 | -0.0016 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4860 | 0.9715 | 0.0547 | 0.5969 | 0.0310 | 0.0363 | 0.9690 | 0.0468 | 0.0768 | 0.0719 | -0.0049 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.5408 | 0.9762 | 0.1384 | 0.5774 | 0.0945 | 0.0363 | 0.9055 | 0.0644 | 0.0788 | 0.1239 | 0.0451 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.5497 | 0.9763 | 0.1347 | 0.7958 | 0.0937 | 0.0377 | 0.9063 | 0.0709 | 0.0864 | 0.1313 | 0.0449 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.5258 | 0.9744 | 0.1213 | 0.8779 | 0.0725 | 0.0394 | 0.9275 | 0.0621 | 0.0764 | 0.1055 | 0.0291 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.5118 | 0.9741 | 0.1218 | 0.8117 | 0.0708 | 0.0556 | 0.9292 | 0.0703 | 0.1011 | 0.1131 | 0.0120 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.4693 | 0.9713 | 0.1511 | 0.9458 | 0.0865 | 0.0619 | 0.9135 | 0.0587 | 0.1199 | 0.1259 | 0.0061 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.4873 | 0.9731 | 0.1762 | 0.9129 | 0.1023 | 0.0708 | 0.8977 | 0.0733 | 0.1102 | 0.1340 | 0.0238 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 4 | 0.4912 | 0.9733 | 0.1081 | 0.8477 | 0.0611 | 0.0393 | 0.9389 | 0.0542 | 0.0825 | 0.1023 | 0.0197 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 5 | 0.5107 | 0.9729 | 0.0994 | 0.9396 | 0.0542 | 0.0468 | 0.9458 | 0.0559 | 0.0786 | 0.0879 | 0.0092 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 6 | 0.4933 | 0.9710 | 0.0805 | 0.6171 | 0.0470 | 0.0485 | 0.9530 | 0.0615 | 0.0829 | 0.0813 | -0.0016 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 7 | 0.4860 | 0.9715 | 0.0547 | 0.5969 | 0.0310 | 0.0363 | 0.9690 | 0.0468 | 0.0768 | 0.0719 | -0.0049 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 8 | 0.5408 | 0.9762 | 0.1384 | 0.5774 | 0.0945 | 0.0363 | 0.9055 | 0.0644 | 0.0788 | 0.1239 | 0.0451 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 9 | 0.5497 | 0.9763 | 0.1347 | 0.7958 | 0.0937 | 0.0377 | 0.9063 | 0.0709 | 0.0864 | 0.1313 | 0.0449 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 10 | 0.5258 | 0.9744 | 0.1213 | 0.8779 | 0.0725 | 0.0394 | 0.9275 | 0.0621 | 0.0764 | 0.1055 | 0.0291 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4924 | 0.9710 | 0.0320 | 0.9519 | 0.0163 | 0.0212 | 0.9837 | 0.0368 | 0.0884 | 0.0792 | -0.0092 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4933 | 0.9745 | 0.0564 | 0.9346 | 0.0300 | 0.0106 | 0.9700 | 0.0619 | 0.0815 | 0.1007 | 0.0191 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4915 | 0.9746 | 0.0775 | 0.9234 | 0.0419 | 0.0166 | 0.9581 | 0.0518 | 0.0820 | 0.1051 | 0.0231 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4909 | 0.9736 | 0.0653 | 0.9286 | 0.0341 | 0.0196 | 0.9659 | 0.0724 | 0.0918 | 0.1067 | 0.0150 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5152 | 0.9738 | 0.0536 | 0.9150 | 0.0280 | 0.0272 | 0.9720 | 0.0552 | 0.0802 | 0.0919 | 0.0118 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5011 | 0.9719 | 0.0310 | 0.9229 | 0.0159 | 0.0256 | 0.9841 | 0.0457 | 0.0825 | 0.0729 | -0.0096 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4677 | 0.9702 | 0.0311 | 0.9325 | 0.0159 | 0.0196 | 0.9841 | 0.0399 | 0.0777 | 0.0658 | -0.0119 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4497 | 0.9691 | 0.0380 | 0.9340 | 0.0195 | 0.0317 | 0.9805 | 0.0401 | 0.0934 | 0.0689 | -0.0245 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4727 | 0.9710 | 0.0562 | 0.9372 | 0.0295 | 0.0151 | 0.9705 | 0.0447 | 0.0853 | 0.0865 | 0.0012 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4485 | 0.9695 | 0.0444 | 0.9226 | 0.0230 | 0.0212 | 0.9770 | 0.0407 | 0.0947 | 0.0801 | -0.0146 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4924 | 0.9710 | 0.0547 | 0.9526 | 0.0282 | 0.0377 | 0.9718 | 0.0368 | 0.0884 | 0.0792 | -0.0092 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4933 | 0.9745 | 0.1068 | 0.9399 | 0.0578 | 0.0242 | 0.9422 | 0.0619 | 0.0815 | 0.1007 | 0.0191 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4915 | 0.9746 | 0.1114 | 0.9177 | 0.0616 | 0.0288 | 0.9384 | 0.0518 | 0.0820 | 0.1051 | 0.0231 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4909 | 0.9736 | 0.1193 | 0.9276 | 0.0646 | 0.0482 | 0.9354 | 0.0724 | 0.0918 | 0.1067 | 0.0150 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5152 | 0.9738 | 0.1033 | 0.9339 | 0.0556 | 0.0393 | 0.9444 | 0.0552 | 0.0802 | 0.0919 | 0.0118 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5011 | 0.9719 | 0.0616 | 0.9261 | 0.0320 | 0.0498 | 0.9680 | 0.0457 | 0.0825 | 0.0729 | -0.0096 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4677 | 0.9702 | 0.0610 | 0.9457 | 0.0316 | 0.0378 | 0.9684 | 0.0399 | 0.0777 | 0.0658 | -0.0119 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4497 | 0.9691 | 0.0658 | 0.9440 | 0.0342 | 0.0513 | 0.9658 | 0.0401 | 0.0934 | 0.0689 | -0.0245 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4727 | 0.9710 | 0.0905 | 0.9454 | 0.0484 | 0.0378 | 0.9516 | 0.0447 | 0.0853 | 0.0865 | 0.0012 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4485 | 0.9695 | 0.0742 | 0.9440 | 0.0389 | 0.0408 | 0.9611 | 0.0407 | 0.0947 | 0.0801 | -0.0146 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 1 | 0.4924 | 0.9710 | 0.0547 | 0.9526 | 0.0282 | 0.0377 | 0.9718 | 0.0368 | 0.0884 | 0.0792 | -0.0092 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 2 | 0.4933 | 0.9745 | 0.1068 | 0.9399 | 0.0578 | 0.0242 | 0.9422 | 0.0619 | 0.0815 | 0.1007 | 0.0191 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 3 | 0.4915 | 0.9746 | 0.1114 | 0.9177 | 0.0616 | 0.0288 | 0.9384 | 0.0518 | 0.0820 | 0.1051 | 0.0231 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 4 | 0.4909 | 0.9736 | 0.1193 | 0.9276 | 0.0646 | 0.0482 | 0.9354 | 0.0724 | 0.0918 | 0.1067 | 0.0150 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 5 | 0.5152 | 0.9738 | 0.1033 | 0.9339 | 0.0556 | 0.0393 | 0.9444 | 0.0552 | 0.0802 | 0.0919 | 0.0118 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 6 | 0.5011 | 0.9719 | 0.0616 | 0.9261 | 0.0320 | 0.0498 | 0.9680 | 0.0457 | 0.0825 | 0.0729 | -0.0096 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 7 | 0.4677 | 0.9702 | 0.0610 | 0.9457 | 0.0316 | 0.0378 | 0.9684 | 0.0399 | 0.0777 | 0.0658 | -0.0119 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 8 | 0.4497 | 0.9691 | 0.0658 | 0.9440 | 0.0342 | 0.0513 | 0.9658 | 0.0401 | 0.0934 | 0.0689 | -0.0245 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 9 | 0.4727 | 0.9710 | 0.0905 | 0.9454 | 0.0484 | 0.0378 | 0.9516 | 0.0447 | 0.0853 | 0.0865 | 0.0012 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 10 | 0.4485 | 0.9695 | 0.0742 | 0.9440 | 0.0389 | 0.0408 | 0.9611 | 0.0407 | 0.0947 | 0.0801 | -0.0146 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4925 | 0.9712 | 0.0468 | 0.9302 | 0.0242 | 0.0286 | 0.9758 | 0.0435 | 0.0965 | 0.0890 | -0.0075 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4957 | 0.9752 | 0.1287 | 0.9377 | 0.0746 | 0.0152 | 0.9254 | 0.0550 | 0.0948 | 0.1345 | 0.0397 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4927 | 0.9747 | 0.1528 | 0.9355 | 0.0910 | 0.0242 | 0.9090 | 0.0453 | 0.1070 | 0.1479 | 0.0409 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4928 | 0.9736 | 0.1609 | 0.9260 | 0.0932 | 0.0362 | 0.9068 | 0.0637 | 0.1124 | 0.1550 | 0.0426 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5177 | 0.9741 | 0.1087 | 0.9022 | 0.0590 | 0.0242 | 0.9410 | 0.0532 | 0.0879 | 0.1213 | 0.0334 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.5057 | 0.9721 | 0.0572 | 0.9268 | 0.0299 | 0.0377 | 0.9701 | 0.0527 | 0.0948 | 0.0941 | -0.0007 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4724 | 0.9705 | 0.0504 | 0.9238 | 0.0262 | 0.0226 | 0.9738 | 0.0416 | 0.0874 | 0.0807 | -0.0067 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4547 | 0.9693 | 0.0517 | 0.9328 | 0.0267 | 0.0317 | 0.9733 | 0.0437 | 0.1009 | 0.0818 | -0.0191 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4764 | 0.9712 | 0.0768 | 0.9311 | 0.0409 | 0.0272 | 0.9591 | 0.0495 | 0.0990 | 0.1018 | 0.0028 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4540 | 0.9697 | 0.0627 | 0.9416 | 0.0329 | 0.0256 | 0.9671 | 0.0435 | 0.1029 | 0.0936 | -0.0093 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4925 | 0.9712 | 0.0720 | 0.9443 | 0.0377 | 0.0422 | 0.9623 | 0.0435 | 0.0965 | 0.0890 | -0.0075 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4957 | 0.9752 | 0.1651 | 0.9237 | 0.0965 | 0.0392 | 0.9035 | 0.0550 | 0.0948 | 0.1345 | 0.0397 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4927 | 0.9747 | 0.1832 | 0.9160 | 0.1098 | 0.0529 | 0.8902 | 0.0453 | 0.1070 | 0.1479 | 0.0409 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4928 | 0.9736 | 0.2047 | 0.9353 | 0.1199 | 0.0633 | 0.8801 | 0.0637 | 0.1124 | 0.1550 | 0.0426 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5177 | 0.9741 | 0.1541 | 0.9262 | 0.0862 | 0.0497 | 0.9138 | 0.0532 | 0.0879 | 0.1213 | 0.0334 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.5057 | 0.9721 | 0.0997 | 0.9277 | 0.0539 | 0.0588 | 0.9461 | 0.0527 | 0.0948 | 0.0941 | -0.0007 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4724 | 0.9705 | 0.0786 | 0.9310 | 0.0415 | 0.0437 | 0.9585 | 0.0416 | 0.0874 | 0.0807 | -0.0067 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4547 | 0.9693 | 0.0797 | 0.9276 | 0.0420 | 0.0497 | 0.9580 | 0.0437 | 0.1009 | 0.0818 | -0.0191 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4764 | 0.9712 | 0.1144 | 0.9472 | 0.0620 | 0.0483 | 0.9380 | 0.0495 | 0.0990 | 0.1018 | 0.0028 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4540 | 0.9697 | 0.0991 | 0.9347 | 0.0528 | 0.0467 | 0.9472 | 0.0435 | 0.1029 | 0.0936 | -0.0093 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.4925 | 0.9712 | 0.0720 | 0.9443 | 0.0377 | 0.0422 | 0.9623 | 0.0435 | 0.0965 | 0.0890 | -0.0075 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.4957 | 0.9752 | 0.1651 | 0.9237 | 0.0965 | 0.0392 | 0.9035 | 0.0550 | 0.0948 | 0.1345 | 0.0397 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.4927 | 0.9747 | 0.1832 | 0.9160 | 0.1098 | 0.0529 | 0.8902 | 0.0453 | 0.1070 | 0.1479 | 0.0409 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.4928 | 0.9736 | 0.2047 | 0.9353 | 0.1199 | 0.0633 | 0.8801 | 0.0637 | 0.1124 | 0.1550 | 0.0426 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.5177 | 0.9741 | 0.1541 | 0.9262 | 0.0862 | 0.0497 | 0.9138 | 0.0532 | 0.0879 | 0.1213 | 0.0334 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.5057 | 0.9721 | 0.0997 | 0.9277 | 0.0539 | 0.0588 | 0.9461 | 0.0527 | 0.0948 | 0.0941 | -0.0007 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.4724 | 0.9705 | 0.0786 | 0.9310 | 0.0415 | 0.0437 | 0.9585 | 0.0416 | 0.0874 | 0.0807 | -0.0067 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.4547 | 0.9693 | 0.0797 | 0.9276 | 0.0420 | 0.0497 | 0.9580 | 0.0437 | 0.1009 | 0.0818 | -0.0191 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.4764 | 0.9712 | 0.1144 | 0.9472 | 0.0620 | 0.0483 | 0.9380 | 0.0495 | 0.0990 | 0.1018 | 0.0028 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.4540 | 0.9697 | 0.0991 | 0.9347 | 0.0528 | 0.0467 | 0.9472 | 0.0435 | 0.1029 | 0.0936 | -0.0093 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.4923 | 0.9708 | 0.0304 | 0.9595 | 0.0155 | 0.0212 | 0.9845 | 0.0368 | 0.0910 | 0.0796 | -0.0114 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.4956 | 0.9729 | 0.0228 | 0.9312 | 0.0116 | 0.0167 | 0.9884 | 0.0530 | 0.0851 | 0.0832 | -0.0019 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.4942 | 0.9725 | 0.0398 | 0.9324 | 0.0208 | 0.0212 | 0.9792 | 0.0436 | 0.0868 | 0.0841 | -0.0027 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.4892 | 0.9723 | 0.0572 | 0.9396 | 0.0298 | 0.0256 | 0.9702 | 0.0630 | 0.1036 | 0.1019 | -0.0017 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.5116 | 0.9736 | 0.0500 | 0.9325 | 0.0261 | 0.0348 | 0.9739 | 0.0570 | 0.0878 | 0.0912 | 0.0033 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.4966 | 0.9712 | 0.0295 | 0.9175 | 0.0152 | 0.0331 | 0.9848 | 0.0446 | 0.0879 | 0.0719 | -0.0160 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.4633 | 0.9698 | 0.0278 | 0.9532 | 0.0141 | 0.0182 | 0.9859 | 0.0429 | 0.0803 | 0.0624 | -0.0179 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.4471 | 0.9689 | 0.0366 | 0.9458 | 0.0187 | 0.0302 | 0.9813 | 0.0385 | 0.0924 | 0.0647 | -0.0278 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.4623 | 0.9708 | 0.0506 | 0.9387 | 0.0264 | 0.0182 | 0.9736 | 0.0431 | 0.0881 | 0.0802 | -0.0079 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.4376 | 0.9692 | 0.0410 | 0.9131 | 0.0211 | 0.0287 | 0.9789 | 0.0411 | 0.1008 | 0.0745 | -0.0263 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.4923 | 0.9708 | 0.0543 | 0.9619 | 0.0279 | 0.0407 | 0.9721 | 0.0368 | 0.0910 | 0.0796 | -0.0114 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.4956 | 0.9729 | 0.0644 | 0.9293 | 0.0337 | 0.0423 | 0.9663 | 0.0530 | 0.0851 | 0.0832 | -0.0019 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.4942 | 0.9725 | 0.0682 | 0.9281 | 0.0368 | 0.0379 | 0.9632 | 0.0436 | 0.0868 | 0.0841 | -0.0027 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.4892 | 0.9723 | 0.1054 | 0.9388 | 0.0571 | 0.0634 | 0.9429 | 0.0630 | 0.1036 | 0.1019 | -0.0017 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.5116 | 0.9736 | 0.0998 | 0.9593 | 0.0537 | 0.0514 | 0.9463 | 0.0570 | 0.0878 | 0.0912 | 0.0033 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.4966 | 0.9712 | 0.0588 | 0.9492 | 0.0305 | 0.0558 | 0.9695 | 0.0446 | 0.0879 | 0.0719 | -0.0160 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.4633 | 0.9698 | 0.0601 | 0.9613 | 0.0310 | 0.0453 | 0.9690 | 0.0429 | 0.0803 | 0.0624 | -0.0179 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.4471 | 0.9689 | 0.0639 | 0.9551 | 0.0331 | 0.0544 | 0.9669 | 0.0385 | 0.0924 | 0.0647 | -0.0278 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.4623 | 0.9708 | 0.0891 | 0.9584 | 0.0474 | 0.0439 | 0.9526 | 0.0431 | 0.0881 | 0.0802 | -0.0079 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.4376 | 0.9692 | 0.0708 | 0.9538 | 0.0370 | 0.0529 | 0.9630 | 0.0411 | 0.1008 | 0.0745 | -0.0263 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.4923 | 0.9708 | 0.0543 | 0.9619 | 0.0279 | 0.0407 | 0.9721 | 0.0368 | 0.0910 | 0.0796 | -0.0114 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.4956 | 0.9729 | 0.0644 | 0.9293 | 0.0337 | 0.0423 | 0.9663 | 0.0530 | 0.0851 | 0.0832 | -0.0019 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.4942 | 0.9725 | 0.0682 | 0.9281 | 0.0368 | 0.0379 | 0.9632 | 0.0436 | 0.0868 | 0.0841 | -0.0027 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.4892 | 0.9723 | 0.1054 | 0.9388 | 0.0571 | 0.0634 | 0.9429 | 0.0630 | 0.1036 | 0.1019 | -0.0017 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.5116 | 0.9736 | 0.0998 | 0.9593 | 0.0537 | 0.0514 | 0.9463 | 0.0570 | 0.0878 | 0.0912 | 0.0033 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.4966 | 0.9712 | 0.0588 | 0.9492 | 0.0305 | 0.0558 | 0.9695 | 0.0446 | 0.0879 | 0.0719 | -0.0160 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.4633 | 0.9698 | 0.0601 | 0.9613 | 0.0310 | 0.0453 | 0.9690 | 0.0429 | 0.0803 | 0.0624 | -0.0179 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.4471 | 0.9689 | 0.0639 | 0.9551 | 0.0331 | 0.0544 | 0.9669 | 0.0385 | 0.0924 | 0.0647 | -0.0278 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.4623 | 0.9708 | 0.0891 | 0.9584 | 0.0474 | 0.0439 | 0.9526 | 0.0431 | 0.0881 | 0.0802 | -0.0079 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.4376 | 0.9692 | 0.0708 | 0.9538 | 0.0370 | 0.0529 | 0.9630 | 0.0411 | 0.1008 | 0.0745 | -0.0263 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `fedavg` approximates collaborative normal-only training without sharing raw vibration windows.
- `fedprox` adds a proximal penalty to reduce local client drift under non-IID data.
- `fedbn` keeps BatchNorm parameters and running statistics local to each client.
- `*-personalized` locally adapts the federated global model before client evaluation.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
