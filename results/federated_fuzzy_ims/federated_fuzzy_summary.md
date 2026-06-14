# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, FedAvg, FedProx, and personalized federated training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| clients_by | training_mode | calibration_scope | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6943 | 0.9930 | 0.6188 | 0.9993 | 0.4788 | 0.0163 | 0.5212 | 0.0395 | 0.0961 | 0.5177 | 0.4216 |
| bearing | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6943 | 0.9930 | 0.6354 | 0.9987 | 0.4956 | 0.0336 | 0.5044 | 0.0395 | 0.0961 | 0.5177 | 0.4216 |
| bearing | centralized | adaptive | cnn-ae | hard_val_p95 | 0.6943 | 0.9930 | 0.6354 | 0.9987 | 0.4956 | 0.0336 | 0.5044 | 0.0395 | 0.0961 | 0.5177 | 0.4216 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7233 | 0.9934 | 0.6631 | 0.9988 | 0.5437 | 0.0298 | 0.4563 | 0.0378 | 0.1057 | 0.5723 | 0.4666 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7233 | 0.9934 | 0.6794 | 0.9980 | 0.5614 | 0.0526 | 0.4386 | 0.0378 | 0.1057 | 0.5723 | 0.4666 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.7233 | 0.9934 | 0.6794 | 0.9980 | 0.5614 | 0.0526 | 0.4386 | 0.0378 | 0.1057 | 0.5723 | 0.4666 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6038 | 0.9901 | 0.4441 | 0.9983 | 0.2856 | 0.0594 | 0.7144 | 0.0515 | 0.1590 | 0.3528 | 0.1938 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6038 | 0.9901 | 0.4723 | 0.9971 | 0.3094 | 0.1057 | 0.6906 | 0.0515 | 0.1590 | 0.3528 | 0.1938 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.6038 | 0.9901 | 0.4723 | 0.9971 | 0.3094 | 0.1057 | 0.6906 | 0.0515 | 0.1590 | 0.3528 | 0.1938 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6652 | 0.9919 | 0.4773 | 0.9975 | 0.3491 | 0.0191 | 0.6509 | 0.0742 | 0.1031 | 0.4170 | 0.3140 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6652 | 0.9919 | 0.5194 | 0.9965 | 0.3852 | 0.0414 | 0.6148 | 0.0742 | 0.1031 | 0.4170 | 0.3140 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.6652 | 0.9919 | 0.5194 | 0.9965 | 0.3852 | 0.0414 | 0.6148 | 0.0742 | 0.1031 | 0.4170 | 0.3140 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6919 | 0.9923 | 0.5294 | 0.9973 | 0.4171 | 0.0310 | 0.5829 | 0.0674 | 0.1116 | 0.4717 | 0.3601 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6919 | 0.9923 | 0.5653 | 0.9960 | 0.4525 | 0.0575 | 0.5475 | 0.0674 | 0.1116 | 0.4717 | 0.3601 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.6919 | 0.9923 | 0.5653 | 0.9960 | 0.4525 | 0.0575 | 0.5475 | 0.0674 | 0.1116 | 0.4717 | 0.3601 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6219 | 0.9899 | 0.3176 | 0.9965 | 0.1956 | 0.0346 | 0.8044 | 0.0763 | 0.1389 | 0.2934 | 0.1544 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6219 | 0.9899 | 0.3639 | 0.9953 | 0.2265 | 0.0772 | 0.7735 | 0.0763 | 0.1389 | 0.2934 | 0.1544 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.6219 | 0.9899 | 0.3639 | 0.9953 | 0.2265 | 0.0772 | 0.7735 | 0.0763 | 0.1389 | 0.2934 | 0.1544 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7598 | 0.9946 | 0.7060 | 0.9992 | 0.5958 | 0.0190 | 0.4042 | 0.0309 | 0.0975 | 0.6202 | 0.5228 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7598 | 0.9946 | 0.7181 | 0.9984 | 0.6097 | 0.0442 | 0.3903 | 0.0309 | 0.0975 | 0.6202 | 0.5228 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.7598 | 0.9946 | 0.7181 | 0.9984 | 0.6097 | 0.0442 | 0.3903 | 0.0309 | 0.0975 | 0.6202 | 0.5228 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.7621 | 0.9947 | 0.6156 | 0.9980 | 0.4987 | 0.0139 | 0.5013 | 0.0627 | 0.1116 | 0.5764 | 0.4648 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7621 | 0.9947 | 0.6434 | 0.9981 | 0.5276 | 0.0280 | 0.4724 | 0.0627 | 0.1116 | 0.5764 | 0.4648 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.7621 | 0.9947 | 0.6434 | 0.9981 | 0.5276 | 0.0280 | 0.4724 | 0.0627 | 0.1116 | 0.5764 | 0.4648 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7777 | 0.9948 | 0.6585 | 0.9983 | 0.5614 | 0.0303 | 0.4386 | 0.0516 | 0.1053 | 0.6039 | 0.4986 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7777 | 0.9948 | 0.6826 | 0.9974 | 0.5873 | 0.0536 | 0.4127 | 0.0516 | 0.1053 | 0.6039 | 0.4986 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.7777 | 0.9948 | 0.6826 | 0.9974 | 0.5873 | 0.0536 | 0.4127 | 0.0516 | 0.1053 | 0.6039 | 0.4986 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.7218 | 0.9931 | 0.3565 | 0.8323 | 0.2425 | 0.0419 | 0.7575 | 0.0889 | 0.1595 | 0.3738 | 0.2143 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7218 | 0.9931 | 0.3936 | 0.8731 | 0.2764 | 0.0780 | 0.7236 | 0.0889 | 0.1595 | 0.3738 | 0.2143 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.7218 | 0.9931 | 0.3936 | 0.8731 | 0.2764 | 0.0780 | 0.7236 | 0.0889 | 0.1595 | 0.3738 | 0.2143 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7592 | 0.9946 | 0.7068 | 0.9990 | 0.5966 | 0.0271 | 0.4034 | 0.0330 | 0.0985 | 0.6212 | 0.5227 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7592 | 0.9946 | 0.7191 | 0.9984 | 0.6110 | 0.0447 | 0.3890 | 0.0330 | 0.0985 | 0.6212 | 0.5227 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.7592 | 0.9946 | 0.7191 | 0.9984 | 0.6110 | 0.0447 | 0.3890 | 0.0330 | 0.0985 | 0.6212 | 0.5227 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6671 | 0.9921 | 0.4799 | 0.9978 | 0.3489 | 0.0194 | 0.6511 | 0.0757 | 0.1011 | 0.4180 | 0.3170 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6671 | 0.9921 | 0.5237 | 0.9972 | 0.3851 | 0.0369 | 0.6149 | 0.0757 | 0.1011 | 0.4180 | 0.3170 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.6671 | 0.9921 | 0.5237 | 0.9972 | 0.3851 | 0.0369 | 0.6149 | 0.0757 | 0.1011 | 0.4180 | 0.3170 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6928 | 0.9926 | 0.5345 | 0.9973 | 0.4177 | 0.0311 | 0.5823 | 0.0710 | 0.1081 | 0.4737 | 0.3656 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6928 | 0.9926 | 0.5754 | 0.9964 | 0.4568 | 0.0533 | 0.5432 | 0.0710 | 0.1081 | 0.4737 | 0.3656 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.6928 | 0.9926 | 0.5754 | 0.9964 | 0.4568 | 0.0533 | 0.5432 | 0.0710 | 0.1081 | 0.4737 | 0.3656 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6293 | 0.9904 | 0.3220 | 0.9972 | 0.1972 | 0.0334 | 0.8028 | 0.0871 | 0.1375 | 0.2995 | 0.1620 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6293 | 0.9904 | 0.3748 | 0.9960 | 0.2329 | 0.0705 | 0.7671 | 0.0871 | 0.1375 | 0.2995 | 0.1620 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.6293 | 0.9904 | 0.3748 | 0.9960 | 0.2329 | 0.0705 | 0.7671 | 0.0871 | 0.1375 | 0.2995 | 0.1620 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7625 | 0.9947 | 0.7116 | 0.9991 | 0.6011 | 0.0224 | 0.3989 | 0.0322 | 0.1055 | 0.6267 | 0.5213 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7625 | 0.9947 | 0.7216 | 0.9984 | 0.6139 | 0.0450 | 0.3861 | 0.0322 | 0.1055 | 0.6267 | 0.5213 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.7625 | 0.9947 | 0.7216 | 0.9984 | 0.6139 | 0.0450 | 0.3861 | 0.0322 | 0.1055 | 0.6267 | 0.5213 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.8154 | 0.9957 | 0.7167 | 0.9988 | 0.6120 | 0.0314 | 0.3880 | 0.0480 | 0.1119 | 0.6549 | 0.5430 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.8154 | 0.9957 | 0.7377 | 0.9982 | 0.6342 | 0.0605 | 0.3658 | 0.0480 | 0.1119 | 0.6549 | 0.5430 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.8154 | 0.9957 | 0.7377 | 0.9982 | 0.6342 | 0.0605 | 0.3658 | 0.0480 | 0.1119 | 0.6549 | 0.5430 |
| condition | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.5984 | 0.9743 | 0.4526 | 0.9963 | 0.2935 | 0.0447 | 0.7065 | 0.0462 | 0.1645 | 0.3605 | 0.1960 |
| condition | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5984 | 0.9743 | 0.4759 | 0.9924 | 0.3134 | 0.1052 | 0.6866 | 0.0462 | 0.1645 | 0.3605 | 0.1960 |
| condition | centralized | adaptive | cnn-ae | hard_val_p95 | 0.5984 | 0.9743 | 0.4759 | 0.9924 | 0.3134 | 0.1052 | 0.6866 | 0.0462 | 0.1645 | 0.3605 | 0.1960 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6041 | 0.9741 | 0.4791 | 0.9957 | 0.3222 | 0.0292 | 0.6778 | 0.0486 | 0.1430 | 0.3866 | 0.2436 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6041 | 0.9741 | 0.5051 | 0.9924 | 0.3470 | 0.0571 | 0.6530 | 0.0486 | 0.1430 | 0.3866 | 0.2436 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.6041 | 0.9741 | 0.5051 | 0.9924 | 0.3470 | 0.0571 | 0.6530 | 0.0486 | 0.1430 | 0.3866 | 0.2436 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.5931 | 0.9740 | 0.4493 | 0.9944 | 0.2902 | 0.1001 | 0.7098 | 0.0514 | 0.2022 | 0.3578 | 0.1556 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.5931 | 0.9740 | 0.4761 | 0.9906 | 0.3134 | 0.1557 | 0.6866 | 0.0514 | 0.2022 | 0.3578 | 0.1556 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.5931 | 0.9740 | 0.4761 | 0.9906 | 0.3134 | 0.1557 | 0.6866 | 0.0514 | 0.2022 | 0.3578 | 0.1556 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6199 | 0.9748 | 0.4209 | 0.9960 | 0.2675 | 0.0262 | 0.7325 | 0.0388 | 0.1371 | 0.3450 | 0.2079 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6199 | 0.9748 | 0.4396 | 0.9926 | 0.2829 | 0.0584 | 0.7171 | 0.0388 | 0.1371 | 0.3450 | 0.2079 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.6199 | 0.9748 | 0.4396 | 0.9926 | 0.2829 | 0.0584 | 0.7171 | 0.0388 | 0.1371 | 0.3450 | 0.2079 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6232 | 0.9747 | 0.4700 | 0.9962 | 0.3149 | 0.0234 | 0.6851 | 0.0477 | 0.1391 | 0.3876 | 0.2485 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6232 | 0.9747 | 0.4946 | 0.9927 | 0.3382 | 0.0534 | 0.6618 | 0.0477 | 0.1391 | 0.3876 | 0.2485 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.6232 | 0.9747 | 0.4946 | 0.9927 | 0.3382 | 0.0534 | 0.6618 | 0.0477 | 0.1391 | 0.3876 | 0.2485 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6176 | 0.9747 | 0.4089 | 0.9949 | 0.2579 | 0.0355 | 0.7421 | 0.0400 | 0.1491 | 0.3354 | 0.1863 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6176 | 0.9747 | 0.4278 | 0.9908 | 0.2733 | 0.0778 | 0.7267 | 0.0400 | 0.1491 | 0.3354 | 0.1863 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.6176 | 0.9747 | 0.4278 | 0.9908 | 0.2733 | 0.0778 | 0.7267 | 0.0400 | 0.1491 | 0.3354 | 0.1863 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6392 | 0.9765 | 0.5138 | 0.9967 | 0.3529 | 0.0272 | 0.6471 | 0.0435 | 0.1418 | 0.4237 | 0.2819 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6392 | 0.9765 | 0.5334 | 0.9943 | 0.3733 | 0.0476 | 0.6267 | 0.0435 | 0.1418 | 0.4237 | 0.2819 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.6392 | 0.9765 | 0.5334 | 0.9943 | 0.3733 | 0.0476 | 0.6267 | 0.0435 | 0.1418 | 0.4237 | 0.2819 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6642 | 0.9791 | 0.3820 | 0.9972 | 0.2505 | 0.0496 | 0.7495 | 0.1541 | 0.1712 | 0.3671 | 0.1960 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6642 | 0.9791 | 0.4098 | 0.9949 | 0.2717 | 0.0931 | 0.7283 | 0.1541 | 0.1712 | 0.3671 | 0.1960 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.6642 | 0.9791 | 0.4098 | 0.9949 | 0.2717 | 0.0931 | 0.7283 | 0.1541 | 0.1712 | 0.3671 | 0.1960 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6680 | 0.9790 | 0.5078 | 0.9968 | 0.3576 | 0.0235 | 0.6424 | 0.0607 | 0.1314 | 0.4323 | 0.3009 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6680 | 0.9790 | 0.5342 | 0.9941 | 0.3832 | 0.0501 | 0.6168 | 0.0607 | 0.1314 | 0.4323 | 0.3009 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.6680 | 0.9790 | 0.5342 | 0.9941 | 0.3832 | 0.0501 | 0.6168 | 0.0607 | 0.1314 | 0.4323 | 0.3009 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6608 | 0.9788 | 0.3949 | 0.7737 | 0.2672 | 0.1096 | 0.7328 | 0.1307 | 0.2221 | 0.3787 | 0.1566 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6608 | 0.9788 | 0.4269 | 0.7713 | 0.3020 | 0.1789 | 0.6980 | 0.1307 | 0.2221 | 0.3787 | 0.1566 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.6608 | 0.9788 | 0.4269 | 0.7713 | 0.3020 | 0.1789 | 0.6980 | 0.1307 | 0.2221 | 0.3787 | 0.1566 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6433 | 0.9768 | 0.5167 | 0.9963 | 0.3553 | 0.0321 | 0.6447 | 0.0432 | 0.1444 | 0.4271 | 0.2827 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6433 | 0.9768 | 0.5373 | 0.9940 | 0.3768 | 0.0546 | 0.6232 | 0.0432 | 0.1444 | 0.4271 | 0.2827 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.6433 | 0.9768 | 0.5373 | 0.9940 | 0.3768 | 0.0546 | 0.6232 | 0.0432 | 0.1444 | 0.4271 | 0.2827 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6218 | 0.9747 | 0.4150 | 0.9957 | 0.2634 | 0.0239 | 0.7366 | 0.0538 | 0.1336 | 0.3441 | 0.2106 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6218 | 0.9747 | 0.4455 | 0.9929 | 0.2878 | 0.0532 | 0.7122 | 0.0538 | 0.1336 | 0.3441 | 0.2106 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.6218 | 0.9747 | 0.4455 | 0.9929 | 0.2878 | 0.0532 | 0.7122 | 0.0538 | 0.1336 | 0.3441 | 0.2106 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6238 | 0.9746 | 0.4578 | 0.9959 | 0.3064 | 0.0230 | 0.6936 | 0.0593 | 0.1375 | 0.3821 | 0.2446 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6238 | 0.9746 | 0.4895 | 0.9927 | 0.3342 | 0.0528 | 0.6658 | 0.0593 | 0.1375 | 0.3821 | 0.2446 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.6238 | 0.9746 | 0.4895 | 0.9927 | 0.3342 | 0.0528 | 0.6658 | 0.0593 | 0.1375 | 0.3821 | 0.2446 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6203 | 0.9747 | 0.3969 | 0.9948 | 0.2487 | 0.0316 | 0.7513 | 0.0628 | 0.1446 | 0.3325 | 0.1879 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6203 | 0.9747 | 0.4330 | 0.9910 | 0.2772 | 0.0715 | 0.7228 | 0.0628 | 0.1446 | 0.3325 | 0.1879 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.6203 | 0.9747 | 0.4330 | 0.9910 | 0.2772 | 0.0715 | 0.7228 | 0.0628 | 0.1446 | 0.3325 | 0.1879 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6346 | 0.9761 | 0.5052 | 0.9968 | 0.3433 | 0.0259 | 0.6567 | 0.0445 | 0.1425 | 0.4145 | 0.2720 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6346 | 0.9761 | 0.5256 | 0.9937 | 0.3643 | 0.0542 | 0.6357 | 0.0445 | 0.1425 | 0.4145 | 0.2720 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.6346 | 0.9761 | 0.5256 | 0.9937 | 0.3643 | 0.0542 | 0.6357 | 0.0445 | 0.1425 | 0.4145 | 0.2720 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7085 | 0.9796 | 0.6224 | 0.9966 | 0.4604 | 0.0309 | 0.5396 | 0.0473 | 0.1438 | 0.5218 | 0.3780 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7085 | 0.9796 | 0.6406 | 0.9940 | 0.4812 | 0.0571 | 0.5188 | 0.0473 | 0.1438 | 0.5218 | 0.3780 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.7085 | 0.9796 | 0.6406 | 0.9940 | 0.4812 | 0.0571 | 0.5188 | 0.0473 | 0.1438 | 0.5218 | 0.3780 |

## Client Stability

| clients_by | training_mode | calibration_scope | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0212 | 0.2218 | 0.0337 | 0.3034 |
| bearing | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0431 | 0.2197 | 0.0337 | 0.3034 |
| bearing | centralized | adaptive | cnn-ae | hard_val_p95 | 0.0431 | 0.2197 | 0.0337 | 0.3034 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0107 | 0.2766 | 0.0206 | 0.2779 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0139 | 0.2745 | 0.0206 | 0.2779 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0139 | 0.2745 | 0.0206 | 0.2779 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.1137 | 0.0000 | 0.0019 | 0.2022 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1962 | 0.0000 | 0.0019 | 0.2022 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.1962 | 0.0000 | 0.0019 | 0.2022 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0197 | 0.2202 | 0.0532 | 0.2713 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0395 | 0.2188 | 0.0532 | 0.2713 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.0395 | 0.2188 | 0.0532 | 0.2713 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0211 | 0.2849 | 0.0401 | 0.2788 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0281 | 0.2906 | 0.0401 | 0.2788 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0281 | 0.2906 | 0.0401 | 0.2788 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0488 | 0.0878 | 0.0497 | 0.1521 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1099 | 0.0681 | 0.0497 | 0.1521 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.1099 | 0.0681 | 0.0497 | 0.1521 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0073 | 0.2817 | 0.0184 | 0.2849 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0148 | 0.2798 | 0.0184 | 0.2849 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0148 | 0.2798 | 0.0184 | 0.2849 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0188 | 0.2687 | 0.0483 | 0.3046 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0364 | 0.2685 | 0.0483 | 0.3046 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.0364 | 0.2685 | 0.0483 | 0.3046 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0174 | 0.3142 | 0.0307 | 0.2998 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0230 | 0.3129 | 0.0307 | 0.2998 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0230 | 0.3129 | 0.0307 | 0.2998 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.1005 | 0.1753 | 0.1126 | 0.1669 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1752 | 0.1966 | 0.1126 | 0.1669 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.1752 | 0.1966 | 0.1126 | 0.1669 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0063 | 0.2813 | 0.0200 | 0.2819 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0104 | 0.2798 | 0.0200 | 0.2819 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0104 | 0.2798 | 0.0200 | 0.2819 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0196 | 0.2141 | 0.0553 | 0.2623 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0351 | 0.2077 | 0.0553 | 0.2623 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.0351 | 0.2077 | 0.0553 | 0.2623 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0140 | 0.2744 | 0.0486 | 0.2696 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0145 | 0.2767 | 0.0486 | 0.2696 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0145 | 0.2767 | 0.0486 | 0.2696 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0512 | 0.0772 | 0.0613 | 0.1475 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1050 | 0.0500 | 0.0613 | 0.1475 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.1050 | 0.0500 | 0.0613 | 0.1475 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0094 | 0.2778 | 0.0174 | 0.2831 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0206 | 0.2792 | 0.0174 | 0.2831 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0206 | 0.2792 | 0.0174 | 0.2831 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0148 | 0.2796 | 0.0229 | 0.2597 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0166 | 0.2716 | 0.0229 | 0.2597 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0166 | 0.2716 | 0.0229 | 0.2597 |
| condition | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0422 | 0.0313 | 0.0152 | 0.1006 |
| condition | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1052 | 0.0243 | 0.0152 | 0.1006 |
| condition | centralized | adaptive | cnn-ae | hard_val_p95 | 0.1052 | 0.0243 | 0.0152 | 0.1006 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0097 | 0.0953 | 0.0187 | 0.0915 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0063 | 0.1062 | 0.0187 | 0.0915 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0063 | 0.1062 | 0.0187 | 0.0915 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.1236 | 0.0000 | 0.0038 | 0.1339 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1728 | 0.0000 | 0.0038 | 0.1339 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.1728 | 0.0000 | 0.0038 | 0.1339 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0228 | 0.0286 | 0.0140 | 0.0677 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0532 | 0.0273 | 0.0140 | 0.0677 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.0532 | 0.0273 | 0.0140 | 0.0677 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0045 | 0.1004 | 0.0251 | 0.0974 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0104 | 0.1112 | 0.0251 | 0.0974 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0104 | 0.1112 | 0.0251 | 0.0974 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0349 | 0.0274 | 0.0087 | 0.0678 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0753 | 0.0243 | 0.0087 | 0.0678 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0753 | 0.0243 | 0.0087 | 0.0678 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0046 | 0.0973 | 0.0287 | 0.0977 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0053 | 0.1130 | 0.0287 | 0.0977 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0053 | 0.1130 | 0.0287 | 0.0977 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0763 | 0.1273 | 0.2066 | 0.1073 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1355 | 0.1240 | 0.2066 | 0.1073 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.1355 | 0.1240 | 0.2066 | 0.1073 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0063 | 0.1478 | 0.0358 | 0.1339 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0150 | 0.1559 | 0.0358 | 0.1339 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0150 | 0.1559 | 0.0358 | 0.1339 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.1476 | 0.1524 | 0.1551 | 0.1128 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.2346 | 0.1917 | 0.1551 | 0.1128 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.2346 | 0.1917 | 0.1551 | 0.1128 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0102 | 0.0961 | 0.0304 | 0.1004 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0151 | 0.1122 | 0.0304 | 0.1004 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0151 | 0.1122 | 0.0304 | 0.1004 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0219 | 0.0399 | 0.0299 | 0.0734 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0467 | 0.0293 | 0.0299 | 0.0734 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.0467 | 0.0293 | 0.0299 | 0.0734 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0056 | 0.1096 | 0.0319 | 0.1043 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0132 | 0.1134 | 0.0319 | 0.1043 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0132 | 0.1134 | 0.0319 | 0.1043 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0326 | 0.0319 | 0.0352 | 0.0654 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0717 | 0.0139 | 0.0352 | 0.0654 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0717 | 0.0139 | 0.0352 | 0.0654 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0058 | 0.0831 | 0.0330 | 0.0893 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0133 | 0.0995 | 0.0330 | 0.0893 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0133 | 0.0995 | 0.0330 | 0.0893 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0068 | 0.1054 | 0.0291 | 0.0964 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0104 | 0.1113 | 0.0291 | 0.0964 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0104 | 0.1113 | 0.0291 | 0.0964 |

## Federated Convergence

| clients_by | training_mode | calibration_scope | model | decision_policy | round | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6274 | 0.9895 | 0.2645 | 0.9940 | 0.1779 | 0.0253 | 0.8221 | 0.1262 | 0.1143 | 0.2885 | 0.1743 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6767 | 0.9930 | 0.5702 | 0.9990 | 0.4221 | 0.0196 | 0.5779 | 0.0540 | 0.1003 | 0.4717 | 0.3714 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6915 | 0.9932 | 0.5971 | 0.9995 | 0.4473 | 0.0122 | 0.5527 | 0.0423 | 0.0946 | 0.4909 | 0.3963 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6274 | 0.9895 | 0.3442 | 0.9927 | 0.2367 | 0.0549 | 0.7633 | 0.1262 | 0.1143 | 0.2885 | 0.1743 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6767 | 0.9930 | 0.5989 | 0.9983 | 0.4524 | 0.0380 | 0.5476 | 0.0540 | 0.1003 | 0.4717 | 0.3714 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6915 | 0.9932 | 0.6151 | 0.9986 | 0.4663 | 0.0313 | 0.5337 | 0.0423 | 0.0946 | 0.4909 | 0.3963 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6274 | 0.9895 | 0.3442 | 0.9927 | 0.2367 | 0.0549 | 0.7633 | 0.1262 | 0.1143 | 0.2885 | 0.1743 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6767 | 0.9930 | 0.5989 | 0.9983 | 0.4524 | 0.0380 | 0.5476 | 0.0540 | 0.1003 | 0.4717 | 0.3714 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6915 | 0.9932 | 0.6151 | 0.9986 | 0.4663 | 0.0313 | 0.5337 | 0.0423 | 0.0946 | 0.4909 | 0.3963 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6347 | 0.9898 | 0.3098 | 0.9940 | 0.2321 | 0.0308 | 0.7679 | 0.0944 | 0.1172 | 0.3172 | 0.1999 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.7118 | 0.9935 | 0.6307 | 0.9988 | 0.4984 | 0.0344 | 0.5016 | 0.0488 | 0.1145 | 0.5368 | 0.4224 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7291 | 0.9936 | 0.6478 | 0.9990 | 0.5208 | 0.0277 | 0.4792 | 0.0589 | 0.1033 | 0.5611 | 0.4579 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6347 | 0.9898 | 0.3697 | 0.9920 | 0.2794 | 0.0663 | 0.7206 | 0.0944 | 0.1172 | 0.3172 | 0.1999 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.7118 | 0.9935 | 0.6539 | 0.9978 | 0.5268 | 0.0586 | 0.4732 | 0.0488 | 0.1145 | 0.5368 | 0.4224 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7291 | 0.9936 | 0.6724 | 0.9981 | 0.5513 | 0.0476 | 0.4487 | 0.0589 | 0.1033 | 0.5611 | 0.4579 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6347 | 0.9898 | 0.3697 | 0.9920 | 0.2794 | 0.0663 | 0.7206 | 0.0944 | 0.1172 | 0.3172 | 0.1999 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.7118 | 0.9935 | 0.6539 | 0.9978 | 0.5268 | 0.0586 | 0.4732 | 0.0488 | 0.1145 | 0.5368 | 0.4224 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.7291 | 0.9936 | 0.6724 | 0.9981 | 0.5513 | 0.0476 | 0.4487 | 0.0589 | 0.1033 | 0.5611 | 0.4579 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6137 | 0.9888 | 0.1356 | 0.9929 | 0.0728 | 0.0359 | 0.9272 | 0.1460 | 0.1329 | 0.2119 | 0.0790 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6375 | 0.9908 | 0.3888 | 0.9983 | 0.2414 | 0.0311 | 0.7586 | 0.0483 | 0.1355 | 0.3263 | 0.1908 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6146 | 0.9902 | 0.4285 | 0.9985 | 0.2728 | 0.0368 | 0.7272 | 0.0346 | 0.1484 | 0.3419 | 0.1935 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6137 | 0.9888 | 0.2326 | 0.9930 | 0.1317 | 0.0656 | 0.8683 | 0.1460 | 0.1329 | 0.2119 | 0.0790 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6375 | 0.9908 | 0.4114 | 0.9963 | 0.2593 | 0.0744 | 0.7407 | 0.0483 | 0.1355 | 0.3263 | 0.1908 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6146 | 0.9902 | 0.4476 | 0.9967 | 0.2886 | 0.0916 | 0.7114 | 0.0346 | 0.1484 | 0.3419 | 0.1935 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.6137 | 0.9888 | 0.2326 | 0.9930 | 0.1317 | 0.0656 | 0.8683 | 0.1460 | 0.1329 | 0.2119 | 0.0790 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.6375 | 0.9908 | 0.4114 | 0.9963 | 0.2593 | 0.0744 | 0.7407 | 0.0483 | 0.1355 | 0.3263 | 0.1908 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.6146 | 0.9902 | 0.4476 | 0.9967 | 0.2886 | 0.0916 | 0.7114 | 0.0346 | 0.1484 | 0.3419 | 0.1935 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.7376 | 0.9940 | 0.4257 | 0.9949 | 0.3429 | 0.0113 | 0.6571 | 0.0920 | 0.1351 | 0.4820 | 0.3469 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.8054 | 0.9959 | 0.7377 | 0.9996 | 0.6007 | 0.0151 | 0.3993 | 0.0578 | 0.1025 | 0.6534 | 0.5509 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7432 | 0.9943 | 0.6835 | 0.9995 | 0.5525 | 0.0152 | 0.4475 | 0.0382 | 0.0972 | 0.5937 | 0.4965 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.7376 | 0.9940 | 0.4699 | 0.9961 | 0.3827 | 0.0228 | 0.6173 | 0.0920 | 0.1351 | 0.4820 | 0.3469 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.8054 | 0.9959 | 0.7613 | 0.9993 | 0.6296 | 0.0314 | 0.3704 | 0.0578 | 0.1025 | 0.6534 | 0.5509 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7432 | 0.9943 | 0.6991 | 0.9990 | 0.5705 | 0.0297 | 0.4295 | 0.0382 | 0.0972 | 0.5937 | 0.4965 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 1 | 0.7376 | 0.9940 | 0.4699 | 0.9961 | 0.3827 | 0.0228 | 0.6173 | 0.0920 | 0.1351 | 0.4820 | 0.3469 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 2 | 0.8054 | 0.9959 | 0.7613 | 0.9993 | 0.6296 | 0.0314 | 0.3704 | 0.0578 | 0.1025 | 0.6534 | 0.5509 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 3 | 0.7432 | 0.9943 | 0.6991 | 0.9990 | 0.5705 | 0.0297 | 0.4295 | 0.0382 | 0.0972 | 0.5937 | 0.4965 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.7339 | 0.9939 | 0.4804 | 0.9967 | 0.4155 | 0.0266 | 0.5845 | 0.0649 | 0.1001 | 0.4769 | 0.3769 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.8320 | 0.9960 | 0.7728 | 0.9990 | 0.6565 | 0.0346 | 0.3435 | 0.0553 | 0.1056 | 0.6956 | 0.5900 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7672 | 0.9946 | 0.7223 | 0.9991 | 0.6122 | 0.0298 | 0.3878 | 0.0345 | 0.1103 | 0.6391 | 0.5288 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.7339 | 0.9939 | 0.5154 | 0.9955 | 0.4459 | 0.0514 | 0.5541 | 0.0649 | 0.1001 | 0.4769 | 0.3769 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.8320 | 0.9960 | 0.7939 | 0.9986 | 0.6840 | 0.0521 | 0.3160 | 0.0553 | 0.1056 | 0.6956 | 0.5900 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7672 | 0.9946 | 0.7386 | 0.9982 | 0.6320 | 0.0573 | 0.3680 | 0.0345 | 0.1103 | 0.6391 | 0.5288 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.7339 | 0.9939 | 0.5154 | 0.9955 | 0.4459 | 0.0514 | 0.5541 | 0.0649 | 0.1001 | 0.4769 | 0.3769 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.8320 | 0.9960 | 0.7939 | 0.9986 | 0.6840 | 0.0521 | 0.3160 | 0.0553 | 0.1056 | 0.6956 | 0.5900 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.7672 | 0.9946 | 0.7386 | 0.9982 | 0.6320 | 0.0573 | 0.3680 | 0.0345 | 0.1103 | 0.6391 | 0.5288 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.7300 | 0.9933 | 0.0776 | 0.4988 | 0.0518 | 0.0325 | 0.9482 | 0.1108 | 0.1723 | 0.2543 | 0.0821 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.7711 | 0.9943 | 0.4900 | 0.9992 | 0.3379 | 0.0355 | 0.6621 | 0.1061 | 0.1435 | 0.4523 | 0.3088 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6643 | 0.9917 | 0.5020 | 0.9987 | 0.3379 | 0.0577 | 0.6621 | 0.0499 | 0.1627 | 0.4147 | 0.2520 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.7300 | 0.9933 | 0.1183 | 0.6229 | 0.0891 | 0.0664 | 0.9109 | 0.1108 | 0.1723 | 0.2543 | 0.0821 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.7711 | 0.9943 | 0.5421 | 0.9985 | 0.3852 | 0.0727 | 0.6148 | 0.1061 | 0.1435 | 0.4523 | 0.3088 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6643 | 0.9917 | 0.5204 | 0.9980 | 0.3548 | 0.0948 | 0.6452 | 0.0499 | 0.1627 | 0.4147 | 0.2520 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.7300 | 0.9933 | 0.1183 | 0.6229 | 0.0891 | 0.0664 | 0.9109 | 0.1108 | 0.1723 | 0.2543 | 0.0821 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.7711 | 0.9943 | 0.5421 | 0.9985 | 0.3852 | 0.0727 | 0.6148 | 0.1061 | 0.1435 | 0.4523 | 0.3088 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.6643 | 0.9917 | 0.5204 | 0.9980 | 0.3548 | 0.0948 | 0.6452 | 0.0499 | 0.1627 | 0.4147 | 0.2520 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6450 | 0.9905 | 0.2947 | 0.9952 | 0.1984 | 0.0211 | 0.8016 | 0.1323 | 0.1059 | 0.3105 | 0.2046 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6684 | 0.9927 | 0.5472 | 0.9989 | 0.4008 | 0.0202 | 0.5992 | 0.0550 | 0.1010 | 0.4537 | 0.3527 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6878 | 0.9932 | 0.5977 | 0.9992 | 0.4476 | 0.0169 | 0.5524 | 0.0397 | 0.0963 | 0.4899 | 0.3936 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6450 | 0.9905 | 0.3839 | 0.9947 | 0.2635 | 0.0439 | 0.7365 | 0.1323 | 0.1059 | 0.3105 | 0.2046 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6684 | 0.9927 | 0.5734 | 0.9984 | 0.4263 | 0.0355 | 0.5737 | 0.0550 | 0.1010 | 0.4537 | 0.3527 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6878 | 0.9932 | 0.6137 | 0.9987 | 0.4655 | 0.0312 | 0.5345 | 0.0397 | 0.0963 | 0.4899 | 0.3936 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6450 | 0.9905 | 0.3839 | 0.9947 | 0.2635 | 0.0439 | 0.7365 | 0.1323 | 0.1059 | 0.3105 | 0.2046 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6684 | 0.9927 | 0.5734 | 0.9984 | 0.4263 | 0.0355 | 0.5737 | 0.0550 | 0.1010 | 0.4537 | 0.3527 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6878 | 0.9932 | 0.6137 | 0.9987 | 0.4655 | 0.0312 | 0.5345 | 0.0397 | 0.0963 | 0.4899 | 0.3936 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6551 | 0.9908 | 0.3442 | 0.9942 | 0.2576 | 0.0301 | 0.7424 | 0.0975 | 0.1062 | 0.3432 | 0.2370 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6981 | 0.9934 | 0.6091 | 0.9988 | 0.4735 | 0.0304 | 0.5265 | 0.0598 | 0.1107 | 0.5167 | 0.4060 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7252 | 0.9936 | 0.6502 | 0.9988 | 0.5221 | 0.0328 | 0.4779 | 0.0557 | 0.1073 | 0.5611 | 0.4538 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6551 | 0.9908 | 0.4147 | 0.9935 | 0.3110 | 0.0571 | 0.6890 | 0.0975 | 0.1062 | 0.3432 | 0.2370 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6981 | 0.9934 | 0.6391 | 0.9978 | 0.5084 | 0.0536 | 0.4916 | 0.0598 | 0.1107 | 0.5167 | 0.4060 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7252 | 0.9936 | 0.6723 | 0.9980 | 0.5509 | 0.0490 | 0.4491 | 0.0557 | 0.1073 | 0.5611 | 0.4538 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6551 | 0.9908 | 0.4147 | 0.9935 | 0.3110 | 0.0571 | 0.6890 | 0.0975 | 0.1062 | 0.3432 | 0.2370 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.6981 | 0.9934 | 0.6391 | 0.9978 | 0.5084 | 0.0536 | 0.4916 | 0.0598 | 0.1107 | 0.5167 | 0.4060 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.7252 | 0.9936 | 0.6723 | 0.9980 | 0.5509 | 0.0490 | 0.4491 | 0.0557 | 0.1073 | 0.5611 | 0.4538 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6256 | 0.9897 | 0.1646 | 0.9949 | 0.0897 | 0.0343 | 0.9103 | 0.1735 | 0.1348 | 0.2378 | 0.1031 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6478 | 0.9912 | 0.3796 | 0.9980 | 0.2343 | 0.0334 | 0.7657 | 0.0493 | 0.1333 | 0.3233 | 0.1900 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6144 | 0.9902 | 0.4219 | 0.9985 | 0.2674 | 0.0326 | 0.7326 | 0.0385 | 0.1443 | 0.3373 | 0.1930 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6256 | 0.9897 | 0.2829 | 0.9950 | 0.1649 | 0.0619 | 0.8351 | 0.1735 | 0.1348 | 0.2378 | 0.1031 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6478 | 0.9912 | 0.3997 | 0.9963 | 0.2500 | 0.0696 | 0.7500 | 0.0493 | 0.1333 | 0.3233 | 0.1900 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6144 | 0.9902 | 0.4419 | 0.9968 | 0.2839 | 0.0801 | 0.7161 | 0.0385 | 0.1443 | 0.3373 | 0.1930 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.6256 | 0.9897 | 0.2829 | 0.9950 | 0.1649 | 0.0619 | 0.8351 | 0.1735 | 0.1348 | 0.2378 | 0.1031 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.6478 | 0.9912 | 0.3997 | 0.9963 | 0.2500 | 0.0696 | 0.7500 | 0.0493 | 0.1333 | 0.3233 | 0.1900 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.6144 | 0.9902 | 0.4419 | 0.9968 | 0.2839 | 0.0801 | 0.7161 | 0.0385 | 0.1443 | 0.3373 | 0.1930 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6444 | 0.9764 | 0.3842 | 0.9955 | 0.2385 | 0.0182 | 0.7615 | 0.0533 | 0.1207 | 0.3340 | 0.2133 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6108 | 0.9742 | 0.4362 | 0.9959 | 0.2794 | 0.0305 | 0.7206 | 0.0322 | 0.1438 | 0.3502 | 0.2064 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6044 | 0.9739 | 0.4424 | 0.9966 | 0.2845 | 0.0301 | 0.7155 | 0.0307 | 0.1467 | 0.3507 | 0.2040 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6444 | 0.9764 | 0.4098 | 0.9917 | 0.2589 | 0.0373 | 0.7411 | 0.0533 | 0.1207 | 0.3340 | 0.2133 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6108 | 0.9742 | 0.4534 | 0.9928 | 0.2939 | 0.0696 | 0.7061 | 0.0322 | 0.1438 | 0.3502 | 0.2064 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6044 | 0.9739 | 0.4557 | 0.9933 | 0.2958 | 0.0684 | 0.7042 | 0.0307 | 0.1467 | 0.3507 | 0.2040 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6444 | 0.9764 | 0.4098 | 0.9917 | 0.2589 | 0.0373 | 0.7411 | 0.0533 | 0.1207 | 0.3340 | 0.2133 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6108 | 0.9742 | 0.4534 | 0.9928 | 0.2939 | 0.0696 | 0.7061 | 0.0322 | 0.1438 | 0.3502 | 0.2064 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6044 | 0.9739 | 0.4557 | 0.9933 | 0.2958 | 0.0684 | 0.7042 | 0.0307 | 0.1467 | 0.3507 | 0.2040 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6454 | 0.9762 | 0.4505 | 0.9962 | 0.3012 | 0.0200 | 0.6988 | 0.0545 | 0.1319 | 0.3856 | 0.2537 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6152 | 0.9741 | 0.4786 | 0.9963 | 0.3209 | 0.0239 | 0.6791 | 0.0447 | 0.1430 | 0.3895 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6089 | 0.9738 | 0.4808 | 0.9963 | 0.3226 | 0.0263 | 0.6774 | 0.0437 | 0.1423 | 0.3878 | 0.2455 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6454 | 0.9762 | 0.4737 | 0.9921 | 0.3235 | 0.0485 | 0.6765 | 0.0545 | 0.1319 | 0.3856 | 0.2537 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6152 | 0.9741 | 0.5036 | 0.9928 | 0.3446 | 0.0568 | 0.6554 | 0.0447 | 0.1430 | 0.3895 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6089 | 0.9738 | 0.5063 | 0.9932 | 0.3465 | 0.0550 | 0.6535 | 0.0437 | 0.1423 | 0.3878 | 0.2455 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6454 | 0.9762 | 0.4737 | 0.9921 | 0.3235 | 0.0485 | 0.6765 | 0.0545 | 0.1319 | 0.3856 | 0.2537 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.6152 | 0.9741 | 0.5036 | 0.9928 | 0.3446 | 0.0568 | 0.6554 | 0.0447 | 0.1430 | 0.3895 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.6089 | 0.9738 | 0.5063 | 0.9932 | 0.3465 | 0.0550 | 0.6535 | 0.0437 | 0.1423 | 0.3878 | 0.2455 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6429 | 0.9762 | 0.3594 | 0.9939 | 0.2193 | 0.0245 | 0.7807 | 0.0512 | 0.1241 | 0.3151 | 0.1911 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6085 | 0.9741 | 0.4297 | 0.9955 | 0.2740 | 0.0331 | 0.7260 | 0.0354 | 0.1578 | 0.3449 | 0.1871 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6013 | 0.9738 | 0.4375 | 0.9952 | 0.2804 | 0.0491 | 0.7196 | 0.0334 | 0.1655 | 0.3463 | 0.1808 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6429 | 0.9762 | 0.3849 | 0.9896 | 0.2389 | 0.0471 | 0.7611 | 0.0512 | 0.1241 | 0.3151 | 0.1911 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6085 | 0.9741 | 0.4480 | 0.9909 | 0.2894 | 0.0906 | 0.7106 | 0.0354 | 0.1578 | 0.3449 | 0.1871 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6013 | 0.9738 | 0.4507 | 0.9918 | 0.2916 | 0.0955 | 0.7084 | 0.0334 | 0.1655 | 0.3463 | 0.1808 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.6429 | 0.9762 | 0.3849 | 0.9896 | 0.2389 | 0.0471 | 0.7611 | 0.0512 | 0.1241 | 0.3151 | 0.1911 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.6085 | 0.9741 | 0.4480 | 0.9909 | 0.2894 | 0.0906 | 0.7106 | 0.0354 | 0.1578 | 0.3449 | 0.1871 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.6013 | 0.9738 | 0.4507 | 0.9918 | 0.2916 | 0.0955 | 0.7084 | 0.0334 | 0.1655 | 0.3463 | 0.1808 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6755 | 0.9810 | 0.1467 | 0.9985 | 0.0828 | 0.0113 | 0.9172 | 0.3793 | 0.1496 | 0.2769 | 0.1273 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6650 | 0.9785 | 0.5109 | 0.9963 | 0.3449 | 0.0824 | 0.6551 | 0.0452 | 0.1938 | 0.4240 | 0.2302 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6520 | 0.9779 | 0.4885 | 0.9968 | 0.3240 | 0.0552 | 0.6760 | 0.0378 | 0.1701 | 0.4005 | 0.2305 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6755 | 0.9810 | 0.1962 | 0.9979 | 0.1150 | 0.0239 | 0.8850 | 0.3793 | 0.1496 | 0.2769 | 0.1273 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6650 | 0.9785 | 0.5290 | 0.9931 | 0.3618 | 0.1494 | 0.6382 | 0.0452 | 0.1938 | 0.4240 | 0.2302 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6520 | 0.9779 | 0.5043 | 0.9938 | 0.3383 | 0.1061 | 0.6617 | 0.0378 | 0.1701 | 0.4005 | 0.2305 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6755 | 0.9810 | 0.1962 | 0.9979 | 0.1150 | 0.0239 | 0.8850 | 0.3793 | 0.1496 | 0.2769 | 0.1273 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6650 | 0.9785 | 0.5290 | 0.9931 | 0.3618 | 0.1494 | 0.6382 | 0.0452 | 0.1938 | 0.4240 | 0.2302 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6520 | 0.9779 | 0.5043 | 0.9938 | 0.3383 | 0.1061 | 0.6617 | 0.0378 | 0.1701 | 0.4005 | 0.2305 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6800 | 0.9810 | 0.4397 | 0.9971 | 0.3167 | 0.0164 | 0.6833 | 0.0898 | 0.1102 | 0.3957 | 0.2855 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6696 | 0.9784 | 0.5513 | 0.9969 | 0.3866 | 0.0253 | 0.6134 | 0.0485 | 0.1438 | 0.4603 | 0.3166 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6544 | 0.9776 | 0.5325 | 0.9964 | 0.3697 | 0.0289 | 0.6303 | 0.0439 | 0.1402 | 0.4408 | 0.3007 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6800 | 0.9810 | 0.4772 | 0.9941 | 0.3494 | 0.0398 | 0.6506 | 0.0898 | 0.1102 | 0.3957 | 0.2855 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6696 | 0.9784 | 0.5735 | 0.9940 | 0.4103 | 0.0588 | 0.5897 | 0.0485 | 0.1438 | 0.4603 | 0.3166 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6544 | 0.9776 | 0.5517 | 0.9943 | 0.3899 | 0.0517 | 0.6101 | 0.0439 | 0.1402 | 0.4408 | 0.3007 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6800 | 0.9810 | 0.4772 | 0.9941 | 0.3494 | 0.0398 | 0.6506 | 0.0898 | 0.1102 | 0.3957 | 0.2855 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.6696 | 0.9784 | 0.5735 | 0.9940 | 0.4103 | 0.0588 | 0.5897 | 0.0485 | 0.1438 | 0.4603 | 0.3166 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.6544 | 0.9776 | 0.5517 | 0.9943 | 0.3899 | 0.0517 | 0.6101 | 0.0439 | 0.1402 | 0.4408 | 0.3007 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6743 | 0.9807 | 0.2089 | 0.3303 | 0.1527 | 0.1111 | 0.8473 | 0.2891 | 0.2510 | 0.3251 | 0.0742 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6615 | 0.9782 | 0.5002 | 0.9957 | 0.3359 | 0.1120 | 0.6641 | 0.0548 | 0.2100 | 0.4179 | 0.2079 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6467 | 0.9776 | 0.4757 | 0.9953 | 0.3131 | 0.1057 | 0.6869 | 0.0481 | 0.2052 | 0.3931 | 0.1879 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6743 | 0.9807 | 0.2640 | 0.3295 | 0.2202 | 0.1997 | 0.7798 | 0.2891 | 0.2510 | 0.3251 | 0.0742 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6615 | 0.9782 | 0.5208 | 0.9924 | 0.3548 | 0.1747 | 0.6452 | 0.0548 | 0.2100 | 0.4179 | 0.2079 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6467 | 0.9776 | 0.4958 | 0.9920 | 0.3310 | 0.1624 | 0.6690 | 0.0481 | 0.2052 | 0.3931 | 0.1879 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.6743 | 0.9807 | 0.2640 | 0.3295 | 0.2202 | 0.1997 | 0.7798 | 0.2891 | 0.2510 | 0.3251 | 0.0742 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.6615 | 0.9782 | 0.5208 | 0.9924 | 0.3548 | 0.1747 | 0.6452 | 0.0548 | 0.2100 | 0.4179 | 0.2079 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.6467 | 0.9776 | 0.4958 | 0.9920 | 0.3310 | 0.1624 | 0.6690 | 0.0481 | 0.2052 | 0.3931 | 0.1879 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6442 | 0.9757 | 0.3790 | 0.9946 | 0.2364 | 0.0174 | 0.7636 | 0.0933 | 0.1188 | 0.3358 | 0.2170 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6176 | 0.9747 | 0.4220 | 0.9961 | 0.2679 | 0.0235 | 0.7321 | 0.0380 | 0.1359 | 0.3446 | 0.2087 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6034 | 0.9739 | 0.4439 | 0.9964 | 0.2857 | 0.0307 | 0.7143 | 0.0302 | 0.1460 | 0.3520 | 0.2060 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6442 | 0.9757 | 0.4344 | 0.9926 | 0.2793 | 0.0341 | 0.7207 | 0.0933 | 0.1188 | 0.3358 | 0.2170 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6176 | 0.9747 | 0.4442 | 0.9925 | 0.2863 | 0.0590 | 0.7137 | 0.0380 | 0.1359 | 0.3446 | 0.2087 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6034 | 0.9739 | 0.4580 | 0.9935 | 0.2977 | 0.0663 | 0.7023 | 0.0302 | 0.1460 | 0.3520 | 0.2060 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6442 | 0.9757 | 0.4344 | 0.9926 | 0.2793 | 0.0341 | 0.7207 | 0.0933 | 0.1188 | 0.3358 | 0.2170 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6176 | 0.9747 | 0.4442 | 0.9925 | 0.2863 | 0.0590 | 0.7137 | 0.0380 | 0.1359 | 0.3446 | 0.2087 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6034 | 0.9739 | 0.4580 | 0.9935 | 0.2977 | 0.0663 | 0.7023 | 0.0302 | 0.1460 | 0.3520 | 0.2060 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6420 | 0.9756 | 0.4167 | 0.9945 | 0.2781 | 0.0189 | 0.7219 | 0.0887 | 0.1299 | 0.3700 | 0.2401 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6207 | 0.9745 | 0.4735 | 0.9964 | 0.3171 | 0.0250 | 0.6829 | 0.0457 | 0.1419 | 0.3885 | 0.2466 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6088 | 0.9738 | 0.4831 | 0.9967 | 0.3241 | 0.0251 | 0.6759 | 0.0434 | 0.1406 | 0.3878 | 0.2471 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6420 | 0.9756 | 0.4675 | 0.9916 | 0.3189 | 0.0528 | 0.6811 | 0.0887 | 0.1299 | 0.3700 | 0.2401 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6207 | 0.9745 | 0.4965 | 0.9926 | 0.3391 | 0.0588 | 0.6609 | 0.0457 | 0.1419 | 0.3885 | 0.2466 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6088 | 0.9738 | 0.5045 | 0.9939 | 0.3447 | 0.0467 | 0.6553 | 0.0434 | 0.1406 | 0.3878 | 0.2471 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6420 | 0.9756 | 0.4675 | 0.9916 | 0.3189 | 0.0528 | 0.6811 | 0.0887 | 0.1299 | 0.3700 | 0.2401 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.6207 | 0.9745 | 0.4965 | 0.9926 | 0.3391 | 0.0588 | 0.6609 | 0.0457 | 0.1419 | 0.3885 | 0.2466 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.6088 | 0.9738 | 0.5045 | 0.9939 | 0.3447 | 0.0467 | 0.6553 | 0.0434 | 0.1406 | 0.3878 | 0.2471 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6441 | 0.9756 | 0.3402 | 0.9940 | 0.2052 | 0.0225 | 0.7948 | 0.1119 | 0.1245 | 0.3126 | 0.1881 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6158 | 0.9745 | 0.4123 | 0.9951 | 0.2601 | 0.0275 | 0.7399 | 0.0439 | 0.1440 | 0.3381 | 0.1941 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6008 | 0.9738 | 0.4381 | 0.9953 | 0.2809 | 0.0448 | 0.7191 | 0.0326 | 0.1653 | 0.3469 | 0.1816 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6441 | 0.9756 | 0.4102 | 0.9904 | 0.2587 | 0.0471 | 0.7413 | 0.1119 | 0.1245 | 0.3126 | 0.1881 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6158 | 0.9745 | 0.4376 | 0.9911 | 0.2808 | 0.0674 | 0.7192 | 0.0439 | 0.1440 | 0.3381 | 0.1941 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6008 | 0.9738 | 0.4513 | 0.9914 | 0.2921 | 0.1000 | 0.7079 | 0.0326 | 0.1653 | 0.3469 | 0.1816 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.6441 | 0.9756 | 0.4102 | 0.9904 | 0.2587 | 0.0471 | 0.7413 | 0.1119 | 0.1245 | 0.3126 | 0.1881 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.6158 | 0.9745 | 0.4376 | 0.9911 | 0.2808 | 0.0674 | 0.7192 | 0.0439 | 0.1440 | 0.3381 | 0.1941 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.6008 | 0.9738 | 0.4513 | 0.9914 | 0.2921 | 0.1000 | 0.7079 | 0.0326 | 0.1653 | 0.3469 | 0.1816 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `fedavg` approximates collaborative normal-only training without sharing raw vibration windows.
- `fedprox` adds a proximal penalty to reduce local client drift under non-IID data.
- `fedbn` keeps BatchNorm parameters and running statistics local to each client.
- `*-personalized` locally adapts the federated global model before client evaluation.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
