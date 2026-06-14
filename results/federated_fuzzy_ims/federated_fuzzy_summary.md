# Federated Fuzzy Health-Index Comparison

This experiment compares local-only, centralized, FedAvg, FedProx, and personalized federated training under the same fuzzy health-index decision layer. Fault windows are audit-only and are not used during training.

## Mean Performance

| clients_by | training_mode | calibration_scope | model | decision_policy | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6812 | 0.9929 | 0.5496 | 0.9994 | 0.3972 | 0.0137 | 0.6028 | 0.0378 | 0.0924 | 0.4452 | 0.3528 |
| bearing | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6812 | 0.9929 | 0.5660 | 0.9988 | 0.4138 | 0.0299 | 0.5862 | 0.0378 | 0.0924 | 0.4452 | 0.3528 |
| bearing | centralized | adaptive | cnn-ae | hard_val_p95 | 0.6812 | 0.9929 | 0.5660 | 0.9988 | 0.4138 | 0.0299 | 0.5862 | 0.0378 | 0.0924 | 0.4452 | 0.3528 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7062 | 0.9930 | 0.6288 | 0.9987 | 0.4949 | 0.0295 | 0.5051 | 0.0491 | 0.1089 | 0.5317 | 0.4228 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7062 | 0.9930 | 0.6495 | 0.9979 | 0.5187 | 0.0525 | 0.4813 | 0.0491 | 0.1089 | 0.5317 | 0.4228 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.7062 | 0.9930 | 0.6495 | 0.9979 | 0.5187 | 0.0525 | 0.4813 | 0.0491 | 0.1089 | 0.5317 | 0.4228 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6313 | 0.9906 | 0.4324 | 0.9984 | 0.2760 | 0.0340 | 0.7240 | 0.0311 | 0.1424 | 0.3417 | 0.1993 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6313 | 0.9906 | 0.4485 | 0.9970 | 0.2893 | 0.0716 | 0.7107 | 0.0311 | 0.1424 | 0.3417 | 0.1993 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.6313 | 0.9906 | 0.4485 | 0.9970 | 0.2893 | 0.0716 | 0.7107 | 0.0311 | 0.1424 | 0.3417 | 0.1993 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6753 | 0.9929 | 0.5146 | 0.9991 | 0.3600 | 0.0130 | 0.6400 | 0.0487 | 0.0920 | 0.4158 | 0.3238 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6753 | 0.9929 | 0.5391 | 0.9985 | 0.3828 | 0.0272 | 0.6172 | 0.0487 | 0.0920 | 0.4158 | 0.3238 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.6753 | 0.9929 | 0.5391 | 0.9985 | 0.3828 | 0.0272 | 0.6172 | 0.0487 | 0.0920 | 0.4158 | 0.3238 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7039 | 0.9932 | 0.6039 | 0.9986 | 0.4679 | 0.0268 | 0.5321 | 0.0586 | 0.1065 | 0.5121 | 0.4056 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7039 | 0.9932 | 0.6313 | 0.9975 | 0.4994 | 0.0534 | 0.5006 | 0.0586 | 0.1065 | 0.5121 | 0.4056 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.7039 | 0.9932 | 0.6313 | 0.9975 | 0.4994 | 0.0534 | 0.5006 | 0.0586 | 0.1065 | 0.5121 | 0.4056 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6243 | 0.9905 | 0.3977 | 0.9980 | 0.2494 | 0.0344 | 0.7506 | 0.0405 | 0.1389 | 0.3244 | 0.1855 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6243 | 0.9905 | 0.4208 | 0.9967 | 0.2669 | 0.0674 | 0.7331 | 0.0405 | 0.1389 | 0.3244 | 0.1855 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.6243 | 0.9905 | 0.4208 | 0.9967 | 0.2669 | 0.0674 | 0.7331 | 0.0405 | 0.1389 | 0.3244 | 0.1855 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7500 | 0.9950 | 0.6871 | 0.9991 | 0.5653 | 0.0266 | 0.4347 | 0.0414 | 0.1071 | 0.5979 | 0.4908 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7500 | 0.9950 | 0.7032 | 0.9981 | 0.5862 | 0.0522 | 0.4138 | 0.0414 | 0.1071 | 0.5979 | 0.4908 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.7500 | 0.9950 | 0.7032 | 0.9981 | 0.5862 | 0.0522 | 0.4138 | 0.0414 | 0.1071 | 0.5979 | 0.4908 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.7517 | 0.9951 | 0.6408 | 0.9995 | 0.4997 | 0.0130 | 0.5003 | 0.0414 | 0.0963 | 0.5509 | 0.4546 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7517 | 0.9951 | 0.6589 | 0.9989 | 0.5188 | 0.0305 | 0.4812 | 0.0414 | 0.0963 | 0.5509 | 0.4546 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.7517 | 0.9951 | 0.6589 | 0.9989 | 0.5188 | 0.0305 | 0.4812 | 0.0414 | 0.0963 | 0.5509 | 0.4546 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7680 | 0.9951 | 0.7084 | 0.9990 | 0.5938 | 0.0269 | 0.4062 | 0.0379 | 0.1073 | 0.6243 | 0.5170 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7680 | 0.9951 | 0.7235 | 0.9980 | 0.6129 | 0.0542 | 0.3871 | 0.0379 | 0.1073 | 0.6243 | 0.5170 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.7680 | 0.9951 | 0.7235 | 0.9980 | 0.6129 | 0.0542 | 0.3871 | 0.0379 | 0.1073 | 0.6243 | 0.5170 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.7173 | 0.9935 | 0.4673 | 0.9860 | 0.3131 | 0.0463 | 0.6869 | 0.0566 | 0.1528 | 0.4159 | 0.2630 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7173 | 0.9935 | 0.4969 | 0.9848 | 0.3390 | 0.0813 | 0.6610 | 0.0566 | 0.1528 | 0.4159 | 0.2630 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.7173 | 0.9935 | 0.4969 | 0.9848 | 0.3390 | 0.0813 | 0.6610 | 0.0566 | 0.1528 | 0.4159 | 0.2630 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7496 | 0.9950 | 0.6879 | 0.9990 | 0.5665 | 0.0281 | 0.4335 | 0.0393 | 0.1077 | 0.5980 | 0.4903 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7496 | 0.9950 | 0.7028 | 0.9980 | 0.5861 | 0.0507 | 0.4139 | 0.0393 | 0.1077 | 0.5980 | 0.4903 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.7496 | 0.9950 | 0.7028 | 0.9980 | 0.5861 | 0.0507 | 0.4139 | 0.0393 | 0.1077 | 0.5980 | 0.4903 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6761 | 0.9930 | 0.5191 | 0.9992 | 0.3638 | 0.0128 | 0.6362 | 0.0486 | 0.0929 | 0.4191 | 0.3262 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6761 | 0.9930 | 0.5432 | 0.9986 | 0.3864 | 0.0288 | 0.6136 | 0.0486 | 0.0929 | 0.4191 | 0.3262 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.6761 | 0.9930 | 0.5432 | 0.9986 | 0.3864 | 0.0288 | 0.6136 | 0.0486 | 0.0929 | 0.4191 | 0.3262 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7052 | 0.9932 | 0.6071 | 0.9987 | 0.4703 | 0.0263 | 0.5297 | 0.0595 | 0.1082 | 0.5145 | 0.4063 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7052 | 0.9932 | 0.6347 | 0.9975 | 0.5023 | 0.0566 | 0.4977 | 0.0595 | 0.1082 | 0.5145 | 0.4063 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.7052 | 0.9932 | 0.6347 | 0.9975 | 0.5023 | 0.0566 | 0.4977 | 0.0595 | 0.1082 | 0.5145 | 0.4063 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6245 | 0.9905 | 0.4003 | 0.9981 | 0.2512 | 0.0342 | 0.7488 | 0.0401 | 0.1389 | 0.3261 | 0.1872 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6245 | 0.9905 | 0.4226 | 0.9967 | 0.2683 | 0.0677 | 0.7317 | 0.0401 | 0.1389 | 0.3261 | 0.1872 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.6245 | 0.9905 | 0.4226 | 0.9967 | 0.2683 | 0.0677 | 0.7317 | 0.0401 | 0.1389 | 0.3261 | 0.1872 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7506 | 0.9950 | 0.6903 | 0.9989 | 0.5696 | 0.0292 | 0.4304 | 0.0400 | 0.1094 | 0.6014 | 0.4920 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7506 | 0.9950 | 0.7064 | 0.9980 | 0.5909 | 0.0582 | 0.4091 | 0.0400 | 0.1094 | 0.6014 | 0.4920 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.7506 | 0.9950 | 0.7064 | 0.9980 | 0.5909 | 0.0582 | 0.4091 | 0.0400 | 0.1094 | 0.6014 | 0.4920 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.8669 | 0.9971 | 0.7704 | 0.9992 | 0.6609 | 0.0240 | 0.3391 | 0.0706 | 0.1006 | 0.7132 | 0.6127 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.8669 | 0.9971 | 0.8021 | 0.9985 | 0.6990 | 0.0488 | 0.3010 | 0.0706 | 0.1006 | 0.7132 | 0.6127 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.8669 | 0.9971 | 0.8021 | 0.9985 | 0.6990 | 0.0488 | 0.3010 | 0.0706 | 0.1006 | 0.7132 | 0.6127 |
| condition | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6277 | 0.9752 | 0.4484 | 0.9968 | 0.2897 | 0.0204 | 0.7103 | 0.0249 | 0.1183 | 0.3437 | 0.2254 |
| condition | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6277 | 0.9752 | 0.4604 | 0.9939 | 0.3000 | 0.0452 | 0.7000 | 0.0249 | 0.1183 | 0.3437 | 0.2254 |
| condition | centralized | adaptive | cnn-ae | hard_val_p95 | 0.6277 | 0.9752 | 0.4604 | 0.9939 | 0.3000 | 0.0452 | 0.7000 | 0.0249 | 0.1183 | 0.3437 | 0.2254 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6183 | 0.9751 | 0.4794 | 0.9961 | 0.3194 | 0.0253 | 0.6806 | 0.0351 | 0.1291 | 0.3709 | 0.2418 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6183 | 0.9751 | 0.4963 | 0.9924 | 0.3352 | 0.0554 | 0.6648 | 0.0351 | 0.1291 | 0.3709 | 0.2418 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.6183 | 0.9751 | 0.4963 | 0.9924 | 0.3352 | 0.0554 | 0.6648 | 0.0351 | 0.1291 | 0.3709 | 0.2418 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6419 | 0.9756 | 0.4338 | 0.9949 | 0.2774 | 0.0295 | 0.7226 | 0.0294 | 0.1291 | 0.3415 | 0.2124 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6419 | 0.9756 | 0.4471 | 0.9909 | 0.2887 | 0.0611 | 0.7113 | 0.0294 | 0.1291 | 0.3415 | 0.2124 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.6419 | 0.9756 | 0.4471 | 0.9909 | 0.2887 | 0.0611 | 0.7113 | 0.0294 | 0.1291 | 0.3415 | 0.2124 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6265 | 0.9751 | 0.4190 | 0.9966 | 0.2659 | 0.0183 | 0.7341 | 0.0302 | 0.1155 | 0.3302 | 0.2148 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6265 | 0.9751 | 0.4339 | 0.9934 | 0.2781 | 0.0402 | 0.7219 | 0.0302 | 0.1155 | 0.3302 | 0.2148 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.6265 | 0.9751 | 0.4339 | 0.9934 | 0.2781 | 0.0402 | 0.7219 | 0.0302 | 0.1155 | 0.3302 | 0.2148 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6216 | 0.9752 | 0.4708 | 0.9960 | 0.3154 | 0.0262 | 0.6846 | 0.0395 | 0.1287 | 0.3734 | 0.2446 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6216 | 0.9752 | 0.4897 | 0.9925 | 0.3336 | 0.0507 | 0.6664 | 0.0395 | 0.1287 | 0.3734 | 0.2446 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.6216 | 0.9752 | 0.4897 | 0.9925 | 0.3336 | 0.0507 | 0.6664 | 0.0395 | 0.1287 | 0.3734 | 0.2446 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6348 | 0.9752 | 0.3995 | 0.9947 | 0.2500 | 0.0267 | 0.7500 | 0.0338 | 0.1261 | 0.3223 | 0.1963 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6348 | 0.9752 | 0.4152 | 0.9901 | 0.2627 | 0.0569 | 0.7373 | 0.0338 | 0.1261 | 0.3223 | 0.1963 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.6348 | 0.9752 | 0.4152 | 0.9901 | 0.2627 | 0.0569 | 0.7373 | 0.0338 | 0.1261 | 0.3223 | 0.1963 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6725 | 0.9796 | 0.5268 | 0.9962 | 0.3706 | 0.0278 | 0.6294 | 0.0347 | 0.1297 | 0.4312 | 0.3015 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6725 | 0.9796 | 0.5419 | 0.9929 | 0.3864 | 0.0538 | 0.6136 | 0.0347 | 0.1297 | 0.4312 | 0.3015 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.6725 | 0.9796 | 0.5419 | 0.9929 | 0.3864 | 0.0538 | 0.6136 | 0.0347 | 0.1297 | 0.4312 | 0.3015 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6858 | 0.9797 | 0.4802 | 0.9966 | 0.3212 | 0.0320 | 0.6788 | 0.0505 | 0.1307 | 0.3998 | 0.2691 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6858 | 0.9797 | 0.5020 | 0.9936 | 0.3408 | 0.0636 | 0.6592 | 0.0505 | 0.1307 | 0.3998 | 0.2691 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.6858 | 0.9797 | 0.5020 | 0.9936 | 0.3408 | 0.0636 | 0.6592 | 0.0505 | 0.1307 | 0.3998 | 0.2691 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6778 | 0.9795 | 0.5316 | 0.9962 | 0.3753 | 0.0275 | 0.6247 | 0.0389 | 0.1306 | 0.4393 | 0.3088 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6778 | 0.9795 | 0.5491 | 0.9931 | 0.3934 | 0.0547 | 0.6066 | 0.0389 | 0.1306 | 0.4393 | 0.3088 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.6778 | 0.9795 | 0.5491 | 0.9931 | 0.3934 | 0.0547 | 0.6066 | 0.0389 | 0.1306 | 0.4393 | 0.3088 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6972 | 0.9803 | 0.4521 | 0.9949 | 0.2973 | 0.0565 | 0.7027 | 0.0612 | 0.1507 | 0.3928 | 0.2422 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6972 | 0.9803 | 0.4821 | 0.9915 | 0.3251 | 0.0887 | 0.6749 | 0.0612 | 0.1507 | 0.3928 | 0.2422 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.6972 | 0.9803 | 0.4821 | 0.9915 | 0.3251 | 0.0887 | 0.6749 | 0.0612 | 0.1507 | 0.3928 | 0.2422 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6724 | 0.9796 | 0.5252 | 0.9961 | 0.3694 | 0.0307 | 0.6306 | 0.0342 | 0.1302 | 0.4304 | 0.3002 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6724 | 0.9796 | 0.5400 | 0.9931 | 0.3850 | 0.0562 | 0.6150 | 0.0342 | 0.1302 | 0.4304 | 0.3002 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.6724 | 0.9796 | 0.5400 | 0.9931 | 0.3850 | 0.0562 | 0.6150 | 0.0342 | 0.1302 | 0.4304 | 0.3002 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.6276 | 0.9752 | 0.4192 | 0.9967 | 0.2660 | 0.0172 | 0.7340 | 0.0304 | 0.1138 | 0.3302 | 0.2164 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6276 | 0.9752 | 0.4342 | 0.9935 | 0.2784 | 0.0382 | 0.7216 | 0.0304 | 0.1138 | 0.3302 | 0.2164 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.6276 | 0.9752 | 0.4342 | 0.9935 | 0.2784 | 0.0382 | 0.7216 | 0.0304 | 0.1138 | 0.3302 | 0.2164 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6224 | 0.9753 | 0.4723 | 0.9961 | 0.3169 | 0.0257 | 0.6831 | 0.0396 | 0.1286 | 0.3746 | 0.2460 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6224 | 0.9753 | 0.4916 | 0.9927 | 0.3354 | 0.0511 | 0.6646 | 0.0396 | 0.1286 | 0.3746 | 0.2460 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.6224 | 0.9753 | 0.4916 | 0.9927 | 0.3354 | 0.0511 | 0.6646 | 0.0396 | 0.1286 | 0.3746 | 0.2460 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.6360 | 0.9753 | 0.3989 | 0.9946 | 0.2496 | 0.0263 | 0.7504 | 0.0344 | 0.1240 | 0.3222 | 0.1982 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6360 | 0.9753 | 0.4153 | 0.9903 | 0.2627 | 0.0530 | 0.7373 | 0.0344 | 0.1240 | 0.3222 | 0.1982 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.6360 | 0.9753 | 0.4153 | 0.9903 | 0.2627 | 0.0530 | 0.7373 | 0.0344 | 0.1240 | 0.3222 | 0.1982 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.6734 | 0.9797 | 0.5262 | 0.9966 | 0.3703 | 0.0243 | 0.6297 | 0.0356 | 0.1280 | 0.4314 | 0.3034 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.6734 | 0.9797 | 0.5409 | 0.9933 | 0.3859 | 0.0517 | 0.6141 | 0.0356 | 0.1280 | 0.4314 | 0.3034 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.6734 | 0.9797 | 0.5409 | 0.9933 | 0.3859 | 0.0517 | 0.6141 | 0.0356 | 0.1280 | 0.4314 | 0.3034 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.7664 | 0.9853 | 0.6292 | 0.9968 | 0.4722 | 0.0324 | 0.5278 | 0.0808 | 0.1227 | 0.5428 | 0.4201 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.7664 | 0.9853 | 0.6639 | 0.9950 | 0.5073 | 0.0546 | 0.4927 | 0.0808 | 0.1227 | 0.5428 | 0.4201 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.7664 | 0.9853 | 0.6639 | 0.9950 | 0.5073 | 0.0546 | 0.4927 | 0.0808 | 0.1227 | 0.5428 | 0.4201 |

## Client Stability

| clients_by | training_mode | calibration_scope | model | decision_policy | false_alarm_rate_std | miss_rate_std | uncertain_rate_std | health_gap_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0241 | 0.1739 | 0.0147 | 0.2481 |
| bearing | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0410 | 0.1782 | 0.0147 | 0.2481 |
| bearing | centralized | adaptive | cnn-ae | hard_val_p95 | 0.0410 | 0.1782 | 0.0147 | 0.2481 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0220 | 0.2429 | 0.0392 | 0.2542 |
| bearing | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0202 | 0.2461 | 0.0392 | 0.2542 |
| bearing | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0202 | 0.2461 | 0.0392 | 0.2542 |
| bearing | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0576 | 0.0027 | 0.0027 | 0.1627 |
| bearing | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1053 | 0.0027 | 0.0027 | 0.1627 |
| bearing | centralized | pooled | cnn-ae | hard_val_p95 | 0.1053 | 0.0027 | 0.0027 | 0.1627 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0187 | 0.1367 | 0.0345 | 0.2166 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0353 | 0.1399 | 0.0345 | 0.2166 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.0353 | 0.1399 | 0.0345 | 0.2166 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0127 | 0.2280 | 0.0395 | 0.2437 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0152 | 0.2395 | 0.0395 | 0.2437 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0152 | 0.2395 | 0.0395 | 0.2437 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0605 | 0.0348 | 0.0409 | 0.1589 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1090 | 0.0149 | 0.0409 | 0.1589 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 0.1090 | 0.0149 | 0.0409 | 0.1589 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0124 | 0.2559 | 0.0386 | 0.2683 |
| bearing | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0191 | 0.2606 | 0.0386 | 0.2683 |
| bearing | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0191 | 0.2606 | 0.0386 | 0.2683 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0222 | 0.2109 | 0.0518 | 0.2776 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0550 | 0.2085 | 0.0518 | 0.2776 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.0550 | 0.2085 | 0.0518 | 0.2776 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0136 | 0.2637 | 0.0300 | 0.2715 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0233 | 0.2657 | 0.0300 | 0.2715 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0233 | 0.2657 | 0.0300 | 0.2715 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.1161 | 0.1033 | 0.0815 | 0.1804 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1561 | 0.1088 | 0.0815 | 0.1804 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 0.1561 | 0.1088 | 0.0815 | 0.1804 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0159 | 0.2562 | 0.0356 | 0.2727 |
| bearing | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0258 | 0.2616 | 0.0356 | 0.2727 |
| bearing | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0258 | 0.2616 | 0.0356 | 0.2727 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0177 | 0.1360 | 0.0342 | 0.2172 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0369 | 0.1391 | 0.0342 | 0.2172 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.0369 | 0.1391 | 0.0342 | 0.2172 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0127 | 0.2254 | 0.0400 | 0.2425 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0186 | 0.2371 | 0.0400 | 0.2425 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0186 | 0.2371 | 0.0400 | 0.2425 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0598 | 0.0316 | 0.0402 | 0.1587 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1095 | 0.0129 | 0.0402 | 0.1587 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 0.1095 | 0.0129 | 0.0402 | 0.1587 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0165 | 0.2574 | 0.0373 | 0.2718 |
| bearing | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0227 | 0.2632 | 0.0373 | 0.2718 |
| bearing | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0227 | 0.2632 | 0.0373 | 0.2718 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0130 | 0.2295 | 0.0492 | 0.2031 |
| bearing | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0190 | 0.2146 | 0.0492 | 0.2031 |
| bearing | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0190 | 0.2146 | 0.0492 | 0.2031 |
| condition | centralized | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0182 | 0.0241 | 0.0056 | 0.0790 |
| condition | centralized | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0351 | 0.0224 | 0.0056 | 0.0790 |
| condition | centralized | adaptive | cnn-ae | hard_val_p95 | 0.0351 | 0.0224 | 0.0056 | 0.0790 |
| condition | centralized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0078 | 0.0710 | 0.0169 | 0.0990 |
| condition | centralized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0093 | 0.0774 | 0.0169 | 0.0990 |
| condition | centralized | client_specific | cnn-ae | hard_val_p95 | 0.0093 | 0.0774 | 0.0169 | 0.0990 |
| condition | centralized | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0318 | 0.0026 | 0.0040 | 0.0656 |
| condition | centralized | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0559 | 0.0014 | 0.0040 | 0.0656 |
| condition | centralized | pooled | cnn-ae | hard_val_p95 | 0.0559 | 0.0014 | 0.0040 | 0.0656 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0196 | 0.0286 | 0.0080 | 0.0801 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0395 | 0.0278 | 0.0080 | 0.0801 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 0.0395 | 0.0278 | 0.0080 | 0.0801 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0081 | 0.0992 | 0.0245 | 0.1195 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0112 | 0.1091 | 0.0245 | 0.1195 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 0.0112 | 0.1091 | 0.0245 | 0.1195 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0333 | 0.0067 | 0.0088 | 0.0670 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0644 | 0.0064 | 0.0088 | 0.0670 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 0.0644 | 0.0064 | 0.0088 | 0.0670 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0059 | 0.1350 | 0.0178 | 0.1514 |
| condition | fedavg-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0148 | 0.1427 | 0.0178 | 0.1514 |
| condition | fedavg-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0148 | 0.1427 | 0.0178 | 0.1514 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0564 | 0.0805 | 0.0644 | 0.1130 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0918 | 0.0844 | 0.0644 | 0.1130 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 0.0918 | 0.0844 | 0.0644 | 0.1130 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0092 | 0.1341 | 0.0187 | 0.1466 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0145 | 0.1415 | 0.0187 | 0.1466 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 0.0145 | 0.1415 | 0.0187 | 0.1466 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.1107 | 0.0805 | 0.0861 | 0.0956 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.1384 | 0.1017 | 0.0861 | 0.0956 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 0.1384 | 0.1017 | 0.0861 | 0.0956 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0108 | 0.1365 | 0.0179 | 0.1542 |
| condition | fedbn-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0127 | 0.1452 | 0.0179 | 0.1542 |
| condition | fedbn-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0127 | 0.1452 | 0.0179 | 0.1542 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 0.0190 | 0.0290 | 0.0097 | 0.0797 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0376 | 0.0278 | 0.0097 | 0.0797 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 0.0376 | 0.0278 | 0.0097 | 0.0797 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0074 | 0.1003 | 0.0244 | 0.1211 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0116 | 0.1105 | 0.0244 | 0.1211 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 0.0116 | 0.1105 | 0.0244 | 0.1211 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 0.0340 | 0.0088 | 0.0112 | 0.0664 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0631 | 0.0067 | 0.0112 | 0.0664 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 0.0631 | 0.0067 | 0.0112 | 0.0664 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0052 | 0.1368 | 0.0178 | 0.1522 |
| condition | fedprox-personalized | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0089 | 0.1453 | 0.0178 | 0.1522 |
| condition | fedprox-personalized | client_specific | cnn-ae | hard_val_p95 | 0.0089 | 0.1453 | 0.0178 | 0.1522 |
| condition | local-only | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 0.0127 | 0.1322 | 0.0498 | 0.1091 |
| condition | local-only | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 0.0129 | 0.1158 | 0.0498 | 0.1091 |
| condition | local-only | client_specific | cnn-ae | hard_val_p95 | 0.0129 | 0.1158 | 0.0498 | 0.1091 |

## Federated Convergence

| clients_by | training_mode | calibration_scope | model | decision_policy | round | roc_auc | pr_auc | f1 | precision | recall | false_alarm_rate | miss_rate | uncertain_rate | mean_health_normal | mean_health_fault | health_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6484 | 0.9911 | 0.3527 | 0.9964 | 0.2313 | 0.0218 | 0.7687 | 0.1251 | 0.1136 | 0.3367 | 0.2231 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6826 | 0.9932 | 0.5904 | 0.9993 | 0.4412 | 0.0139 | 0.5588 | 0.0400 | 0.0947 | 0.4840 | 0.3893 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6857 | 0.9932 | 0.5700 | 0.9994 | 0.4151 | 0.0133 | 0.5849 | 0.0516 | 0.0928 | 0.4653 | 0.3725 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6770 | 0.9931 | 0.5233 | 0.9995 | 0.3636 | 0.0103 | 0.6364 | 0.0424 | 0.0895 | 0.4163 | 0.3268 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6759 | 0.9931 | 0.5190 | 0.9995 | 0.3587 | 0.0098 | 0.6413 | 0.0394 | 0.0885 | 0.4104 | 0.3219 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6762 | 0.9931 | 0.5180 | 0.9994 | 0.3576 | 0.0120 | 0.6424 | 0.0385 | 0.0885 | 0.4090 | 0.3205 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6763 | 0.9931 | 0.5180 | 0.9994 | 0.3576 | 0.0116 | 0.6424 | 0.0375 | 0.0881 | 0.4087 | 0.3207 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6767 | 0.9931 | 0.5179 | 0.9994 | 0.3576 | 0.0126 | 0.6424 | 0.0375 | 0.0882 | 0.4087 | 0.3205 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6773 | 0.9931 | 0.5183 | 0.9994 | 0.3580 | 0.0119 | 0.6420 | 0.0372 | 0.0884 | 0.4090 | 0.3206 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6774 | 0.9931 | 0.5190 | 0.9994 | 0.3588 | 0.0128 | 0.6412 | 0.0379 | 0.0882 | 0.4099 | 0.3218 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6484 | 0.9911 | 0.4265 | 0.9959 | 0.2878 | 0.0436 | 0.7122 | 0.1251 | 0.1136 | 0.3367 | 0.2231 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6826 | 0.9932 | 0.6099 | 0.9986 | 0.4623 | 0.0292 | 0.5377 | 0.0400 | 0.0947 | 0.4840 | 0.3893 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6857 | 0.9932 | 0.5943 | 0.9988 | 0.4415 | 0.0285 | 0.5585 | 0.0516 | 0.0928 | 0.4653 | 0.3725 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6770 | 0.9931 | 0.5446 | 0.9988 | 0.3847 | 0.0260 | 0.6153 | 0.0424 | 0.0895 | 0.4163 | 0.3268 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6759 | 0.9931 | 0.5381 | 0.9988 | 0.3775 | 0.0251 | 0.6225 | 0.0394 | 0.0885 | 0.4104 | 0.3219 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6762 | 0.9931 | 0.5363 | 0.9989 | 0.3756 | 0.0243 | 0.6244 | 0.0385 | 0.0885 | 0.4090 | 0.3205 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6763 | 0.9931 | 0.5356 | 0.9989 | 0.3748 | 0.0241 | 0.6252 | 0.0375 | 0.0881 | 0.4087 | 0.3207 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6767 | 0.9931 | 0.5350 | 0.9989 | 0.3742 | 0.0242 | 0.6258 | 0.0375 | 0.0882 | 0.4087 | 0.3205 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6773 | 0.9931 | 0.5351 | 0.9989 | 0.3743 | 0.0239 | 0.6257 | 0.0372 | 0.0884 | 0.4090 | 0.3206 |
| bearing | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6774 | 0.9931 | 0.5357 | 0.9989 | 0.3751 | 0.0227 | 0.6249 | 0.0379 | 0.0882 | 0.4099 | 0.3218 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6484 | 0.9911 | 0.4265 | 0.9959 | 0.2878 | 0.0436 | 0.7122 | 0.1251 | 0.1136 | 0.3367 | 0.2231 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6826 | 0.9932 | 0.6099 | 0.9986 | 0.4623 | 0.0292 | 0.5377 | 0.0400 | 0.0947 | 0.4840 | 0.3893 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6857 | 0.9932 | 0.5943 | 0.9988 | 0.4415 | 0.0285 | 0.5585 | 0.0516 | 0.0928 | 0.4653 | 0.3725 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 4 | 0.6770 | 0.9931 | 0.5446 | 0.9988 | 0.3847 | 0.0260 | 0.6153 | 0.0424 | 0.0895 | 0.4163 | 0.3268 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 5 | 0.6759 | 0.9931 | 0.5381 | 0.9988 | 0.3775 | 0.0251 | 0.6225 | 0.0394 | 0.0885 | 0.4104 | 0.3219 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 6 | 0.6762 | 0.9931 | 0.5363 | 0.9989 | 0.3756 | 0.0243 | 0.6244 | 0.0385 | 0.0885 | 0.4090 | 0.3205 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 7 | 0.6763 | 0.9931 | 0.5356 | 0.9989 | 0.3748 | 0.0241 | 0.6252 | 0.0375 | 0.0881 | 0.4087 | 0.3207 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 8 | 0.6767 | 0.9931 | 0.5350 | 0.9989 | 0.3742 | 0.0242 | 0.6258 | 0.0375 | 0.0882 | 0.4087 | 0.3205 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 9 | 0.6773 | 0.9931 | 0.5351 | 0.9989 | 0.3743 | 0.0239 | 0.6257 | 0.0372 | 0.0884 | 0.4090 | 0.3206 |
| bearing | fedavg | adaptive | cnn-ae | hard_val_p95 | 10 | 0.6774 | 0.9931 | 0.5357 | 0.9989 | 0.3751 | 0.0227 | 0.6249 | 0.0379 | 0.0882 | 0.4099 | 0.3218 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6691 | 0.9914 | 0.4021 | 0.9960 | 0.2845 | 0.0260 | 0.7155 | 0.1089 | 0.1121 | 0.3726 | 0.2604 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.7257 | 0.9937 | 0.6429 | 0.9990 | 0.5139 | 0.0284 | 0.4861 | 0.0549 | 0.1025 | 0.5535 | 0.4509 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7228 | 0.9935 | 0.6455 | 0.9989 | 0.5156 | 0.0271 | 0.4844 | 0.0506 | 0.1041 | 0.5525 | 0.4484 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.7060 | 0.9933 | 0.6190 | 0.9989 | 0.4782 | 0.0251 | 0.5218 | 0.0582 | 0.1058 | 0.5207 | 0.4149 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.7031 | 0.9933 | 0.6184 | 0.9989 | 0.4767 | 0.0261 | 0.5233 | 0.0555 | 0.1056 | 0.5176 | 0.4120 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.7026 | 0.9932 | 0.6197 | 0.9989 | 0.4784 | 0.0267 | 0.5216 | 0.0540 | 0.1074 | 0.5187 | 0.4113 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.7016 | 0.9933 | 0.6215 | 0.9989 | 0.4808 | 0.0265 | 0.5192 | 0.0520 | 0.1071 | 0.5199 | 0.4128 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.7028 | 0.9933 | 0.6228 | 0.9988 | 0.4827 | 0.0277 | 0.5173 | 0.0508 | 0.1070 | 0.5210 | 0.4140 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.7025 | 0.9933 | 0.6234 | 0.9988 | 0.4834 | 0.0272 | 0.5166 | 0.0503 | 0.1069 | 0.5216 | 0.4147 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.7032 | 0.9933 | 0.6240 | 0.9988 | 0.4845 | 0.0267 | 0.5155 | 0.0505 | 0.1062 | 0.5225 | 0.4162 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6691 | 0.9914 | 0.4630 | 0.9951 | 0.3349 | 0.0554 | 0.6651 | 0.1089 | 0.1121 | 0.3726 | 0.2604 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.7257 | 0.9937 | 0.6655 | 0.9981 | 0.5429 | 0.0496 | 0.4571 | 0.0549 | 0.1025 | 0.5535 | 0.4509 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7228 | 0.9935 | 0.6669 | 0.9979 | 0.5431 | 0.0512 | 0.4569 | 0.0506 | 0.1041 | 0.5525 | 0.4484 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.7060 | 0.9933 | 0.6462 | 0.9977 | 0.5122 | 0.0554 | 0.4878 | 0.0582 | 0.1058 | 0.5207 | 0.4149 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.7031 | 0.9933 | 0.6437 | 0.9978 | 0.5080 | 0.0532 | 0.4920 | 0.0555 | 0.1056 | 0.5176 | 0.4120 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.7026 | 0.9932 | 0.6448 | 0.9977 | 0.5096 | 0.0552 | 0.4904 | 0.0540 | 0.1074 | 0.5187 | 0.4113 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.7016 | 0.9933 | 0.6454 | 0.9978 | 0.5103 | 0.0535 | 0.4897 | 0.0520 | 0.1071 | 0.5199 | 0.4128 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.7028 | 0.9933 | 0.6455 | 0.9978 | 0.5107 | 0.0534 | 0.4893 | 0.0508 | 0.1070 | 0.5210 | 0.4140 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.7025 | 0.9933 | 0.6456 | 0.9977 | 0.5108 | 0.0544 | 0.4892 | 0.0503 | 0.1069 | 0.5216 | 0.4147 |
| bearing | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.7032 | 0.9933 | 0.6461 | 0.9978 | 0.5117 | 0.0527 | 0.4883 | 0.0505 | 0.1062 | 0.5225 | 0.4162 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6691 | 0.9914 | 0.4630 | 0.9951 | 0.3349 | 0.0554 | 0.6651 | 0.1089 | 0.1121 | 0.3726 | 0.2604 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.7257 | 0.9937 | 0.6655 | 0.9981 | 0.5429 | 0.0496 | 0.4571 | 0.0549 | 0.1025 | 0.5535 | 0.4509 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.7228 | 0.9935 | 0.6669 | 0.9979 | 0.5431 | 0.0512 | 0.4569 | 0.0506 | 0.1041 | 0.5525 | 0.4484 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.7060 | 0.9933 | 0.6462 | 0.9977 | 0.5122 | 0.0554 | 0.4878 | 0.0582 | 0.1058 | 0.5207 | 0.4149 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.7031 | 0.9933 | 0.6437 | 0.9978 | 0.5080 | 0.0532 | 0.4920 | 0.0555 | 0.1056 | 0.5176 | 0.4120 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.7026 | 0.9932 | 0.6448 | 0.9977 | 0.5096 | 0.0552 | 0.4904 | 0.0540 | 0.1074 | 0.5187 | 0.4113 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.7016 | 0.9933 | 0.6454 | 0.9978 | 0.5103 | 0.0535 | 0.4897 | 0.0520 | 0.1071 | 0.5199 | 0.4128 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.7028 | 0.9933 | 0.6455 | 0.9978 | 0.5107 | 0.0534 | 0.4893 | 0.0508 | 0.1070 | 0.5210 | 0.4140 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.7025 | 0.9933 | 0.6456 | 0.9977 | 0.5108 | 0.0544 | 0.4892 | 0.0503 | 0.1069 | 0.5216 | 0.4147 |
| bearing | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.7032 | 0.9933 | 0.6461 | 0.9978 | 0.5117 | 0.0527 | 0.4883 | 0.0505 | 0.1062 | 0.5225 | 0.4162 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6345 | 0.9903 | 0.2542 | 0.9962 | 0.1458 | 0.0416 | 0.8542 | 0.1614 | 0.1459 | 0.2842 | 0.1383 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6207 | 0.9903 | 0.4061 | 0.9983 | 0.2549 | 0.0327 | 0.7451 | 0.0435 | 0.1396 | 0.3315 | 0.1919 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6181 | 0.9902 | 0.4197 | 0.9983 | 0.2657 | 0.0352 | 0.7343 | 0.0292 | 0.1426 | 0.3331 | 0.1905 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6234 | 0.9905 | 0.4108 | 0.9982 | 0.2586 | 0.0339 | 0.7414 | 0.0253 | 0.1385 | 0.3269 | 0.1885 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6238 | 0.9905 | 0.4123 | 0.9983 | 0.2598 | 0.0330 | 0.7402 | 0.0240 | 0.1373 | 0.3273 | 0.1900 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6238 | 0.9906 | 0.4131 | 0.9983 | 0.2605 | 0.0331 | 0.7395 | 0.0247 | 0.1372 | 0.3277 | 0.1904 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6244 | 0.9906 | 0.4144 | 0.9983 | 0.2615 | 0.0333 | 0.7385 | 0.0242 | 0.1372 | 0.3280 | 0.1909 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6247 | 0.9906 | 0.4153 | 0.9982 | 0.2622 | 0.0338 | 0.7378 | 0.0239 | 0.1370 | 0.3284 | 0.1914 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6249 | 0.9906 | 0.4155 | 0.9983 | 0.2624 | 0.0337 | 0.7376 | 0.0241 | 0.1374 | 0.3284 | 0.1910 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6251 | 0.9907 | 0.4157 | 0.9983 | 0.2625 | 0.0337 | 0.7375 | 0.0244 | 0.1368 | 0.3285 | 0.1917 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6345 | 0.9903 | 0.3649 | 0.9958 | 0.2234 | 0.0721 | 0.7766 | 0.1614 | 0.1459 | 0.2842 | 0.1383 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6207 | 0.9903 | 0.4290 | 0.9968 | 0.2734 | 0.0684 | 0.7266 | 0.0435 | 0.1396 | 0.3315 | 0.1919 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6181 | 0.9902 | 0.4347 | 0.9965 | 0.2780 | 0.0772 | 0.7220 | 0.0292 | 0.1426 | 0.3331 | 0.1905 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6234 | 0.9905 | 0.4227 | 0.9967 | 0.2682 | 0.0657 | 0.7318 | 0.0253 | 0.1385 | 0.3269 | 0.1885 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6238 | 0.9905 | 0.4242 | 0.9968 | 0.2695 | 0.0647 | 0.7305 | 0.0240 | 0.1373 | 0.3273 | 0.1900 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6238 | 0.9906 | 0.4251 | 0.9968 | 0.2701 | 0.0637 | 0.7299 | 0.0247 | 0.1372 | 0.3277 | 0.1904 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6244 | 0.9906 | 0.4260 | 0.9968 | 0.2709 | 0.0648 | 0.7291 | 0.0242 | 0.1372 | 0.3280 | 0.1909 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6247 | 0.9906 | 0.4268 | 0.9968 | 0.2716 | 0.0655 | 0.7284 | 0.0239 | 0.1370 | 0.3284 | 0.1914 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6249 | 0.9906 | 0.4271 | 0.9968 | 0.2718 | 0.0662 | 0.7282 | 0.0241 | 0.1374 | 0.3284 | 0.1910 |
| bearing | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6251 | 0.9907 | 0.4274 | 0.9968 | 0.2720 | 0.0661 | 0.7280 | 0.0244 | 0.1368 | 0.3285 | 0.1917 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.6345 | 0.9903 | 0.3649 | 0.9958 | 0.2234 | 0.0721 | 0.7766 | 0.1614 | 0.1459 | 0.2842 | 0.1383 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.6207 | 0.9903 | 0.4290 | 0.9968 | 0.2734 | 0.0684 | 0.7266 | 0.0435 | 0.1396 | 0.3315 | 0.1919 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.6181 | 0.9902 | 0.4347 | 0.9965 | 0.2780 | 0.0772 | 0.7220 | 0.0292 | 0.1426 | 0.3331 | 0.1905 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.6234 | 0.9905 | 0.4227 | 0.9967 | 0.2682 | 0.0657 | 0.7318 | 0.0253 | 0.1385 | 0.3269 | 0.1885 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.6238 | 0.9905 | 0.4242 | 0.9968 | 0.2695 | 0.0647 | 0.7305 | 0.0240 | 0.1373 | 0.3273 | 0.1900 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.6238 | 0.9906 | 0.4251 | 0.9968 | 0.2701 | 0.0637 | 0.7299 | 0.0247 | 0.1372 | 0.3277 | 0.1904 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.6244 | 0.9906 | 0.4260 | 0.9968 | 0.2709 | 0.0648 | 0.7291 | 0.0242 | 0.1372 | 0.3280 | 0.1909 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.6247 | 0.9906 | 0.4268 | 0.9968 | 0.2716 | 0.0655 | 0.7284 | 0.0239 | 0.1370 | 0.3284 | 0.1914 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.6249 | 0.9906 | 0.4271 | 0.9968 | 0.2718 | 0.0662 | 0.7282 | 0.0241 | 0.1374 | 0.3284 | 0.1910 |
| bearing | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.6251 | 0.9907 | 0.4274 | 0.9968 | 0.2720 | 0.0661 | 0.7280 | 0.0244 | 0.1368 | 0.3285 | 0.1917 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.8144 | 0.9960 | 0.6250 | 0.9991 | 0.4974 | 0.0147 | 0.5026 | 0.1323 | 0.1408 | 0.6106 | 0.4698 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.7896 | 0.9954 | 0.7029 | 0.9996 | 0.5679 | 0.0161 | 0.4321 | 0.0426 | 0.0945 | 0.6157 | 0.5212 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7725 | 0.9952 | 0.6864 | 0.9996 | 0.5491 | 0.0138 | 0.4509 | 0.0336 | 0.0952 | 0.5959 | 0.5007 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.7388 | 0.9948 | 0.6451 | 0.9995 | 0.5022 | 0.0126 | 0.4978 | 0.0287 | 0.0916 | 0.5442 | 0.4526 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.7343 | 0.9948 | 0.6338 | 0.9995 | 0.4899 | 0.0128 | 0.5101 | 0.0285 | 0.0919 | 0.5324 | 0.4405 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.7326 | 0.9948 | 0.6278 | 0.9996 | 0.4833 | 0.0106 | 0.5167 | 0.0285 | 0.0903 | 0.5260 | 0.4357 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.7329 | 0.9949 | 0.6228 | 0.9995 | 0.4778 | 0.0121 | 0.5222 | 0.0297 | 0.0890 | 0.5216 | 0.4326 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.7328 | 0.9950 | 0.6217 | 0.9994 | 0.4767 | 0.0139 | 0.5233 | 0.0294 | 0.0906 | 0.5205 | 0.4299 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.7338 | 0.9950 | 0.6214 | 0.9995 | 0.4763 | 0.0114 | 0.5237 | 0.0297 | 0.0894 | 0.5205 | 0.4311 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.7353 | 0.9951 | 0.6213 | 0.9995 | 0.4762 | 0.0119 | 0.5238 | 0.0307 | 0.0896 | 0.5212 | 0.4316 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.8144 | 0.9960 | 0.6934 | 0.9990 | 0.5649 | 0.0458 | 0.4351 | 0.1323 | 0.1408 | 0.6106 | 0.4698 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.7896 | 0.9954 | 0.7208 | 0.9991 | 0.5880 | 0.0356 | 0.4120 | 0.0426 | 0.0945 | 0.6157 | 0.5212 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7725 | 0.9952 | 0.6987 | 0.9991 | 0.5632 | 0.0281 | 0.4368 | 0.0336 | 0.0952 | 0.5959 | 0.5007 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.7388 | 0.9948 | 0.6561 | 0.9989 | 0.5146 | 0.0268 | 0.4854 | 0.0287 | 0.0916 | 0.5442 | 0.4526 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.7343 | 0.9948 | 0.6454 | 0.9988 | 0.5025 | 0.0310 | 0.4975 | 0.0285 | 0.0919 | 0.5324 | 0.4405 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.7326 | 0.9948 | 0.6395 | 0.9989 | 0.4959 | 0.0286 | 0.5041 | 0.0285 | 0.0903 | 0.5260 | 0.4357 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.7329 | 0.9949 | 0.6348 | 0.9990 | 0.4906 | 0.0249 | 0.5094 | 0.0297 | 0.0890 | 0.5216 | 0.4326 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.7328 | 0.9950 | 0.6336 | 0.9989 | 0.4894 | 0.0287 | 0.5106 | 0.0294 | 0.0906 | 0.5205 | 0.4299 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.7338 | 0.9950 | 0.6332 | 0.9988 | 0.4889 | 0.0276 | 0.5111 | 0.0297 | 0.0894 | 0.5205 | 0.4311 |
| bearing | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.7353 | 0.9951 | 0.6337 | 0.9988 | 0.4896 | 0.0282 | 0.5104 | 0.0307 | 0.0896 | 0.5212 | 0.4316 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 1 | 0.8144 | 0.9960 | 0.6934 | 0.9990 | 0.5649 | 0.0458 | 0.4351 | 0.1323 | 0.1408 | 0.6106 | 0.4698 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 2 | 0.7896 | 0.9954 | 0.7208 | 0.9991 | 0.5880 | 0.0356 | 0.4120 | 0.0426 | 0.0945 | 0.6157 | 0.5212 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 3 | 0.7725 | 0.9952 | 0.6987 | 0.9991 | 0.5632 | 0.0281 | 0.4368 | 0.0336 | 0.0952 | 0.5959 | 0.5007 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 4 | 0.7388 | 0.9948 | 0.6561 | 0.9989 | 0.5146 | 0.0268 | 0.4854 | 0.0287 | 0.0916 | 0.5442 | 0.4526 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 5 | 0.7343 | 0.9948 | 0.6454 | 0.9988 | 0.5025 | 0.0310 | 0.4975 | 0.0285 | 0.0919 | 0.5324 | 0.4405 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 6 | 0.7326 | 0.9948 | 0.6395 | 0.9989 | 0.4959 | 0.0286 | 0.5041 | 0.0285 | 0.0903 | 0.5260 | 0.4357 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 7 | 0.7329 | 0.9949 | 0.6348 | 0.9990 | 0.4906 | 0.0249 | 0.5094 | 0.0297 | 0.0890 | 0.5216 | 0.4326 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 8 | 0.7328 | 0.9950 | 0.6336 | 0.9989 | 0.4894 | 0.0287 | 0.5106 | 0.0294 | 0.0906 | 0.5205 | 0.4299 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 9 | 0.7338 | 0.9950 | 0.6332 | 0.9988 | 0.4889 | 0.0276 | 0.5111 | 0.0297 | 0.0894 | 0.5205 | 0.4311 |
| bearing | fedbn | adaptive | cnn-ae | hard_val_p95 | 10 | 0.7353 | 0.9951 | 0.6337 | 0.9988 | 0.4896 | 0.0282 | 0.5104 | 0.0307 | 0.0896 | 0.5212 | 0.4316 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.8161 | 0.9958 | 0.6896 | 0.9984 | 0.5891 | 0.0246 | 0.4109 | 0.0682 | 0.1072 | 0.6415 | 0.5343 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.8007 | 0.9953 | 0.7587 | 0.9992 | 0.6481 | 0.0253 | 0.3519 | 0.0332 | 0.1069 | 0.6754 | 0.5685 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7843 | 0.9952 | 0.7394 | 0.9991 | 0.6288 | 0.0253 | 0.3712 | 0.0294 | 0.1051 | 0.6541 | 0.5490 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.7569 | 0.9949 | 0.7125 | 0.9990 | 0.5976 | 0.0288 | 0.4024 | 0.0305 | 0.1085 | 0.6223 | 0.5137 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.7534 | 0.9949 | 0.7038 | 0.9990 | 0.5870 | 0.0278 | 0.4130 | 0.0330 | 0.1081 | 0.6136 | 0.5055 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.7516 | 0.9949 | 0.6997 | 0.9991 | 0.5821 | 0.0265 | 0.4179 | 0.0336 | 0.1076 | 0.6097 | 0.5022 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.7527 | 0.9949 | 0.6963 | 0.9990 | 0.5780 | 0.0286 | 0.4220 | 0.0360 | 0.1072 | 0.6071 | 0.4999 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.7536 | 0.9950 | 0.6953 | 0.9990 | 0.5763 | 0.0289 | 0.4237 | 0.0371 | 0.1091 | 0.6063 | 0.4972 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.7539 | 0.9951 | 0.6946 | 0.9990 | 0.5757 | 0.0261 | 0.4243 | 0.0375 | 0.1064 | 0.6060 | 0.4996 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.7563 | 0.9951 | 0.6945 | 0.9990 | 0.5756 | 0.0272 | 0.4244 | 0.0400 | 0.1074 | 0.6071 | 0.4997 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.8161 | 0.9958 | 0.7214 | 0.9979 | 0.6231 | 0.0565 | 0.3769 | 0.0682 | 0.1072 | 0.6415 | 0.5343 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.8007 | 0.9953 | 0.7717 | 0.9983 | 0.6647 | 0.0544 | 0.3353 | 0.0332 | 0.1069 | 0.6754 | 0.5685 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7843 | 0.9952 | 0.7505 | 0.9982 | 0.6434 | 0.0510 | 0.3566 | 0.0294 | 0.1051 | 0.6541 | 0.5490 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.7569 | 0.9949 | 0.7239 | 0.9980 | 0.6131 | 0.0538 | 0.3869 | 0.0305 | 0.1085 | 0.6223 | 0.5137 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.7534 | 0.9949 | 0.7168 | 0.9979 | 0.6044 | 0.0553 | 0.3956 | 0.0330 | 0.1081 | 0.6136 | 0.5055 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.7516 | 0.9949 | 0.7133 | 0.9980 | 0.6001 | 0.0528 | 0.3999 | 0.0336 | 0.1076 | 0.6097 | 0.5022 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.7527 | 0.9949 | 0.7100 | 0.9981 | 0.5961 | 0.0541 | 0.4039 | 0.0360 | 0.1072 | 0.6071 | 0.4999 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.7536 | 0.9950 | 0.7094 | 0.9980 | 0.5948 | 0.0575 | 0.4052 | 0.0371 | 0.1091 | 0.6063 | 0.4972 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.7539 | 0.9951 | 0.7088 | 0.9981 | 0.5941 | 0.0514 | 0.4059 | 0.0375 | 0.1064 | 0.6060 | 0.4996 |
| bearing | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.7563 | 0.9951 | 0.7093 | 0.9980 | 0.5950 | 0.0551 | 0.4050 | 0.0400 | 0.1074 | 0.6071 | 0.4997 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.8161 | 0.9958 | 0.7214 | 0.9979 | 0.6231 | 0.0565 | 0.3769 | 0.0682 | 0.1072 | 0.6415 | 0.5343 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.8007 | 0.9953 | 0.7717 | 0.9983 | 0.6647 | 0.0544 | 0.3353 | 0.0332 | 0.1069 | 0.6754 | 0.5685 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.7843 | 0.9952 | 0.7505 | 0.9982 | 0.6434 | 0.0510 | 0.3566 | 0.0294 | 0.1051 | 0.6541 | 0.5490 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 4 | 0.7569 | 0.9949 | 0.7239 | 0.9980 | 0.6131 | 0.0538 | 0.3869 | 0.0305 | 0.1085 | 0.6223 | 0.5137 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 5 | 0.7534 | 0.9949 | 0.7168 | 0.9979 | 0.6044 | 0.0553 | 0.3956 | 0.0330 | 0.1081 | 0.6136 | 0.5055 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 6 | 0.7516 | 0.9949 | 0.7133 | 0.9980 | 0.6001 | 0.0528 | 0.3999 | 0.0336 | 0.1076 | 0.6097 | 0.5022 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 7 | 0.7527 | 0.9949 | 0.7100 | 0.9981 | 0.5961 | 0.0541 | 0.4039 | 0.0360 | 0.1072 | 0.6071 | 0.4999 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 8 | 0.7536 | 0.9950 | 0.7094 | 0.9980 | 0.5948 | 0.0575 | 0.4052 | 0.0371 | 0.1091 | 0.6063 | 0.4972 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 9 | 0.7539 | 0.9951 | 0.7088 | 0.9981 | 0.5941 | 0.0514 | 0.4059 | 0.0375 | 0.1064 | 0.6060 | 0.4996 |
| bearing | fedbn | client_specific | cnn-ae | hard_val_p95 | 10 | 0.7563 | 0.9951 | 0.7093 | 0.9980 | 0.5950 | 0.0551 | 0.4050 | 0.0400 | 0.1074 | 0.6071 | 0.4997 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.8062 | 0.9954 | 0.3027 | 0.8742 | 0.2296 | 0.0972 | 0.7704 | 0.2214 | 0.2129 | 0.4215 | 0.2086 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.7579 | 0.9946 | 0.5504 | 0.9991 | 0.3845 | 0.0590 | 0.6155 | 0.0804 | 0.1623 | 0.4877 | 0.3254 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7406 | 0.9941 | 0.5419 | 0.9986 | 0.3740 | 0.0516 | 0.6260 | 0.0566 | 0.1603 | 0.4752 | 0.3149 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.7025 | 0.9931 | 0.4856 | 0.9984 | 0.3212 | 0.0386 | 0.6788 | 0.0332 | 0.1460 | 0.4145 | 0.2685 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6969 | 0.9930 | 0.4741 | 0.9983 | 0.3111 | 0.0383 | 0.6889 | 0.0296 | 0.1437 | 0.4023 | 0.2586 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6936 | 0.9930 | 0.4688 | 0.9983 | 0.3065 | 0.0359 | 0.6935 | 0.0288 | 0.1412 | 0.3955 | 0.2544 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6933 | 0.9930 | 0.4644 | 0.9984 | 0.3027 | 0.0342 | 0.6973 | 0.0292 | 0.1405 | 0.3920 | 0.2515 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6929 | 0.9930 | 0.4625 | 0.9983 | 0.3012 | 0.0355 | 0.6988 | 0.0282 | 0.1409 | 0.3899 | 0.2490 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6939 | 0.9931 | 0.4620 | 0.9983 | 0.3008 | 0.0360 | 0.6992 | 0.0286 | 0.1401 | 0.3898 | 0.2497 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6955 | 0.9932 | 0.4610 | 0.9983 | 0.2999 | 0.0363 | 0.7001 | 0.0301 | 0.1403 | 0.3903 | 0.2500 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.8062 | 0.9954 | 0.4627 | 0.8738 | 0.3631 | 0.1310 | 0.6369 | 0.2214 | 0.2129 | 0.4215 | 0.2086 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.7579 | 0.9946 | 0.5905 | 0.9980 | 0.4236 | 0.1012 | 0.5764 | 0.0804 | 0.1623 | 0.4877 | 0.3254 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7406 | 0.9941 | 0.5628 | 0.9974 | 0.3944 | 0.0913 | 0.6056 | 0.0566 | 0.1603 | 0.4752 | 0.3149 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.7025 | 0.9931 | 0.4977 | 0.9970 | 0.3320 | 0.0767 | 0.6680 | 0.0332 | 0.1460 | 0.4145 | 0.2685 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6969 | 0.9930 | 0.4844 | 0.9969 | 0.3202 | 0.0727 | 0.6798 | 0.0296 | 0.1437 | 0.4023 | 0.2586 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6936 | 0.9930 | 0.4791 | 0.9970 | 0.3155 | 0.0682 | 0.6845 | 0.0288 | 0.1412 | 0.3955 | 0.2544 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6933 | 0.9930 | 0.4750 | 0.9970 | 0.3120 | 0.0676 | 0.6880 | 0.0292 | 0.1405 | 0.3920 | 0.2515 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6929 | 0.9930 | 0.4726 | 0.9969 | 0.3099 | 0.0700 | 0.6901 | 0.0282 | 0.1409 | 0.3899 | 0.2490 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6939 | 0.9931 | 0.4719 | 0.9970 | 0.3093 | 0.0656 | 0.6907 | 0.0286 | 0.1401 | 0.3898 | 0.2497 |
| bearing | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6955 | 0.9932 | 0.4726 | 0.9969 | 0.3099 | 0.0683 | 0.6901 | 0.0301 | 0.1403 | 0.3903 | 0.2500 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.8062 | 0.9954 | 0.4627 | 0.8738 | 0.3631 | 0.1310 | 0.6369 | 0.2214 | 0.2129 | 0.4215 | 0.2086 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.7579 | 0.9946 | 0.5905 | 0.9980 | 0.4236 | 0.1012 | 0.5764 | 0.0804 | 0.1623 | 0.4877 | 0.3254 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.7406 | 0.9941 | 0.5628 | 0.9974 | 0.3944 | 0.0913 | 0.6056 | 0.0566 | 0.1603 | 0.4752 | 0.3149 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 4 | 0.7025 | 0.9931 | 0.4977 | 0.9970 | 0.3320 | 0.0767 | 0.6680 | 0.0332 | 0.1460 | 0.4145 | 0.2685 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 5 | 0.6969 | 0.9930 | 0.4844 | 0.9969 | 0.3202 | 0.0727 | 0.6798 | 0.0296 | 0.1437 | 0.4023 | 0.2586 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 6 | 0.6936 | 0.9930 | 0.4791 | 0.9970 | 0.3155 | 0.0682 | 0.6845 | 0.0288 | 0.1412 | 0.3955 | 0.2544 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 7 | 0.6933 | 0.9930 | 0.4750 | 0.9970 | 0.3120 | 0.0676 | 0.6880 | 0.0292 | 0.1405 | 0.3920 | 0.2515 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 8 | 0.6929 | 0.9930 | 0.4726 | 0.9969 | 0.3099 | 0.0700 | 0.6901 | 0.0282 | 0.1409 | 0.3899 | 0.2490 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 9 | 0.6939 | 0.9931 | 0.4719 | 0.9970 | 0.3093 | 0.0656 | 0.6907 | 0.0286 | 0.1401 | 0.3898 | 0.2497 |
| bearing | fedbn | pooled | cnn-ae | hard_val_p95 | 10 | 0.6955 | 0.9932 | 0.4726 | 0.9969 | 0.3099 | 0.0683 | 0.6901 | 0.0301 | 0.1403 | 0.3903 | 0.2500 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6544 | 0.9916 | 0.3795 | 0.9978 | 0.2524 | 0.0156 | 0.7476 | 0.1223 | 0.1083 | 0.3531 | 0.2447 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6831 | 0.9932 | 0.5905 | 0.9993 | 0.4410 | 0.0145 | 0.5590 | 0.0403 | 0.0963 | 0.4842 | 0.3879 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6869 | 0.9932 | 0.5740 | 0.9994 | 0.4196 | 0.0136 | 0.5804 | 0.0509 | 0.0926 | 0.4691 | 0.3765 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6758 | 0.9930 | 0.5252 | 0.9994 | 0.3654 | 0.0123 | 0.6346 | 0.0423 | 0.0911 | 0.4175 | 0.3264 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6760 | 0.9931 | 0.5210 | 0.9994 | 0.3607 | 0.0122 | 0.6393 | 0.0396 | 0.0906 | 0.4123 | 0.3217 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6757 | 0.9931 | 0.5204 | 0.9994 | 0.3599 | 0.0120 | 0.6401 | 0.0385 | 0.0905 | 0.4111 | 0.3207 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6766 | 0.9931 | 0.5192 | 0.9994 | 0.3587 | 0.0127 | 0.6413 | 0.0383 | 0.0899 | 0.4100 | 0.3201 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6769 | 0.9931 | 0.5197 | 0.9994 | 0.3593 | 0.0117 | 0.6407 | 0.0376 | 0.0897 | 0.4103 | 0.3206 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6775 | 0.9931 | 0.5209 | 0.9994 | 0.3606 | 0.0117 | 0.6394 | 0.0377 | 0.0900 | 0.4117 | 0.3217 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6776 | 0.9931 | 0.5207 | 0.9994 | 0.3604 | 0.0119 | 0.6396 | 0.0384 | 0.0899 | 0.4118 | 0.3219 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6544 | 0.9916 | 0.4509 | 0.9971 | 0.3081 | 0.0330 | 0.6919 | 0.1223 | 0.1083 | 0.3531 | 0.2447 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6831 | 0.9932 | 0.6099 | 0.9986 | 0.4623 | 0.0302 | 0.5377 | 0.0403 | 0.0963 | 0.4842 | 0.3879 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6869 | 0.9932 | 0.5976 | 0.9988 | 0.4452 | 0.0270 | 0.5548 | 0.0509 | 0.0926 | 0.4691 | 0.3765 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6758 | 0.9930 | 0.5449 | 0.9987 | 0.3848 | 0.0275 | 0.6152 | 0.0423 | 0.0911 | 0.4175 | 0.3264 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6760 | 0.9931 | 0.5399 | 0.9987 | 0.3792 | 0.0273 | 0.6208 | 0.0396 | 0.0906 | 0.4123 | 0.3217 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6757 | 0.9931 | 0.5387 | 0.9987 | 0.3778 | 0.0290 | 0.6222 | 0.0385 | 0.0905 | 0.4111 | 0.3207 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6766 | 0.9931 | 0.5375 | 0.9988 | 0.3765 | 0.0272 | 0.6235 | 0.0383 | 0.0899 | 0.4100 | 0.3201 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6769 | 0.9931 | 0.5369 | 0.9987 | 0.3760 | 0.0297 | 0.6240 | 0.0376 | 0.0897 | 0.4103 | 0.3206 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6775 | 0.9931 | 0.5379 | 0.9988 | 0.3771 | 0.0280 | 0.6229 | 0.0377 | 0.0900 | 0.4117 | 0.3217 |
| bearing | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6776 | 0.9931 | 0.5378 | 0.9987 | 0.3772 | 0.0289 | 0.6228 | 0.0384 | 0.0899 | 0.4118 | 0.3219 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6544 | 0.9916 | 0.4509 | 0.9971 | 0.3081 | 0.0330 | 0.6919 | 0.1223 | 0.1083 | 0.3531 | 0.2447 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6831 | 0.9932 | 0.6099 | 0.9986 | 0.4623 | 0.0302 | 0.5377 | 0.0403 | 0.0963 | 0.4842 | 0.3879 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6869 | 0.9932 | 0.5976 | 0.9988 | 0.4452 | 0.0270 | 0.5548 | 0.0509 | 0.0926 | 0.4691 | 0.3765 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 4 | 0.6758 | 0.9930 | 0.5449 | 0.9987 | 0.3848 | 0.0275 | 0.6152 | 0.0423 | 0.0911 | 0.4175 | 0.3264 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 5 | 0.6760 | 0.9931 | 0.5399 | 0.9987 | 0.3792 | 0.0273 | 0.6208 | 0.0396 | 0.0906 | 0.4123 | 0.3217 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 6 | 0.6757 | 0.9931 | 0.5387 | 0.9987 | 0.3778 | 0.0290 | 0.6222 | 0.0385 | 0.0905 | 0.4111 | 0.3207 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 7 | 0.6766 | 0.9931 | 0.5375 | 0.9988 | 0.3765 | 0.0272 | 0.6235 | 0.0383 | 0.0899 | 0.4100 | 0.3201 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 8 | 0.6769 | 0.9931 | 0.5369 | 0.9987 | 0.3760 | 0.0297 | 0.6240 | 0.0376 | 0.0897 | 0.4103 | 0.3206 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 9 | 0.6775 | 0.9931 | 0.5379 | 0.9988 | 0.3771 | 0.0280 | 0.6229 | 0.0377 | 0.0900 | 0.4117 | 0.3217 |
| bearing | fedprox | adaptive | cnn-ae | hard_val_p95 | 10 | 0.6776 | 0.9931 | 0.5378 | 0.9987 | 0.3772 | 0.0289 | 0.6228 | 0.0384 | 0.0899 | 0.4118 | 0.3219 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6739 | 0.9919 | 0.4329 | 0.9972 | 0.3108 | 0.0239 | 0.6892 | 0.1068 | 0.1096 | 0.3933 | 0.2837 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.7241 | 0.9937 | 0.6411 | 0.9990 | 0.5107 | 0.0265 | 0.4893 | 0.0560 | 0.1037 | 0.5517 | 0.4480 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7233 | 0.9936 | 0.6460 | 0.9989 | 0.5161 | 0.0276 | 0.4839 | 0.0528 | 0.1043 | 0.5540 | 0.4498 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.7077 | 0.9933 | 0.6197 | 0.9989 | 0.4788 | 0.0282 | 0.5212 | 0.0591 | 0.1085 | 0.5212 | 0.4127 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.7047 | 0.9933 | 0.6188 | 0.9988 | 0.4767 | 0.0276 | 0.5233 | 0.0570 | 0.1088 | 0.5185 | 0.4097 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.7035 | 0.9933 | 0.6199 | 0.9988 | 0.4781 | 0.0269 | 0.5219 | 0.0554 | 0.1092 | 0.5190 | 0.4099 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.7037 | 0.9933 | 0.6213 | 0.9989 | 0.4800 | 0.0253 | 0.5200 | 0.0534 | 0.1095 | 0.5198 | 0.4103 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.7035 | 0.9933 | 0.6226 | 0.9988 | 0.4822 | 0.0264 | 0.5178 | 0.0520 | 0.1091 | 0.5213 | 0.4122 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.7035 | 0.9933 | 0.6243 | 0.9989 | 0.4845 | 0.0253 | 0.5155 | 0.0513 | 0.1098 | 0.5229 | 0.4131 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.7041 | 0.9933 | 0.6248 | 0.9989 | 0.4855 | 0.0252 | 0.5145 | 0.0511 | 0.1092 | 0.5234 | 0.4143 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6739 | 0.9919 | 0.4918 | 0.9962 | 0.3607 | 0.0514 | 0.6393 | 0.1068 | 0.1096 | 0.3933 | 0.2837 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.7241 | 0.9937 | 0.6648 | 0.9980 | 0.5415 | 0.0506 | 0.4585 | 0.0560 | 0.1037 | 0.5517 | 0.4480 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7233 | 0.9936 | 0.6687 | 0.9979 | 0.5453 | 0.0526 | 0.4547 | 0.0528 | 0.1043 | 0.5540 | 0.4498 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.7077 | 0.9933 | 0.6454 | 0.9977 | 0.5106 | 0.0572 | 0.4894 | 0.0591 | 0.1085 | 0.5212 | 0.4127 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.7047 | 0.9933 | 0.6445 | 0.9976 | 0.5084 | 0.0579 | 0.4916 | 0.0570 | 0.1088 | 0.5185 | 0.4097 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.7035 | 0.9933 | 0.6453 | 0.9976 | 0.5094 | 0.0583 | 0.4906 | 0.0554 | 0.1092 | 0.5190 | 0.4099 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.7037 | 0.9933 | 0.6458 | 0.9976 | 0.5104 | 0.0605 | 0.4896 | 0.0534 | 0.1095 | 0.5198 | 0.4103 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.7035 | 0.9933 | 0.6460 | 0.9977 | 0.5111 | 0.0583 | 0.4889 | 0.0520 | 0.1091 | 0.5213 | 0.4122 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.7035 | 0.9933 | 0.6471 | 0.9976 | 0.5125 | 0.0598 | 0.4875 | 0.0513 | 0.1098 | 0.5229 | 0.4131 |
| bearing | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.7041 | 0.9933 | 0.6474 | 0.9976 | 0.5133 | 0.0592 | 0.4867 | 0.0511 | 0.1092 | 0.5234 | 0.4143 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6739 | 0.9919 | 0.4918 | 0.9962 | 0.3607 | 0.0514 | 0.6393 | 0.1068 | 0.1096 | 0.3933 | 0.2837 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.7241 | 0.9937 | 0.6648 | 0.9980 | 0.5415 | 0.0506 | 0.4585 | 0.0560 | 0.1037 | 0.5517 | 0.4480 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.7233 | 0.9936 | 0.6687 | 0.9979 | 0.5453 | 0.0526 | 0.4547 | 0.0528 | 0.1043 | 0.5540 | 0.4498 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.7077 | 0.9933 | 0.6454 | 0.9977 | 0.5106 | 0.0572 | 0.4894 | 0.0591 | 0.1085 | 0.5212 | 0.4127 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.7047 | 0.9933 | 0.6445 | 0.9976 | 0.5084 | 0.0579 | 0.4916 | 0.0570 | 0.1088 | 0.5185 | 0.4097 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.7035 | 0.9933 | 0.6453 | 0.9976 | 0.5094 | 0.0583 | 0.4906 | 0.0554 | 0.1092 | 0.5190 | 0.4099 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.7037 | 0.9933 | 0.6458 | 0.9976 | 0.5104 | 0.0605 | 0.4896 | 0.0534 | 0.1095 | 0.5198 | 0.4103 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.7035 | 0.9933 | 0.6460 | 0.9977 | 0.5111 | 0.0583 | 0.4889 | 0.0520 | 0.1091 | 0.5213 | 0.4122 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.7035 | 0.9933 | 0.6471 | 0.9976 | 0.5125 | 0.0598 | 0.4875 | 0.0513 | 0.1098 | 0.5229 | 0.4131 |
| bearing | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.7041 | 0.9933 | 0.6474 | 0.9976 | 0.5133 | 0.0592 | 0.4867 | 0.0511 | 0.1092 | 0.5234 | 0.4143 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6387 | 0.9907 | 0.2737 | 0.9971 | 0.1590 | 0.0343 | 0.8410 | 0.1573 | 0.1418 | 0.2933 | 0.1514 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6204 | 0.9903 | 0.4088 | 0.9983 | 0.2570 | 0.0342 | 0.7430 | 0.0440 | 0.1408 | 0.3332 | 0.1924 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6172 | 0.9902 | 0.4208 | 0.9981 | 0.2666 | 0.0390 | 0.7334 | 0.0288 | 0.1423 | 0.3342 | 0.1919 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6229 | 0.9904 | 0.4104 | 0.9982 | 0.2583 | 0.0338 | 0.7417 | 0.0257 | 0.1396 | 0.3274 | 0.1878 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6236 | 0.9905 | 0.4128 | 0.9983 | 0.2602 | 0.0330 | 0.7398 | 0.0241 | 0.1381 | 0.3280 | 0.1899 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6234 | 0.9905 | 0.4147 | 0.9982 | 0.2617 | 0.0338 | 0.7383 | 0.0238 | 0.1384 | 0.3288 | 0.1904 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6243 | 0.9906 | 0.4144 | 0.9983 | 0.2614 | 0.0320 | 0.7386 | 0.0248 | 0.1375 | 0.3287 | 0.1911 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6243 | 0.9906 | 0.4158 | 0.9982 | 0.2626 | 0.0351 | 0.7374 | 0.0237 | 0.1373 | 0.3291 | 0.1918 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6252 | 0.9906 | 0.4157 | 0.9983 | 0.2625 | 0.0337 | 0.7375 | 0.0244 | 0.1368 | 0.3293 | 0.1925 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6251 | 0.9906 | 0.4157 | 0.9983 | 0.2625 | 0.0330 | 0.7375 | 0.0246 | 0.1368 | 0.3291 | 0.1922 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6387 | 0.9907 | 0.3761 | 0.9961 | 0.2319 | 0.0687 | 0.7681 | 0.1573 | 0.1418 | 0.2933 | 0.1514 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6204 | 0.9903 | 0.4312 | 0.9968 | 0.2751 | 0.0708 | 0.7249 | 0.0440 | 0.1408 | 0.3332 | 0.1924 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6172 | 0.9902 | 0.4355 | 0.9967 | 0.2786 | 0.0744 | 0.7214 | 0.0288 | 0.1423 | 0.3342 | 0.1919 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6229 | 0.9904 | 0.4230 | 0.9966 | 0.2685 | 0.0694 | 0.7315 | 0.0257 | 0.1396 | 0.3274 | 0.1878 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6236 | 0.9905 | 0.4240 | 0.9967 | 0.2693 | 0.0662 | 0.7307 | 0.0241 | 0.1381 | 0.3280 | 0.1899 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6234 | 0.9905 | 0.4264 | 0.9966 | 0.2712 | 0.0690 | 0.7288 | 0.0238 | 0.1384 | 0.3288 | 0.1904 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6243 | 0.9906 | 0.4268 | 0.9968 | 0.2716 | 0.0660 | 0.7284 | 0.0248 | 0.1375 | 0.3287 | 0.1911 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6243 | 0.9906 | 0.4276 | 0.9968 | 0.2722 | 0.0657 | 0.7278 | 0.0237 | 0.1373 | 0.3291 | 0.1918 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6252 | 0.9906 | 0.4278 | 0.9969 | 0.2724 | 0.0634 | 0.7276 | 0.0244 | 0.1368 | 0.3293 | 0.1925 |
| bearing | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6251 | 0.9906 | 0.4277 | 0.9969 | 0.2723 | 0.0634 | 0.7277 | 0.0246 | 0.1368 | 0.3291 | 0.1922 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.6387 | 0.9907 | 0.3761 | 0.9961 | 0.2319 | 0.0687 | 0.7681 | 0.1573 | 0.1418 | 0.2933 | 0.1514 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.6204 | 0.9903 | 0.4312 | 0.9968 | 0.2751 | 0.0708 | 0.7249 | 0.0440 | 0.1408 | 0.3332 | 0.1924 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.6172 | 0.9902 | 0.4355 | 0.9967 | 0.2786 | 0.0744 | 0.7214 | 0.0288 | 0.1423 | 0.3342 | 0.1919 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.6229 | 0.9904 | 0.4230 | 0.9966 | 0.2685 | 0.0694 | 0.7315 | 0.0257 | 0.1396 | 0.3274 | 0.1878 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.6236 | 0.9905 | 0.4240 | 0.9967 | 0.2693 | 0.0662 | 0.7307 | 0.0241 | 0.1381 | 0.3280 | 0.1899 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.6234 | 0.9905 | 0.4264 | 0.9966 | 0.2712 | 0.0690 | 0.7288 | 0.0238 | 0.1384 | 0.3288 | 0.1904 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.6243 | 0.9906 | 0.4268 | 0.9968 | 0.2716 | 0.0660 | 0.7284 | 0.0248 | 0.1375 | 0.3287 | 0.1911 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.6243 | 0.9906 | 0.4276 | 0.9968 | 0.2722 | 0.0657 | 0.7278 | 0.0237 | 0.1373 | 0.3291 | 0.1918 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.6252 | 0.9906 | 0.4278 | 0.9969 | 0.2724 | 0.0634 | 0.7276 | 0.0244 | 0.1368 | 0.3293 | 0.1925 |
| bearing | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.6251 | 0.9906 | 0.4277 | 0.9969 | 0.2723 | 0.0634 | 0.7277 | 0.0246 | 0.1368 | 0.3291 | 0.1922 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6239 | 0.9745 | 0.3942 | 0.9957 | 0.2460 | 0.0261 | 0.7540 | 0.0483 | 0.1347 | 0.3288 | 0.1940 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6153 | 0.9743 | 0.4301 | 0.9965 | 0.2747 | 0.0255 | 0.7253 | 0.0344 | 0.1325 | 0.3436 | 0.2110 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6262 | 0.9747 | 0.4179 | 0.9965 | 0.2649 | 0.0192 | 0.7351 | 0.0311 | 0.1179 | 0.3306 | 0.2127 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6304 | 0.9753 | 0.4208 | 0.9966 | 0.2674 | 0.0159 | 0.7326 | 0.0285 | 0.1090 | 0.3296 | 0.2206 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6293 | 0.9753 | 0.4195 | 0.9968 | 0.2663 | 0.0144 | 0.7337 | 0.0288 | 0.1084 | 0.3283 | 0.2198 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6284 | 0.9753 | 0.4195 | 0.9967 | 0.2663 | 0.0159 | 0.7337 | 0.0277 | 0.1091 | 0.3276 | 0.2184 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6282 | 0.9753 | 0.4200 | 0.9967 | 0.2667 | 0.0155 | 0.7333 | 0.0265 | 0.1096 | 0.3275 | 0.2179 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6278 | 0.9753 | 0.4209 | 0.9967 | 0.2674 | 0.0163 | 0.7326 | 0.0260 | 0.1102 | 0.3277 | 0.2175 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6279 | 0.9753 | 0.4228 | 0.9967 | 0.2689 | 0.0171 | 0.7311 | 0.0254 | 0.1115 | 0.3288 | 0.2173 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6280 | 0.9754 | 0.4246 | 0.9967 | 0.2703 | 0.0172 | 0.7297 | 0.0256 | 0.1117 | 0.3298 | 0.2182 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6239 | 0.9745 | 0.4174 | 0.9918 | 0.2646 | 0.0622 | 0.7354 | 0.0483 | 0.1347 | 0.3288 | 0.1940 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6153 | 0.9743 | 0.4475 | 0.9933 | 0.2892 | 0.0557 | 0.7108 | 0.0344 | 0.1325 | 0.3436 | 0.2110 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6262 | 0.9747 | 0.4335 | 0.9935 | 0.2777 | 0.0405 | 0.7223 | 0.0311 | 0.1179 | 0.3306 | 0.2127 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6304 | 0.9753 | 0.4351 | 0.9940 | 0.2790 | 0.0322 | 0.7210 | 0.0285 | 0.1090 | 0.3296 | 0.2206 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6293 | 0.9753 | 0.4334 | 0.9937 | 0.2778 | 0.0331 | 0.7222 | 0.0288 | 0.1084 | 0.3283 | 0.2198 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6284 | 0.9753 | 0.4330 | 0.9938 | 0.2774 | 0.0333 | 0.7226 | 0.0277 | 0.1091 | 0.3276 | 0.2184 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6282 | 0.9753 | 0.4330 | 0.9937 | 0.2774 | 0.0340 | 0.7226 | 0.0265 | 0.1096 | 0.3275 | 0.2179 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6278 | 0.9753 | 0.4336 | 0.9936 | 0.2779 | 0.0354 | 0.7221 | 0.0260 | 0.1102 | 0.3277 | 0.2175 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6279 | 0.9753 | 0.4354 | 0.9934 | 0.2793 | 0.0373 | 0.7207 | 0.0254 | 0.1115 | 0.3288 | 0.2173 |
| condition | fedavg | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6280 | 0.9754 | 0.4373 | 0.9933 | 0.2809 | 0.0385 | 0.7191 | 0.0256 | 0.1117 | 0.3298 | 0.2182 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6239 | 0.9745 | 0.4174 | 0.9918 | 0.2646 | 0.0622 | 0.7354 | 0.0483 | 0.1347 | 0.3288 | 0.1940 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6153 | 0.9743 | 0.4475 | 0.9933 | 0.2892 | 0.0557 | 0.7108 | 0.0344 | 0.1325 | 0.3436 | 0.2110 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6262 | 0.9747 | 0.4335 | 0.9935 | 0.2777 | 0.0405 | 0.7223 | 0.0311 | 0.1179 | 0.3306 | 0.2127 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 4 | 0.6304 | 0.9753 | 0.4351 | 0.9940 | 0.2790 | 0.0322 | 0.7210 | 0.0285 | 0.1090 | 0.3296 | 0.2206 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 5 | 0.6293 | 0.9753 | 0.4334 | 0.9937 | 0.2778 | 0.0331 | 0.7222 | 0.0288 | 0.1084 | 0.3283 | 0.2198 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 6 | 0.6284 | 0.9753 | 0.4330 | 0.9938 | 0.2774 | 0.0333 | 0.7226 | 0.0277 | 0.1091 | 0.3276 | 0.2184 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 7 | 0.6282 | 0.9753 | 0.4330 | 0.9937 | 0.2774 | 0.0340 | 0.7226 | 0.0265 | 0.1096 | 0.3275 | 0.2179 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 8 | 0.6278 | 0.9753 | 0.4336 | 0.9936 | 0.2779 | 0.0354 | 0.7221 | 0.0260 | 0.1102 | 0.3277 | 0.2175 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 9 | 0.6279 | 0.9753 | 0.4354 | 0.9934 | 0.2793 | 0.0373 | 0.7207 | 0.0254 | 0.1115 | 0.3288 | 0.2173 |
| condition | fedavg | adaptive | cnn-ae | hard_val_p95 | 10 | 0.6280 | 0.9754 | 0.4373 | 0.9933 | 0.2809 | 0.0385 | 0.7191 | 0.0256 | 0.1117 | 0.3298 | 0.2182 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6232 | 0.9744 | 0.4362 | 0.9956 | 0.2852 | 0.0253 | 0.7148 | 0.0474 | 0.1331 | 0.3614 | 0.2283 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6187 | 0.9743 | 0.4756 | 0.9966 | 0.3190 | 0.0241 | 0.6810 | 0.0415 | 0.1372 | 0.3833 | 0.2462 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6211 | 0.9746 | 0.4677 | 0.9963 | 0.3119 | 0.0258 | 0.6881 | 0.0390 | 0.1302 | 0.3715 | 0.2413 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6246 | 0.9755 | 0.4756 | 0.9961 | 0.3198 | 0.0253 | 0.6802 | 0.0389 | 0.1265 | 0.3752 | 0.2487 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6233 | 0.9756 | 0.4756 | 0.9961 | 0.3201 | 0.0257 | 0.6799 | 0.0386 | 0.1262 | 0.3749 | 0.2487 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6218 | 0.9755 | 0.4748 | 0.9960 | 0.3193 | 0.0265 | 0.6807 | 0.0390 | 0.1269 | 0.3738 | 0.2470 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6216 | 0.9755 | 0.4745 | 0.9960 | 0.3189 | 0.0266 | 0.6811 | 0.0380 | 0.1259 | 0.3732 | 0.2473 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6209 | 0.9755 | 0.4753 | 0.9959 | 0.3194 | 0.0274 | 0.6806 | 0.0376 | 0.1267 | 0.3732 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6207 | 0.9754 | 0.4761 | 0.9959 | 0.3199 | 0.0287 | 0.6801 | 0.0374 | 0.1272 | 0.3730 | 0.2459 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6206 | 0.9755 | 0.4770 | 0.9960 | 0.3207 | 0.0269 | 0.6793 | 0.0378 | 0.1274 | 0.3740 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6232 | 0.9744 | 0.4603 | 0.9920 | 0.3066 | 0.0498 | 0.6934 | 0.0474 | 0.1331 | 0.3614 | 0.2283 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6187 | 0.9743 | 0.4982 | 0.9930 | 0.3405 | 0.0545 | 0.6595 | 0.0415 | 0.1372 | 0.3833 | 0.2462 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6211 | 0.9746 | 0.4867 | 0.9927 | 0.3300 | 0.0496 | 0.6700 | 0.0390 | 0.1302 | 0.3715 | 0.2413 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6246 | 0.9755 | 0.4940 | 0.9924 | 0.3374 | 0.0509 | 0.6626 | 0.0389 | 0.1265 | 0.3752 | 0.2487 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6233 | 0.9756 | 0.4939 | 0.9925 | 0.3380 | 0.0498 | 0.6620 | 0.0386 | 0.1262 | 0.3749 | 0.2487 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6218 | 0.9755 | 0.4930 | 0.9924 | 0.3371 | 0.0514 | 0.6629 | 0.0390 | 0.1269 | 0.3738 | 0.2470 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6216 | 0.9755 | 0.4921 | 0.9928 | 0.3362 | 0.0487 | 0.6638 | 0.0380 | 0.1259 | 0.3732 | 0.2473 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6209 | 0.9755 | 0.4922 | 0.9928 | 0.3360 | 0.0497 | 0.6640 | 0.0376 | 0.1267 | 0.3732 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6207 | 0.9754 | 0.4926 | 0.9926 | 0.3360 | 0.0500 | 0.6640 | 0.0374 | 0.1272 | 0.3730 | 0.2459 |
| condition | fedavg | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6206 | 0.9755 | 0.4945 | 0.9923 | 0.3379 | 0.0524 | 0.6621 | 0.0378 | 0.1274 | 0.3740 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6232 | 0.9744 | 0.4603 | 0.9920 | 0.3066 | 0.0498 | 0.6934 | 0.0474 | 0.1331 | 0.3614 | 0.2283 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 2 | 0.6187 | 0.9743 | 0.4982 | 0.9930 | 0.3405 | 0.0545 | 0.6595 | 0.0415 | 0.1372 | 0.3833 | 0.2462 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 3 | 0.6211 | 0.9746 | 0.4867 | 0.9927 | 0.3300 | 0.0496 | 0.6700 | 0.0390 | 0.1302 | 0.3715 | 0.2413 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 4 | 0.6246 | 0.9755 | 0.4940 | 0.9924 | 0.3374 | 0.0509 | 0.6626 | 0.0389 | 0.1265 | 0.3752 | 0.2487 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 5 | 0.6233 | 0.9756 | 0.4939 | 0.9925 | 0.3380 | 0.0498 | 0.6620 | 0.0386 | 0.1262 | 0.3749 | 0.2487 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 6 | 0.6218 | 0.9755 | 0.4930 | 0.9924 | 0.3371 | 0.0514 | 0.6629 | 0.0390 | 0.1269 | 0.3738 | 0.2470 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 7 | 0.6216 | 0.9755 | 0.4921 | 0.9928 | 0.3362 | 0.0487 | 0.6638 | 0.0380 | 0.1259 | 0.3732 | 0.2473 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 8 | 0.6209 | 0.9755 | 0.4922 | 0.9928 | 0.3360 | 0.0497 | 0.6640 | 0.0376 | 0.1267 | 0.3732 | 0.2465 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 9 | 0.6207 | 0.9754 | 0.4926 | 0.9926 | 0.3360 | 0.0500 | 0.6640 | 0.0374 | 0.1272 | 0.3730 | 0.2459 |
| condition | fedavg | client_specific | cnn-ae | hard_val_p95 | 10 | 0.6206 | 0.9755 | 0.4945 | 0.9923 | 0.3379 | 0.0524 | 0.6621 | 0.0378 | 0.1274 | 0.3740 | 0.2465 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6243 | 0.9744 | 0.3843 | 0.9945 | 0.2382 | 0.0337 | 0.7618 | 0.0532 | 0.1436 | 0.3233 | 0.1797 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6135 | 0.9742 | 0.4153 | 0.9956 | 0.2624 | 0.0318 | 0.7376 | 0.0408 | 0.1424 | 0.3334 | 0.1910 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6328 | 0.9748 | 0.4001 | 0.9949 | 0.2504 | 0.0262 | 0.7496 | 0.0366 | 0.1270 | 0.3228 | 0.1959 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6406 | 0.9754 | 0.3993 | 0.9947 | 0.2498 | 0.0246 | 0.7502 | 0.0322 | 0.1202 | 0.3216 | 0.2014 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6400 | 0.9754 | 0.3968 | 0.9946 | 0.2478 | 0.0251 | 0.7522 | 0.0317 | 0.1198 | 0.3195 | 0.1996 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6397 | 0.9754 | 0.3967 | 0.9947 | 0.2478 | 0.0245 | 0.7522 | 0.0304 | 0.1202 | 0.3190 | 0.1989 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6391 | 0.9755 | 0.3980 | 0.9947 | 0.2487 | 0.0245 | 0.7513 | 0.0291 | 0.1209 | 0.3193 | 0.1983 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6387 | 0.9755 | 0.3983 | 0.9946 | 0.2490 | 0.0247 | 0.7510 | 0.0288 | 0.1211 | 0.3196 | 0.1985 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6393 | 0.9755 | 0.4023 | 0.9944 | 0.2521 | 0.0263 | 0.7479 | 0.0273 | 0.1227 | 0.3218 | 0.1991 |
| condition | fedavg | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6397 | 0.9756 | 0.4044 | 0.9946 | 0.2538 | 0.0258 | 0.7462 | 0.0277 | 0.1230 | 0.3232 | 0.2002 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6243 | 0.9744 | 0.4089 | 0.9901 | 0.2577 | 0.0793 | 0.7423 | 0.0532 | 0.1436 | 0.3233 | 0.1797 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6135 | 0.9742 | 0.4344 | 0.9912 | 0.2782 | 0.0731 | 0.7218 | 0.0408 | 0.1424 | 0.3334 | 0.1910 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6328 | 0.9748 | 0.4173 | 0.9905 | 0.2644 | 0.0553 | 0.7356 | 0.0366 | 0.1270 | 0.3228 | 0.1959 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6406 | 0.9754 | 0.4143 | 0.9902 | 0.2620 | 0.0496 | 0.7380 | 0.0322 | 0.1202 | 0.3216 | 0.2014 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6400 | 0.9754 | 0.4113 | 0.9902 | 0.2596 | 0.0484 | 0.7404 | 0.0317 | 0.1198 | 0.3195 | 0.1996 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6397 | 0.9754 | 0.4107 | 0.9900 | 0.2591 | 0.0503 | 0.7409 | 0.0304 | 0.1202 | 0.3190 | 0.1989 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6391 | 0.9755 | 0.4112 | 0.9900 | 0.2595 | 0.0507 | 0.7405 | 0.0291 | 0.1209 | 0.3193 | 0.1983 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6387 | 0.9755 | 0.4118 | 0.9898 | 0.2600 | 0.0518 | 0.7400 | 0.0288 | 0.1211 | 0.3196 | 0.1985 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6393 | 0.9755 | 0.4150 | 0.9897 | 0.2625 | 0.0545 | 0.7375 | 0.0273 | 0.1227 | 0.3218 | 0.1991 |
| condition | fedavg | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6397 | 0.9756 | 0.4174 | 0.9896 | 0.2645 | 0.0559 | 0.7355 | 0.0277 | 0.1230 | 0.3232 | 0.2002 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 1 | 0.6243 | 0.9744 | 0.4089 | 0.9901 | 0.2577 | 0.0793 | 0.7423 | 0.0532 | 0.1436 | 0.3233 | 0.1797 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 2 | 0.6135 | 0.9742 | 0.4344 | 0.9912 | 0.2782 | 0.0731 | 0.7218 | 0.0408 | 0.1424 | 0.3334 | 0.1910 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 3 | 0.6328 | 0.9748 | 0.4173 | 0.9905 | 0.2644 | 0.0553 | 0.7356 | 0.0366 | 0.1270 | 0.3228 | 0.1959 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 4 | 0.6406 | 0.9754 | 0.4143 | 0.9902 | 0.2620 | 0.0496 | 0.7380 | 0.0322 | 0.1202 | 0.3216 | 0.2014 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 5 | 0.6400 | 0.9754 | 0.4113 | 0.9902 | 0.2596 | 0.0484 | 0.7404 | 0.0317 | 0.1198 | 0.3195 | 0.1996 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 6 | 0.6397 | 0.9754 | 0.4107 | 0.9900 | 0.2591 | 0.0503 | 0.7409 | 0.0304 | 0.1202 | 0.3190 | 0.1989 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 7 | 0.6391 | 0.9755 | 0.4112 | 0.9900 | 0.2595 | 0.0507 | 0.7405 | 0.0291 | 0.1209 | 0.3193 | 0.1983 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 8 | 0.6387 | 0.9755 | 0.4118 | 0.9898 | 0.2600 | 0.0518 | 0.7400 | 0.0288 | 0.1211 | 0.3196 | 0.1985 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 9 | 0.6393 | 0.9755 | 0.4150 | 0.9897 | 0.2625 | 0.0545 | 0.7375 | 0.0273 | 0.1227 | 0.3218 | 0.1991 |
| condition | fedavg | pooled | cnn-ae | hard_val_p95 | 10 | 0.6397 | 0.9756 | 0.4174 | 0.9896 | 0.2645 | 0.0559 | 0.7355 | 0.0277 | 0.1230 | 0.3232 | 0.2002 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6863 | 0.9790 | 0.4248 | 0.9958 | 0.2929 | 0.0745 | 0.7071 | 0.1709 | 0.1797 | 0.4131 | 0.2334 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6976 | 0.9811 | 0.5079 | 0.9974 | 0.3417 | 0.0554 | 0.6583 | 0.0608 | 0.1666 | 0.4305 | 0.2639 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6936 | 0.9803 | 0.4973 | 0.9969 | 0.3334 | 0.0355 | 0.6666 | 0.0417 | 0.1385 | 0.4122 | 0.2738 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6851 | 0.9794 | 0.4880 | 0.9965 | 0.3259 | 0.0253 | 0.6741 | 0.0336 | 0.1200 | 0.3973 | 0.2773 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6800 | 0.9791 | 0.4812 | 0.9962 | 0.3199 | 0.0245 | 0.6801 | 0.0330 | 0.1186 | 0.3904 | 0.2718 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6790 | 0.9791 | 0.4778 | 0.9966 | 0.3170 | 0.0217 | 0.6830 | 0.0331 | 0.1173 | 0.3876 | 0.2703 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6802 | 0.9793 | 0.4781 | 0.9967 | 0.3173 | 0.0207 | 0.6827 | 0.0322 | 0.1176 | 0.3880 | 0.2704 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6822 | 0.9794 | 0.4806 | 0.9964 | 0.3198 | 0.0209 | 0.6802 | 0.0321 | 0.1177 | 0.3907 | 0.2729 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6847 | 0.9799 | 0.4818 | 0.9965 | 0.3209 | 0.0211 | 0.6791 | 0.0334 | 0.1170 | 0.3927 | 0.2757 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6890 | 0.9803 | 0.4844 | 0.9966 | 0.3232 | 0.0204 | 0.6768 | 0.0343 | 0.1141 | 0.3961 | 0.2819 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6863 | 0.9790 | 0.4974 | 0.9937 | 0.3567 | 0.1302 | 0.6433 | 0.1709 | 0.1797 | 0.4131 | 0.2334 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6976 | 0.9811 | 0.5367 | 0.9939 | 0.3683 | 0.1238 | 0.6317 | 0.0608 | 0.1666 | 0.4305 | 0.2639 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6936 | 0.9803 | 0.5164 | 0.9933 | 0.3509 | 0.0796 | 0.6491 | 0.0417 | 0.1385 | 0.4122 | 0.2738 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6851 | 0.9794 | 0.5025 | 0.9935 | 0.3391 | 0.0465 | 0.6609 | 0.0336 | 0.1200 | 0.3973 | 0.2773 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6800 | 0.9791 | 0.4944 | 0.9934 | 0.3320 | 0.0441 | 0.6680 | 0.0330 | 0.1186 | 0.3904 | 0.2718 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6790 | 0.9791 | 0.4914 | 0.9936 | 0.3293 | 0.0427 | 0.6707 | 0.0331 | 0.1173 | 0.3876 | 0.2703 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6802 | 0.9793 | 0.4913 | 0.9936 | 0.3293 | 0.0441 | 0.6707 | 0.0322 | 0.1176 | 0.3880 | 0.2704 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6822 | 0.9794 | 0.4943 | 0.9935 | 0.3322 | 0.0440 | 0.6678 | 0.0321 | 0.1177 | 0.3907 | 0.2729 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6847 | 0.9799 | 0.4961 | 0.9935 | 0.3337 | 0.0443 | 0.6663 | 0.0334 | 0.1170 | 0.3927 | 0.2757 |
| condition | fedbn | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6890 | 0.9803 | 0.4994 | 0.9943 | 0.3368 | 0.0371 | 0.6632 | 0.0343 | 0.1141 | 0.3961 | 0.2819 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6863 | 0.9790 | 0.4974 | 0.9937 | 0.3567 | 0.1302 | 0.6433 | 0.1709 | 0.1797 | 0.4131 | 0.2334 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6976 | 0.9811 | 0.5367 | 0.9939 | 0.3683 | 0.1238 | 0.6317 | 0.0608 | 0.1666 | 0.4305 | 0.2639 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6936 | 0.9803 | 0.5164 | 0.9933 | 0.3509 | 0.0796 | 0.6491 | 0.0417 | 0.1385 | 0.4122 | 0.2738 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 4 | 0.6851 | 0.9794 | 0.5025 | 0.9935 | 0.3391 | 0.0465 | 0.6609 | 0.0336 | 0.1200 | 0.3973 | 0.2773 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 5 | 0.6800 | 0.9791 | 0.4944 | 0.9934 | 0.3320 | 0.0441 | 0.6680 | 0.0330 | 0.1186 | 0.3904 | 0.2718 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 6 | 0.6790 | 0.9791 | 0.4914 | 0.9936 | 0.3293 | 0.0427 | 0.6707 | 0.0331 | 0.1173 | 0.3876 | 0.2703 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 7 | 0.6802 | 0.9793 | 0.4913 | 0.9936 | 0.3293 | 0.0441 | 0.6707 | 0.0322 | 0.1176 | 0.3880 | 0.2704 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 8 | 0.6822 | 0.9794 | 0.4943 | 0.9935 | 0.3322 | 0.0440 | 0.6678 | 0.0321 | 0.1177 | 0.3907 | 0.2729 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 9 | 0.6847 | 0.9799 | 0.4961 | 0.9935 | 0.3337 | 0.0443 | 0.6663 | 0.0334 | 0.1170 | 0.3927 | 0.2757 |
| condition | fedbn | adaptive | cnn-ae | hard_val_p95 | 10 | 0.6890 | 0.9803 | 0.4994 | 0.9943 | 0.3368 | 0.0371 | 0.6632 | 0.0343 | 0.1141 | 0.3961 | 0.2819 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6854 | 0.9788 | 0.5344 | 0.9938 | 0.3860 | 0.0260 | 0.6140 | 0.0560 | 0.1342 | 0.4558 | 0.3216 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6930 | 0.9808 | 0.5532 | 0.9969 | 0.3896 | 0.0248 | 0.6104 | 0.0498 | 0.1350 | 0.4656 | 0.3306 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6838 | 0.9800 | 0.5384 | 0.9966 | 0.3796 | 0.0279 | 0.6204 | 0.0388 | 0.1319 | 0.4467 | 0.3149 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6765 | 0.9793 | 0.5317 | 0.9965 | 0.3747 | 0.0265 | 0.6253 | 0.0354 | 0.1289 | 0.4356 | 0.3067 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6710 | 0.9790 | 0.5275 | 0.9962 | 0.3712 | 0.0279 | 0.6288 | 0.0345 | 0.1289 | 0.4309 | 0.3020 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6700 | 0.9790 | 0.5252 | 0.9965 | 0.3693 | 0.0276 | 0.6307 | 0.0350 | 0.1287 | 0.4291 | 0.3004 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6710 | 0.9791 | 0.5249 | 0.9965 | 0.3690 | 0.0268 | 0.6310 | 0.0345 | 0.1296 | 0.4295 | 0.2999 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6729 | 0.9793 | 0.5259 | 0.9963 | 0.3701 | 0.0287 | 0.6299 | 0.0339 | 0.1305 | 0.4312 | 0.3007 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6754 | 0.9796 | 0.5262 | 0.9961 | 0.3704 | 0.0301 | 0.6296 | 0.0354 | 0.1292 | 0.4326 | 0.3034 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6791 | 0.9800 | 0.5290 | 0.9964 | 0.3731 | 0.0283 | 0.6269 | 0.0357 | 0.1286 | 0.4363 | 0.3077 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6854 | 0.9788 | 0.5623 | 0.9919 | 0.4142 | 0.0538 | 0.5858 | 0.0560 | 0.1342 | 0.4558 | 0.3216 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6930 | 0.9808 | 0.5770 | 0.9938 | 0.4141 | 0.0579 | 0.5859 | 0.0498 | 0.1350 | 0.4656 | 0.3306 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6838 | 0.9800 | 0.5572 | 0.9934 | 0.3987 | 0.0601 | 0.6013 | 0.0388 | 0.1319 | 0.4467 | 0.3149 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6765 | 0.9793 | 0.5472 | 0.9929 | 0.3908 | 0.0545 | 0.6092 | 0.0354 | 0.1289 | 0.4356 | 0.3067 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6710 | 0.9790 | 0.5414 | 0.9931 | 0.3858 | 0.0526 | 0.6142 | 0.0345 | 0.1289 | 0.4309 | 0.3020 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6700 | 0.9790 | 0.5397 | 0.9933 | 0.3844 | 0.0518 | 0.6156 | 0.0350 | 0.1287 | 0.4291 | 0.3004 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6710 | 0.9791 | 0.5396 | 0.9930 | 0.3844 | 0.0551 | 0.6156 | 0.0345 | 0.1296 | 0.4295 | 0.2999 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6729 | 0.9793 | 0.5408 | 0.9929 | 0.3858 | 0.0553 | 0.6142 | 0.0339 | 0.1305 | 0.4312 | 0.3007 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6754 | 0.9796 | 0.5411 | 0.9933 | 0.3859 | 0.0527 | 0.6141 | 0.0354 | 0.1292 | 0.4326 | 0.3034 |
| condition | fedbn | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6791 | 0.9800 | 0.5446 | 0.9934 | 0.3895 | 0.0529 | 0.6105 | 0.0357 | 0.1286 | 0.4363 | 0.3077 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6854 | 0.9788 | 0.5623 | 0.9919 | 0.4142 | 0.0538 | 0.5858 | 0.0560 | 0.1342 | 0.4558 | 0.3216 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 2 | 0.6930 | 0.9808 | 0.5770 | 0.9938 | 0.4141 | 0.0579 | 0.5859 | 0.0498 | 0.1350 | 0.4656 | 0.3306 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 3 | 0.6838 | 0.9800 | 0.5572 | 0.9934 | 0.3987 | 0.0601 | 0.6013 | 0.0388 | 0.1319 | 0.4467 | 0.3149 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 4 | 0.6765 | 0.9793 | 0.5472 | 0.9929 | 0.3908 | 0.0545 | 0.6092 | 0.0354 | 0.1289 | 0.4356 | 0.3067 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 5 | 0.6710 | 0.9790 | 0.5414 | 0.9931 | 0.3858 | 0.0526 | 0.6142 | 0.0345 | 0.1289 | 0.4309 | 0.3020 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 6 | 0.6700 | 0.9790 | 0.5397 | 0.9933 | 0.3844 | 0.0518 | 0.6156 | 0.0350 | 0.1287 | 0.4291 | 0.3004 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 7 | 0.6710 | 0.9791 | 0.5396 | 0.9930 | 0.3844 | 0.0551 | 0.6156 | 0.0345 | 0.1296 | 0.4295 | 0.2999 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 8 | 0.6729 | 0.9793 | 0.5408 | 0.9929 | 0.3858 | 0.0553 | 0.6142 | 0.0339 | 0.1305 | 0.4312 | 0.3007 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 9 | 0.6754 | 0.9796 | 0.5411 | 0.9933 | 0.3859 | 0.0527 | 0.6141 | 0.0354 | 0.1292 | 0.4326 | 0.3034 |
| condition | fedbn | client_specific | cnn-ae | hard_val_p95 | 10 | 0.6791 | 0.9800 | 0.5446 | 0.9934 | 0.3895 | 0.0529 | 0.6105 | 0.0357 | 0.1286 | 0.4363 | 0.3077 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6783 | 0.9783 | 0.3747 | 0.9953 | 0.2689 | 0.1594 | 0.7311 | 0.2383 | 0.2375 | 0.4258 | 0.1884 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6987 | 0.9810 | 0.4834 | 0.9955 | 0.3213 | 0.1276 | 0.6787 | 0.0763 | 0.2131 | 0.4234 | 0.2103 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.7043 | 0.9808 | 0.4682 | 0.9951 | 0.3068 | 0.0689 | 0.6932 | 0.0472 | 0.1641 | 0.4011 | 0.2370 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.7001 | 0.9802 | 0.4592 | 0.9947 | 0.2989 | 0.0320 | 0.7011 | 0.0366 | 0.1313 | 0.3861 | 0.2548 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6967 | 0.9800 | 0.4533 | 0.9945 | 0.2940 | 0.0318 | 0.7060 | 0.0355 | 0.1290 | 0.3800 | 0.2510 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6959 | 0.9801 | 0.4515 | 0.9944 | 0.2924 | 0.0305 | 0.7076 | 0.0347 | 0.1274 | 0.3781 | 0.2507 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6965 | 0.9801 | 0.4520 | 0.9947 | 0.2929 | 0.0294 | 0.7071 | 0.0340 | 0.1272 | 0.3786 | 0.2514 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6977 | 0.9803 | 0.4553 | 0.9946 | 0.2957 | 0.0297 | 0.7043 | 0.0346 | 0.1268 | 0.3809 | 0.2541 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.7005 | 0.9807 | 0.4595 | 0.9947 | 0.2994 | 0.0289 | 0.7006 | 0.0361 | 0.1267 | 0.3852 | 0.2585 |
| condition | fedbn | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.7039 | 0.9812 | 0.4635 | 0.9951 | 0.3029 | 0.0268 | 0.6971 | 0.0386 | 0.1235 | 0.3892 | 0.2656 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6783 | 0.9783 | 0.5302 | 0.9961 | 0.4191 | 0.1911 | 0.5809 | 0.2383 | 0.2375 | 0.4258 | 0.1884 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6987 | 0.9810 | 0.5183 | 0.9929 | 0.3524 | 0.1787 | 0.6476 | 0.0763 | 0.2131 | 0.4234 | 0.2103 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.7043 | 0.9808 | 0.4877 | 0.9911 | 0.3238 | 0.1123 | 0.6762 | 0.0472 | 0.1641 | 0.4011 | 0.2370 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.7001 | 0.9802 | 0.4730 | 0.9907 | 0.3110 | 0.0623 | 0.6890 | 0.0366 | 0.1313 | 0.3861 | 0.2548 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6967 | 0.9800 | 0.4657 | 0.9904 | 0.3048 | 0.0592 | 0.6952 | 0.0355 | 0.1290 | 0.3800 | 0.2510 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6959 | 0.9801 | 0.4632 | 0.9907 | 0.3027 | 0.0551 | 0.6973 | 0.0347 | 0.1274 | 0.3781 | 0.2507 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6965 | 0.9801 | 0.4640 | 0.9904 | 0.3033 | 0.0585 | 0.6967 | 0.0340 | 0.1272 | 0.3786 | 0.2514 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6977 | 0.9803 | 0.4678 | 0.9907 | 0.3066 | 0.0576 | 0.6934 | 0.0346 | 0.1268 | 0.3809 | 0.2541 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.7005 | 0.9807 | 0.4731 | 0.9905 | 0.3114 | 0.0589 | 0.6886 | 0.0361 | 0.1267 | 0.3852 | 0.2585 |
| condition | fedbn | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.7039 | 0.9812 | 0.4781 | 0.9912 | 0.3157 | 0.0534 | 0.6843 | 0.0386 | 0.1235 | 0.3892 | 0.2656 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 1 | 0.6783 | 0.9783 | 0.5302 | 0.9961 | 0.4191 | 0.1911 | 0.5809 | 0.2383 | 0.2375 | 0.4258 | 0.1884 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 2 | 0.6987 | 0.9810 | 0.5183 | 0.9929 | 0.3524 | 0.1787 | 0.6476 | 0.0763 | 0.2131 | 0.4234 | 0.2103 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 3 | 0.7043 | 0.9808 | 0.4877 | 0.9911 | 0.3238 | 0.1123 | 0.6762 | 0.0472 | 0.1641 | 0.4011 | 0.2370 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 4 | 0.7001 | 0.9802 | 0.4730 | 0.9907 | 0.3110 | 0.0623 | 0.6890 | 0.0366 | 0.1313 | 0.3861 | 0.2548 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 5 | 0.6967 | 0.9800 | 0.4657 | 0.9904 | 0.3048 | 0.0592 | 0.6952 | 0.0355 | 0.1290 | 0.3800 | 0.2510 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 6 | 0.6959 | 0.9801 | 0.4632 | 0.9907 | 0.3027 | 0.0551 | 0.6973 | 0.0347 | 0.1274 | 0.3781 | 0.2507 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 7 | 0.6965 | 0.9801 | 0.4640 | 0.9904 | 0.3033 | 0.0585 | 0.6967 | 0.0340 | 0.1272 | 0.3786 | 0.2514 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 8 | 0.6977 | 0.9803 | 0.4678 | 0.9907 | 0.3066 | 0.0576 | 0.6934 | 0.0346 | 0.1268 | 0.3809 | 0.2541 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 9 | 0.7005 | 0.9807 | 0.4731 | 0.9905 | 0.3114 | 0.0589 | 0.6886 | 0.0361 | 0.1267 | 0.3852 | 0.2585 |
| condition | fedbn | pooled | cnn-ae | hard_val_p95 | 10 | 0.7039 | 0.9812 | 0.4781 | 0.9912 | 0.3157 | 0.0534 | 0.6843 | 0.0386 | 0.1235 | 0.3892 | 0.2656 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6298 | 0.9751 | 0.3854 | 0.9959 | 0.2393 | 0.0208 | 0.7607 | 0.0552 | 0.1264 | 0.3265 | 0.2001 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6163 | 0.9743 | 0.4300 | 0.9966 | 0.2745 | 0.0239 | 0.7255 | 0.0332 | 0.1303 | 0.3426 | 0.2123 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6249 | 0.9747 | 0.4186 | 0.9968 | 0.2654 | 0.0177 | 0.7346 | 0.0290 | 0.1151 | 0.3301 | 0.2150 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6309 | 0.9753 | 0.4215 | 0.9969 | 0.2679 | 0.0147 | 0.7321 | 0.0274 | 0.1085 | 0.3297 | 0.2212 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6311 | 0.9755 | 0.4224 | 0.9967 | 0.2686 | 0.0154 | 0.7314 | 0.0279 | 0.1082 | 0.3301 | 0.2219 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6297 | 0.9754 | 0.4221 | 0.9965 | 0.2684 | 0.0174 | 0.7316 | 0.0269 | 0.1096 | 0.3291 | 0.2195 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6289 | 0.9754 | 0.4216 | 0.9968 | 0.2680 | 0.0155 | 0.7320 | 0.0265 | 0.1096 | 0.3282 | 0.2187 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6282 | 0.9753 | 0.4219 | 0.9968 | 0.2682 | 0.0157 | 0.7318 | 0.0260 | 0.1095 | 0.3279 | 0.2184 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6277 | 0.9753 | 0.4235 | 0.9969 | 0.2695 | 0.0152 | 0.7305 | 0.0260 | 0.1105 | 0.3287 | 0.2183 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6283 | 0.9753 | 0.4248 | 0.9968 | 0.2705 | 0.0160 | 0.7295 | 0.0257 | 0.1109 | 0.3295 | 0.2186 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6298 | 0.9751 | 0.4140 | 0.9919 | 0.2619 | 0.0518 | 0.7381 | 0.0552 | 0.1264 | 0.3265 | 0.2001 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6163 | 0.9743 | 0.4469 | 0.9934 | 0.2886 | 0.0509 | 0.7114 | 0.0332 | 0.1303 | 0.3426 | 0.2123 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6249 | 0.9747 | 0.4331 | 0.9935 | 0.2773 | 0.0378 | 0.7227 | 0.0290 | 0.1151 | 0.3301 | 0.2150 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6309 | 0.9753 | 0.4350 | 0.9938 | 0.2789 | 0.0339 | 0.7211 | 0.0274 | 0.1085 | 0.3297 | 0.2212 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6311 | 0.9755 | 0.4364 | 0.9937 | 0.2801 | 0.0332 | 0.7199 | 0.0279 | 0.1082 | 0.3301 | 0.2219 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6297 | 0.9754 | 0.4351 | 0.9935 | 0.2791 | 0.0342 | 0.7209 | 0.0269 | 0.1096 | 0.3291 | 0.2195 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6289 | 0.9754 | 0.4347 | 0.9936 | 0.2788 | 0.0342 | 0.7212 | 0.0265 | 0.1096 | 0.3282 | 0.2187 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6282 | 0.9753 | 0.4342 | 0.9938 | 0.2784 | 0.0339 | 0.7216 | 0.0260 | 0.1095 | 0.3279 | 0.2184 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6277 | 0.9753 | 0.4357 | 0.9936 | 0.2796 | 0.0353 | 0.7204 | 0.0260 | 0.1105 | 0.3287 | 0.2183 |
| condition | fedprox | adaptive | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6283 | 0.9753 | 0.4374 | 0.9937 | 0.2810 | 0.0365 | 0.7190 | 0.0257 | 0.1109 | 0.3295 | 0.2186 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 1 | 0.6298 | 0.9751 | 0.4140 | 0.9919 | 0.2619 | 0.0518 | 0.7381 | 0.0552 | 0.1264 | 0.3265 | 0.2001 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 2 | 0.6163 | 0.9743 | 0.4469 | 0.9934 | 0.2886 | 0.0509 | 0.7114 | 0.0332 | 0.1303 | 0.3426 | 0.2123 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 3 | 0.6249 | 0.9747 | 0.4331 | 0.9935 | 0.2773 | 0.0378 | 0.7227 | 0.0290 | 0.1151 | 0.3301 | 0.2150 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 4 | 0.6309 | 0.9753 | 0.4350 | 0.9938 | 0.2789 | 0.0339 | 0.7211 | 0.0274 | 0.1085 | 0.3297 | 0.2212 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 5 | 0.6311 | 0.9755 | 0.4364 | 0.9937 | 0.2801 | 0.0332 | 0.7199 | 0.0279 | 0.1082 | 0.3301 | 0.2219 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 6 | 0.6297 | 0.9754 | 0.4351 | 0.9935 | 0.2791 | 0.0342 | 0.7209 | 0.0269 | 0.1096 | 0.3291 | 0.2195 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 7 | 0.6289 | 0.9754 | 0.4347 | 0.9936 | 0.2788 | 0.0342 | 0.7212 | 0.0265 | 0.1096 | 0.3282 | 0.2187 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 8 | 0.6282 | 0.9753 | 0.4342 | 0.9938 | 0.2784 | 0.0339 | 0.7216 | 0.0260 | 0.1095 | 0.3279 | 0.2184 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 9 | 0.6277 | 0.9753 | 0.4357 | 0.9936 | 0.2796 | 0.0353 | 0.7204 | 0.0260 | 0.1105 | 0.3287 | 0.2183 |
| condition | fedprox | adaptive | cnn-ae | hard_val_p95 | 10 | 0.6283 | 0.9753 | 0.4374 | 0.9937 | 0.2810 | 0.0365 | 0.7190 | 0.0257 | 0.1109 | 0.3295 | 0.2186 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6291 | 0.9751 | 0.4373 | 0.9960 | 0.2879 | 0.0213 | 0.7121 | 0.0524 | 0.1281 | 0.3658 | 0.2377 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6179 | 0.9741 | 0.4758 | 0.9963 | 0.3189 | 0.0260 | 0.6811 | 0.0403 | 0.1357 | 0.3821 | 0.2464 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6213 | 0.9747 | 0.4701 | 0.9964 | 0.3139 | 0.0242 | 0.6861 | 0.0376 | 0.1301 | 0.3726 | 0.2425 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6246 | 0.9755 | 0.4773 | 0.9960 | 0.3212 | 0.0259 | 0.6788 | 0.0389 | 0.1269 | 0.3763 | 0.2495 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6244 | 0.9757 | 0.4780 | 0.9961 | 0.3222 | 0.0251 | 0.6778 | 0.0385 | 0.1259 | 0.3768 | 0.2509 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6226 | 0.9757 | 0.4776 | 0.9959 | 0.3217 | 0.0275 | 0.6783 | 0.0378 | 0.1276 | 0.3758 | 0.2481 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6221 | 0.9756 | 0.4766 | 0.9960 | 0.3209 | 0.0266 | 0.6791 | 0.0375 | 0.1276 | 0.3746 | 0.2469 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6209 | 0.9756 | 0.4760 | 0.9962 | 0.3201 | 0.0267 | 0.6799 | 0.0381 | 0.1268 | 0.3736 | 0.2468 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6205 | 0.9756 | 0.4766 | 0.9961 | 0.3205 | 0.0267 | 0.6795 | 0.0380 | 0.1284 | 0.3739 | 0.2455 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6205 | 0.9755 | 0.4777 | 0.9961 | 0.3212 | 0.0267 | 0.6788 | 0.0370 | 0.1287 | 0.3743 | 0.2457 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6291 | 0.9751 | 0.4638 | 0.9928 | 0.3119 | 0.0440 | 0.6881 | 0.0524 | 0.1281 | 0.3658 | 0.2377 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6179 | 0.9741 | 0.4963 | 0.9937 | 0.3385 | 0.0479 | 0.6615 | 0.0403 | 0.1357 | 0.3821 | 0.2464 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6213 | 0.9747 | 0.4888 | 0.9926 | 0.3319 | 0.0512 | 0.6681 | 0.0376 | 0.1301 | 0.3726 | 0.2425 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6246 | 0.9755 | 0.4957 | 0.9928 | 0.3390 | 0.0503 | 0.6610 | 0.0389 | 0.1269 | 0.3763 | 0.2495 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6244 | 0.9757 | 0.4968 | 0.9928 | 0.3404 | 0.0494 | 0.6596 | 0.0385 | 0.1259 | 0.3768 | 0.2509 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6226 | 0.9757 | 0.4956 | 0.9924 | 0.3393 | 0.0529 | 0.6607 | 0.0378 | 0.1276 | 0.3758 | 0.2481 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6221 | 0.9756 | 0.4947 | 0.9925 | 0.3383 | 0.0534 | 0.6617 | 0.0375 | 0.1276 | 0.3746 | 0.2469 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6209 | 0.9756 | 0.4935 | 0.9926 | 0.3372 | 0.0513 | 0.6628 | 0.0381 | 0.1268 | 0.3736 | 0.2468 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6205 | 0.9756 | 0.4946 | 0.9923 | 0.3381 | 0.0553 | 0.6619 | 0.0380 | 0.1284 | 0.3739 | 0.2455 |
| condition | fedprox | client_specific | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6205 | 0.9755 | 0.4960 | 0.9924 | 0.3392 | 0.0553 | 0.6608 | 0.0370 | 0.1287 | 0.3743 | 0.2457 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 1 | 0.6291 | 0.9751 | 0.4638 | 0.9928 | 0.3119 | 0.0440 | 0.6881 | 0.0524 | 0.1281 | 0.3658 | 0.2377 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 2 | 0.6179 | 0.9741 | 0.4963 | 0.9937 | 0.3385 | 0.0479 | 0.6615 | 0.0403 | 0.1357 | 0.3821 | 0.2464 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 3 | 0.6213 | 0.9747 | 0.4888 | 0.9926 | 0.3319 | 0.0512 | 0.6681 | 0.0376 | 0.1301 | 0.3726 | 0.2425 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 4 | 0.6246 | 0.9755 | 0.4957 | 0.9928 | 0.3390 | 0.0503 | 0.6610 | 0.0389 | 0.1269 | 0.3763 | 0.2495 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 5 | 0.6244 | 0.9757 | 0.4968 | 0.9928 | 0.3404 | 0.0494 | 0.6596 | 0.0385 | 0.1259 | 0.3768 | 0.2509 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 6 | 0.6226 | 0.9757 | 0.4956 | 0.9924 | 0.3393 | 0.0529 | 0.6607 | 0.0378 | 0.1276 | 0.3758 | 0.2481 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 7 | 0.6221 | 0.9756 | 0.4947 | 0.9925 | 0.3383 | 0.0534 | 0.6617 | 0.0375 | 0.1276 | 0.3746 | 0.2469 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 8 | 0.6209 | 0.9756 | 0.4935 | 0.9926 | 0.3372 | 0.0513 | 0.6628 | 0.0381 | 0.1268 | 0.3736 | 0.2468 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 9 | 0.6205 | 0.9756 | 0.4946 | 0.9923 | 0.3381 | 0.0553 | 0.6619 | 0.0380 | 0.1284 | 0.3739 | 0.2455 |
| condition | fedprox | client_specific | cnn-ae | hard_val_p95 | 10 | 0.6205 | 0.9755 | 0.4960 | 0.9924 | 0.3392 | 0.0553 | 0.6608 | 0.0370 | 0.1287 | 0.3743 | 0.2457 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 1 | 0.6298 | 0.9750 | 0.3703 | 0.9947 | 0.2276 | 0.0265 | 0.7724 | 0.0629 | 0.1333 | 0.3182 | 0.1849 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 2 | 0.6150 | 0.9741 | 0.4156 | 0.9955 | 0.2626 | 0.0293 | 0.7374 | 0.0392 | 0.1390 | 0.3331 | 0.1941 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 3 | 0.6315 | 0.9748 | 0.3999 | 0.9950 | 0.2503 | 0.0249 | 0.7497 | 0.0342 | 0.1240 | 0.3219 | 0.1979 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 4 | 0.6410 | 0.9755 | 0.3997 | 0.9946 | 0.2501 | 0.0248 | 0.7499 | 0.0314 | 0.1194 | 0.3219 | 0.2025 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 5 | 0.6416 | 0.9756 | 0.3996 | 0.9945 | 0.2500 | 0.0255 | 0.7500 | 0.0312 | 0.1197 | 0.3216 | 0.2020 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 6 | 0.6409 | 0.9755 | 0.3990 | 0.9943 | 0.2496 | 0.0263 | 0.7504 | 0.0300 | 0.1204 | 0.3207 | 0.2003 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 7 | 0.6402 | 0.9755 | 0.3991 | 0.9943 | 0.2497 | 0.0264 | 0.7503 | 0.0296 | 0.1203 | 0.3202 | 0.1999 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 8 | 0.6398 | 0.9755 | 0.4007 | 0.9943 | 0.2509 | 0.0269 | 0.7491 | 0.0281 | 0.1210 | 0.3206 | 0.1996 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 9 | 0.6397 | 0.9755 | 0.4020 | 0.9944 | 0.2519 | 0.0264 | 0.7481 | 0.0284 | 0.1213 | 0.3211 | 0.1998 |
| condition | fedprox | pooled | cnn-ae | fuzzy_fault_only_hi75 | 10 | 0.6403 | 0.9756 | 0.4035 | 0.9945 | 0.2531 | 0.0264 | 0.7469 | 0.0286 | 0.1212 | 0.3223 | 0.2011 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 1 | 0.6298 | 0.9750 | 0.4018 | 0.9907 | 0.2520 | 0.0611 | 0.7480 | 0.0629 | 0.1333 | 0.3182 | 0.1849 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 2 | 0.6150 | 0.9741 | 0.4347 | 0.9910 | 0.2784 | 0.0703 | 0.7216 | 0.0392 | 0.1390 | 0.3331 | 0.1941 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 3 | 0.6315 | 0.9748 | 0.4157 | 0.9906 | 0.2631 | 0.0514 | 0.7369 | 0.0342 | 0.1240 | 0.3219 | 0.1979 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 4 | 0.6410 | 0.9755 | 0.4156 | 0.9906 | 0.2629 | 0.0468 | 0.7371 | 0.0314 | 0.1194 | 0.3219 | 0.2025 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 5 | 0.6416 | 0.9756 | 0.4154 | 0.9903 | 0.2628 | 0.0480 | 0.7372 | 0.0312 | 0.1197 | 0.3216 | 0.2020 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 6 | 0.6409 | 0.9755 | 0.4132 | 0.9900 | 0.2611 | 0.0502 | 0.7389 | 0.0300 | 0.1204 | 0.3207 | 0.2003 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 7 | 0.6402 | 0.9755 | 0.4126 | 0.9900 | 0.2606 | 0.0502 | 0.7394 | 0.0296 | 0.1203 | 0.3202 | 0.1999 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 8 | 0.6398 | 0.9755 | 0.4129 | 0.9899 | 0.2609 | 0.0508 | 0.7391 | 0.0281 | 0.1210 | 0.3206 | 0.1996 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 9 | 0.6397 | 0.9755 | 0.4137 | 0.9900 | 0.2615 | 0.0507 | 0.7385 | 0.0284 | 0.1213 | 0.3211 | 0.1998 |
| condition | fedprox | pooled | cnn-ae | fuzzy_warning_as_alarm_hi50 | 10 | 0.6403 | 0.9756 | 0.4170 | 0.9903 | 0.2641 | 0.0502 | 0.7359 | 0.0286 | 0.1212 | 0.3223 | 0.2011 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 1 | 0.6298 | 0.9750 | 0.4018 | 0.9907 | 0.2520 | 0.0611 | 0.7480 | 0.0629 | 0.1333 | 0.3182 | 0.1849 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 2 | 0.6150 | 0.9741 | 0.4347 | 0.9910 | 0.2784 | 0.0703 | 0.7216 | 0.0392 | 0.1390 | 0.3331 | 0.1941 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 3 | 0.6315 | 0.9748 | 0.4157 | 0.9906 | 0.2631 | 0.0514 | 0.7369 | 0.0342 | 0.1240 | 0.3219 | 0.1979 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 4 | 0.6410 | 0.9755 | 0.4156 | 0.9906 | 0.2629 | 0.0468 | 0.7371 | 0.0314 | 0.1194 | 0.3219 | 0.2025 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 5 | 0.6416 | 0.9756 | 0.4154 | 0.9903 | 0.2628 | 0.0480 | 0.7372 | 0.0312 | 0.1197 | 0.3216 | 0.2020 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 6 | 0.6409 | 0.9755 | 0.4132 | 0.9900 | 0.2611 | 0.0502 | 0.7389 | 0.0300 | 0.1204 | 0.3207 | 0.2003 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 7 | 0.6402 | 0.9755 | 0.4126 | 0.9900 | 0.2606 | 0.0502 | 0.7394 | 0.0296 | 0.1203 | 0.3202 | 0.1999 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 8 | 0.6398 | 0.9755 | 0.4129 | 0.9899 | 0.2609 | 0.0508 | 0.7391 | 0.0281 | 0.1210 | 0.3206 | 0.1996 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 9 | 0.6397 | 0.9755 | 0.4137 | 0.9900 | 0.2615 | 0.0507 | 0.7385 | 0.0284 | 0.1213 | 0.3211 | 0.1998 |
| condition | fedprox | pooled | cnn-ae | hard_val_p95 | 10 | 0.6403 | 0.9756 | 0.4170 | 0.9903 | 0.2641 | 0.0502 | 0.7359 | 0.0286 | 0.1212 | 0.3223 | 0.2011 |

## Interpretation

- `local-only` shows how each private site performs without collaboration.
- `centralized` is the upper-reference setting that pools normal data and would require data sharing.
- `fedavg` approximates collaborative normal-only training without sharing raw vibration windows.
- `fedprox` adds a proximal penalty to reduce local client drift under non-IID data.
- `fedbn` keeps BatchNorm parameters and running statistics local to each client.
- `*-personalized` locally adapts the federated global model before client evaluation.
- Client stability is evaluated through the standard deviation of false alarm rate, miss rate, uncertain rate, and fuzzy health gap across clients.
- The fuzzy layer is model-agnostic here because it is applied to both VAE and CNN-AE reconstruction scores.
