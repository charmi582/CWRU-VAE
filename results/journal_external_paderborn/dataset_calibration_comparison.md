# Dataset-Specific Calibration Comparison

This comparison summarizes threshold calibration at the policy level rather than by transferring one fixed numeric threshold across datasets.

Raw anomaly-score scales are not directly interchangeable across CWRU and Paderborn because the model is fitted on each dataset and the score distribution changes with sampling rate, segmentation length, bearing condition, operating condition, and feature distribution. Therefore, the defensible journal claim is:

> The calibration policy must be selected and validated per dataset and protocol; fixed CWRU thresholds should not be reused blindly on Paderborn.

## Compared Calibration Scopes

- CWRU calibration: mean Isolation Forest behavior over strict CWRU protocols.
- Paderborn calibration: mean Isolation Forest behavior over expanded Paderborn protocols.
- Protocol-specific calibration: reported separately in `paderborn_expanded_protocol_threshold_summary.csv`.
- Pooled calibration: represented by the expanded-protocol mean in `paderborn_expanded_threshold_summary.csv`; this is useful for an overall operating-policy summary, but it hides condition-wise, bearing-wise, and file-wise differences.

## Main Finding

| Dataset | Practical policy | F1 | False alarm rate | Miss rate | Interpretation |
| --- | --- | --- | --- | --- | --- |
| CWRU | val_p95 | 0.8491 | 0.0696 | 0.1005 | CWRU strict splits remain comparatively easy, and validation-based P95 is a practical operating point. |
| Paderborn | val_p95 | 0.5200 | 0.1194 | 0.5950 | Paderborn is much harder; the same P95 policy keeps false alarms moderate but misses many faults. |
| CWRU | val_p99 | 0.8377 | 0.0147 | 0.2397 | P99 reduces false alarms with a moderate miss-rate penalty on CWRU. |
| Paderborn | val_p99 | 0.1184 | 0.0377 | 0.9256 | P99 is too conservative for expanded Paderborn and is not acceptable as the main threshold policy. |

## Paper-Level Interpretation

The results support making threshold calibration a core contribution instead of an auxiliary experiment. ROC-AUC and PR-AUC describe ranking quality, but the operating threshold determines whether a deployed system produces acceptable false alarm and miss rates.

For the journal version, the recommended wording is:

- Use dataset-specific validation calibration as the default deployment rule.
- Report P95 and P99 to expose the false-alarm versus miss-rate tradeoff.
- Use protocol-specific calibration tables to show how the threshold behaves under condition-wise, bearing-wise, and file-wise shifts.
- Avoid claiming that a threshold value learned from CWRU can be numerically reused on Paderborn.

## Files

- `dataset_calibration_policy_comparison.csv`
- `paderborn_expanded_threshold_summary.csv`
- `paderborn_expanded_protocol_threshold_summary.csv`
- `paderborn_expanded_heldout_calibration_detail.csv`
