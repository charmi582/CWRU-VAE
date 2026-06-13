# Journal Extension Plan

Target positioning: Q3 upper-tier engineering / vibration / condition-monitoring journal.

The journal version should not be framed as a new state-of-the-art VAE model.
The core contribution should be:

> A reproducible, threshold-calibrated unsupervised anomaly detection workflow
> for bearing vibration monitoring under load, fault-size, file, and noise shifts.

## Target Journal Strategy

Primary direction:

- Journal of Vibration Engineering & Technologies or a similar vibration /
  rotating-machinery engineering journal.

Backup direction:

- International Journal of Advanced Manufacturing Technology.
- Journal of the Brazilian Society of Mechanical Sciences and Engineering.

Not recommended as first targets:

- Sensors / Machines / IEEE Access, because APC may be required.
- Engineering Applications of Artificial Intelligence / MSSP, because the current
  method novelty is not strong enough for a safe first submission.

## Required Experiment Blocks

### 1. Strict CWRU Generalisation

Implemented in:

```bash
python strict_cwru_journal_experiments.py \
  --protocols fault-size-wise file-wise load-wise \
  --models vae cnn-ae lstm-ae iforest
```

Outputs:

- `results/journal_strict_cwru/strict_cwru_metrics.csv`
- `results/journal_strict_cwru/models/*.pt`

Protocols:

- `load-wise`: hold out one load condition.
- `fault-size-wise`: hold out one fault diameter: 0.007, 0.014, or 0.021 inch.
- `file-wise`: GroupKFold by original `.mat` file to reduce sliding-window leakage.

Each protocol reports:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- False alarm rate
- Miss rate
- Multiple threshold calibration strategies

### 2. Additional Baselines

Implemented:

- VAE
- CNN-AE
- LSTM-AE
- Isolation Forest with 12 statistical time/frequency features

Still useful if time allows:

- OC-SVM
- LOF
- CNN-LSTM-AE or TCN-AE

### 3. Threshold Calibration as Main Contribution

Threshold strategies implemented in the strict experiment output:

- train normal P95
- train normal P99
- validation normal P95
- validation normal P99
- robust train median + 5 MAD

The journal paper should present threshold calibration as a central finding:

> AUC evaluates ranking, but false alarm rate and miss rate expose whether a
> threshold transfers across loads, files, fault sizes, and noise levels.

Fuzzy decision-layer extension:

- Implemented in `fuzzy_health_index_experiment.py`.
- The fuzzy layer uses validation-normal score anchors P50, P90, P95, and P99.
- P95 is treated as the warning center; P99 is treated as high-confidence fault
  membership.
- The output is a fuzzy health index in `[0, 1]` plus three practical decision
  policies:
  - `hard_val_p95`
  - `fuzzy_warning_as_alarm_hi50`
  - `fuzzy_fault_only_hi75`
- This extension should be framed as a calibrated decision layer that exposes
  warning and uncertain regions, not as a replacement for anomaly-score models.

Current fuzzy result summary:

- CWRU:
  - `hard_val_p95`: F1 0.8491, FAR 0.0696, miss rate 0.1005.
  - `fuzzy_fault_only_hi75`: F1 0.8557, FAR 0.0331, miss rate 0.1722.
  - Interpretation: high-confidence fuzzy alarms reduce false alarms but accept
    more missed early/weak alarms.
- Paderborn:
  - `hard_val_p95`: F1 0.5200, FAR 0.1194, miss rate 0.5950.
  - `fuzzy_fault_only_hi75`: F1 0.3911, FAR 0.0683, miss rate 0.7212.
  - Interpretation: expanded Paderborn exposes a large gray zone; fuzzy health
    indexing is most useful for graded warnings and calibration diagnostics, not
    as a claim of automatic F1 improvement.

### 4. Deployment Analysis

Implemented in strict deep-model runs:

- parameter count
- approximate model size
- CPU inference latency per window
- training time per fold

Still useful if time allows:

- CPU batch throughput
- memory usage
- end-to-end stream detection latency

### 5. External Dataset

Candidate order:

1. Paderborn Bearing Dataset
2. IMS Bearing Dataset
3. FEMTO-ST / PRONOSTIA

Selection criteria:

- Download stability and licensing clarity.
- Contains normal and faulty bearing vibration signals.
- Can be framed as cross-dataset validation rather than retraining-only result.
- Does not require APC-related publication venue.

Recommended first integration target: Paderborn, because it is a bearing dataset
with well-known fault categories and is frequently used in fault diagnosis.

Initial integration implemented:

- `paderborn_data_loader.py`
- `paderborn_external_experiments.py`

The Paderborn loader supports the official Bearing DataCenter archive layout.
It converts extracted MATLAB files into the same `(X, y_binary, y_multi,
metadata)` format used by CWRU, with metadata fields for operating condition,
bearing code, measurement file, and label name.

Suggested pilot command after downloading and extracting a subset:

```bash
python paderborn_external_experiments.py \
  --protocols condition-wise bearing-wise file-wise \
  --models vae cnn-ae lstm-ae iforest \
  --bearings K001 K002 KA01 KI01 \
  --max-files-per-condition 2
```

If 7-Zip is available locally, the script can also download and extract the
selected official archives:

```bash
python paderborn_external_experiments.py \
  --download --extract \
  --protocols condition-wise bearing-wise file-wise \
  --models vae cnn-ae lstm-ae iforest \
  --bearings K001 K002 KA01 KI01 \
  --max-files-per-condition 2
```

Raw Paderborn archives and extracted MATLAB files are intentionally ignored by
git because the external dataset is several GB.

Pilot training status:

- Completed on `K001`, `K002`, `KA01`, and `KI01`, using two files per operating
  condition.
- Outputs:
  - `results/journal_external_paderborn/paderborn_metrics.csv`
  - `results/journal_external_paderborn/paderborn_pilot_summary.md`
- Follow-up diagnostics:
  - `results/journal_external_paderborn/paderborn_window_sensitivity_metrics.csv`
  - `results/journal_external_paderborn/paderborn_window_sensitivity_summary.md`
  - `results/journal_external_paderborn/paderborn_metrics_window_8192_deep.csv`
  - `results/journal_external_paderborn/paderborn_8192_deep_summary.md`
- The pilot result is substantially harder than CWRU. This should be framed as
  an important journal finding: CWRU can show near-perfect ranking, but external
  validation exposes cross-dataset transfer and threshold calibration limits.
- Window-size sensitivity shows that the CWRU-style 1024-point window is too
  short for Paderborn. Increasing the Paderborn window to 8192 points improves
  the Isolation Forest ROC-AUC/PR-AUC and F1 substantially, so the journal paper
  should discuss sampling-rate-aware segmentation instead of treating all
  bearing datasets with the same window length.
- In the 8192-point external pilot, Isolation Forest remains the strongest and
  most stable baseline on this small Paderborn subset. This means the journal
  manuscript should avoid claiming deep reconstruction models are universally
  superior; the stronger claim is that the proposed pipeline exposes when
  dataset-specific calibration and classical baselines are necessary.

Expanded Paderborn main-experiment status:

- Completed on `K001-K006`, `KA01`, `KA03`, `KA05`, `KA07`, `KI01`, `KI03`,
  `KI05`, and `KI07`.
- Main segmentation uses an 8192-point window and 4096-point stride.
- To keep this first expanded pass tractable, four files per operating
  condition are used.
- Protocols completed:
  - `condition-wise`
  - `bearing-wise`
  - `file-wise`
- Output files:
  - `results/journal_external_paderborn/paderborn_expanded_8192_iforest_metrics.csv`
  - `results/journal_external_paderborn/paderborn_expanded_8192_iforest_summary.md`
  - `results/journal_external_paderborn/paderborn_expanded_threshold_summary.csv`
  - `results/journal_external_paderborn/paderborn_expanded_protocol_threshold_summary.csv`
  - `results/journal_external_paderborn/dataset_calibration_comparison.md`

Reproduction command:

```bash
python paderborn_external_experiments.py \
  --window-size 8192 \
  --stride 4096 \
  --bearings K001 K002 K003 K004 K005 K006 KA01 KA03 KA05 KA07 KI01 KI03 KI05 KI07 \
  --protocols condition-wise bearing-wise file-wise \
  --models iforest \
  --max-files-per-condition 4 \
  --n-file-splits 6 \
  --output-name paderborn_expanded_8192_iforest_metrics.csv
```

Expanded Paderborn interpretation:

- The expanded external dataset contains 13,704 windows in this tractable pass:
  5,874 normal windows and 7,830 fault windows.
- The overall Isolation Forest ranking quality is moderate rather than
  CWRU-perfect: ROC-AUC 0.7854 and PR-AUC 0.8359.
- `train_p95` and `val_p95` are the practical threshold policies. They keep
  false alarm rates near 0.12, but miss rates remain high, near 0.59.
- `train_p99`, `val_p99`, and MAD-based thresholds are too conservative on the
  expanded Paderborn setting and miss most fault windows.
- This should be presented as evidence that dataset-specific and
  protocol-specific calibration is necessary for external deployment.

### 6. Federated Normal-Only Deployment Extension

Implemented:

- `federated_normal_experiment.py`

Motivation:

> Each company or factory site may own private normal bearing data. Federated
> learning allows clients to train local normal-only anomaly detectors and share
> only model parameters for aggregation, without exposing raw vibration windows.

Current pilot:

```bash
python federated_normal_experiment.py \
  --model cnn-ae \
  --clients-by bearing \
  --bearings K001 K002 KA01 KI01 \
  --window-size 8192 \
  --stride 4096 \
  --max-files-per-condition 1 \
  --rounds 3 \
  --local-epochs 1 \
  --include-fault-audit
```

Outputs:

- `results/federated_normal/federated_normal_metrics.csv`
- `results/federated_normal/federated_normal_summary.md`

Interpretation:

- Training remains normal-only and privacy-preserving.
- Raw client windows are never shared with the server.
- The primary normal-only metric is client false alarm rate.
- Fault data, when available, are used only as an audit set and not for
  federated training.
- The first pilot should be framed as a deployment architecture proof, not as
  an accuracy improvement claim. The current CNN-AE pilot has low optional fault
  recall, so future work should compare FedAvg with local-only and centralized
  normal-only baselines before making any performance claim.

Formal federated fuzzy comparison:

- Implemented in `federated_fuzzy_experiment.py`.
- The experiment compares:
  - `local-only`: each client trains its own normal-only model.
  - `centralized`: one model trains on pooled normal windows.
  - `federated`: clients train locally and share only model weights through
    FedAvg.
- The fuzzy health-index layer is applied to both `cnn-ae` and `vae`
  reconstruction scores, making the fuzzy decision layer model-agnostic.
- Fault windows are audit-only and are not used for local, centralized, or
  federated training.
- Main outputs:
  - `results/federated_fuzzy/federated_fuzzy_metrics.csv`
  - `results/federated_fuzzy/federated_fuzzy_summary.csv`
  - `results/federated_fuzzy/federated_fuzzy_client_stability.csv`
  - `results/federated_fuzzy/federated_fuzzy_client_detail.csv`
  - `results/federated_fuzzy/federated_fuzzy_summary.md`

Current Paderborn formal pass:

```bash
python federated_fuzzy_experiment.py \
  --models cnn-ae vae \
  --rounds 2 \
  --local-epochs 1 \
  --centralized-epochs 2 \
  --max-files-per-condition 1
```

Initial result interpretation:

- FedAvg should not be claimed as an automatic accuracy improvement.
- For CNN-AE with `fuzzy_warning_as_alarm_hi50`, local-only currently gives the
  strongest mean F1, while federated training provides the privacy-preserving
  collaborative reference.
- Client stability is analyzed through the standard deviation of false alarm
  rate, miss rate, uncertain rate, and fuzzy health gap.
- VAE is unstable in the low-epoch federated setting, which supports reporting
  model choice and client heterogeneity as practical deployment concerns.
- This completes the missing comparison that reviewers usually expect for
  federated learning: local-only versus centralized versus federated.

## Formal Experiment Order

1. Run CWRU strict protocols with `iforest` first to validate all splits.
2. Run `vae` and `cnn-ae` for all CWRU strict protocols.
3. Run `lstm-ae` only after the first two deep baselines are stable, because it
   is slower.
4. Add external dataset loader. **Done for Paderborn scaffold.**
5. Run Paderborn external validation experiments after data download/extraction.
   **Pilot and expanded 14-bearing Isolation Forest pass are done.**
6. Run expanded Paderborn deep baselines after the threshold-calibration section
   is finalized.
7. Run federated normal-only deployment experiments. **Initial FedAvg pilot
   done.**
8. Run cross-dataset experiments if signal preprocessing assumptions remain
   comparable.
9. Rewrite journal manuscript around threshold calibration and deployment
   stability.

## Recommended Main Claim

Do not claim:

> VAE is superior to all existing methods on CWRU.

Claim:

> A reproducible anomaly-detection pipeline must evaluate not only ranking
> performance but also threshold transferability, false alarm behavior, and
> robustness under realistic operating shifts.
