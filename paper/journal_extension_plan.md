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

## Formal Experiment Order

1. Run CWRU strict protocols with `iforest` first to validate all splits.
2. Run `vae` and `cnn-ae` for all CWRU strict protocols.
3. Run `lstm-ae` only after the first two deep baselines are stable, because it
   is slower.
4. Add external dataset loader.
5. Run cross-dataset experiments.
6. Rewrite journal manuscript around threshold calibration and deployment
   stability.

## Recommended Main Claim

Do not claim:

> VAE is superior to all existing methods on CWRU.

Claim:

> A reproducible anomaly-detection pipeline must evaluate not only ranking
> performance but also threshold transferability, false alarm behavior, and
> robustness under realistic operating shifts.
