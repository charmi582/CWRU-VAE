# 基於變分自編碼器之軸承故障無監督異常偵測

> **Unsupervised Bearing Fault Detection Using Variational Autoencoder (VAE) with KL-Regularised Latent Normal Manifold**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-1.0000-brightgreen)]()

---

## 研究摘要

本專案提出一種**純無監督**的軸承故障偵測方法：

- 訓練時**僅使用正常振動訊號**，無需任何故障標籤
- 利用 **VAE（Variational Autoencoder）** 的 KL 散度將正常訊號投影至潛在空間 *N*(0, **I**)
- 異常訊號進入時，無法被正確重建，**重建誤差顯著升高**，藉此實現故障偵測
- 在 CWRU 標準資料集上達到 **AUC-ROC = 1.0000（5 折全滿分）**

---

## 專案結構

```
cwru_vae/
├── config.py           # 所有超參數與資料集設定
├── data_loader.py      # CWRU 資料集下載與切窗
├── preprocessor.py     # 資料平衡、正規化、K-Fold 切分
├── vae_model.py        # VAE 模型（Encoder / Reparameterize / Decoder）
├── trainer.py          # K-Fold 訓練主流程
├── load_wise_experiment.py # CWRU load-wise 泛化驗證
├── threshold_calibration_experiment.py # 跨負載閾值校準分析
├── noise_robustness_experiment.py # SNR 噪音魯棒性分析
├── ae_load_wise_experiment.py # AE baseline 的 load-wise 對照
├── reviewer_response_experiments.py # 評審回覆補充分析
├── visualizer.py       # 10 張視覺化圖表生成
├── main.py             # 主程式入口（執行全流程）
├── make_pptx_v2.py     # 自動生成中文 PPT 與學術海報
├── requirements.txt    # Python 相依套件
├── run.sh              # 一鍵執行腳本
└── results/
    ├── plots/          # 10 張視覺化圖片（自動生成）
    ├── models/         # 訓練完成的最佳模型（best_vae.pt）
    └── data/           # 下載的 .mat 原始資料（自動下載）
```

---

## 資料集：CWRU Bearing Dataset

| 項目 | 說明 |
|------|------|
| 來源 | Case Western Reserve University（官方伺服器直接下載）|
| 軸承型號 | SKF 6205 深溝球軸承 |
| 採樣頻率 | 12,000 Hz |
| 故障製造 | 電火花加工（EDM）|
| 故障尺寸 | 0.007" / 0.014" / 0.021" 直徑 |
| 負載條件 | 0–3 HP（1,720–1,797 RPM）|
| 檔案數量 | 40 個 .mat 檔案（自動下載）|

### 故障類型

| 標籤 | 名稱 | 說明 |
|------|------|------|
| 0 | Normal（正常）| 無故障基準線 |
| 1 | Inner Race Fault（內圈故障）| 內軌道損傷 |
| 2 | Ball Fault（滾動體故障）| 鋼球表面損傷 |
| 3 | Outer Race Fault（外圈故障）| 外軌道損傷 |

---

## 資料前處理

```
原始訊號 (12 kHz)
    ↓  滑動視窗（1024 點 ≈ 85 ms，50% 重疊）
11,832 片段（Normal: 3,310 / Fault: 8,522）
    ↓  工廠情境平衡（保留全部正常 + 隨機 10% 故障）
4,162 片段（Normal: 3,310 / Fault: 852）
    ↓  每樣本 Z-score 標準化
```

---

## VAE 模型架構

### Encoder

```
Input  (B, 1, 1024)
  → Conv1d(1→16,  k=7, s=2, p=3) + BN + ReLU  → (16, 512)
  → Conv1d(16→32, k=5, s=2, p=2) + BN + ReLU  → (32, 256)
  → Conv1d(32→64, k=3, s=2, p=1) + BN + ReLU  → (64, 128)
  → Conv1d(64→128,k=3, s=2, p=1) + BN + ReLU  → (128, 64)
  → Flatten → FC(8192→256) → ReLU
  → FC(256→32) : μ（均值）
  → FC(256→32) : log σ²（對數變異數）
```

### Reparameterization

```
z = μ + ε · exp(0.5 · log σ²),   ε ~ N(0, I)
```

### Decoder

```
z (B, 32)
  → FC(32→256) → ReLU
  → FC(256→8192) → ReLU → Reshape(128, 64)
  → ConvTranspose1d(128→64, k=3, s=2) + BN + ReLU
  → ConvTranspose1d(64→32,  k=3, s=2) + BN + ReLU
  → ConvTranspose1d(32→16,  k=5, s=2) + BN + ReLU
  → ConvTranspose1d(16→1,   k=7, s=2) + Tanh
Output (B, 1, 1024)
```

### 損失函數

$$\mathcal{L} = \underbrace{\text{MSE}(x, \hat{x}) \times 1024}_{\text{重建損失}} + \beta \cdot \underbrace{D_{KL}\left(q(z|x) \| \mathcal{N}(0, I)\right)}_{\text{KL 散度}}$$

$$D_{KL} = -\frac{1}{2} \cdot \text{mean}\left(1 + \log\sigma^2 - \mu^2 - \exp(\log\sigma^2)\right)$$

---

## 訓練設定

| 超參數 | 值 |
|--------|-----|
| Latent Dimension | 32 |
| Beta（KL 權重）| 1.0 |
| Batch Size | 64 |
| Max Epochs | 100 |
| Learning Rate | 1e-3 |
| K-Folds | 5 |
| Early Stopping Patience | 20 |
| Optimizer | Adam（weight_decay=1e-5）|
| LR Scheduler | ReduceLROnPlateau（patience=10, factor=0.5）|
| 訓練設備 | Apple MPS / CUDA / CPU（自動選擇）|

> **關鍵設計**：VAE 訓練期間**只使用正常資料**，故障資料僅用於最終評估。

---

## 實驗結果

> **結果解讀提醒**：CWRU 是高度標準化的實驗室資料集，在目前視窗層級切分與前處理設定下，正常與故障訊號高度可分。因此，本專案將 AUC-ROC = 1.0000 定位為「可重現軟體流程的案例驗證」，而非宣稱 VAE 在所有工業場景中顯著優於其他方法。若用於一般論文或真實部署，建議進一步採用 file-wise、load-wise 或 cross-domain split。

### 最終指標

| 指標 | 數值 |
|------|------|
| **K-Fold AUC-ROC** | **1.0000 ± 0.0000** |
| **Final AUC-ROC** | **1.0000** |
| 異常閾值（P95）| 0.477769 |
| 正常重建誤差 mean±std | 0.2007 ± 0.0950 |
| 故障重建誤差 mean±std | 1.0493 ± 0.0103 |
| **故障/正常誤差比** | **~5.2×** |

### 補充實務指標（回應評審建議）

以正常資料重建誤差百分位數作為閾值時：

| 閾值 | Precision | Recall | F1-score | False alarm rate | Miss rate |
|------|-----------|--------|----------|------------------|-----------|
| P90 | 72.02% | 100.00% | 83.67% | 10.00% | 0.00% |
| P95 | 83.69% | 100.00% | 91.12% | 5.02% | 0.00% |
| P97.5 | 91.12% | 100.00% | 95.35% | 2.51% | 0.00% |
| P99 | 96.17% | 100.00% | 98.05% | 1.03% | 0.00% |

額外腳本 `reviewer_response_experiments.py` 可在完成 `python main.py` 後產生：

- `results/reviewer_response/reviewer_metrics.csv`
- `results/reviewer_response/strict_split_protocol.txt`

```bash
pip install -r requirements.txt
python main.py
python reviewer_response_experiments.py
```

該腳本補充 PR-AUC、Precision、Recall、F1、false alarm rate、miss rate、P90/P95/P97.5/P99 閾值敏感度、95:5 與 99:1 極端不平衡壓力測試，並輸出 file-wise / load-wise GroupKFold 的切分協議。

### Load-wise 泛化驗證

為避免 50% 重疊滑動視窗在隨機切分時造成過度樂觀，本專案新增 CWRU load-wise split 實驗：

- 每一折保留一個負載條件作為完全未見測試集
- VAE 只使用其餘三個負載條件中的正常資料訓練
- 閾值由訓練端切出的正常 validation set 估計，不使用測試負載資料
- 輸出 ROC-AUC、PR-AUC、Precision、Recall、F1、false alarm rate、miss rate

```bash
pip install -r requirements.txt
python load_wise_experiment.py
```

輸出：

- `results/load_wise/load_wise_metrics.csv`
- `results/models/load_wise_heldout_load0.pt` 到 `load_wise_heldout_load3.pt`

目前正式執行結果如下：

| Held-out load | ROC-AUC | PR-AUC | Precision | Recall | F1-score | False alarm rate | Miss rate |
|---------------|---------|--------|-----------|--------|----------|------------------|-----------|
| 0 | 1.0000 | 1.0000 | 74.55% | 100.00% | 85.42% | 14.74% | 0.00% |
| 1 | 1.0000 | 1.0000 | 99.54% | 100.00% | 99.77% | 0.11% | 0.00% |
| 2 | 1.0000 | 1.0000 | 99.11% | 100.00% | 99.55% | 0.21% | 0.00% |
| 3 | 1.0000 | 1.0000 | 18.21% | 100.00% | 30.80% | 99.16% | 0.00% |
| **Mean ± std** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** | - | - | **78.89% ± 28.36%** | - | - |

此結果顯示，load-wise split 下異常分數排序能力仍高，但固定以 validation normal P95 作為閾值時，部分未見負載可能出現高誤報率。因此，修正版論文不應只報 AUC，還需要討論跨負載閾值校準、false alarm rate 與實務警報成本。

若只是快速檢查流程，可先降低訓練輪數：

```bash
python load_wise_experiment.py --max-epochs 3 --patience 2
```

### 期刊版嚴格實驗擴充

期刊版不再只以 CWRU 視窗層級 AUC 作為主要貢獻，而是將
**threshold calibration**、跨負載、跨故障尺寸、file-wise 切分與部署延遲
作為核心分析。新增腳本如下：

```bash
python strict_cwru_journal_experiments.py \
  --protocols fault-size-wise file-wise load-wise \
  --models vae cnn-ae lstm-ae iforest
```

若要先快速檢查流程，可使用：

```bash
python strict_cwru_journal_experiments.py \
  --protocols fault-size-wise \
  --models iforest \
  --max-epochs 1 --patience 1
```

輸出：

- `results/journal_strict_cwru/strict_cwru_metrics.csv`
- `results/journal_strict_cwru/models/`（模型權重預設不納入 Git）

目前支援的 protocol：

- `load-wise`：保留一個負載條件作為未見測試集。
- `fault-size-wise`：保留一個故障尺寸（0.007、0.014、0.021 inch）作為未見測試集。
- `file-wise`：以原始 `.mat` 檔案作為 group，降低重疊視窗造成的資料洩漏風險。

目前支援的 baseline：

- `vae`：卷積式變分自編碼器。
- `cnn-ae`：卷積式 deterministic autoencoder。
- `lstm-ae`：LSTM autoencoder。
- `iforest`：Isolation Forest，使用 12 維時域/頻域統計特徵。

### Threshold calibration

load-wise 結果顯示，跨負載時最大的問題不是 AUC，而是閾值轉移。新增腳本：

```bash
python threshold_calibration_experiment.py
```

輸出：

- `results/threshold_calibration/threshold_calibration_metrics.csv`

平均結果摘要：

| Calibration strategy | Mean F1 | Mean false alarm rate |
|----------------------|---------|-----------------------|
| Train normal P95 | 61.40% | 54.32% |
| Validation normal P95 | 78.89% | 28.55% |
| Validation normal P99 | 83.46% | 20.25% |
| Target-load 10% normal P95 | 90.88% | 6.13% |
| Target-load 10% normal P99 | 98.05% | 1.14% |

這代表若部署場域允許蒐集少量目標負載的正常資料作為 calibration set，誤報率可大幅降低。此結論比單純報告 AUC 更貼近工業部署。

### Noise robustness

為模擬真實工廠雜訊，本專案對 load-wise 測試訊號加入高斯噪音：

```bash
python noise_robustness_experiment.py
```

輸出：

- `results/noise_robustness/noise_robustness_metrics.csv`

平均結果摘要：

| SNR | Mean F1 | Mean false alarm rate |
|-----|---------|-----------------------|
| Clean | 78.89% | 28.55% |
| 30 dB | 78.75% | 28.71% |
| 20 dB | 77.97% | 29.76% |
| 10 dB | 61.38% | 52.90% |
| 5 dB | 35.08% | 100.00% |

排序指標 ROC-AUC / PR-AUC 在此設定下仍維持 1.0000，但固定閾值在低 SNR 下會造成誤報率上升，因此實務部署需要噪音感知的閾值校準或前處理。

### AE vs VAE under load-wise split

新增 deterministic convolutional AE baseline：

```bash
python ae_load_wise_experiment.py
```

輸出：

- `results/ae_load_wise/ae_load_wise_metrics.csv`

平均結果摘要：

| Model | ROC-AUC | PR-AUC | Mean F1 | Mean false alarm rate | Mean miss rate |
|-------|---------|--------|---------|-----------------------|----------------|
| VAE | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 78.89% ± 28.36% | 28.55% | 0.00% |
| AE | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 80.52% ± 28.73% | 26.52% | 0.00% |

在 CWRU load-wise split 下，AE 與 VAE 的排序能力皆達滿分，兩者差異主要反映在閾值與誤報率，而非 AUC。修正版論文因此不應宣稱 VAE 明顯優於 AE，而應將 VAE 定位為具機率潛在空間與可解釋性的可替換模型元件。

### 各 Fold 結果

| Fold | AUC-ROC | Val Loss | 備註 |
|------|---------|----------|------|
| 1 | 1.0000 | - | Best fold |
| 2 | 1.0000 | - | |
| 3 | 1.0000 | - | |
| 4 | 1.0000 | - | |
| 5 | 1.0000 | 339.63 | Early stop @ epoch 60 |

---

## 視覺化圖表

| 圖表 | 說明 |
|------|------|
| `01_time_domain.png` | 時域振動波形（4 類對比）|
| `02_fft_spectrum.png` | FFT 頻域分析（4 類對比）|
| `03_spectrogram.png` | STFT 時頻譜圖 |
| `04_class_distribution.png` | 平衡前後資料分布對比 |
| `05_tsne_features.png` | 原始特徵空間 t-SNE |
| `06_kfold_results.png` | K-Fold AUC & 驗證損失 |
| `07_training_curves.png` | Total / Recon / KL 損失曲線 |
| `08_reconstruction_error.png` | 正常 vs 故障重建誤差分布 |
| `09_roc_curve.png` | ROC 曲線（AUC = 1.0000）|
| `10_latent_space.png` | 潛在空間 t-SNE 投影 |

---

## 快速開始

### 1. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 2. 執行完整訓練流程

```bash
python main.py
```

或使用一鍵腳本：

```bash
bash run.sh
```

**執行流程：**
1. 自動從 CWRU 官方伺服器下載 40 個 .mat 檔案
2. 滑動視窗切割 + 工廠情境平衡
3. 生成 10 張資料視覺化圖表
4. 5-Fold 交叉驗證 VAE 訓練（僅正常資料）
5. 異常偵測評估（AUC-ROC、重建誤差分布）
6. 儲存最佳模型至 `results/models/best_vae.pt`

### 3. 生成 PPT 與學術海報（可選）

```bash
pip install python-pptx
python make_pptx_v2.py
```

輸出：
- `results/CWRU_VAE_Presentation_ZH.pptx`（14 張中文投影片）
- `results/CWRU_VAE_Poster_ZH.pptx`（A0 學術海報）

---

## 異常偵測原理

```
訓練階段（僅正常資料）：
  正常訊號 → Encoder → z ~ N(μ,σ²) ≈ N(0,I) → Decoder → 重建訊號
  KL 散度將正常資料的潛在分布推向 N(0,I)，形成「正常流形」

推論階段（含異常資料）：
  正常訊號 → 重建誤差低（≈0.2007）→ 低於閾值 → 判為正常 ✓
  異常訊號 → 超出正常流形 → 重建誤差高（≈1.0493）→ 超過閾值 → 判為異常 ✗

閾值 = P95（正常資料重建誤差）= 0.477769
```

---

## 系統需求

- Python 3.9+
- PyTorch 1.9+（支援 CUDA / Apple MPS / CPU）
- 磁碟空間：約 150 MB（資料集）+ 16 MB（模型）
- 記憶體：建議 8 GB+

---

## 授權

本專案採用 [MIT License](LICENSE)。

---

## 參考文獻

1. Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.
2. Smith, W.A. & Randall, R.B. (2015). Rolling element bearing diagnostics using the Case Western Reserve University data. *Mechanical Systems and Signal Processing, 64*, 100–131.
3. Loparo, K.A. (2012). Bearings Vibration Data Set. *Case Western Reserve University*.
4. An, J. & Cho, S. (2015). Variational Autoencoder based Anomaly Detection using Reconstruction Probability. *SNU Data Mining Center*.
---

## Journal Extension: Paderborn External Validation

The journal extension now includes a Paderborn Bearing Dataset integration
scaffold. Raw external data are intentionally ignored by git.

Official source:

- https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download

Default local layout:

```text
results/external/paderborn/raw/   # official .rar archives
results/external/paderborn/mat/   # extracted .mat files
```

Pilot external-validation run after data are available:

```bash
python paderborn_external_experiments.py \
  --protocols condition-wise bearing-wise file-wise \
  --models vae cnn-ae iforest \
  --bearings K001 K002 K003 KA01 KA03 KI01 KI03 \
  --max-files-per-condition 2
```

If 7-Zip is installed locally, selected official archives can be downloaded and
extracted directly:

```bash
python paderborn_external_experiments.py \
  --download --extract \
  --protocols condition-wise bearing-wise file-wise \
  --models vae cnn-ae lstm-ae iforest \
  --bearings K001 K002 KA01 KI01 \
  --max-files-per-condition 2
```

Outputs:

- `results/journal_external_paderborn/paderborn_metrics.csv`
- `results/journal_external_paderborn/models/*.pt`
- `results/journal_external_paderborn/paderborn_pilot_summary.md`

Supported Paderborn protocols:

- `condition-wise`: hold out one operating condition.
- `bearing-wise`: hold out one damaged bearing state plus a normal test subset.
- `file-wise`: hold out complete measurement files.

Pilot result note:

- A 4-bearing Paderborn pilot has been run with `K001`, `K002`, `KA01`, and
  `KI01`, using two files per operating condition.
- The pilot loaded 16,070 windows and produced 200 metric rows.
- Unlike CWRU, the external subset is not trivially separable; the result should
  be discussed as evidence that CWRU-only evaluation overestimates deployment
  readiness.

Window-size diagnostic:

```bash
python paderborn_window_sensitivity.py \
  --window-sizes 1024 2048 4096 8192 \
  --models iforest \
  --protocols condition-wise bearing-wise file-wise \
  --bearings K001 K002 KA01 KI01 \
  --max-files-per-condition 2
```

The diagnostic shows that longer Paderborn windows improve external validation.
The 8192-point follow-up can be reproduced with:

```bash
python paderborn_external_experiments.py \
  --window-size 8192 \
  --stride 4096 \
  --bearings K001 K002 KA01 KI01 \
  --protocols condition-wise bearing-wise file-wise \
  --models vae cnn-ae lstm-ae iforest \
  --max-files-per-condition 2 \
  --output-name paderborn_metrics_window_8192_deep.csv
```

---

## Journal Extension: Federated Normal-Only Deployment

The federated extension simulates multiple companies or factory sites as
clients. Each client keeps its own normal bearing windows locally, trains a
local reconstruction model, and shares only model parameters with the server for
FedAvg aggregation. Raw vibration windows are not transmitted.

Pilot run:

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

Important framing:

- Federated training uses normal data only.
- Fault data are optional audit data and are not used during training.
- The main normal-only metric is false alarm rate under client/site shift.
- This extension supports a privacy-preserving deployment story; it should not
  be claimed as an automatic accuracy improvement without local-only and
  centralized comparisons.
