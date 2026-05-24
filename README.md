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

### 最終指標

| 指標 | 數值 |
|------|------|
| **K-Fold AUC-ROC** | **1.0000 ± 0.0000** |
| **Final AUC-ROC** | **1.0000** |
| 異常閾值（P95）| 0.477769 |
| 正常重建誤差 mean±std | 0.2007 ± 0.0950 |
| 故障重建誤差 mean±std | 1.0493 ± 0.0103 |
| **故障/正常誤差比** | **~5.2×** |

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
