# 審查意見修正摘要

本文依據兩位評審意見，將修正重點由「模型效能突破」調整為「可重現無監督異常偵測軟體流程」之案例驗證，並補強實務部署相關評估。

## 主要修正

1. **重新定位論文貢獻**
   - 已在摘要、討論與結論中說明：CWRU 屬於高度標準化資料集，AUC-ROC = 1.0000 應視為流程驗證結果，而非 VAE 對所有方法具有顯著效能優勢。
   - 明確強調貢獻為可重現流程、正常資料訓練策略、可替換模型管線與研究/教學基準。

2. **補充實務指標**
   - 新增 Precision、Recall、F1-score、False Alarm Rate、Miss Rate。
   - 新增 P90、P95、P97.5、P99 閾值敏感度分析，說明不同閾值下誤報率與漏報率的權衡。

3. **回應極端不平衡資料建議**
   - 新增 `reviewer_response_experiments.py`，可產生 95:5 與 99:1 故障比例壓力測試。
   - 結果輸出至 `results/reviewer_response/reviewer_metrics.csv`。

4. **補充跨負載閾值校準**
   - 新增 `threshold_calibration_experiment.py`。
   - 比較 train normal P95/P99、validation normal P95/P99、median-MAD robust threshold、以及目標負載 10% normal calibration。
   - 結果顯示 target-load 10% normal P99 可將平均 false alarm rate 降至 1.14%，平均 F1 提升至 98.05%，明確指出跨負載部署時需要 calibration set。

5. **補充噪音魯棒性**
   - 新增 `noise_robustness_experiment.py`。
   - 測試 clean、30 dB、20 dB、10 dB、5 dB 高斯噪音。
   - 結果顯示 30/20 dB 下 F1 接近 clean，但 10/5 dB 固定閾值會明顯增加誤報率，支持論文中「真實工廠需噪音感知閾值校準」的修正說法。

6. **補充 AE vs VAE 嚴格設定比較**
   - 新增 `ae_model.py` 與 `ae_load_wise_experiment.py`。
   - AE 使用與 VAE 對齊的卷積 encoder/decoder，只移除 KL 與 reparameterization。
   - 在 load-wise split 下 AE 與 VAE 皆達 ROC-AUC / PR-AUC = 1.0000，平均 F1 分別為 80.52% 與 78.89%。因此修正版不宣稱 VAE 顯著優於 AE，而是將 VAE 定位為具機率潛在空間與可解釋性的模型元件。

7. **回應資料洩漏與嚴格切分建議**
   - `data_loader.py` 新增 `download_and_load_with_metadata()`，保留每個 window 的來源檔案、負載條件與類別名稱。
   - `preprocessor.py` 新增 `group_kfold_splits()`，支援 file-wise / load-wise GroupKFold。
   - `reviewer_response_experiments.py` 輸出 `strict_split_protocol.txt`，列出 file-wise 與 load-wise split 的 fold 規模，作為下一版完整重訓的切分基礎。
   - 新增 `load_wise_experiment.py`，正式執行 CWRU load-wise split：每折保留一個負載條件作為未見測試集，僅使用其他負載的正常資料訓練，並輸出 ROC-AUC、PR-AUC、Precision、Recall、F1、False Alarm Rate 與 Miss Rate。
   - 正式 load-wise 結果已輸出至 `results/load_wise/load_wise_metrics.csv`。四個 held-out load 的 ROC-AUC 與 PR-AUC 皆為 1.0000，但 load 3 在 P95 validation threshold 下出現 99.16% false alarm rate，顯示 AUC 滿分仍需搭配閾值校準與誤報分析。

8. **補強公平比較說明**
   - README 與論文均補充：傳統方法使用 12 維手工統計特徵，AE/VAE 使用原始訊號，因此比較主要作為流程參考，不應解讀為完全公平的模型優劣排名。

9. **補強研究限制**
   - 討論中新增：視窗層級隨機切分可能受 50% 重疊視窗影響。
   - 新增未來工作：file-wise split、load-wise split、cross-domain evaluation、global normalization、robust scaling、無標準化比較，以及 FEMTO-ST / PRONOSTIA 外部資料驗證。

## 對應評審意見

| 評審建議 | 已修正內容 |
|---|---|
| 更嚴格資料切分 | 新增 metadata loader、GroupKFold 工具與 strict split protocol |
| 50% 重疊視窗可能高估 | 討論與結論中明確承認，並新增且執行 load-wise split 實驗 |
| 95:5、99:1 極端不平衡 | 新增壓力測試腳本 |
| 補充 PR-AUC、F1、誤報率、漏報率 | 新增分析腳本與論文閾值敏感度表 |
| VAE 優勢不明顯 | 改寫定位，不宣稱效能顯著優於 AE/傳統方法 |
| VAE 與 AE 在嚴格設定下比較 | 新增 AE load-wise baseline，顯示兩者 AUC 皆滿分，差異主要在閾值/誤報 |
| 傳統特徵與原始訊號比較公平性 | README/論文補充比較限制 |
| 閾值來源與敏感度 | 新增 P90/P95/P97.5/P99 分析與 target-load calibration |
| CWRU 太乾淨 | 新增 30/20/10/5 dB noise robustness 實驗 |
| 邊緣部署指標不足 | 討論保守改為「部署雛形」，未宣稱已完成即時部署 |
| 補充近年研究與定位 | 討論中強調流程基準與未來跨域驗證 |
| 避免強調效能突破 | 摘要、討論與結論已改為流程貢獻導向 |
