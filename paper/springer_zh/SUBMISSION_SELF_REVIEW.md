# 投稿可行性自評

## 本篇主軸

本稿已依照目前研究收斂方向撰寫：

- 嚴格泛化驗證
- Paderborn 外部資料集
- threshold calibration
- fuzzy health index
- 可重複實現流程

聯邦式學習已從主文移除，只保留於未來工作，避免論文主線過度發散。

## 如果以這個方向投稿，會上嗎？

理性判斷：**有投稿機會，但目前仍不應視為保證會上。**

若投稿到 Q3 前段或偏應用型 vibration / mechanical engineering journal，本研究的優勢是：

1. 不再只是在 CWRU 上報告高 AUC，而是補上 load-wise、fault-size-wise、file-wise 等較嚴格設定。
2. 加入 Paderborn 外部資料集，降低單一 benchmark 過度樂觀的問題。
3. 將 threshold calibration 放成核心問題，能回應工業部署中的 false alarm 與 miss rate。
4. fuzzy health index 與 warning/uncertain region 讓研究更貼近健康監測，而不是單純二元分類。
5. 程式流程與圖表結果可重複，適合包裝為 reproducible software pipeline。

主要風險是：

1. 方法創新性仍屬於 workflow/system integration，而不是全新模型理論。
2. Paderborn 上的數值不漂亮，必須誠實包裝成 external validation 與 calibration challenge，不能宣稱模型效能突破。
3. 目前中文稿還需要轉成完整英文期刊稿，並補強近年 predictive maintenance、open-set fault detection、domain adaptation、self-supervised anomaly detection 文獻。
4. fuzzy health index 目前是 calibrated decision layer，若審稿人期待更複雜 fuzzy rule system，可能會要求更多消融。

## 建議投稿定位

建議不要投強調 AI 模型創新的期刊，而應投：

- vibration engineering
- mechanical systems health monitoring
- applied condition monitoring
- engineering software workflow

比較適合的論文定位是：

> A reproducible threshold-calibrated fuzzy health-index workflow for unsupervised bearing anomaly detection under strict and external validation protocols.

## 目前最需要再補強的地方

投稿前建議優先完成：

1. 將中文稿轉成正式英文稿。
2. 補 2021--2026 年相關文獻。
3. 在方法中更明確說明 fuzzy membership anchors 的合理性。
4. 將表格分成主表與附錄表，避免正文過長。
5. 補 GitHub repository URL 與 reproducibility checklist。

## 結論

以目前成果而言，若主打「可重現流程 + 嚴格驗證 + Paderborn 外部資料集 + threshold calibration + fuzzy health index」，投稿 Q3 前段應用型期刊是合理的。

但若要提高接受率，最關鍵不是再加新模組，而是把論文故事寫得更集中、文獻補足、英文表達成熟，並避免宣稱 VAE 或 fuzzy theory 在所有情境下都提升準確率。
