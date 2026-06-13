# Springer 中文期刊初稿 Overleaf 使用方式

本資料夾是 Springer Nature `sn-jnl` LaTeX 模板風格的中文初稿。

## Overleaf 設定

1. 在 Overleaf 建立 Springer Nature Journal Article template 專案，或上傳 `sn-jnl.cls` 至本專案。
2. 將本資料夾內容上傳至 Overleaf：
   - `main.tex`
   - `references.bib`
   - `figures/`
   - `tables/`
3. Menu -> Compiler 選擇 `XeLaTeX`。
4. 主檔設定為 `main.tex`。

## 字型

目前 `main.tex` 會優先使用：

- 英文：Times New Roman；若 Overleaf 找不到則使用 TeX Gyre Termes。
- 中文：Noto Serif CJK TC；若找不到則使用 AR PL UMing TW。

若學校或期刊要求中文標楷體，可在本機或 Overleaf 專案上傳標楷體字型後，將：

```tex
\setCJKmainfont{Noto Serif CJK TC}
```

改為對應字型名稱或檔案。

## 本版主軸

本篇已依目前討論收斂為：

- 不將聯邦式學習放入主文。
- 主文聚焦 CWRU/Paderborn 嚴格驗證。
- 核心貢獻為 threshold calibration 與 fuzzy health index。
- 聯邦式學習放在未來研究。

## 注意

這是中文初稿與 Overleaf 程式碼骨架，投稿前仍需：

- 補作者、單位、email。
- 依目標期刊確認 Springer 模板選項。
- 補完整英文摘要或改成全英文稿。
- 檢查參考文獻格式與 BibTeX 欄位。
