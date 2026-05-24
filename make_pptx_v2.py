"""
make_pptx_v2.py
產生：
  1. CWRU_VAE_Presentation_ZH.pptx  ── 14 張中文投影片（深藍主題）
  2. CWRU_VAE_Poster_ZH.pptx        ── A0 直式學術海報（標準研討會規範）
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

os.chdir("/Users/guset/deep/cwru_vae")
PLOT = "results/plots"
OUT  = "results"

# ── 色彩系統 ──────────────────────────────────────────────────────────────────
def rgb(r,g,b): return RGBColor(r,g,b)

NAVY    = rgb(0x0D, 0x1B, 0x3E)
MID     = rgb(0x11, 0x2D, 0x60)
BLUE    = rgb(0x21, 0x96, 0xF3)
WHITE   = rgb(0xFF, 0xFF, 0xFF)
LGRAY   = rgb(0xB0, 0xBE, 0xC5)
GOLD    = rgb(0xFF, 0xD6, 0x00)
GREEN   = rgb(0x00, 0xC8, 0x5A)
RED     = rgb(0xF4, 0x43, 0x36)
ORANGE  = rgb(0xFF, 0x98, 0x00)
PURPLE  = rgb(0x9C, 0x27, 0xB0)
DARK    = rgb(0x1A, 0x1A, 0x2E)
DKBLUE  = rgb(0x0D, 0x47, 0xA1)
LTBLUE  = rgb(0xE3, 0xF2, 0xFD)
LTGRAY  = rgb(0xF5, 0xF5, 0xF5)
MIDGRAY = rgb(0xE0, 0xE0, 0xE0)


# ══════════════════════════════════════════════════════════════════════════════
#  工具函式
# ══════════════════════════════════════════════════════════════════════════════

def _rect(slide, x, y, w, h, fill=None, line_color=None, line_w=None):
    shp = slide.shapes.add_shape(1,
          Inches(x), Inches(y), Inches(w), Inches(h))
    sf = shp.fill
    if fill: sf.solid(); sf.fore_color.rgb = fill
    else:    sf.background()
    if line_color:
        shp.line.color.rgb = line_color
        if line_w: shp.line.width = Pt(line_w)
    else:
        shp.line.fill.background()
    return shp


def _txt(slide, text, x, y, w, h, size=14, bold=False, color=WHITE,
         align=PP_ALIGN.LEFT, italic=False, font="Microsoft JhengHei"):
    txb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size    = Pt(size)
    run.font.bold    = bold
    run.font.italic  = italic
    run.font.color.rgb = color
    # CJK font fallback
    try: run.font.name = font
    except: pass
    return txb


def _img(slide, path, x, y, w, h=None):
    if h: slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    else: slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w))


def dark_bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY


def sec_bar(slide, title, y=0.0, w=10.0):
    _rect(slide, 0, y, w, 0.52, fill=MID)
    _txt(slide, title, 0.18, y+0.05, w-0.3, 0.42,
         size=20, bold=True, color=BLUE)


def bullet(slide, items, x, y, w, size=11, color=WHITE, gap=0.3):
    for i, item in enumerate(items):
        _txt(slide, f"▸  {item}", x, y+i*gap, w, gap+0.05, size=size, color=color)


def metric_card(slide, val, label, x, y, w=2.2, h=0.9,
                vc=GOLD, bg=MID):
    _rect(slide, x, y, w, h, fill=bg)
    _txt(slide, val,   x+0.08, y+0.04, w-0.16, 0.46,
         size=22, bold=True, color=vc, align=PP_ALIGN.CENTER)
    _txt(slide, label, x+0.08, y+0.5,  w-0.16, 0.36,
         size=9,  color=LGRAY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
#  中文 PPT（14 張，16:9，深藍主題）
# ══════════════════════════════════════════════════════════════════════════════

def make_ppt_zh():
    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(5.625)
    BL = prs.slide_layouts[6]

    # ── S01 標題 ──────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    _rect(s, 0, 0, 10, 1.05, fill=MID)
    _txt(s, "CWRU 資料集  ｜  VAE 變分自編碼器  ｜  異常偵測",
         0.3, 0.18, 9.4, 0.5, size=10, color=LGRAY, align=PP_ALIGN.CENTER)
    _txt(s, "基於變分自編碼器之軸承故障無監督異常偵測",
         0.4, 1.2, 9.2, 1.3, size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _txt(s, "KL 損失正則化潛在空間・工業設備健康監測應用",
         0.4, 2.6, 9.2, 0.55, size=14, color=BLUE, italic=True, align=PP_ALIGN.CENTER)
    _txt(s, "凱斯西儲大學（CWRU）軸承資料集 ／ PyTorch ／ Apple MPS",
         0.4, 3.2, 9.2, 0.4, size=11, color=LGRAY, align=PP_ALIGN.CENTER)
    metric_card(s, "AUC = 1.0000", "完美異常偵測", 1.0, 4.1, vc=GREEN)
    metric_card(s, "5-Fold CV",    "交叉驗證",     3.4, 4.1)
    metric_card(s, "5.2 倍",       "故障/正常誤差比", 5.8, 4.1, vc=GOLD)
    metric_card(s, "僅正常資料",   "訓練策略",     8.0, 4.1, vc=BLUE,
                w=1.8, bg=DKBLUE)

    # ── S02 研究動機 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "研究背景與動機")
    _txt(s, "為何需要軸承故障偵測？", 0.3, 0.65, 5.5, 0.4,
         size=14, bold=True, color=BLUE)
    bullet(s, [
        "軸承是旋轉機械（馬達、風機、幫浦）的核心零件",
        "未及時偵測故障 → 非計畫停機、安全風險、鉅額損失",
        "及早偵測可降低維修成本達 30%",
        "工業 IoT 趨勢要求自動化、即時的狀態監測方案",
    ], 0.3, 1.1, 5.5)

    _txt(s, "核心挑戰", 0.3, 2.6, 5.5, 0.4, size=14, bold=True, color=BLUE)
    bullet(s, [
        "故障事件極為罕見 → 資料嚴重不平衡",
        "故障標籤需靠專家標注，成本高昂且難以取得",
        "傳統監督式學習在不平衡資料下表現不佳",
    ], 0.3, 3.05, 5.5)

    _txt(s, "本研究解決方案", 0.3, 4.1, 5.5, 0.35, size=13, bold=True, color=GREEN)
    _txt(s, "VAE 僅用正常訊號訓練，學習「正常流形」\n"
            "→ 故障訊號無法被良好重建 → 高重建誤差 → 自動標記異常",
         0.3, 4.48, 5.5, 0.8, size=11, color=WHITE)

    _rect(s, 5.9, 0.65, 3.8, 4.6, fill=MID)
    _txt(s, "資料分布統計", 6.0, 0.75, 3.6, 0.4, size=13, bold=True,
         color=GOLD, align=PP_ALIGN.CENTER)
    rows = [
        ("原始片段總數",  "11,832", WHITE),
        ("平衡後總數",    "4,162",  WHITE),
        ("正常（訓練用）","3,310 (79.5%)", GREEN),
        ("故障（僅評估）","852 (20.5%)",   LGRAY),
        ("內圈故障",      "282 片段",      RED),
        ("滾動體故障",    "271 片段",      ORANGE),
        ("外圈故障",      "299 片段",      PURPLE),
    ]
    for i, (k, v, vc) in enumerate(rows):
        yy = 1.25 + i*0.5
        _txt(s, k, 6.0, yy, 2.2, 0.38, size=10, color=LGRAY)
        _txt(s, v, 8.0, yy, 1.7, 0.38, size=10, bold=True, color=vc,
             align=PP_ALIGN.RIGHT)

    # ── S03 資料集 ────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "CWRU 軸承資料集介紹")
    _img(s, f"{PLOT}/01_time_domain.png", 5.1, 0.6, 4.8)
    _txt(s, "實驗設置", 0.3, 0.65, 4.6, 0.35, size=13, bold=True, color=BLUE)
    rows2 = [
        ("軸承型號", "SKF 6205 深溝球軸承"),
        ("採樣頻率", "12,000 Hz"),
        ("故障製造", "電火花加工（EDM）"),
        ("故障尺寸", '0.007" / 0.014" / 0.021" 直徑'),
        ("負載條件", "0、1、2、3 HP（1720-1797 RPM）"),
        ("檔案總數", "40 個 .mat 檔案"),
    ]
    for i, (k, v) in enumerate(rows2):
        yy = 1.1 + i*0.52
        _rect(s, 0.3, yy, 4.6, 0.45,
              fill=MID if i%2==0 else rgb(0x0D,0x1B,0x3E))
        _txt(s, k, 0.4,  yy+0.07, 1.8, 0.32, size=10, color=LGRAY)
        _txt(s, v, 2.25, yy+0.07, 2.6, 0.32, size=10, bold=True, color=WHITE)

    _txt(s, "故障類型分類", 0.3, 4.22, 4.6, 0.3, size=12, bold=True, color=BLUE)
    for i, (nm, cl) in enumerate([
        ("正常",    GREEN), ("內圈故障", RED),
        ("滾動體",  ORANGE),("外圈故障", PURPLE),
    ]):
        _rect(s, 0.3+i*1.18, 4.55, 1.1, 0.65, fill=MID)
        _txt(s, nm, 0.32+i*1.18, 4.6, 1.06, 0.55,
             size=9, bold=True, color=cl, align=PP_ALIGN.CENTER)

    # ── S04 資料前處理 ────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "資料前處理與工廠情境模擬")
    _img(s, f"{PLOT}/04_class_distribution.png", 4.75, 0.6, 5.1)
    steps = [
        ("① 滑動視窗切割",
         "視窗大小：1,024 點（≈85 ms @ 12 kHz）\n步長：512 點（50% 重疊）"),
        ("② 工廠分布模擬",
         "保留全部正常片段（3,310）\n隨機保留 10% 故障片段（852）\n真實工廠中故障罕見，正常占比 79.5%"),
        ("③ 逐樣本 Z-Score 正規化",
         "每個視窗獨立：減去均值，除以標準差\n確保各片段振幅尺度一致"),
        ("④ 訓練 / 評估分離",
         "VAE 僅以正常片段訓練（3,310 片）\n故障資料僅用於評估 AUC-ROC"),
    ]
    y = 0.65
    for t, b in steps:
        _rect(s, 0.25, y, 4.35, 1.1, fill=MID)
        _txt(s, t, 0.35, y+0.06, 4.15, 0.36, size=11, bold=True, color=BLUE)
        _txt(s, b, 0.35, y+0.44, 4.15, 0.62, size=9,  color=WHITE)
        y += 1.2

    # ── S05 訊號分析 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "訊號分析：FFT 頻譜 與 STFT 時頻譜圖")
    _img(s, f"{PLOT}/02_fft_spectrum.png", 0.1, 0.6, 4.85)
    _img(s, f"{PLOT}/03_spectrogram.png",  5.1, 0.6, 4.8)
    _txt(s, "◀  FFT 頻域分析：故障訊號出現明顯諧波頻率峰值",
         0.15, 5.15, 4.85, 0.4, size=9, color=LGRAY, align=PP_ALIGN.CENTER)
    _txt(s, "STFT 時頻分析：故障訊號出現週期性能量聚集  ▶",
         5.05, 5.15, 4.85, 0.4, size=9, color=LGRAY, align=PP_ALIGN.CENTER)

    # ── S06 t-SNE ─────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "t-SNE 特徵空間視覺化（原始訊號）")
    _img(s, f"{PLOT}/05_tsne_features.png", 0.5, 0.65, 9.0)
    _txt(s, "故障訊號在原始特徵空間已具備部分可分性；VAE 潛在空間可達完全分離。",
         0.5, 5.22, 9.0, 0.35, size=10, color=LGRAY, italic=True, align=PP_ALIGN.CENTER)

    # ── S07 VAE 架構 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "VAE 架構設計：一維卷積編碼器─解碼器")

    _rect(s, 0.2, 0.65, 3.0, 4.5, fill=MID)
    _txt(s, "編碼器 Encoder", 0.3, 0.72, 2.8, 0.38,
         size=13, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    for i, l in enumerate([
        "輸入 (B, 1, 1024)", "Conv1d 1→16  k7 s2",
        "Conv1d 16→32 k5 s2","Conv1d 32→64 k3 s2",
        "Conv1d 64→128 k3 s2","攤平 → 8,192",
        "全連接 → 256","FC→μ(32)  FC→logσ²(32)",
    ]):
        _txt(s, l, 0.3, 1.15+i*0.38, 2.8, 0.34, size=9,
             color=GOLD if "μ" in l else WHITE, align=PP_ALIGN.CENTER)

    _rect(s, 3.4, 0.65, 3.2, 4.5, fill=rgb(0x0A,0x23,0x4F))
    _txt(s, "潛在空間  z ∈ ℝ³²", 3.5, 0.78, 3.0, 0.38,
         size=12, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
    _txt(s, "重參數化技巧", 3.5, 1.28, 3.0, 0.28,
         size=10, bold=True, color=LGRAY, align=PP_ALIGN.CENTER)
    _txt(s, "z = μ + ε·exp(0.5·logσ²)\nε ~ N(0, I)",
         3.5, 1.6, 3.0, 0.6, size=11, color=WHITE, italic=True, align=PP_ALIGN.CENTER)
    _rect(s, 3.55, 2.48, 2.9, 1.55, fill=DARK)
    _txt(s, "損失函數：", 3.65, 2.56, 2.7, 0.3,
         size=10, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    _txt(s, "L = MSE(x,x̂)×1024\n  + β·KL(q ‖ N(0,I))\nβ = 1.0（標準 VAE）",
         3.65, 2.9, 2.7, 0.75, size=10, color=WHITE, italic=True, align=PP_ALIGN.CENTER)
    _txt(s, "異常分數 = 重建誤差 MSE\n高誤差 → 異常訊號",
         3.55, 4.2, 2.9, 0.7, size=10, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

    _rect(s, 6.8, 0.65, 3.0, 4.5, fill=MID)
    _txt(s, "解碼器 Decoder", 6.9, 0.72, 2.8, 0.38,
         size=13, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    for i, l in enumerate([
        "z (B, 32)", "全連接 → 256",
        "全連接 → 8,192","重塑 (128, 64)",
        "ConvT 128→64 k3 s2","ConvT 64→32  k3 s2",
        "ConvT 32→16  k5 s2","ConvT 16→1   k7 s2 + Tanh",
    ]):
        _txt(s, l, 6.9, 1.15+i*0.38, 2.8, 0.34, size=9,
             color=GREEN if i==7 else WHITE, align=PP_ALIGN.CENTER)

    # ── S08 訓練策略 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "訓練策略：無監督正常流形學習")
    cards = [
        ("僅正常資料訓練",    "VAE 訓練期間不接觸任何故障訊號\n純無監督學習，不需故障標籤"),
        ("5 折交叉驗證",     "KFold(n=5, shuffle=True)\n每折：2,648 訓練 / 662 驗證"),
        ("最佳化器設定",     "Adam  lr=1×10⁻³  weight_decay=1×10⁻⁵\nReduceLROnPlateau(patience=10)"),
        ("早停法",           "耐心值 = 20 個 epoch\n自動載入最佳驗證損失的模型"),
        ("批次與訓練次數",   "批次大小 = 64，最大 100 epochs\nFold 5 於第 60 epoch 觸發早停"),
        ("運算硬體",         "Apple MPS（Metal Performance Shaders）\nMacBook Apple Silicon GPU 加速"),
    ]
    for i, (t, b) in enumerate(cards):
        c = i%3; r = i//3
        x = 0.2+c*3.27; y = 0.7+r*2.3
        _rect(s, x, y, 3.1, 2.1, fill=MID)
        _txt(s, t, x+0.12, y+0.1,  2.86, 0.4, size=12, bold=True, color=BLUE)
        _txt(s, b, x+0.12, y+0.55, 2.86, 1.3, size=10, color=WHITE)

    # ── S09 訓練曲線 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "VAE 訓練曲線（最佳 Fold 1）")
    _img(s, f"{PLOT}/07_training_curves.png", 0.15, 0.62, 9.7)
    _txt(s, "總損失：1,172 → 176　｜　重建損失：1,172 → 173　｜　KL 散度穩定於 3.79",
         0.3, 5.28, 9.4, 0.3, size=10, color=LGRAY, italic=True, align=PP_ALIGN.CENTER)

    # ── S10 K-Fold 結果 ───────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "5 折交叉驗證結果")
    _img(s, f"{PLOT}/06_kfold_results.png", 0.15, 0.62, 9.7)
    metric_card(s, "1.0000 ± 0.0000", "平均 AUC-ROC（5 折全滿分）",
                1.5, 5.08, w=3.4, vc=GREEN)
    metric_card(s, "342.61 ± 3.72",   "平均驗證損失",
                5.5, 5.08, w=3.0)

    # ── S11 重建誤差 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "異常偵測：重建誤差分布分析")
    _img(s, f"{PLOT}/08_reconstruction_error.png", 0.15, 0.62, 9.7)
    for val, lbl, x, vc in [
        ("0.2007", "正常 平均值", 0.2,  BLUE),
        ("0.0950", "正常 標準差", 1.85, LGRAY),
        ("1.0493", "故障 平均值", 3.5,  RED),
        ("0.0103", "故障 標準差", 5.15, LGRAY),
        ("5.2 倍", "誤差倍率",   6.8,  GOLD),
        ("0.4778", "偵測閾值",   8.45, GREEN),
    ]:
        _rect(s, x, 4.95, 1.55, 0.63, fill=MID)
        _txt(s, val, x+0.05, 4.98, 1.45, 0.3,
             size=14, bold=True, color=vc, align=PP_ALIGN.CENTER)
        _txt(s, lbl, x+0.05, 5.27, 1.45, 0.28,
             size=8,  color=LGRAY, align=PP_ALIGN.CENTER)

    # ── S12 ROC 曲線 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "ROC 曲線 — VAE 異常偵測器")
    _img(s, f"{PLOT}/09_roc_curve.png", 1.0, 0.62, 8.0)
    _txt(s, "AUC = 1.0000 — 正常與故障訊號完美分離，零誤報率",
         0.4, 5.33, 9.2, 0.3, size=11, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

    # ── S13 潛在空間 ──────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "VAE 潛在空間視覺化（t-SNE 投影）")
    _img(s, f"{PLOT}/10_latent_space.png", 0.15, 0.62, 9.7)
    _txt(s, "正常資料（藍）形成環形分布，符合 N(0,I) 先驗；故障資料聚集於中央，落在正常重建流形之外。",
         0.4, 5.1, 9.2, 0.48, size=10, color=LGRAY, italic=True, align=PP_ALIGN.CENTER)

    # ── S14 結論 ──────────────────────────────────────────────────────────────
    s = prs.slides.add_slide(BL); dark_bg(s)
    sec_bar(s, "結論與未來展望")
    _txt(s, "主要成果", 0.3, 0.65, 5.5, 0.35, size=14, bold=True, color=BLUE)
    bullet(s, [
        "5 折交叉驗證 AUC-ROC 全部達 1.0000，標準差為 0",
        "故障重建誤差為正常的 5.2 倍，可靠分離正常與故障",
        "閾值 0.4778 可完美標記所有三種故障類型",
        "純無監督訓練，不需任何故障標籤",
        "資料分布符合真實工廠場景（79.5% 正常）",
    ], 0.3, 1.05, 5.5, size=11)

    _txt(s, "量化指標摘要", 5.8, 0.65, 3.9, 0.35, size=14, bold=True, color=BLUE)
    for label, val, yy, vc in [
        ("K-Fold AUC",    "1.0000 ± 0.000", 1.08, GREEN),
        ("最終 AUC",      "1.0000",          1.6,  GREEN),
        ("正常誤差 μ",    "0.2007",          2.12, BLUE),
        ("故障誤差 μ",    "1.0493",          2.64, RED),
        ("誤差倍率",      "5.2 倍",          3.16, GOLD),
        ("偵測閾值 P95",  "0.4778",          3.68, GREEN),
    ]:
        _rect(s, 5.8, yy, 3.8, 0.44, fill=MID)
        _txt(s, label, 5.9, yy+0.07, 2.0, 0.3, size=10, color=LGRAY)
        _txt(s, val,   7.8, yy+0.07, 1.7, 0.3, size=10, bold=True, color=vc,
             align=PP_ALIGN.RIGHT)

    _txt(s, "未來展望", 0.3, 4.18, 9.2, 0.35, size=14, bold=True, color=BLUE)
    bullet(s, [
        "部署於 IoT 串流感測器，實現即時自適應閾值調整",
        "擴展至多感測器融合（振動 + 溫度 + 電流）",
        "引入 Transformer 編碼器捕捉長距離時序依賴",
    ], 0.3, 4.56, 9.4, size=10, gap=0.3)

    path = f"{OUT}/CWRU_VAE_Presentation_ZH.pptx"
    prs.save(path)
    print(f"  PPT 已儲存 → {path}（{len(prs.slides)} 張）")


# ══════════════════════════════════════════════════════════════════════════════
#  學術海報（A0 直式，符合研討會規範）
#  規範來源：
#    - Colin Purrington (colinpurrington.com)：圖:文:留白 = 40:20:40，≤1000字
#    - UCLA/UChicago Library：標題 3m 可讀，內文 1m 可讀
#    - 國際研討會標準：3 欄版面，從左至右閱讀
# ══════════════════════════════════════════════════════════════════════════════

def make_poster_zh():
    # A0 直式：84cm × 119cm = 33.07" × 46.81"
    PW, PH = 33.07, 46.81

    prs = Presentation()
    prs.slide_width  = Inches(PW)
    prs.slide_height = Inches(PH)
    s = prs.slides.add_slide(prs.slide_layouts[6])

    # 白色底
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE

    # ── 頂部標題區（高度約 13%）─────────────────────────────────────────────
    HDR_H = 6.0
    _rect(s, 0, 0, PW, HDR_H, fill=DKBLUE)
    _txt(s, "基於變分自編碼器之軸承故障無監督異常偵測研究",
         0.5, 0.4, PW-1.0, 2.5, size=58, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER)
    _txt(s, "Unsupervised Bearing Fault Detection Using Variational Autoencoder with KL-Regularised Latent Normal Manifold",
         0.5, 2.9, PW-1.0, 1.1, size=22, color=rgb(0xBB,0xDE,0xFB),
         italic=True, align=PP_ALIGN.CENTER)
    # 關鍵詞標籤
    tags = ["CWRU 資料集", "VAE", "KL 散度", "無監督學習", "異常偵測", "AUC = 1.0000"]
    tw = 4.5
    tx_start = (PW - len(tags)*tw) / 2
    for i, tag in enumerate(tags):
        _rect(s, tx_start + i*tw, 4.15, tw-0.2, 0.85,
              fill=rgb(0x1A,0x6B,0xC9))
        _txt(s, tag, tx_start+i*tw+0.1, 4.2, tw-0.3, 0.7,
             size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # ── 摘要區（橫跨全寬，標題下方）────────────────────────────────────────
    ABS_Y = HDR_H + 0.35
    ABS_H = 2.8
    _rect(s, 0.4, ABS_Y, PW-0.8, ABS_H, fill=LTBLUE,
          line_color=DKBLUE, line_w=1.5)
    _txt(s, "【摘要 Abstract】", 0.7, ABS_Y+0.15, PW-1.4, 0.6,
         size=22, bold=True, color=DKBLUE)
    _txt(s,
         "本研究提出一種基於變分自編碼器（VAE）的無監督軸承故障異常偵測方法。"
         "以凱斯西儲大學（CWRU）軸承振動資料集為實驗對象，"
         "模擬真實工廠場景中資料嚴重不平衡的條件（正常資料佔 79.5%），"
         "VAE 僅使用正常振動訊號進行訓練，透過 KL 損失函數將正常資料的潛在表示"
         "壓縮至標準常態分布 N(0,I) 空間。當故障訊號輸入時，"
         "因超出正常流形範圍而產生顯著偏高的重建誤差（正常/故障誤差比達 5.2 倍），"
         "實現零標籤需求的精準異常偵測。5 折交叉驗證結果顯示 AUC-ROC 全數達到 1.0000。",
         0.7, ABS_Y+0.8, PW-1.4, 1.9, size=17, color=DARK)

    # ── 3 欄主體佈局 ────────────────────────────────────────────────────────
    COL_W   = 9.8
    COL_GAP = 0.55
    COL_X   = [0.4, 0.4+COL_W+COL_GAP, 0.4+2*(COL_W+COL_GAP)]
    BODY_Y  = ABS_Y + ABS_H + 0.5
    BODY_H  = PH - BODY_Y - 1.5   # 底部留 footer

    # ── Section header helper ──────────────────────────────────────────────
    def sh(slide, no, title, x, y, w, bg=DKBLUE):
        _rect(slide, x, y, w, 0.75, fill=bg)
        _txt(slide, f"{no}  {title}", x+0.18, y+0.1, w-0.3, 0.55,
             size=20, bold=True, color=WHITE)
        return y + 0.75 + 0.2

    # ── Text block helper ──────────────────────────────────────────────────
    def tb(slide, text, x, y, w, size=15, color=DARK):
        _txt(slide, text, x, y, w, 8, size=size, color=color)

    # ── Card with border ──────────────────────────────────────────────────
    def card(slide, x, y, w, h, bg=LTGRAY):
        _rect(slide, x, y, w, h, fill=bg, line_color=MIDGRAY, line_w=0.5)

    G = 0.28   # 段落間距

    # ════════════════════ 左欄 ════════════════════════════════════════════════
    cx, cw = COL_X[0], COL_W
    y = BODY_Y

    # § 1 研究動機
    y = sh(s, "§1", "研究動機與目的", cx, y, cw)
    tb(s, "軸承故障是旋轉機械最常見的失效原因之一，"
          "約佔機械設備維護成本的 40%。"
          "傳統監督式學習需大量故障標籤，"
          "但實際工廠中故障事件極為罕見且標注成本昂貴。\n\n"
          "本研究目標：建立一種不依賴故障標籤的純無監督異常偵測系統，"
          "透過學習正常軸承訊號的潛在分布，"
          "使任何偏離「正常流形」的訊號均可被自動識別為潛在故障。",
       cx, y, cw); y += 2.5 + G

    # § 2 資料集
    y = sh(s, "§2", "CWRU 軸承資料集", cx, y, cw)
    card(s, cx, y, cw, 4.2, LTGRAY)
    tbl_data = [
        ("參數",       "數值"),
        ("軸承型號",   "SKF 6205 深溝球軸承"),
        ("採樣頻率",   "12,000 Hz"),
        ("故障製造法", "電火花加工（EDM）"),
        ("故障尺寸",   '0.007" / 0.014" / 0.021" 直徑'),
        ("負載條件",   "0–3 HP（1720–1797 RPM）"),
        ("故障類型",   "正常 / 內圈 / 滾動體 / 外圈"),
        ("檔案數量",   "40 個 .mat 檔案"),
    ]
    row_h = 0.48
    for ri, (k, v) in enumerate(tbl_data):
        ry = y + ri*row_h
        bg_r = DKBLUE if ri==0 else (rgb(0xE8,0xF0,0xFE) if ri%2==0 else WHITE)
        _rect(s, cx+0.05, ry, cw-0.1, row_h, fill=bg_r)
        tc = WHITE if ri==0 else DARK
        _txt(s, k, cx+0.15, ry+0.07, 3.8, row_h-0.1, size=14, bold=(ri==0), color=tc)
        _txt(s, v, cx+4.1,  ry+0.07, cw-4.2, row_h-0.1, size=14, bold=(ri==0), color=tc)
    y += len(tbl_data)*row_h + 0.15 + G

    # § 3 資料前處理
    y = sh(s, "§3", "資料前處理", cx, y, cw)
    tb(s, "① 滑動視窗：大小 1,024 點（≈85 ms）、步長 512 點（50% 重疊）\n"
          "② 工廠分布模擬：保留全部正常片段（3,310），\n"
          "   隨機抽取 10% 故障片段（852），反映真實工廠不平衡狀況\n"
          "③ Z-Score 正規化：每片段獨立標準化（μ=0，σ=1）",
       cx, y, cw, size=15); y += 1.9 + G
    _img(s, f"{PLOT}/04_class_distribution.png", cx, y, cw); y += 6.3 + G

    # § 4 時域訊號
    y = sh(s, "§4", "時域振動訊號比較", cx, y, cw)
    tb(s, "四類訊號在時域上呈現明顯差異：\n"
          "正常訊號振幅小且規律；故障訊號振幅偏大，\n"
          "外圈故障（紫）可見明顯週期性衝擊現象。",
       cx, y, cw, size=15); y += 1.3 + G
    _img(s, f"{PLOT}/01_time_domain.png", cx, y, cw)

    # ════════════════════ 中欄 ════════════════════════════════════════════════
    cx, cw = COL_X[1], COL_W
    y = BODY_Y

    # § 5 研究方法
    y = sh(s, "§5", "研究方法：VAE 架構設計", cx, y, cw)
    tb(s, "採用一維卷積 VAE，由編碼器、重參數化採樣層與解碼器三部分組成。"
          "編碼器將 1,024 維振動片段壓縮至 32 維潛在向量，"
          "並輸出均值 μ 與對數方差 logσ²；"
          "解碼器從潛在向量重建原始訊號。",
       cx, y, cw, size=15); y += 1.9 + G

    # 架構卡片
    card(s, cx, y, cw, 8.5, LTGRAY)
    arch_blocks = [
        ("編碼器 Encoder",  DKBLUE, [
            "輸入：(B, 1, 1,024)",
            "Conv1d 1→16   k=7 stride=2  → (16, 512)",
            "Conv1d 16→32  k=5 stride=2  → (32, 256)",
            "Conv1d 32→64  k=3 stride=2  → (64, 128)",
            "Conv1d 64→128 k=3 stride=2  → (128, 64)",
            "攤平 Flatten → 8,192",
            "全連接 FC 8192→256  ReLU",
            "輸出：μ(32)  &  logσ²(32)",
        ]),
        ("潛在空間 Latent Space", rgb(0x2E,0x7D,0x32), [
            "重參數化：z = μ + ε·exp(0.5·logσ²)",
            "ε ~ N(0, I)  （訓練時採樣，推論時用 μ）",
            "損失函數：L = MSE(x,x̂)×1024",
            "              + β·KL(q(z|x) ‖ N(0,I))",
            "KL = -½·Σ(1+logσ²−μ²−σ²),  β=1.0",
        ]),
        ("解碼器 Decoder", rgb(0x78,0x00,0x26), [
            "輸入：z(32)",
            "全連接 FC 32→256  ReLU",
            "全連接 FC 256→8,192  ReLU",
            "重塑 Reshape → (128, 64)",
            "ConvT 128→64 k=3 s=2 → (64,128)",
            "ConvT 64→32  k=3 s=2 → (32,256)",
            "ConvT 32→16  k=5 s=2 → (16,512)",
            "ConvT 16→1   k=7 s=2 + Tanh → (1,1024)",
        ]),
    ]
    ay = y + 0.15
    for title, bg, lines in arch_blocks:
        _rect(s, cx+0.1, ay, cw-0.2, 0.45, fill=bg)
        _txt(s, title, cx+0.2, ay+0.06, cw-0.35, 0.33,
             size=15, bold=True, color=WHITE)
        ay += 0.48
        for line in lines:
            _txt(s, f"  {line}", cx+0.15, ay, cw-0.3, 0.38,
                 size=13, color=DARK)
            ay += 0.36
        ay += 0.1
    y += 8.5 + G

    # § 6 訓練策略
    y = sh(s, "§6", "訓練策略", cx, y, cw)
    tb(s, "• 僅以正常振動訊號訓練 VAE（無監督）\n"
          "• 5 折交叉驗證（KFold, shuffle=True）\n"
          "• Adam 最佳化器，lr=1×10⁻³，weight_decay=1×10⁻⁵\n"
          "• ReduceLROnPlateau 學習率調度（patience=10）\n"
          "• 早停法：patience=20，Fold 5 於第 60 epoch 停止\n"
          "• 批次大小：64，最大訓練次數：100\n"
          "• 硬體：Apple MPS（Metal Performance Shaders）",
       cx, y, cw, size=15); y += 2.8 + G

    # § 7 FFT
    y = sh(s, "§7", "FFT 頻域分析", cx, y, cw)
    tb(s, "故障訊號在頻域出現明顯諧波峰值，\n"
          "正常訊號頻譜相對平坦乾淨。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/02_fft_spectrum.png", cx, y, cw); y += 7.2 + G

    # § 8 STFT 時頻圖
    y = sh(s, "§8", "STFT 時頻譜分析", cx, y, cw)
    tb(s, "時頻圖顯示故障訊號具有週期性能量聚集現象，\n"
          "正常訊號時頻分布均勻。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/03_spectrogram.png", cx, y, cw); y += 6.5 + G

    # § 9 t-SNE
    y = sh(s, "§9", "t-SNE 特徵空間分析", cx, y, cw)
    tb(s, "原始訊號特徵空間中，故障與正常訊號已具備部分可分性，"
          "VAE 潛在空間可進一步強化此分離。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/05_tsne_features.png", cx, y, cw)

    # ════════════════════ 右欄 ════════════════════════════════════════════════
    cx, cw = COL_X[2], COL_W
    y = BODY_Y

    # § 10 訓練曲線
    y = sh(s, "§10", "訓練曲線（最佳 Fold 1）", cx, y, cw)
    tb(s, "總損失由 1,172 快速收斂至 176；"
          "KL 損失穩定於 3.79，顯示潛在空間已收斂至 N(0,I) 先驗分布。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/07_training_curves.png", cx, y, cw); y += 5.3 + G

    # § 11 K-Fold
    y = sh(s, "§11", "5 折交叉驗證結果", cx, y, cw)
    tb(s, "5 折實驗中每一折 AUC-ROC 均達 1.0000，"
          "驗證損失穩定（342.61±3.72），模型具備良好泛化能力。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/06_kfold_results.png", cx, y, cw); y += 6.2 + G

    # § 12 重建誤差
    y = sh(s, "§12", "重建誤差分布分析", cx, y, cw)
    tb(s, "正常訊號重建誤差均值 0.2007，故障訊號為 1.0493（約 5.2 倍）。"
          "兩類分布完全不重疊，P95 閾值 0.4778 可無誤差分割。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/08_reconstruction_error.png", cx, y, cw); y += 6.2 + G

    # § 13 ROC
    y = sh(s, "§13", "ROC 曲線", cx, y, cw)
    tb(s, "ROC 曲線貼頂角，AUC = 1.0000，達成完美異常偵測。\n"
          "閾值 0.4778 下，假陽性率（FPR）= 0，真陽性率（TPR）= 1。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/09_roc_curve.png", cx, y, cw); y += 6.3 + G

    # § 14 潛在空間
    y = sh(s, "§14", "潛在空間視覺化", cx, y, cw)
    tb(s, "正常資料（藍）形成符合 N(0,I) 先驗的環形分布，"
          "故障資料（紅/橙/紫）聚集於中心，落在正常流形之外。",
       cx, y, cw, size=15); y += 0.9 + G
    _img(s, f"{PLOT}/10_latent_space.png", cx, y, cw); y += 6.3 + G

    # § 15 結論
    y = sh(s, "§15", "結論", cx, y, cw, bg=rgb(0x1B,0x5E,0x20))
    # 結果表格
    results = [
        ("K-Fold AUC-ROC",      "1.0000 ± 0.000", GREEN),
        ("最終 AUC-ROC",        "1.0000",          GREEN),
        ("正常重建誤差 μ ± σ",  "0.2007 ± 0.0950", BLUE),
        ("故障重建誤差 μ ± σ",  "1.0493 ± 0.0103", RED),
        ("誤差倍率",             "約 5.2 倍",        GOLD),
        ("異常偵測閾值（P95）",  "0.4778",           GREEN),
    ]
    for k, v, vc in results:
        card(s, cx, y, cw, 0.72, bg=rgb(0xE8,0xF5,0xE9))
        _txt(s, k, cx+0.12, y+0.12, 6.0, 0.5, size=14, color=DARK)
        _txt(s, v, cx+6.1,  y+0.12, cw-6.3, 0.5, size=15,
             bold=True, color=vc, align=PP_ALIGN.RIGHT)
        y += 0.77

    y += 0.3
    tb(s, "本研究驗證 VAE 結合 KL 正則化可在完全無監督條件下，"
          "達成軸承故障的完美異常偵測（AUC=1.0000）。"
          "方法適用於真實工廠不平衡資料場景，無需任何故障標籤。",
       cx, y, cw, size=15, color=DARK)

    # ── 底部 Footer ──────────────────────────────────────────────────────────
    FY = PH - 1.35
    _rect(s, 0, FY, PW, 1.35, fill=DKBLUE)
    _txt(s, "【參考文獻】  "
            "[1] Smith, W.A. & Randall, R.B. (2015). Rolling element bearing diagnostics using the Case Western Reserve University data. "
            "Mech. Syst. Signal Process., 64–65, 100–131.  "
            "[2] Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR 2014.  "
            "[3] Loparo, K.A. Case Western Reserve University Bearing Data Center. https://engineering.case.edu/bearingdatacenter",
         0.4, FY+0.12, PW-0.8, 0.7, size=12, color=rgb(0xBB,0xDE,0xFB))
    _txt(s, "資料集：Case Western Reserve University Bearing Data Center  ·  "
            "框架：PyTorch  ·  硬體：Apple MPS  ·  "
            "程式碼：/Users/guset/deep/cwru_vae/",
         0.4, FY+0.82, PW-0.8, 0.45, size=12,
         color=rgb(0x90,0xCA,0xF9), align=PP_ALIGN.CENTER)

    path = f"{OUT}/CWRU_VAE_Poster_ZH.pptx"
    prs.save(path)
    print(f"  海報已儲存 → {path}（A0 直式）")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═"*50)
    print("  製作中文 PPT 投影片…")
    make_ppt_zh()
    print("  製作學術海報…")
    make_poster_zh()
    print("═"*50)
    print("  完成！")
