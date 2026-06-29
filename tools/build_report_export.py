#!/usr/bin/env python3
"""Build submission-ready, single-language exports of the summary report and
render them to PDF.

For each of {en, zh}:
  * fix the language (drop the zh/en toggle + the lang script);
  * insert a colored Executive-Summary block *before* the Contents, with three
    figures (Ensemble-vs-v3 monthly IC, Ensemble-vs-v3 buckets, all-model IC/ICIR);
  * repoint the factor link to the public https://autoalpha.cn/ page;
  * inline every local <img> as a base64 data URI (self-contained);
  * write report/summary_{lang}.html and report/summary_{lang}.pdf.

Author: Shuo Yin <yins25@mails.tsinghua.edu.cn>
"""
import os
import re
import io
import base64
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "summary_src.html")
OUTDIR = os.path.join(ROOT, "report")
os.makedirs(OUTDIR, exist_ok=True)
MAX_W = 1500
FACTORS_URL = "https://autoalpha.cn/cn_future_alpha/factors.html"

_cache = {}
def _encode(path):
    if path in _cache:
        return _cache[path]
    full = os.path.join(ROOT, path)
    im = Image.open(full).convert("RGBA")
    if im.width > MAX_W:
        h = round(im.height * MAX_W / im.width)
        im = im.resize((MAX_W, h), Image.LANCZOS)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1])
    buf = io.BytesIO()
    bg.save(buf, format="PNG", optimize=True)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    _cache[path] = uri
    return uri

def _inline_images(html):
    def repl(m):
        src = m.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        return 'src="' + _encode(src) + '"'
    return re.sub(r'src="([^"]+)"', repl, html)

EXEC_CSS = """
  .execsum{background:linear-gradient(180deg,#eef4ff,#f8fbff);border:1px solid #c7d7f5;
           border-left:6px solid var(--brand);border-radius:14px;padding:20px 24px 10px;margin:28px 0 6px}
  .execsum h2{border:0;margin:0 0 6px;color:var(--brand);display:block}
  .execsum .kpis{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
  .execsum .kpi{background:#fff;border:1px solid #c7d7f5;border-radius:10px;padding:8px 14px;font-size:13.5px}
  .execsum .kpi b{color:var(--brand);font-size:17px;font-variant-numeric:tabular-nums}
  .execsum .figs{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0 4px}
  .execsum .figs figure{margin:0}
  .execsum .figs .wide{grid-column:1 / -1}
  @media print{
    .topbar{display:none!important}
    section,figure,.execsum,table,.tbl{break-inside:avoid}
    h2,h3,h4{break-after:avoid}
    a{color:#0b3a66}
  }
"""

EXEC_BLOCK = """
<section id="execsum" class="execsum">
  <h2><span class="zh">总结 · Executive Summary</span><span class="en">Executive Summary</span></h2>

  <div class="zh">
    <p>本 Project 用两条技术线预测中国期货 30 分钟收益。<b>经典基线</b>：Ridge / LightGBM / MLP 三个互补单模型（各带日内时段后校准），再用一个小而可审计的 signed-ridge 严格堆叠成 <b>ML Ensemble</b>。<b>端到端</b>：把 Transformer 从 smoke test 一路推进到 v1→v3。</p>
    <p><b>有效的端到端尝试（均通过「双指标 + 2019 验证」门）：</b>多尺度 patch embedding、时间偏置注意力（ALiBi 金融时序化）、双路 pooling 读出、SwiGLU + LayerScale 稳定残差、市场状态门控，以及在线可微的 FactorOperatorBank（低秩交互、丰富输入字段）。被消融拒绝的：一次性大改写、RevIN、跨层/截面/跨变量注意力、低秩残差、元数据嵌入、MoE 头。</p>
    <p><b>最终 IC（2020 测试，Pooled IC）：</b>ML Ensemble <b>0.0573</b>（总分最佳，月度 ICIR 5.67）；End2End v3 <b>0.0548</b>（最强单一深度模型，月度 ICIR 6.01）。两者都稳定越过 0.05，v3 已基本补齐与最强基线的差距，且逐月稳定性（ICIR）甚至更高。</p>
  </div>
  <div class="en">
    <p>This project predicts 30-minute China-futures returns along two technical lines. <b>Classical baseline:</b> three complementary single models (Ridge / LightGBM / MLP), each with intraday post-calibration, combined into a small, auditable signed-ridge strict stack — the <b>ML Ensemble</b>. <b>End-to-end:</b> a Transformer carried from a smoke test through v1→v3.</p>
    <p><b>Effective end-to-end ideas (all passing the two-metric / 2019-validation gate):</b> multi-scale patch embedding, time-biased attention (ALiBi adapted to finance time), dual pooling readout, SwiGLU + LayerScale stable residual, market-state gating, and an online differentiable FactorOperatorBank (low-rank interaction enriching the input). Ablated away: the one-shot rewrite, RevIN, cross-layer / cross-section / cross-variate attention, low-rank residual, metadata embeddings, and MoE heads.</p>
    <p><b>Final IC (2020 test, Pooled IC):</b> ML Ensemble <b>0.0573</b> (best overall, monthly ICIR 5.67); End2End v3 <b>0.0548</b> (best single deep model, monthly ICIR 6.01). Both clear 0.05 comfortably; v3 closes most of the gap to the strongest baseline and is even steadier month-to-month (higher ICIR).</p>
  </div>

  <div class="kpis">
    <div class="kpi"><b>0.0573</b> · <span class="zh">ML Ensemble Pooled IC</span><span class="en">ML Ensemble Pooled IC</span></div>
    <div class="kpi"><b>0.0548</b> · <span class="zh">End2End v3 Pooled IC</span><span class="en">End2End v3 Pooled IC</span></div>
    <div class="kpi"><b>1,144</b> · <span class="zh">基础因子字段</span><span class="en">baseline factor fields</span></div>
    <div class="kpi"><b>v3&gt;v2&gt;v1</b> · <span class="zh">验证→测试排序不变</span><span class="en">order holds val→test</span></div>
  </div>

  <div class="figs">
    <figure>
      <img src="summary_assets/fig_sum_monthly.png" alt="ensemble vs v3 monthly ic">
      <figcaption><span class="zh"><b>图 S1</b> ML Ensemble 与 End2End v3 的 2020 逐月 Pooled IC。</span><span class="en"><b>Fig S1</b> Monthly Pooled IC across 2020 — ML Ensemble vs End2End v3.</span></figcaption>
    </figure>
    <figure>
      <img src="summary_assets/fig_sum_bin.png" alt="ensemble vs v3 buckets">
      <figcaption><span class="zh"><b>图 S2</b> 20 桶收益单调性：Ensemble（柱）与 v3（线）。</span><span class="en"><b>Fig S2</b> 20-bucket return monotonicity — Ensemble (bars) vs v3 (line).</span></figcaption>
    </figure>
    <figure class="wide">
      <img src="summary_assets/fig_sum_allmodels.png" alt="all model ic icir">
      <figcaption><span class="zh"><b>图 S3</b> 全体模型在 2020 测试上的对比：左为 Pooled IC，右为月度 ICIR（mean/std）。</span><span class="en"><b>Fig S3</b> All-model comparison on 2020 test: Pooled IC (left) and monthly ICIR = mean/std (right).</span></figcaption>
    </figure>
  </div>
</section>
"""

def build_lang(lang):
    with open(SRC, encoding="utf-8") as f:
        html = f.read()

    # 1) inject exec-summary CSS before </style>
    html = html.replace("</style>", EXEC_CSS + "</style>", 1)
    # 2) fix language on <body>
    html = re.sub(r'<body class="lang-\w+">', f'<body class="lang-{lang}">', html, count=1)
    # 3) drop the sticky topbar (title + toggle)
    html = re.sub(r'<div class="topbar">.*?</div>\s*</div>\s*</div>', '', html, count=1, flags=re.S)
    # 4) drop the language <script> at the end so the fixed class sticks
    html = re.sub(r'<script>.*?</script>', '', html, count=1, flags=re.S)
    # 5) insert the exec-summary block before the Contents nav
    html = html.replace('<nav class="toc">', EXEC_BLOCK + '\n<nav class="toc">', 1)
    # 6) repoint the factor link to the public page
    html = html.replace('href="factors.html"', f'href="{FACTORS_URL}"')
    # 7) self-contain: inline all local images
    html = _inline_images(html)

    out = os.path.join(OUTDIR, f"summary_{lang}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote {os.path.relpath(out, ROOT)}  ({len(html)/1e6:.2f} MB)")
    return out

if __name__ == "__main__":
    for lang in ("en", "zh"):
        build_lang(lang)
    print("HTML exports done.")
