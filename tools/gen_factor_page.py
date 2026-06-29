#!/usr/bin/env python3
"""Generate factors.html — a standalone, searchable reference page listing the
1,144 retained baseline factor fields (factor name, cross-sectional view, base
factor, family). Linked from the Baseline Features section of the summary.

Source: ../../autodl-tmp/quant/artifacts/factor_catalog.csv (factor,view,base,family)
Author: Shuo Yin <yins25@mails.tsinghua.edu.cn>
"""
import os
import csv
import html

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# Prefer a repo-local copy if present, else fall back to the artifacts path.
CANDIDATES = [
    os.path.join(ROOT, "summary_assets", "factor_catalog.csv"),
    "/root/autodl-tmp/quant/artifacts/factor_catalog.csv",
]
SRC = next((p for p in CANDIDATES if os.path.isfile(p)), CANDIDATES[-1])
OUT = os.path.join(ROOT, "factors.html")

rows = []
with open(SRC, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((r["factor"], r["view"], r["base"], r["family"]))

families = sorted({r[3] for r in rows})
views = {}
for r in rows:
    views[r[1]] = views.get(r[1], 0) + 1
view_summary = " · ".join(f"{k} ({v})" for k, v in sorted(views.items(), key=lambda x: -x[1]))

body_rows = []
for i, (factor, view, base, family) in enumerate(rows, 1):
    hay = f"{factor} {view} {base} {family}".lower()
    body_rows.append(
        f'<tr data-h="{html.escape(hay, quote=True)}" data-fam="{html.escape(family, quote=True)}">'
        f'<td class="idx">{i}</td>'
        f'<td class="fac"><code>{html.escape(factor)}</code></td>'
        f'<td>{html.escape(view)}</td>'
        f'<td><code>{html.escape(base)}</code></td>'
        f'<td><span class="fam">{html.escape(family)}</span></td></tr>'
    )

chips = ['<button class="chip active" data-fam="*">All ({})</button>'.format(len(rows))]
for fam in families:
    n = sum(1 for r in rows if r[3] == fam)
    chips.append('<button class="chip" data-fam="{0}">{0} ({1})</button>'.format(html.escape(fam), n))

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cn-future-alpha · 1,144 Baseline Factor Fields</title>
<style>
  :root{{--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--soft:#f8fafc;--soft2:#f1f5f9;
        --brand:#0f3d6e;--brand2:#0ea5e9;--maxw:1060px;
        --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue","PingFang SC","Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;
        --mono:"SF Mono",ui-monospace,Menlo,Consolas,"Courier New",monospace;}}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:var(--sans);color:var(--ink);background:#fff;line-height:1.6;font-size:16px}}
  .wrap{{max-width:var(--maxw);margin:0 auto;padding:0 22px}}
  header{{padding:34px 0 16px;border-bottom:1px solid var(--line)}}
  h1{{font-size:24px;margin:0 0 .25em;color:var(--brand)}}
  .sub{{color:var(--muted);font-size:14.5px}}
  a{{color:var(--brand2);text-decoration:none}} a:hover{{text-decoration:underline}}
  .back{{display:inline-block;margin-top:10px;font-size:14px}}
  .tools{{position:sticky;top:0;background:rgba(255,255,255,.95);backdrop-filter:blur(6px);
         padding:14px 0 10px;border-bottom:1px solid var(--line);z-index:10}}
  input[type=search]{{width:100%;padding:10px 13px;font-size:15px;border:1px solid #cbd5e1;
         border-radius:9px;font-family:inherit;outline:none}}
  input[type=search]:focus{{border-color:var(--brand2);box-shadow:0 0 0 3px rgba(14,165,233,.15)}}
  .chips{{display:flex;flex-wrap:wrap;gap:7px;margin-top:11px}}
  .chip{{border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:999px;
        padding:4px 11px;font-size:12.5px;cursor:pointer;font-family:inherit}}
  .chip.active{{background:var(--brand);color:#fff;border-color:var(--brand)}}
  .count{{font-size:13px;color:var(--muted);margin:12px 0 6px}}
  table{{border-collapse:collapse;width:100%;font-size:13.5px}}
  th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
  thead th{{position:sticky;top:84px;background:var(--soft2);color:#374151;font-weight:700;
           border-bottom:2px solid #cbd5e1;z-index:5}}
  tbody tr:hover{{background:var(--soft)}}
  td.idx{{color:#9ca3af;font-variant-numeric:tabular-nums;text-align:right;width:48px}}
  td.fac code{{color:#0b3a66}}
  code{{font-family:var(--mono);background:var(--soft2);padding:1px 5px;border-radius:4px;font-size:.88em}}
  td code{{background:transparent;padding:0}}
  .fam{{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:999px;
       background:#ecfeff;color:#0e7490;font-weight:600}}
  .nores{{padding:24px 0;color:var(--muted)}}
  footer{{border-top:1px solid var(--line);padding:20px 0 40px;margin-top:24px;color:var(--muted);font-size:13.5px}}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>1,144 Baseline Factor Fields</h1>
    <div class="sub">cn-future-alpha · the retained leak-free factor library after 0.9-correlation dedup.<br>
      Views: {view_summary}.</div>
    <a class="back" href="summary.html">← back to the summary report</a>
  </div>
</header>

<div class="tools">
  <div class="wrap">
    <input type="search" id="q" placeholder="Search factor / base / family (e.g. macd, csz, volatility, oi) …" autocomplete="off">
    <div class="chips" id="chips">{chips}</div>
  </div>
</div>

<main class="wrap">
  <div class="count" id="count"></div>
  <table>
    <thead><tr><th>#</th><th>Factor field</th><th>View</th><th>Base factor</th><th>Family</th></tr></thead>
    <tbody id="tb">
{body}
    </tbody>
  </table>
  <div class="nores" id="nores" style="display:none">No factors match your filter.</div>
</main>

<footer><div class="wrap">Generated from <code>factor_catalog.csv</code> · {n} fields · cn-future-alpha.</div></footer>

<script>
  var rowsEl = Array.prototype.slice.call(document.querySelectorAll('#tb tr'));
  var q = document.getElementById('q');
  var countEl = document.getElementById('count');
  var nores = document.getElementById('nores');
  var fam = '*';
  function apply(){{
    var s = q.value.trim().toLowerCase();
    var shown = 0;
    for (var i=0;i<rowsEl.length;i++){{
      var tr = rowsEl[i];
      var okF = (fam==='*') || (tr.getAttribute('data-fam')===fam);
      var okS = !s || tr.getAttribute('data-h').indexOf(s) !== -1;
      var vis = okF && okS;
      tr.style.display = vis ? '' : 'none';
      if (vis) shown++;
    }}
    countEl.textContent = 'Showing ' + shown + ' of ' + rowsEl.length + ' factor fields';
    nores.style.display = shown ? 'none' : 'block';
  }}
  q.addEventListener('input', apply);
  document.getElementById('chips').addEventListener('click', function(e){{
    if (e.target.classList.contains('chip')){{
      fam = e.target.getAttribute('data-fam');
      var cs = document.querySelectorAll('.chip');
      for (var i=0;i<cs.length;i++) cs[i].classList.remove('active');
      e.target.classList.add('active');
      apply();
    }}
  }});
  apply();
</script>
</body>
</html>
""".format(view_summary=html.escape(view_summary),
           chips="\n".join(chips),
           body="\n".join(body_rows),
           n=len(rows))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
# keep a repo-local copy of the source CSV so the page is reproducible in-repo
local_csv = os.path.join(ROOT, "summary_assets", "factor_catalog.csv")
if not os.path.isfile(local_csv):
    with open(SRC, encoding="utf-8") as a, open(local_csv, "w", encoding="utf-8") as b:
        b.write(a.read())
print("WROTE", os.path.relpath(OUT, ROOT), "with", len(rows), "factors;", len(families), "families")
