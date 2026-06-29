#!/usr/bin/env python3
"""Inline all local <img src="..."> references in summary_src.html as base64 data URIs,
producing a fully self-contained, portable single-file summary.html.

Large diagrams are downscaled (max width 1500px) and PNG-optimized to keep the
output reasonable. Author: Shuo Yin <yins25@mails.tsinghua.edu.cn>
"""
import os
import re
import io
import base64
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "summary_src.html")
OUT = os.path.join(ROOT, "summary.html")
MAX_W = 1500

with open(SRC, "r", encoding="utf-8") as f:
    html = f.read()

cache = {}

def encode(path):
    if path in cache:
        return cache[path]
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        raise FileNotFoundError(full)
    im = Image.open(full).convert("RGBA")
    if im.width > MAX_W:
        h = round(im.height * MAX_W / im.width)
        im = im.resize((MAX_W, h), Image.LANCZOS)
    # flatten onto white to drop alpha bloat
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[-1])
    buf = io.BytesIO()
    bg.save(buf, format="PNG", optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    uri = "data:image/png;base64," + data
    cache[path] = uri
    print(f"  inlined {path:70s} {len(data)//1024:5d} KB")
    return uri

def repl(m):
    src = m.group(1)
    if src.startswith("data:") or src.startswith("http"):
        return m.group(0)
    return 'src="' + encode(src) + '"'

print("Inlining images into standalone summary.html ...")
html2 = re.sub(r'src="([^"]+)"', repl, html)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html2)

size_mb = os.path.getsize(OUT) / 1e6
print(f"\nWROTE {os.path.relpath(OUT, ROOT)}  ({size_mb:.2f} MB, {len(cache)} images embedded)")
