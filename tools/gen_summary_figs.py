#!/usr/bin/env python3
"""Generate supplementary visualization assets for the Jump take-home summary report.

All chart labels are in English (the surrounding HTML report is bilingual) to avoid
CJK font dependencies. Figures are saved to ../summary_assets/ as PNG.

Author: Shuo Yin <yins25@mails.tsinghua.edu.cn>
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#9aa0a6",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6e8eb",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "summary_assets")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Color palette
# ----------------------------------------------------------------------------
C = {
    "ridge":   "#94a3b8",
    "lgb":     "#38bdf8",
    "mlp":     "#34d399",
    "ens":     "#f59e0b",
    "single":  "#a78bfa",
    "v1":      "#fca5a5",
    "v2":      "#f87171",
    "v3":      "#dc2626",
    "thresh":  "#475569",
    "ink":     "#1f2937",
    "sn":      "#6366f1",
    "pooled":  "#0ea5e9",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(p, os.path.join(HERE, "..")))


# ============================================================================
# DATA (2020 test window unless noted) — exact values from project metrics CSVs
# ============================================================================

# Cross-family headline (2020). (label, pooled_ic, sn_nonoverlap_ic, color, family)
HEADLINE = [
    ("Ridge\n(simplex)",        0.042481, 0.064183, C["ridge"],  "ML"),
    ("LightGBM",                0.050034, 0.065138, C["lgb"],    "ML"),
    ("MLP",                     0.050756, 0.065097, C["mlp"],    "ML"),
    ("ML Ensemble\n(strict stack)",0.057293, None, C["ens"],     "ML"),
    ("E2E single*",             0.039660, 0.031582, C["single"], "E2E"),
    ("E2E v1",                  0.043578, 0.059084, C["v1"],     "E2E"),
    ("E2E v2",                  0.048159, 0.061365, C["v2"],     "E2E"),
    ("E2E v3",                  0.054808, 0.061614, C["v3"],     "E2E"),
]

# Monthly IC across 2020 (pooled), 12 months
MONTHLY_2020 = {
    "MLP":      [0.08615,0.04571,0.05025,0.04232,0.06403,0.06003,0.05352,0.05315,0.05365,0.03850,0.03341,0.05474],
    "LightGBM": [0.07097,0.06677,0.03706,0.03651,0.06261,0.05550,0.06066,0.04511,0.06022,0.04772,0.04134,0.04736],
    "Ridge":    [0.07363,0.04297,0.05376,0.01566,0.05262,0.05108,0.04737,0.04608,0.05314,0.02571,0.02806,0.04245],
    "Ensemble": [0.08221,0.07054,0.05629,0.04452,0.07053,0.07094,0.07066,0.05849,0.05893,0.04927,0.04826,0.06012],
}

# 20-bucket mean forward return (in raw units). Convert to bps (x1e4).
BIN20 = {
    "Ensemble": [-0.0005685,-0.0003538,-0.0002486,-0.0001982,-0.0001800,-0.0001247,
                 -0.00009120,-0.00008956,-0.00006050,-0.00002835,-0.000005723,0.00001931,
                 0.00004121,0.00006592,0.00008997,0.0001034,0.0001505,0.0001728,0.0002303,0.0003485],
    "MLP": [-0.0005046,-0.0003138,-0.0002209,-0.0001764,-0.0001445,-0.0001094,
            -0.00007494,-0.00006634,-0.00004427,-0.00001238,-0.00001360,-0.000007579,
            0.00003297,0.00004709,0.00008455,0.00009501,0.0001334,0.0001415,0.0001925,0.0002803],
    "LightGBM": [-0.0004334,-0.0002894,-0.0002244,-0.0002002,-0.0001525,-0.0001232,
                 -0.00007486,-0.00007894,-0.00004688,-0.00004683,-0.00001187,0.000003989,
                 0.00003129,0.00005642,0.00008129,0.00008871,0.0001313,0.0001507,0.0001952,0.0002774],
    "Ridge": [-0.0004298,-0.0002505,-0.0001924,-0.0001530,-0.0001272,-0.0001182,
              -0.00006706,-0.00007156,-0.00004794,-0.00003710,-0.000009783,-0.000002826,
              0.00004025,0.00004468,0.00005823,0.00009302,0.0001131,0.0001365,0.0001661,0.0002241],
}

# Transformer ladder validation (2019) vs test (2020)
LADDER = {
    "v1": dict(val_p=0.054609, test_p=0.043578, val_sn=0.064411, test_sn=0.059084, params=5.50),
    "v2": dict(val_p=0.062069, test_p=0.048159, val_sn=0.069863, test_sn=0.061365, params=5.03),
    "v3": dict(val_p=0.064172, test_p=0.054808, val_sn=0.070858, test_sn=0.061614, params=5.23),
}

# Attempts NOT retained vs v3 (2019 validation): (label, pooled, sn, retained)
ABLATION = [
    ("v3 retained\n(FactorBank k=96, scaled)", 0.064172, 0.070858, True),
    ("LowRank residual\n(small-scale branch)", 0.064778, 0.069182, False),
    ("Conservative metadata\n(symbol+minute)", 0.060745, 0.070177, False),
    ("MoE head\n(2 experts)",                  0.060294, 0.064352, False),
    ("FactorBank k=160\n(no small scale)",     0.059374, 0.067541, False),
    ("MoE head\n(4 experts + balance)",        0.058668, 0.067202, False),
    ("Full metadata\n(sym+min+day+month)",     0.055533, 0.067231, False),
    ("LowRank interaction\n(replace input)",   0.054888, 0.064736, False),
]

# end2end_single feasibility monthly IC (2020-05..2020-12)
SINGLE_MONTHS = ["May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SINGLE_IC = [0.07208,0.03129,0.05838,0.04772,0.02811,0.02375,0.02147,0.04947]
SINGLE_BIN = [-0.0001826,-0.0001565,-0.0001598,-0.0001098,-0.00009449,-0.00007138,
              -0.00004086,-0.00003965,-0.00003564,-0.000008862,-0.000005930,-0.00001621,
              -0.00001369,0.00001683,0.000003159,0.00001775,0.00003996,0.00005481,0.00009075]

# End2End ladder (v1/v2/v3) on the full 2020 test, computed from the archived
# OOS prediction parquets. Monthly = pooled (cosine) IC per calendar month;
# full-year pooled IC ties out to 0.043578 / 0.048159 / 0.054808 respectively.
E2E_MONTHLY = {
    "v1": [0.06731,0.04604,0.01748,0.03818,0.05741,0.04926,0.04980,0.05862,0.04243,0.02297,0.04744,0.04707],
    "v2": [0.05138,0.04085,0.02488,0.05827,0.06800,0.05069,0.05467,0.05905,0.04664,0.03733,0.04339,0.04911],
    "v3": [0.04740,0.06076,0.03174,0.05422,0.06652,0.06624,0.05024,0.06048,0.06142,0.05173,0.05618,0.06110],
}
# 20-bucket mean realized 30-min return (raw units) on full 2020.
E2E_BIN = {
    "v1": [-0.000435198,-0.000254121,-0.00020305,-0.000135768,-0.000132068,-8.0089e-05,
           -6.4916e-05,-4.1691e-05,-1.5672e-05,2e-06,8.98e-06,2.859e-05,5.6809e-05,
           6.4825e-05,7.53e-05,0.00010301,0.000110436,0.000139218,0.000139441,0.000273725],
    "v2": [-0.000466124,-0.000277621,-0.000227107,-0.000167185,-0.000137922,-0.000106174,
           -7.7757e-05,-4.6757e-05,-2.411e-06,1.4612e-05,2.2642e-05,3.0138e-05,4.9032e-05,
           6.3026e-05,7.9695e-05,0.000113899,0.000114805,0.000151639,0.000190318,0.000319014],
    "v3": [-0.000517108,-0.000334827,-0.000243847,-0.000184464,-0.000148764,-0.000105786,
           -6.9312e-05,-5.1886e-05,-3.2922e-05,1.183e-06,5.333e-06,2.8065e-05,5.1908e-05,
           8.3252e-05,0.000106838,0.000123287,0.000156279,0.000169757,0.000249537,0.000353237],
}


# ============================================================================
# FIG 1 — Cross-family headline IC comparison (2020)
# ============================================================================
def fig_headline():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    labels = [h[0] for h in HEADLINE]
    pooled = [h[1] for h in HEADLINE]
    sn = [np.nan if h[2] is None else h[2] for h in HEADLINE]
    cols = [h[3] for h in HEADLINE]
    y = np.arange(len(labels))[::-1]
    bw = 0.38

    ax.barh(y + bw/2, pooled, height=bw, color=cols, edgecolor="white",
            linewidth=0.6, label="Pooled IC (headline metric)")
    ax.barh(y - bw/2, sn, height=bw, color=cols, alpha=0.45, edgecolor="white",
            linewidth=0.6, label="SN non-overlap IC (robustness)")

    for yi, p, s in zip(y, pooled, sn):
        ax.text(p + 0.0012, yi + bw/2, f"{p:.4f}", va="center", fontsize=8.5, color=C["ink"])
        if np.isfinite(s):
            ax.text(s + 0.0012, yi - bw/2, f"{s:.4f}", va="center", fontsize=8.5, color="#6b7280")
        else:
            ax.text(0.0012, yi - bw/2, "n/a", va="center", fontsize=8.5, color="#6b7280")

    ax.axvline(0.05, color=C["thresh"], ls="--", lw=1.4)
    ax.text(0.0508, len(labels)-0.35, "0.05",
            color=C["thresh"], fontsize=10, fontweight="bold", va="top", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Information Coefficient")
    ax.set_title("Cross-family model comparison on 2020 test")
    ax.set_xlim(0, 0.092)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9.5)
    ax.grid(axis="y", visible=False)
    fig.text(0.012, -0.085,
             "* E2E single is scored on 2020-05..2020-12; strict ML ensemble SN detail was not archived, so it is shown as n/a.",
             fontsize=8, color="#6b7280")
    save(fig, "fig1_headline.png")


# ============================================================================
# FIG 2 — Monthly (rolling) IC across 2020
# ============================================================================
def fig_monthly():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    order = [("Ridge", C["ridge"], "o", 1.6),
             ("LightGBM", C["lgb"], "s", 1.6),
             ("MLP", C["mlp"], "^", 1.6)]
    x = np.arange(12)
    for name, col, mk, lw in order:
        v = MONTHLY_2020[name]
        ir = np.mean(v) / np.std(v)
        ax.plot(x, v, marker=mk, color=col, lw=lw, ms=5,
                label=f"{name}  (mean/std = {ir:.2f})",
                zorder=5 if name == "Ensemble" else 3)
    ax.axhline(0.05, color=C["thresh"], ls="--", lw=1.2)
    ax.text(11.2, 0.0505, "0.05", color=C["thresh"], fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Pooled IC (per month)")
    ax.set_title("Monthly IC across 2020 — strict ML single models")
    ax.set_ylim(0, 0.095)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    save(fig, "fig2_monthly_ic.png")


# ============================================================================
# FIG 3 — 20-bucket monotonicity (economic significance)
# ============================================================================
def fig_bin20():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(1, 21)
    for name, col, mk in [("MLP", C["mlp"], "^"), ("LightGBM", C["lgb"], "s"), ("Ridge", C["ridge"], "o")]:
        ax.plot(x, np.array(BIN20[name]) * 1e4, marker=mk, ms=4, lw=1.4, color=col, label=name)
    ax.axhline(0, color="#374151", lw=0.9)
    ax.set_xticks(x)
    ax.set_xlabel("Prediction bucket (1 = most bearish → 20 = most bullish)")
    ax.set_ylabel("Mean realized 30-min return (bps)")
    ax.set_title("Single-model bucketed returns — strict ensemble bucket detail not archived")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig3_bin20.png")


# ============================================================================
# FIG 4 — Transformer ladder: validation -> test persistence
# ============================================================================
def fig_ladder():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    vers = ["v1", "v2", "v3"]
    vcol = [C["v1"], C["v2"], C["v3"]]
    x = np.arange(3)
    bw = 0.36

    # Panel A: pooled IC val vs test
    valp = [LADDER[v]["val_p"] for v in vers]
    testp = [LADDER[v]["test_p"] for v in vers]
    ax1.bar(x - bw/2, valp, bw, color=vcol, alpha=0.5, edgecolor="white", label="2019 validation")
    ax1.bar(x + bw/2, testp, bw, color=vcol, edgecolor="white", label="2020 test (OOS)")
    for xi, a, b in zip(x, valp, testp):
        ax1.text(xi - bw/2, a + 0.0008, f"{a:.4f}", ha="center", fontsize=8)
        ax1.text(xi + bw/2, b + 0.0008, f"{b:.4f}", ha="center", fontsize=8)
    ax1.axhline(0.05, color=C["thresh"], ls="--", lw=1.1)
    ax1.set_xticks(x); ax1.set_xticklabels(vers)
    ax1.set_ylabel("Pooled IC")
    ax1.set_title("Pooled IC: validation → test")
    ax1.set_ylim(0, 0.075)
    ax1.legend(fontsize=9, loc="upper left")

    # Panel B: SN non-overlap IC val vs test
    vals = [LADDER[v]["val_sn"] for v in vers]
    tests = [LADDER[v]["test_sn"] for v in vers]
    ax2.bar(x - bw/2, vals, bw, color=vcol, alpha=0.5, edgecolor="white", label="2019 validation")
    ax2.bar(x + bw/2, tests, bw, color=vcol, edgecolor="white", label="2020 test (OOS)")
    for xi, a, b in zip(x, vals, tests):
        ax2.text(xi - bw/2, a + 0.0008, f"{a:.4f}", ha="center", fontsize=8)
        ax2.text(xi + bw/2, b + 0.0008, f"{b:.4f}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(vers)
    ax2.set_ylabel("SN non-overlap IC")
    ax2.set_title("SN non-overlap IC: validation → test")
    ax2.set_ylim(0, 0.085)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle("End-to-end Transformer ladder — ordering v3 > v2 > v1 persists out of sample",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig4_ladder.png")


# ============================================================================
# FIG 5 — Ablation: attempts not retained vs v3 (2019 validation)
# ============================================================================
def fig_ablation():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    labels = [a[0] for a in ABLATION]
    pooled = [a[1] for a in ABLATION]
    sn = [a[2] for a in ABLATION]
    retained = [a[3] for a in ABLATION]
    # sort by pooled descending for visual
    order = np.argsort(pooled)
    labels = [labels[i] for i in order]
    pooled = [pooled[i] for i in order]
    sn = [sn[i] for i in order]
    retained = [retained[i] for i in order]
    y = np.arange(len(labels))
    bw = 0.38

    pcol = [C["v3"] if r else "#cbd5e1" for r in retained]
    scol = [C["sn"] if r else "#e2e8f0" for r in retained]
    ax.barh(y + bw/2, pooled, bw, color=pcol, edgecolor="white", label="Pooled IC")
    ax.barh(y - bw/2, sn, bw, color=scol, edgecolor="white", label="SN non-overlap IC")
    for yi, p, s in zip(y, pooled, sn):
        ax.text(p + 0.0004, yi + bw/2, f"{p:.4f}", va="center", fontsize=8)
        ax.text(s + 0.0004, yi - bw/2, f"{s:.4f}", va="center", fontsize=8)

    # reference lines = v3 retained values
    ax.axvline(0.064172, color=C["v3"], ls=":", lw=1.2)
    ax.axvline(0.070858, color=C["sn"], ls=":", lw=1.2)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Information Coefficient (2019 validation)")
    ax.set_title("End-to-end attempts on 2019 validation — v3 retained, others not adopted")
    ax.set_xlim(0.05, 0.076)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_ablation.png")


# ============================================================================
# FIG 6 — Stability (ICIR) of the transformer ladder
# ============================================================================
def fig_icir():
    # dense ICIR + SN ICIR for v1/v2/v3 on 2019 and 2020
    data = {
        "v1": dict(dense19=0.275, dense20=0.256, sn19=0.291, sn20=0.262),
        "v2": dict(dense19=0.308, dense20=0.260, sn19=0.328, sn20=0.276),
        "v3": dict(dense19=0.311, dense20=0.276, sn19=0.317, sn20=0.281),
    }
    vers = ["v1", "v2", "v3"]
    vcol = [C["v1"], C["v2"], C["v3"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(3); bw = 0.36
    d19 = [data[v]["dense19"] for v in vers]
    d20 = [data[v]["dense20"] for v in vers]
    ax1.bar(x - bw/2, d19, bw, color=vcol, alpha=0.5, edgecolor="white", label="2019 val")
    ax1.bar(x + bw/2, d20, bw, color=vcol, edgecolor="white", label="2020 test")
    for xi, a, b in zip(x, d19, d20):
        ax1.text(xi - bw/2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
        ax1.text(xi + bw/2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(vers)
    ax1.set_title("Dense cross-sectional ICIR")
    ax1.set_ylabel("ICIR = mean(IC)/std(IC) per timestamp")
    ax1.set_ylim(0, 0.38); ax1.legend(fontsize=9)

    s19 = [data[v]["sn19"] for v in vers]
    s20 = [data[v]["sn20"] for v in vers]
    ax2.bar(x - bw/2, s19, bw, color=vcol, alpha=0.5, edgecolor="white", label="2019 val")
    ax2.bar(x + bw/2, s20, bw, color=vcol, edgecolor="white", label="2020 test")
    for xi, a, b in zip(x, s19, s20):
        ax2.text(xi - bw/2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
        ax2.text(xi + bw/2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(vers)
    ax2.set_title("SN non-overlap ICIR")
    ax2.set_ylim(0, 0.38); ax2.legend(fontsize=9)
    fig.suptitle("Signal stability improves along the ladder (higher = steadier)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, "fig6_icir.png")


# ============================================================================
# FIG 7 — end2end_single feasibility
# ============================================================================
def fig_single():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # ----- Panel 1: monthly IC (May..Dec), E2E single vs ML singles + ensemble -----
    # The ML single/ensemble monthly series are full-year; slice to May..Dec (idx 4..11)
    # so they align with the E2E single's rolling window.
    x = np.arange(len(SINGLE_MONTHS))
    overlay = [("MLP", C["mlp"], "^"), ("LightGBM", C["lgb"], "s"),
               ("Ridge", C["ridge"], "o"), ("Ensemble", C["ens"], "D")]
    for name, col, mk in overlay:
        v = np.array(MONTHLY_2020[name])[4:]
        ax1.plot(x, v, marker=mk, ms=4, lw=1.4, color=col, alpha=0.9, label=name)
    ax1.plot(x, SINGLE_IC, marker="o", color=C["single"], lw=2.6, ms=7,
             label="E2E single", zorder=6)
    ax1.fill_between(x, SINGLE_IC, 0, color=C["single"], alpha=0.10)
    ax1.axhline(0.05, color=C["thresh"], ls="--", lw=1.1)
    ax1.text(len(x) - 0.8, 0.0505, "0.05", color=C["thresh"], fontsize=9, va="bottom", ha="right")
    ax1.set_xticks(x); ax1.set_xticklabels(SINGLE_MONTHS)
    ax1.set_ylabel("Pooled IC")
    ax1.set_title("Monthly IC (2020-05..12): E2E single vs baselines")
    ax1.set_ylim(0, 0.095)
    ax1.legend(loc="upper right", ncol=2, fontsize=8.2)

    # ----- Panel 2: bucketed returns — E2E single as bars, others as lines -----
    xb = np.arange(1, len(SINGLE_BIN) + 1)
    ax2.bar(xb, np.array(SINGLE_BIN) * 1e4, color=C["single"], alpha=0.55,
            edgecolor="white", linewidth=0.4, label="E2E single (bars)", zorder=2)
    xl = np.arange(1, 21)
    for name, col, mk in [("MLP", C["mlp"], "^"), ("LightGBM", C["lgb"], "s"),
                          ("Ridge", C["ridge"], "o"), ("Ensemble", C["ens"], "D")]:
        ax2.plot(xl, np.array(BIN20[name]) * 1e4, marker=mk, ms=3.5, lw=1.4,
                 color=col, alpha=0.9, label=name, zorder=4)
    ax2.axhline(0, color="#374151", lw=0.9)
    ax2.set_xticks([1, 5, 10, 15, 20])
    ax2.set_xlabel("Prediction bucket (1 = most bearish → top = most bullish)")
    ax2.set_ylabel("Mean 30-min return (bps)")
    ax2.set_title("Bucketed return monotonicity vs baselines")
    ax2.legend(loc="upper left", fontsize=8.2)
    fig.suptitle("End-to-end feasibility check passed before scaling up — tracks the ML baselines",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, "fig7_single.png")


# ============================================================================
# FIG 9 — End2End ladder (v1/v2/v3) vs Ensemble, in the style of Fig 3.1
#   E2E models drawn as lines; the Ensemble (strongest baseline) drawn with a
#   different encoding — dashed line for monthly IC, bars for the buckets.
# ============================================================================
def fig_e2e_compare():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    e2e = [("E2E v1", "v1", C["v1"], "o"),
           ("E2E v2", "v2", C["v2"], "s"),
           ("E2E v3", "v3", C["v3"], "^")]

    # ----- Panel 1: monthly IC across 2020 — E2E as lines, Ensemble dashed -----
    x = np.arange(12)
    for name, key, col, mk in e2e:
        v = E2E_MONTHLY[key]
        ir = np.mean(v) / np.std(v)
        lw = 2.6 if key == "v3" else 1.6
        ax1.plot(x, v, marker=mk, ms=5, lw=lw, color=col,
                 label=f"{name}  (mean/std = {ir:.2f})", zorder=5 if key == "v3" else 3)
    ens = MONTHLY_2020["Ensemble"]
    ax1.plot(x, ens, marker="D", ms=5, lw=2.0, color=C["ens"], ls="--",
             label=f"Ensemble baseline  (mean/std = {np.mean(ens)/np.std(ens):.2f})", zorder=6)
    ax1.axhline(0.05, color=C["thresh"], ls="--", lw=1.1)
    ax1.text(11.2, 0.0505, "0.05", color=C["thresh"], fontsize=9, va="bottom", ha="right")
    ax1.set_xticks(x); ax1.set_xticklabels(MONTHS)
    ax1.set_ylabel("Pooled IC (per month)")
    ax1.set_title("Monthly IC across 2020: End2End ladder vs Ensemble")
    ax1.set_ylim(0, 0.095)
    ax1.legend(loc="upper right", ncol=1, fontsize=8.4)

    # ----- Panel 2: 20-bucket returns — Ensemble as bars, E2E as lines -----
    xb = np.arange(1, 21)
    ax2.bar(xb, np.array(BIN20["Ensemble"]) * 1e4, color=C["ens"], alpha=0.5,
            edgecolor="white", linewidth=0.4, label="Ensemble baseline (bars)", zorder=2)
    for name, key, col, mk in e2e:
        lw = 2.4 if key == "v3" else 1.4
        ax2.plot(xb, np.array(E2E_BIN[key]) * 1e4, marker=mk, ms=3.8, lw=lw,
                 color=col, alpha=0.95, label=name, zorder=5 if key == "v3" else 4)
    ax2.axhline(0, color="#374151", lw=0.9)
    ax2.set_xticks([1, 5, 10, 15, 20])
    ax2.set_xlabel("Prediction bucket (1 = most bearish → 20 = most bullish)")
    ax2.set_ylabel("Mean 30-min return (bps)")
    ax2.set_title("Bucketed return monotonicity vs Ensemble")
    ax2.legend(loc="upper left", fontsize=8.4)
    fig.suptitle("End-to-end ladder approaches the Ensemble — v3 closes most of the gap (2020 test)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, "fig9_e2e_compare.png")


# ============================================================================
# FIG 8 — Ensemble lift vs best single (waterfall-ish)
# ============================================================================
def fig_lift():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    names = ["Ridge", "LightGBM", "MLP", "Ensemble"]
    vals = [0.042481, 0.050034, 0.050756, 0.057293]
    cols = [C["ridge"], C["lgb"], C["mlp"], C["ens"]]
    x = np.arange(4)
    bars = ax.bar(x, vals, color=cols, edgecolor="white", width=0.62)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.0009, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.05, color=C["thresh"], ls="--", lw=1.2)
    ax.text(3.45, 0.0505, "0.05", color=C["thresh"], fontsize=9, va="bottom", ha="right")
    # lift annotation
    ax.annotate("", xy=(3, 0.057293), xytext=(3, 0.050756),
                arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.3))
    ax.text(3.08, 0.0540, "+0.0065\n(+12.9%)", fontsize=9, color=C["ink"], va="center")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Pooled IC (2020)")
    ax.set_title("Ensemble lift over the best single ML model")
    ax.set_ylim(0, 0.07)
    save(fig, "fig8_lift.png")


# ============================================================================
# REPORT-SUMMARY FIGURES (used in the export/PDF summary block)
#   S1: Ensemble vs End2End v3 — monthly IC (2020)
#   S2: Ensemble vs End2End v3 — 20-bucket return monotonicity
#   S3: All-model comparison — Pooled IC and monthly ICIR
# ============================================================================
def _icir(series):
    return float(np.mean(series) / np.std(series))


def fig_sum_monthly():
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    x = np.arange(12)
    v3 = E2E_MONTHLY["v3"]; ens = MONTHLY_2020["Ensemble"]
    ax.plot(x, ens, marker="D", ms=5.5, lw=2.2, color=C["ens"], ls="--",
            label=f"ML Ensemble  (ICIR {_icir(ens):.2f})", zorder=4)
    ax.plot(x, v3, marker="^", ms=6, lw=2.6, color=C["v3"],
            label=f"End2End v3  (ICIR {_icir(v3):.2f})", zorder=5)
    ax.axhline(0.05, color=C["thresh"], ls="--", lw=1.1)
    ax.text(11.2, 0.0505, "0.05", color=C["thresh"], fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Pooled IC (per month)")
    ax.set_title("Monthly IC across 2020 — Ensemble vs End2End v3")
    ax.set_ylim(0, 0.095)
    ax.legend(loc="upper right", fontsize=9.5)
    save(fig, "fig_sum_monthly.png")


def fig_sum_bin():
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    xb = np.arange(1, 21)
    ax.bar(xb, np.array(BIN20["Ensemble"]) * 1e4, color=C["ens"], alpha=0.55,
           edgecolor="white", linewidth=0.4, label="ML Ensemble (bars)", zorder=2)
    ax.plot(xb, np.array(E2E_BIN["v3"]) * 1e4, marker="^", ms=4.5, lw=2.4,
            color=C["v3"], label="End2End v3 (line)", zorder=5)
    ax.axhline(0, color="#374151", lw=0.9)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Prediction bucket (1 = most bearish → 20 = most bullish)")
    ax.set_ylabel("Mean 30-min return (bps)")
    ax.set_title("Bucketed return monotonicity — Ensemble vs End2End v3")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, "fig_sum_bin.png")


def fig_sum_allmodels():
    # (label, pooled_ic, monthly_series_key, source_dict, family)
    rows = [
        ("Ridge",        0.042481, "Ridge",    MONTHLY_2020, "ML"),
        ("LightGBM",     0.050034, "LightGBM", MONTHLY_2020, "ML"),
        ("MLP",          0.050756, "MLP",      MONTHLY_2020, "ML"),
        ("ML Ensemble",  0.057293, "Ensemble", MONTHLY_2020, "ENS"),
        ("E2E v1",       0.043578, "v1",       E2E_MONTHLY,  "E2E"),
        ("E2E v2",       0.048159, "v2",       E2E_MONTHLY,  "E2E"),
        ("E2E v3",       0.054808, "v3",       E2E_MONTHLY,  "E2E"),
    ]
    famcol = {"ML": C["lgb"], "ENS": C["ens"], "E2E": C["v3"]}
    labels = [r[0] for r in rows]
    ic = [r[1] for r in rows]
    icir = [_icir(r[3][r[2]]) for r in rows]
    cols = [famcol[r[4]] for r in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0))
    ax1.bar(x, ic, color=cols, edgecolor="white", width=0.66)
    for xi, v in zip(x, ic):
        ax1.text(xi, v + 0.0009, f"{v:.4f}", ha="center", fontsize=8.6, fontweight="bold")
    ax1.axhline(0.05, color=C["thresh"], ls="--", lw=1.1)
    ax1.text(len(rows)-0.5, 0.0505, "0.05", color=C["thresh"], fontsize=9, va="bottom", ha="right")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax1.set_ylabel("Pooled IC (2020)")
    ax1.set_title("Pooled IC — all models")
    ax1.set_ylim(0, 0.066)

    ax2.bar(x, icir, color=cols, edgecolor="white", width=0.66)
    for xi, v in zip(x, icir):
        ax2.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=8.6, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax2.set_ylabel("Monthly ICIR = mean(IC)/std(IC)")
    ax2.set_title("Monthly ICIR — all models")
    ax2.set_ylim(0, max(icir) * 1.18)

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=C["lgb"], label="ML single"),
               mpatches.Patch(color=C["ens"], label="ML Ensemble"),
               mpatches.Patch(color=C["v3"], label="End2End")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("All-model comparison on 2020 test — Pooled IC and stability (ICIR)",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig_sum_allmodels.png")


if __name__ == "__main__":
    fig_headline()
    fig_monthly()
    fig_bin20()
    fig_ladder()
    fig_ablation()
    fig_icir()
    fig_single()
    fig_e2e_compare()
    fig_lift()
    fig_sum_monthly()
    fig_sum_bin()
    fig_sum_allmodels()
    print("ALL FIGURES DONE ->", os.path.relpath(OUT, os.path.join(HERE, "..")))
